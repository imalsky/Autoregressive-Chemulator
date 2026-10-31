// XLA FFI handlers for the solver's call and its derivatives: "kinjax_call" (the call kernel of lib/kincuda.py, its
// guard and the device-side recovery of the flagged cells; or, when per-cell log rate multipliers are given, the
// tangent kernel with no directions, the guard and jvp_rec, the recovery's path with the multipliers),
// "kinjax_jvp" (the tangent kernel of kincuda_ad, up to DMAX directions per launch, the guard and jvp_rec for the
// flagged cells) and "kinjax_vjp" (per cotangent the adjoint kernel, the guard and vjp_rec, then the Arrhenius-table
// reduction when its cotangent is requested). Each enqueues on the CUDA stream XLA hands it and never synchronises;
// every buffer a kernel writes is an output of the call (XLA allocates it per invocation). CUDA driver API through
// dlopen, so the build needs only g++, cuda.h and the XLA FFI headers.
//
// Python side (kinjax.py): KinjaxDevice reads the device's limits; KinjaxLoad loads one cubin (kind 0 call + guard +
// recover, 1 jvp + guard + jvp_rec, 2 vjp + guard + vjp_rec + arrhg) into the device's primary context under a handle
// shared by the three kinds; the handlers take that handle as the attribute "module". Loaded modules are read-only after
// KinjaxLoad.
//
// Optional operands are empty arrays when absent: lm (per-cell log multipliers [B, nr]), the tangents of lm [D, B, nr]
// and of rc [D, nr, 9], the cotangent of h_out [D, B]. Optional outputs are empty when not requested: the cotangents of
// lm [D, B, nr] (then glm_scratch [B nr] holds the kernel's per-cell values) and of rc [D, groups, nr, 9] (then the
// reduction is not launched).
#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>

#include "cuda.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kDmax = 16;   // tangent directions per launch of the jvp kernel (kincuda_ad.DMAX)

struct Driver {
  decltype(&cuInit) Init;
  decltype(&cuDeviceGet) DeviceGet;
  decltype(&cuDeviceGetAttribute) DeviceGetAttribute;
  decltype(&cuDevicePrimaryCtxRetain) PrimaryCtxRetain;
  decltype(&cuCtxPushCurrent_v2) CtxPush;
  decltype(&cuCtxPopCurrent_v2) CtxPop;
  decltype(&cuCtxGetCurrent) CtxGetCurrent;
  decltype(&cuModuleLoadData) ModuleLoadData;
  decltype(&cuModuleGetFunction) ModuleGetFunction;
  decltype(&cuFuncSetAttribute) FuncSetAttribute;
  decltype(&cuOccupancyMaxActiveBlocksPerMultiprocessor) Occupancy;
  decltype(&cuLaunchKernel) LaunchKernel;
  decltype(&cuMemsetD32Async) MemsetD32Async;
  decltype(&cuMemcpyDtoDAsync_v2) MemcpyDtoDAsync;
  decltype(&cuGetErrorString) GetErrorString;
  std::string error;   // empty when every symbol was found
};

const Driver& drv() {
  static Driver d = [] {
    Driver r{};
    void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) { r.error = "cannot load libcuda.so.1"; return r; }
    auto get = [&](auto& fn, const char* name) {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(h, name));
      if (!fn) r.error += std::string(" missing ") + name;
    };
    get(r.Init, "cuInit");
    get(r.DeviceGet, "cuDeviceGet");
    get(r.DeviceGetAttribute, "cuDeviceGetAttribute");
    get(r.PrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    get(r.CtxPush, "cuCtxPushCurrent_v2");
    get(r.CtxPop, "cuCtxPopCurrent_v2");
    get(r.CtxGetCurrent, "cuCtxGetCurrent");
    get(r.ModuleLoadData, "cuModuleLoadData");
    get(r.ModuleGetFunction, "cuModuleGetFunction");
    get(r.FuncSetAttribute, "cuFuncSetAttribute");
    get(r.Occupancy, "cuOccupancyMaxActiveBlocksPerMultiprocessor");
    get(r.LaunchKernel, "cuLaunchKernel");
    get(r.MemsetD32Async, "cuMemsetD32Async");
    get(r.MemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2");
    get(r.GetErrorString, "cuGetErrorString");
    if (r.error.empty() && r.Init(0) != CUDA_SUCCESS) r.error = "cuInit failed";
    return r;
  }();
  return d;
}

std::string cu_error(CUresult r, const char* what) {
  const char* s = nullptr;
  drv().GetErrorString(r, &s);
  return std::string("CUDA driver, ") + what + ": error " + std::to_string(static_cast<int>(r)) + " (" +
         (s ? s : "") + ")";
}

const char* const kMain[3] = {"call", "jvp", "vjp"};
const char* const kRec[3] = {"recover", "jvp_rec", "vjp_rec"};

struct Module {          // one cubin of one kind loaded into one device's primary context
  int64_t id;
  int device, kind;
  CUcontext ctx;
  CUmodule mod;
  CUfunction main, guard, rec, aux;       // rec: the recovery's kernel; aux: arrhg (kind 2)
  int ns, nr, cells, threads, smem;       // the main kernel: cells and threads per block, dynamic shared memory
  int rec_threads, rec_smem;              // the recovery's kernel: threads per block, shared memory (kinds 1, 2: the
                                          // main kernel's; it runs on the main kernel's grid and scratch slots)
  int sround, jround, ground, nl;         // rounds of the S v, Jacobian and gather streams; lanes of the first two
  int max_blocks, rec_blocks;             // resident blocks on the device
};

std::mutex registry_mutex;
std::deque<Module> registry;   // append-only; a handle names one network's three cubins, loaded once per device

void set_message(char* err, int errlen, const std::string& m) {
  if (err && errlen > 0) std::snprintf(err, errlen, "%s", m.c_str());
}

bool find(int64_t id, int device, int kind, Module& m) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  for (const Module& q : registry)
    if (q.id == id && q.device == device && q.kind == kind) { m = q; return true; }
  return false;
}

// The module's context made current for the handler's scope (XLA's is normally current already).
struct Scope {
  bool push = false;
  CUresult r = CUDA_SUCCESS;
  explicit Scope(CUcontext ctx) {
    CUcontext cur = nullptr;
    drv().CtxGetCurrent(&cur);
    push = cur != ctx;
    if (push) r = drv().CtxPush(ctx);
  }
  ~Scope() {
    if (push && r == CUDA_SUCCESS) {
      CUcontext popped;
      drv().CtxPop(&popped);
    }
  }
};

void* ptr(const ffi::AnyBuffer& b) { return b.element_count() ? b.untyped_data() : nullptr; }

CUresult launch(CUfunction f, unsigned gx, unsigned gy, unsigned block, unsigned smem, CUstream s, void** args) {
  return drv().LaunchKernel(f, gx, gy, 1, block, 1, 1, smem, s, args, nullptr);
}

// The operands every handler takes: the call's inputs and the mechanism tables.
struct Primal {
  ffi::AnyBuffer x, T, P, dt, h, lm, rc, perm, blob, rf, nasa, sst, jt, lu, optr, atoms, gt, xg, didx;
};

ffi::Error check_primal(const Module& m, const Primal& a, int64_t& B) {
  const auto xd = a.x.dimensions();
  B = xd.size() == 2 ? xd[0] : -1;
  if (B < 0 || xd[1] != m.ns) return ffi::Error::InvalidArgument("kinjax: x must be [B, ns]");
  if (B > INT32_MAX) return ffi::Error::InvalidArgument("kinjax: more than 2^31 - 1 cells");
  for (const ffi::AnyBuffer* q : {&a.T, &a.P, &a.dt, &a.h})
    if (q->element_count() != static_cast<size_t>(B)) return ffi::Error::InvalidArgument("kinjax: T, P, dt, h [B]");
  if (a.lm.element_count() && a.lm.element_count() != static_cast<size_t>(B) * m.nr)
    return ffi::Error::InvalidArgument("kinjax: lm must be [B, nr] or empty");
  if (a.sst.element_count() != static_cast<size_t>(m.sround) * m.nl ||
      2 * a.jt.element_count() < static_cast<size_t>(m.jround) * m.nl ||
      a.gt.element_count() != static_cast<size_t>(m.ground) * 32 || a.rc.element_count() != 9u * m.nr)
    return ffi::Error::InvalidArgument("kinjax: tables do not match the module");
  return ffi::Error::Success();
}

int blocks(const Module& m, int64_t B) {
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>((B + m.cells - 1) / m.cells, m.max_blocks)));
}

// One launch of the jvp kernel (rec false) or of jvp_rec over the guard's list (rec true; max_steps = the recovery's
// tries) for D directions (D = 0: the call alone) with the tangent pointers given.
CUresult launch_jvp(const Module& m, bool rec, CUstream s, const Primal& a, int B, void* dx, void* dT, void* dP,
                    void* ddt, void* dh, void* dlm, void* drc, int D, void* xo, void* ho, void* done, void* nok,
                    void* dxo, void* dho, void* count, double rtol, double atol, int max_steps, int adapt,
                    void* scratch, void* redo, void* list, void* wscr) {
  void *px = a.x.untyped_data(), *pT = a.T.untyped_data(), *pP = a.P.untyped_data(), *pdt = a.dt.untyped_data(),
       *ph = a.h.untyped_data(), *plm = ptr(a.lm), *prc = a.rc.untyped_data(), *pperm = a.perm.untyped_data(),
       *pblob = a.blob.untyped_data(), *prf = a.rf.untyped_data(), *pnasa = a.nasa.untyped_data(),
       *psst = a.sst.untyped_data(), *pjt = a.jt.untyped_data(), *plu = a.lu.untyped_data(),
       *poptr = a.optr.untyped_data(), *pdidx = a.didx.untyped_data(), *patoms = a.atoms.untyped_data();
  int sround = m.sround, jround = m.jround;
  void* args[] = {&px,  &pT,   &pP,  &pdt,   &ph,    &plm,  &prc,    &dx,    &dT,    &dP,    &ddt,  &dh,   &dlm,
                  &drc, &D,    &xo,  &ho,    &done,  &nok,  &dxo,    &dho,   &count, &B,     &rtol, &atol, &max_steps,
                  &adapt, &pperm, &pblob, &prf, &pnasa, &scratch, &psst, &pjt, &plu, &poptr, &sround, &jround,
                  &redo, &list, &wscr, &pdidx, &patoms};    // the last five: jvp_rec only
  return launch(rec ? m.rec : m.main, blocks(m, B), 1, m.threads, m.smem, s, args);
}

CUresult launch_guard(const Module& m, CUstream s, const Primal& a, int B, double atol, void* xo, void* done,
                      void* nok, void* redo, void* list, void* count) {
  void *px = a.x.untyped_data(), *pT = a.T.untyped_data(), *pP = a.P.untyped_data(), *pdt = a.dt.untyped_data(),
       *patoms = a.atoms.untyped_data();
  void* args[] = {&px, &pT, &pP, &pdt, &xo, &done, &nok, &patoms, &redo, &list, &count, &B, &atol};
  return launch(m.guard, static_cast<unsigned>((B + 127) / 128), 1, 128, 0, s, args);
}

#define KINJAX_PRIMAL_ARGS                                                                                         \
  .Arg<ffi::AnyBuffer>()     /* x [B, ns] */                                                                       \
      .Arg<ffi::AnyBuffer>() /* T [B] K */                                                                         \
      .Arg<ffi::AnyBuffer>() /* P [B] dyn cm^-2 */                                                                 \
      .Arg<ffi::AnyBuffer>() /* dt [B] s */                                                                        \
      .Arg<ffi::AnyBuffer>() /* h [B] s */                                                                         \
      .Arg<ffi::AnyBuffer>() /* lm [B, nr] or empty */                                                             \
      .Arg<ffi::AnyBuffer>() /* rc [nr, 9] */                                                                      \
      .Arg<ffi::AnyBuffer>() /* perm */                                                                            \
      .Arg<ffi::AnyBuffer>() /* blob */                                                                            \
      .Arg<ffi::AnyBuffer>() /* rf */                                                                              \
      .Arg<ffi::AnyBuffer>() /* nasa */                                                                            \
      .Arg<ffi::AnyBuffer>() /* sst [sround, 32] */                                                                \
      .Arg<ffi::AnyBuffer>() /* jt [jround, 32] */                                                                 \
      .Arg<ffi::AnyBuffer>() /* lu */                                                                              \
      .Arg<ffi::AnyBuffer>() /* optr */                                                                            \
      .Arg<ffi::AnyBuffer>() /* atoms [ns, nel] */                                                                 \
      .Arg<ffi::AnyBuffer>() /* gt [ground, 32] */                                                                 \
      .Arg<ffi::AnyBuffer>() /* xg */                                                                              \
      .Arg<ffi::AnyBuffer>() /* didx (dense position of each LU slot) */

#define KINJAX_PRIMAL_PARAMS                                                                                       \
  ffi::AnyBuffer x, ffi::AnyBuffer T, ffi::AnyBuffer P, ffi::AnyBuffer dt, ffi::AnyBuffer h, ffi::AnyBuffer lm,   \
      ffi::AnyBuffer rc, ffi::AnyBuffer perm, ffi::AnyBuffer blob, ffi::AnyBuffer rf, ffi::AnyBuffer nasa,         \
      ffi::AnyBuffer sst, ffi::AnyBuffer jt, ffi::AnyBuffer lu, ffi::AnyBuffer optr, ffi::AnyBuffer atoms,         \
      ffi::AnyBuffer gt, ffi::AnyBuffer xg, ffi::AnyBuffer didx

#define KINJAX_PRIMAL {x, T, P, dt, h, lm, rc, perm, blob, rf, nasa, sst, jt, lu, optr, atoms, gt, xg, didx}

}  // namespace

// out[0..3]: compute capability major, minor, multiprocessors, opt-in shared memory per block (bytes). 0 = success.
extern "C" int KinjaxDevice(int device, int* out, char* err, int errlen) {
  const Driver& d = drv();
  if (!d.error.empty()) { set_message(err, errlen, d.error); return 1; }
  CUdevice dev;
  CUresult r = d.DeviceGet(&dev, device);
  const CUdevice_attribute at[4] = {CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN};
  for (int i = 0; i < 4 && r == CUDA_SUCCESS; i++) r = d.DeviceGetAttribute(out + i, at[i], dev);
  if (r != CUDA_SUCCESS) { set_message(err, errlen, cu_error(r, "device attributes")); return 1; }
  return 0;
}

// Loads a cubin of `kind` (0 call, 1 jvp, 2 vjp) into the primary context of `device` and returns the handle (>= 0; -1
// on error, message in err): a new one for handle -1, else `handle`. geo: ns, nr, cells and threads per block of the
// main kernel, its dynamic shared memory (bytes), the recovery kernel's threads per block and shared memory, the rounds
// of the S v, Jacobian and gather streams, the lanes of the first two. blocks[0] = resident blocks of the main kernel
// on the device, blocks[1] = of the recovery kernel.
extern "C" int64_t KinjaxLoad(int64_t handle, int device, int kind, const void* cubin, const int* geo, int* blocks,
                              char* err, int errlen) {
  const Driver& d = drv();
  if (!d.error.empty()) { set_message(err, errlen, d.error); return -1; }
  if (kind < 0 || kind > 2) { set_message(err, errlen, "kind must be 0, 1 or 2"); return -1; }
  Module m{handle, device, kind, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, geo[0], geo[1], geo[2], geo[3],
           geo[4], geo[5], geo[6], geo[7], geo[8], geo[9], geo[10], 0, 0};
  CUdevice dev;
  CUresult r = d.DeviceGet(&dev, device);
  if (r == CUDA_SUCCESS) r = d.PrimaryCtxRetain(&m.ctx, dev);   // kept for the life of the process
  if (r != CUDA_SUCCESS) { set_message(err, errlen, cu_error(r, "primary context")); return -1; }
  if ((r = d.CtxPush(m.ctx)) != CUDA_SUCCESS) { set_message(err, errlen, cu_error(r, "push context")); return -1; }
  int occ = 0, occ_rec = 1, sms = 0;
  std::string what = "load";
  r = d.ModuleLoadData(&m.mod, cubin);
  if (r == CUDA_SUCCESS) { what = std::string("function ") + kMain[kind]; r = d.ModuleGetFunction(&m.main, m.mod, kMain[kind]); }
  if (r == CUDA_SUCCESS) { what = "function guard"; r = d.ModuleGetFunction(&m.guard, m.mod, "guard"); }
  if (r == CUDA_SUCCESS) { what = std::string("function ") + kRec[kind]; r = d.ModuleGetFunction(&m.rec, m.mod, kRec[kind]); }
  if (r == CUDA_SUCCESS && kind == 2) { what = "function arrhg"; r = d.ModuleGetFunction(&m.aux, m.mod, "arrhg"); }
  if (r == CUDA_SUCCESS) {
    what = "shared memory";
    r = d.FuncSetAttribute(m.main, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, m.smem);
  }
  if (r == CUDA_SUCCESS) r = d.FuncSetAttribute(m.rec, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, m.rec_smem);
  if (r == CUDA_SUCCESS) { what = "occupancy"; r = d.Occupancy(&occ, m.main, m.threads, m.smem); }
  if (r == CUDA_SUCCESS) r = d.Occupancy(&occ_rec, m.rec, m.rec_threads, m.rec_smem);
  if (r == CUDA_SUCCESS) {
    what = "multiprocessors";
    r = d.DeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
  }
  CUcontext popped;
  d.CtxPop(&popped);
  if (r != CUDA_SUCCESS) { set_message(err, errlen, cu_error(r, what.c_str())); return -1; }
  if (occ < 1 || occ_rec < 1) { set_message(err, errlen, "the kernel does not fit on this device"); return -1; }
  m.max_blocks = sms * occ;
  m.rec_blocks = sms * occ_rec;
  blocks[0] = m.max_blocks;
  blocks[1] = m.rec_blocks;
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (handle < 0) {
    m.id = 0;
    for (const Module& q : registry) m.id = q.id >= m.id ? q.id + 1 : m.id;
  } else {
    for (const Module& q : registry)
      if (q.id == handle && q.device == device && q.kind == kind) {
        set_message(err, errlen, "already loaded");
        return -1;
      }
  }
  registry.push_back(m);
  return m.id;
}

namespace {

// The call: without lm the call kernel (h copied to h_out, which it updates in place), the guard (which lists the
// flagged cells) and the recover kernel (it recomputes the listed cells from the inputs; flag 1 recovered, 2 still
// failed); with lm the jvp kernel with no directions (the same arithmetic; exp(0) multipliers leave every rate bitwise
// unchanged), the guard and jvp_rec with no directions (the recover kernel's arithmetic with the multipliers; dense W
// per warp in rscratch).
ffi::Error CallImpl(CUstream stream, int32_t device, KINJAX_PRIMAL_PARAMS, ffi::Result<ffi::AnyBuffer> xo,
                    ffi::Result<ffi::AnyBuffer> ho, ffi::Result<ffi::AnyBuffer> done, ffi::Result<ffi::AnyBuffer> nok,
                    ffi::Result<ffi::AnyBuffer> redo, ffi::Result<ffi::AnyBuffer> count,
                    ffi::Result<ffi::AnyBuffer> scratch, ffi::Result<ffi::AnyBuffer> list,
                    ffi::Result<ffi::AnyBuffer> rscratch, int64_t module, double rtol, double atol,
                    int64_t max_steps, int64_t rec_steps) {
  const Driver& d = drv();
  if (!d.error.empty()) return ffi::Error::Internal(d.error);
  const Primal a KINJAX_PRIMAL;
  const bool with_lm = lm.element_count() > 0;
  Module m;
  if (!find(module, device, with_lm ? 1 : 0, m))
    return ffi::Error::InvalidArgument("kinjax: module handle not loaded on this device");
  int64_t B;
  if (ffi::Error e = check_primal(m, a, B); e.failure()) return e;
  if (count->element_count() != 10 || list->element_count() != static_cast<size_t>(B))
    return ffi::Error::InvalidArgument("kinjax: count [10], list [B]");
  const size_t per_cell = (with_lm ? 3 : 2) * static_cast<size_t>(m.nr) + 2, nw = static_cast<size_t>(blocks(m, B)) * m.cells;
  if (scratch->element_count() < nw * per_cell) return ffi::Error::InvalidArgument("kinjax: rate scratch too small");
  if (rscratch->element_count() < (with_lm ? nw * m.ns * m.ns : static_cast<size_t>(m.rec_blocks) * (2 * m.nr + 2)))
    return ffi::Error::InvalidArgument("kinjax: recovery scratch too small");
  Scope sc(m.ctx);
  if (sc.r != CUDA_SUCCESS) return ffi::Error::Internal(cu_error(sc.r, "push context"));
  const char* what = "zero counts";
  void *pcount = count->untyped_data(), *pxo = xo->untyped_data(), *pho = ho->untyped_data(),
       *pdone = done->untyped_data(), *pnok = nok->untyped_data(), *predo = redo->untyped_data(),
       *pscr = scratch->untyped_data(), *plist = list->untyped_data(), *prs = rscratch->untyped_data();
  CUresult r = d.MemsetD32Async(reinterpret_cast<CUdeviceptr>(pcount), 0, 10, stream);
  const int nB = static_cast<int>(B), ms = static_cast<int>(max_steps);
  if (r == CUDA_SUCCESS && B > 0 && with_lm) {
    what = "launch jvp (no directions)";
    r = launch_jvp(m, false, stream, a, nB, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, pxo, pho,
                   pdone, pnok, nullptr, nullptr, pcount, rtol, atol, ms, 1, pscr, nullptr, nullptr, nullptr);
    if (r == CUDA_SUCCESS) {
      what = "launch guard";
      r = launch_guard(m, stream, a, nB, atol, pxo, pdone, pnok, predo, plist, pcount);
    }
    if (r == CUDA_SUCCESS) {   // returns at once when no cell is listed
      what = "launch jvp_rec (no directions)";
      r = launch_jvp(m, true, stream, a, nB, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, pxo,
                     pho, pdone, pnok, nullptr, nullptr, pcount, rtol, atol, static_cast<int>(rec_steps), 1, pscr,
                     predo, plist, prs);
    }
  } else if (r == CUDA_SUCCESS && B > 0) {
    what = "copy h";
    r = d.MemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(pho), reinterpret_cast<CUdeviceptr>(h.untyped_data()),
                          B * sizeof(double), stream);
    if (r == CUDA_SUCCESS) {
      what = "launch call";
      void *px = x.untyped_data(), *pT = T.untyped_data(), *pP = P.untyped_data(), *pdt = dt.untyped_data(),
           *pperm = perm.untyped_data(), *pblob = blob.untyped_data(), *prc = rc.untyped_data(),
           *prf = rf.untyped_data(), *pnasa = nasa.untyped_data(), *psst = sst.untyped_data(),
           *pjt = jt.untyped_data(), *plu = lu.untyped_data(), *poptr = optr.untyped_data();
      int sround = m.sround, jround = m.jround;
      double rt = rtol, at = atol;
      int nB2 = nB, ms2 = ms;
      void* args[] = {&px,   &pT,    &pP,   &pdt, &pho,   &pxo,  &pdone, &pnok, &pcount, &nB2, &rt,  &at,     &ms2,
                      &pperm, &pblob, &prc, &prf, &pnasa, &pscr, &psst,  &pjt,  &plu,    &poptr, &sround, &jround};
      r = launch(m.main, blocks(m, B), 1, m.threads, m.smem, stream, args);
    }
    if (r == CUDA_SUCCESS) {
      what = "launch guard";
      r = launch_guard(m, stream, a, nB, atol, pxo, pdone, pnok, predo, plist, pcount);
    }
  }
  if (r == CUDA_SUCCESS && B > 0 && !with_lm) {   // returns at once when no cell is listed
    what = "launch recover";
    void *px = x.untyped_data(), *pT = T.untyped_data(), *pP = P.untyped_data(), *pdt = dt.untyped_data(),
         *ph = h.untyped_data(), *pperm = perm.untyped_data(), *pblob = blob.untyped_data(), *prc = rc.untyped_data(),
         *prf = rf.untyped_data(), *pnasa = nasa.untyped_data(), *psst = sst.untyped_data(), *pjt = jt.untyped_data(),
         *pdidx = didx.untyped_data(), *patoms = atoms.untyped_data();
    int sround = m.sround, jround = m.jround, rs = static_cast<int>(rec_steps);
    double rt = rtol, at = atol;
    void* rargs[] = {&px,  &pT,    &pP,    &pdt,  &ph,  &pxo,   &pho, &pdone, &pnok,  &predo,   &plist,  &pcount, &rt,
                     &at,  &rs,    &pperm, &pblob, &prc, &prf, &pnasa, &prs, &psst,  &pjt,   &pdidx, &patoms, &sround,
                     &jround};
    r = launch(m.rec, static_cast<unsigned>(std::min<int64_t>(B, m.rec_blocks)), 1, m.rec_threads, m.rec_smem, stream,
               rargs);
  }
  if (r != CUDA_SUCCESS) return ffi::Error::Internal(cu_error(r, what));
  return ffi::Error::Success();
}

// Tangents of (x_out, h_out) for D directions: per DMAX directions the jvp kernel (it recomputes the call on the same
// branch), the guard and jvp_rec (the flagged cells redone on the recovery's path with their tangents; flag 2: NaN
// tangents); its (x, h, done, steps, flag, counts) are outputs too (bitwise those of kinjax_call).
ffi::Error JvpImpl(CUstream stream, int32_t device, KINJAX_PRIMAL_PARAMS, ffi::AnyBuffer tx, ffi::AnyBuffer tT,
                   ffi::AnyBuffer tP, ffi::AnyBuffer tdt, ffi::AnyBuffer th, ffi::AnyBuffer tlm, ffi::AnyBuffer trc,
                   ffi::Result<ffi::AnyBuffer> dxo, ffi::Result<ffi::AnyBuffer> dho, ffi::Result<ffi::AnyBuffer> xs,
                   ffi::Result<ffi::AnyBuffer> hs, ffi::Result<ffi::AnyBuffer> dones,
                   ffi::Result<ffi::AnyBuffer> noks, ffi::Result<ffi::AnyBuffer> redo,
                   ffi::Result<ffi::AnyBuffer> count, ffi::Result<ffi::AnyBuffer> scratch,
                   ffi::Result<ffi::AnyBuffer> list, ffi::Result<ffi::AnyBuffer> wscr, int64_t module, double rtol,
                   double atol, int64_t max_steps, int64_t adapt, int64_t rec_steps) {
  const Driver& d = drv();
  if (!d.error.empty()) return ffi::Error::Internal(d.error);
  const Primal a KINJAX_PRIMAL;
  Module m;
  if (!find(module, device, 1, m)) return ffi::Error::InvalidArgument("kinjax: module handle not loaded on this device");
  int64_t B;
  if (ffi::Error e = check_primal(m, a, B); e.failure()) return e;
  const auto td = tx.dimensions();
  const int64_t D = td.size() == 3 ? td[0] : -1;
  if (D < 0 || td[1] != B || td[2] != m.ns) return ffi::Error::InvalidArgument("kinjax_jvp: tangent of x [D, B, ns]");
  const size_t DB = static_cast<size_t>(D) * B;
  for (const ffi::AnyBuffer* q : {&tT, &tP, &tdt, &th})
    if (q->element_count() != DB) return ffi::Error::InvalidArgument("kinjax_jvp: tangents of T, P, dt, h [D, B]");
  if ((tlm.element_count() && tlm.element_count() != DB * m.nr) ||
      (trc.element_count() && trc.element_count() != static_cast<size_t>(D) * m.nr * 9))
    return ffi::Error::InvalidArgument("kinjax_jvp: tangents of lm [D, B, nr], rc [D, nr, 9] or empty");
  const size_t dc = static_cast<size_t>(std::min<int64_t>(D, kDmax)), nw = static_cast<size_t>(blocks(m, B)) * m.cells;
  if (scratch->element_count() < nw * (3 * m.nr + 2 + dc * (2 * m.nr + m.ns)) ||
      wscr->element_count() < nw * m.ns * m.ns || count->element_count() != 10 ||
      list->element_count() != static_cast<size_t>(B))
    return ffi::Error::InvalidArgument("kinjax_jvp: scratch too small");
  if (D == 0 || B == 0) return ffi::Error::Success();
  Scope sc(m.ctx);
  if (sc.r != CUDA_SUCCESS) return ffi::Error::Internal(cu_error(sc.r, "push context"));
  auto at = [](const ffi::AnyBuffer& b, size_t off) -> void* {
    return b.element_count() ? static_cast<double*>(b.untyped_data()) + off : nullptr;
  };
  auto ot = [](ffi::Result<ffi::AnyBuffer>& b, size_t off) -> void* {
    return static_cast<double*>(b->untyped_data()) + off;
  };
  void *pxs = xs->untyped_data(), *phs = hs->untyped_data(), *pdones = dones->untyped_data(),
       *pnoks = noks->untyped_data(), *predo = redo->untyped_data(), *pcount = count->untyped_data(),
       *pscr = scratch->untyped_data(), *plist = list->untyped_data(), *pw = wscr->untyped_data();
  const int nB = static_cast<int>(B);
  CUresult r = CUDA_SUCCESS;
  const char* what = "zero counts";
  for (int64_t d0 = 0; d0 < D && r == CUDA_SUCCESS; d0 += kDmax) {
    const int n = static_cast<int>(std::min<int64_t>(kDmax, D - d0));
    const size_t o = static_cast<size_t>(d0) * B;
    what = "zero counts";
    r = d.MemsetD32Async(reinterpret_cast<CUdeviceptr>(pcount), 0, 10, stream);
    for (int k = 0; k < 2 && r == CUDA_SUCCESS; k++) {   // jvp, then jvp_rec after the guard
      what = k ? "launch jvp_rec" : "launch jvp";
      r = launch_jvp(m, k == 1, stream, a, nB, at(tx, o * m.ns), at(tT, o), at(tP, o), at(tdt, o), at(th, o),
                     at(tlm, o * m.nr), at(trc, static_cast<size_t>(d0) * m.nr * 9), n, pxs, phs, pdones, pnoks,
                     ot(dxo, o * m.ns), ot(dho, o), pcount, rtol, atol, static_cast<int>(k ? rec_steps : max_steps),
                     static_cast<int>(adapt), pscr, predo, plist, pw);
      if (r == CUDA_SUCCESS && k == 0) {
        what = "launch guard";
        r = launch_guard(m, stream, a, nB, atol, pxs, pdones, pnoks, predo, plist, pcount);
      }
    }
  }
  if (r != CUDA_SUCCESS) return ffi::Error::Internal(cu_error(r, what));
  return ffi::Error::Success();
}

// Cotangents of (x, T, P, dt, h[, lm][, rc]) for D cotangents of (x_out[, h_out]): per cotangent one launch of the
// vjp kernel (it recomputes the call and reverses it), the guard and vjp_rec (the flagged cells redone on the
// recovery's path and reversed; flag 2: NaN cotangents), then, when the rc cotangent is requested, the reduction of the
// per-cell lm cotangents to the Arrhenius table over `groups` equal consecutive groups of cells.
ffi::Error VjpImpl(CUstream stream, int32_t device, KINJAX_PRIMAL_PARAMS, ffi::AnyBuffer cx, ffi::AnyBuffer ch,
                   ffi::Result<ffi::AnyBuffer> gx, ffi::Result<ffi::AnyBuffer> gT, ffi::Result<ffi::AnyBuffer> gP,
                   ffi::Result<ffi::AnyBuffer> gdt, ffi::Result<ffi::AnyBuffer> gh, ffi::Result<ffi::AnyBuffer> glm,
                   ffi::Result<ffi::AnyBuffer> grc, ffi::Result<ffi::AnyBuffer> xs, ffi::Result<ffi::AnyBuffer> hs,
                   ffi::Result<ffi::AnyBuffer> dones, ffi::Result<ffi::AnyBuffer> noks,
                   ffi::Result<ffi::AnyBuffer> redo, ffi::Result<ffi::AnyBuffer> count,
                   ffi::Result<ffi::AnyBuffer> scratch, ffi::Result<ffi::AnyBuffer> rec,
                   ffi::Result<ffi::AnyBuffer> glms, ffi::Result<ffi::AnyBuffer> list,
                   ffi::Result<ffi::AnyBuffer> wscr, ffi::Result<ffi::AnyBuffer> rrec, int64_t module, double rtol,
                   double atol, int64_t max_steps, int64_t adapt, int64_t groups, int64_t cap, int64_t rec_steps) {
  const Driver& d = drv();
  if (!d.error.empty()) return ffi::Error::Internal(d.error);
  const Primal a KINJAX_PRIMAL;
  Module m;
  if (!find(module, device, 2, m)) return ffi::Error::InvalidArgument("kinjax: module handle not loaded on this device");
  int64_t B;
  if (ffi::Error e = check_primal(m, a, B); e.failure()) return e;
  const auto cd = cx.dimensions();
  const int64_t D = cd.size() == 3 ? cd[0] : -1;
  if (D < 0 || cd[1] != B || cd[2] != m.ns) return ffi::Error::InvalidArgument("kinjax_vjp: cotangent of x [D, B, ns]");
  const size_t DB = static_cast<size_t>(D) * B, nw = static_cast<size_t>(blocks(m, B)) * m.cells;
  if (ch.element_count() && ch.element_count() != DB)
    return ffi::Error::InvalidArgument("kinjax_vjp: cotangent of h [D, B] or empty");
  const bool want_lm = glm->element_count() > 0, want_rc = grc->element_count() > 0;
  if (groups < 1 || B % groups || (want_lm && glm->element_count() != DB * m.nr) ||
      (!want_lm && glms->element_count() != static_cast<size_t>(B) * m.nr) ||
      (want_rc && grc->element_count() != static_cast<size_t>(D) * groups * m.nr * 9))
    return ffi::Error::InvalidArgument("kinjax_vjp: lm / rc cotangent outputs do not match");
  if (scratch->element_count() < nw * (12 * m.nr + 2) || rec->element_count() < nw * 2 * cap * (m.ns + 3) ||
      rrec->element_count() < nw * 4 * cap * (m.ns + 3) || wscr->element_count() < nw * m.ns * m.ns ||
      count->element_count() != 10 || list->element_count() != static_cast<size_t>(B))
    return ffi::Error::InvalidArgument("kinjax_vjp: scratch too small");
  if (D == 0) return ffi::Error::Success();
  Scope sc(m.ctx);
  if (sc.r != CUDA_SUCCESS) return ffi::Error::Internal(cu_error(sc.r, "push context"));
  if (B == 0) {                                     // no cells: the Arrhenius cotangent is 0
    CUresult r = want_rc ? d.MemsetD32Async(reinterpret_cast<CUdeviceptr>(grc->untyped_data()), 0,
                                            2 * grc->element_count(), stream)
                         : CUDA_SUCCESS;
    return r == CUDA_SUCCESS ? ffi::Error::Success() : ffi::Error::Internal(cu_error(r, "zero rc cotangent"));
  }
  void *px = x.untyped_data(), *pT = T.untyped_data(), *pP = P.untyped_data(), *pdt = dt.untyped_data(),
       *ph = h.untyped_data(), *plm = ptr(lm), *prc = rc.untyped_data(), *pperm = perm.untyped_data(),
       *pblob = blob.untyped_data(), *prf = rf.untyped_data(), *pnasa = nasa.untyped_data(),
       *psst = sst.untyped_data(), *pjt = jt.untyped_data(), *plu = lu.untyped_data(), *poptr = optr.untyped_data(),
       *pgt = gt.untyped_data(), *pxg = xg.untyped_data(), *pxs = xs->untyped_data(), *phs = hs->untyped_data(),
       *pdones = dones->untyped_data(), *pnoks = noks->untyped_data(), *pcount = count->untyped_data(),
       *pscr = scratch->untyped_data(), *prec = rec->untyped_data(), *predo = redo->untyped_data(),
       *plist = list->untyped_data(), *pw = wscr->untyped_data(), *prrec = rrec->untyped_data(),
       *pdidx = didx.untyped_data(), *patoms = atoms.untyped_data();
  int nB = static_cast<int>(B), ms = static_cast<int>(max_steps), ad = static_cast<int>(adapt), sround = m.sround,
      jround = m.jround, ground = m.ground, nG = static_cast<int>(groups), rs = static_cast<int>(rec_steps);
  double rt = rtol, att = atol;
  auto off = [](void* p, size_t o) -> void* { return p ? static_cast<double*>(p) + o : nullptr; };
  CUresult r = CUDA_SUCCESS;
  const char* what = "zero queue";
  for (int64_t k = 0; k < D && r == CUDA_SUCCESS; k++) {
    const size_t o = static_cast<size_t>(k) * B;
    void *pcx = off(cx.untyped_data(), o * m.ns), *pch = off(ptr(ch), o), *pgx = off(gx->untyped_data(), o * m.ns),
         *pgT = off(gT->untyped_data(), o), *pgP = off(gP->untyped_data(), o), *pgdt = off(gdt->untyped_data(), o),
         *pgh = off(gh->untyped_data(), o),
         *pglm = want_lm ? off(glm->untyped_data(), o * m.nr) : glms->untyped_data();
    what = "zero counts";
    r = d.MemsetD32Async(reinterpret_cast<CUdeviceptr>(pcount), 0, 10, stream);
    if (r != CUDA_SUCCESS) break;
    what = "launch vjp";
    void* args[] = {&px,   &pT,    &pP,     &pdt,   &ph,    &plm,   &prc,    &pcx,    &pch,   &pxs,   &phs,
                    &pdones, &pnoks, &pgx,  &pgT,   &pgP,   &pgdt,  &pgh,    &pglm,   &pcount, &nB,   &rt,
                    &att,  &ms,    &ad,     &pperm, &pblob, &prf,   &pnasa,  &pscr,   &prec,  &psst,  &pjt,
                    &plu,  &poptr, &pgt,    &pxg,   &sround, &jround, &ground, &predo, &plist, &pw,    &pdidx,
                    &patoms};    // the last five: vjp_rec only
    r = launch(m.main, blocks(m, B), 1, m.threads, m.smem, stream, args);
    if (r == CUDA_SUCCESS) {
      what = "launch guard";
      r = launch_guard(m, stream, a, nB, att, pxs, pdones, pnoks, predo, plist, pcount);
    }
    if (r == CUDA_SUCCESS) {   // vjp_rec: max_steps (args[23]) = the recovery's tries, rec (args[30]) = rrec, CAPR = 2 cap
      what = "launch vjp_rec";
      args[23] = &rs;
      args[30] = &prrec;
      r = launch(m.rec, blocks(m, B), 1, m.threads, m.smem, stream, args);
    }
    if (r == CUDA_SUCCESS && want_rc) {
      what = "launch arrhg";
      void* pgrc = off(grc->untyped_data(), static_cast<size_t>(k) * groups * m.nr * 9);
      void* gargs[] = {&pglm, &pT, &pP, &prc, &prf, &pgrc, &nB, &nG};
      r = launch(m.aux, m.nr, static_cast<unsigned>(std::min<int64_t>(groups, 65535)), 256, 0, stream, gargs);
    }
  }
  if (r != CUDA_SUCCESS) return ffi::Error::Internal(cu_error(r, what));
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(KinjaxCall, CallImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<CUstream>>()
                                  .Ctx<ffi::DeviceOrdinal>() KINJAX_PRIMAL_ARGS
                                  .Ret<ffi::AnyBuffer>()    // x_out [B, ns]
                                  .Ret<ffi::AnyBuffer>()    // h_out [B]
                                  .Ret<ffi::AnyBuffer>()    // done [B] int32
                                  .Ret<ffi::AnyBuffer>()    // accepted steps [B] float64
                                  .Ret<ffi::AnyBuffer>()    // flag [B] int32: 0 passed, 1 recovered, 2 still failed
                                  .Ret<ffi::AnyBuffer>()    // count [10] int32: queue, guard and recovery counts
                                  .Ret<ffi::AnyBuffer>()    // rate scratch
                                  .Ret<ffi::AnyBuffer>()    // list of flagged cells [B] int32 (scratch)
                                  .Ret<ffi::AnyBuffer>()    // recovery scratch (rates; with lm the dense W per warp)
                                  .Attr<int64_t>("module")
                                  .Attr<double>("rtol")
                                  .Attr<double>("atol")
                                  .Attr<int64_t>("max_steps")
                                  .Attr<int64_t>("rec_steps"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(KinjaxJvp, JvpImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<CUstream>>()
                                  .Ctx<ffi::DeviceOrdinal>() KINJAX_PRIMAL_ARGS
                                  .Arg<ffi::AnyBuffer>()    // tangent of x [D, B, ns]
                                  .Arg<ffi::AnyBuffer>()    // of T [D, B]
                                  .Arg<ffi::AnyBuffer>()    // of P [D, B]
                                  .Arg<ffi::AnyBuffer>()    // of dt [D, B]
                                  .Arg<ffi::AnyBuffer>()    // of h [D, B]
                                  .Arg<ffi::AnyBuffer>()    // of lm [D, B, nr] or empty
                                  .Arg<ffi::AnyBuffer>()    // of rc [D, nr, 9] or empty
                                  .Ret<ffi::AnyBuffer>()    // tangent of x_out [D, B, ns]
                                  .Ret<ffi::AnyBuffer>()    // of h_out [D, B]
                                  .Ret<ffi::AnyBuffer>()    // scratch: x_out [B, ns]
                                  .Ret<ffi::AnyBuffer>()    // h_out [B]
                                  .Ret<ffi::AnyBuffer>()    // done [B]
                                  .Ret<ffi::AnyBuffer>()    // steps [B]
                                  .Ret<ffi::AnyBuffer>()    // flag [B] int32 (as kinjax_call's)
                                  .Ret<ffi::AnyBuffer>()    // count [10] int32 (as kinjax_call's)
                                  .Ret<ffi::AnyBuffer>()    // rates and tangent work
                                  .Ret<ffi::AnyBuffer>()    // list of flagged cells [B] int32 (scratch)
                                  .Ret<ffi::AnyBuffer>()    // dense W per warp (scratch)
                                  .Attr<int64_t>("module")
                                  .Attr<double>("rtol")
                                  .Attr<double>("atol")
                                  .Attr<int64_t>("max_steps")
                                  .Attr<int64_t>("adapt")
                                  .Attr<int64_t>("rec_steps"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(KinjaxVjp, VjpImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<CUstream>>()
                                  .Ctx<ffi::DeviceOrdinal>() KINJAX_PRIMAL_ARGS
                                  .Arg<ffi::AnyBuffer>()    // cotangent of x_out [D, B, ns]
                                  .Arg<ffi::AnyBuffer>()    // of h_out [D, B] or empty
                                  .Ret<ffi::AnyBuffer>()    // cotangent of x [D, B, ns]
                                  .Ret<ffi::AnyBuffer>()    // of T [D, B]
                                  .Ret<ffi::AnyBuffer>()    // of P [D, B]
                                  .Ret<ffi::AnyBuffer>()    // of dt [D, B]
                                  .Ret<ffi::AnyBuffer>()    // of h [D, B]
                                  .Ret<ffi::AnyBuffer>()    // of lm [D, B, nr] or empty
                                  .Ret<ffi::AnyBuffer>()    // of rc [D, groups, nr, 9] or empty
                                  .Ret<ffi::AnyBuffer>()    // scratch: x_out [B, ns]
                                  .Ret<ffi::AnyBuffer>()    // h_out [B]
                                  .Ret<ffi::AnyBuffer>()    // done [B]
                                  .Ret<ffi::AnyBuffer>()    // steps [B]
                                  .Ret<ffi::AnyBuffer>()    // flag [B] int32 (as kinjax_call's)
                                  .Ret<ffi::AnyBuffer>()    // count [10] int32 (as kinjax_call's)
                                  .Ret<ffi::AnyBuffer>()    // rates and rate cotangents
                                  .Ret<ffi::AnyBuffer>()    // attempt records
                                  .Ret<ffi::AnyBuffer>()    // per-cell lm cotangents when not requested [B nr] or empty
                                  .Ret<ffi::AnyBuffer>()    // list of flagged cells [B] int32 (scratch)
                                  .Ret<ffi::AnyBuffer>()    // dense W per warp (scratch)
                                  .Ret<ffi::AnyBuffer>()    // attempt records of the recovered cells (2 cap per warp)
                                  .Attr<int64_t>("module")
                                  .Attr<double>("rtol")
                                  .Attr<double>("atol")
                                  .Attr<int64_t>("max_steps")
                                  .Attr<int64_t>("adapt")
                                  .Attr<int64_t>("groups")
                                  .Attr<int64_t>("cap")
                                  .Attr<int64_t>("rec_steps"));
