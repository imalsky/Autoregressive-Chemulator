#!/usr/bin/env python3
"""
dataset.py - Preloaded rollout windows for autoregressive training.

This module provides the dataset and dataloader helper for autoregressive
flow-map training on *pre-normalized* trajectory shards.

What the model expects (per model.py and trainer.py):
- y is already in z-space (normalized). For log-standardized species this means:
      z = (log10(y_phys) - log_mean) / log_std
  and log10(y_phys) can be recovered via:
      log10(y_phys) = z * log_std + log_mean

- dt is already normalized to [0, 1] using log10 + min-max (see preprocessing.py):
      dt_norm = (log10(dt_phys) - log_min) / (log_max - log_min)

- g (globals) is already normalized (method depends on your preprocessing/manifest).

Accordingly, this dataset does NOT apply NormalizationHelper during sampling.
The normalization manifest is still required for training/inference elsewhere
(e.g., loss conversion to log10-space, and post-hoc denormalization).

Data format (per shard NPZ; processing/preprocessing.py writes uncompressed ZIP
archives with np.savez):
    y_mat       : [N, T, S] float32  (z-space trajectories)
    globals     : [N, G]    float32  (normalized globals)
    dt_norm_mat : [N, T-1]  float32  (normalized dt per step in [0,1])

Every split is preloaded into contiguous float32 tensors (the shard dtype) on the
preload device; preload_nbytes() gives a split's preload size from the shard headers
alone, so the caller can check a memory budget before anything is loaded.
Two vectorized single-process streams wrap it and yield pre-batched tensors:
    {
        "y":  [B, K+1, S],   # includes the initial state at index 0
        "dt": [B, K],
        "g":  [B, G],
    }
- create_train_dataloader(): a fixed number of batches per epoch, each B windows drawn
  uniformly at random WITH replacement (a window is a trajectory and a start index), from a
  generator reseeded per epoch. With a JumpSampler, every batch also carries long-jump pairs
  drawn from the raw-trajectory store by a second generator, so the windows are the same with
  and without them:
    {
        "jump_z0":      [Bj, S] float64,  # z at a time t
        "jump_z1":      [Bj, S] float64,  # z at t + D
        "jump_dt_norm": [Bj]    float32,  # D normalized like dt, not clamped (may exceed 1)
        "jump_g":       [Bj, G] float32,
    }
- create_val_dataloader(): a fixed window set, defined by the data and one count only: the
  N_val * n_starts (trajectory, start) windows are enumerated by index and an even subset of
  n_windows of them (all when n_windows >= N_val * n_starts) is scored in index order. No
  generator is involved, so the scored windows do not depend on the seed, the batch size, the
  epoch, a resume or the preload device.

The raw-trajectory store (<processed_dir>/raw_trajectories/, written by processing/preprocessing.py:
every kept train trajectory on its full raw time grid as log10 y in float64) is loaded whole onto
the preload device by RawTrajectories; raw_store_nbytes() gives its size from the .npy headers.

There is no loading path for data larger than RAM. np.load cannot memory-map
.npz members, but because the members are stored uncompressed they can be opened
with np.memmap at their data offset (offsets are not float32-aligned), an input
for any future beyond-RAM loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import zipfile
from numpy.lib import format as npy_format

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

# Preloaded tensors keep the shard dtype (float32): bf16 storage would quantize the z-space
# states, and float64 would only double the memory.
_STORAGE_DTYPE = torch.float32
# The training stream reseeds every epoch from SeedSequence([seed, _TRAIN_STREAM_ID, epoch]), so a
# resumed run draws the same batches as an uninterrupted one (the id keeps it apart from any
# other stream derived from the same seed).
_TRAIN_STREAM_ID = 0
# The long-jump pairs reseed their own generator every epoch from SeedSequence([seed, _JUMP_STREAM_ID,
# epoch]): resume-safe, and the window draws do not depend on whether pairs are drawn.
_JUMP_STREAM_ID = 1
# Largest int64 + 1: the validation window index k * total must stay below it.
_INT64_LIMIT = 2**63
# The raw-trajectory store under processed_dir (processing/preprocessing.py writes it).
_RAW_STORE_DIR = "raw_trajectories"
_RAW_STORE_FILE = "train.npz"
_RAW_STORE_GRID = "t_grid.npy"


# -----------------------------------------------------------------------------
# NPZ header helper (reads only the .npy header of a shard member)
# -----------------------------------------------------------------------------

def _npz_read_array_header(path: Path, *, key: str) -> Tuple[Tuple[int, ...], np.dtype]:
    """Return (shape, dtype) for `key` inside an .npz from its .npy header only.

    np.load(...)[key] reads the whole member; building the shard index only needs the
    header. Header versions other than 1.0 / 2.0 are rejected (regenerate the shards).
    """
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open(f"{key}.npy") as f:
            ver = npy_format.read_magic(f)
            if ver == (1, 0):
                shape, _, dtype = npy_format.read_array_header_1_0(f)
            elif ver == (2, 0):
                shape, _, dtype = npy_format.read_array_header_2_0(f)
            else:
                raise ValueError(
                    f"{path}:{key}.npy has unsupported .npy header version {ver}; "
                    "regenerate the shards with processing/preprocessing.py."
                )
    return tuple(int(s) for s in shape), np.dtype(dtype)


@dataclass(frozen=True)
class ShardIndex:
    """Metadata for a single data shard."""
    path: Path
    n: int


def _scan_shards(split_dir: Path) -> Tuple[List[ShardIndex], int, int, int]:
    """Index a split's shards from header-only reads: (shards, T, S, G).

    The first shard fixes T, S and G; every shard must match them and be float32.
    """
    shard_paths = sorted(split_dir.glob("*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard NPZ files found in {split_dir}")

    shards: List[ShardIndex] = []
    T = S = G = 0
    for p in shard_paths:
        y_shape, y_dtype = _npz_read_array_header(p, key="y_mat")
        g_shape, g_dtype = _npz_read_array_header(p, key="globals")
        dt_shape, dt_dtype = _npz_read_array_header(p, key="dt_norm_mat")

        if len(y_shape) != 3:
            raise ValueError(f"Shard {p} y_mat must be 3D [N,T,S], got {y_shape}")
        if len(g_shape) != 2:
            raise ValueError(f"Shard {p} globals must be 2D [N,G], got {g_shape}")
        if len(dt_shape) != 2:
            raise ValueError(f"Shard {p} dt_norm_mat must be 2D [N,T-1], got {dt_shape}")

        n, t, s = (int(y_shape[0]), int(y_shape[1]), int(y_shape[2]))
        if not shards:
            T, S, G = t, s, int(g_shape[1])
        if t != T or s != S:
            raise ValueError(f"Shard {p} y_mat shape is {y_shape}; expected [N,{T},{S}].")
        # Strict dtypes: avoid silent CPU copies / perf cliffs.
        if y_dtype != np.float32:
            raise ValueError(f"Shard {p} y_mat dtype must be float32, got {y_dtype}. Regenerate preprocessing outputs.")
        if g_dtype != np.float32:
            raise ValueError(f"Shard {p} globals dtype must be float32, got {g_dtype}. Regenerate preprocessing outputs.")
        if dt_dtype != np.float32:
            raise ValueError(f"Shard {p} dt_norm_mat dtype must be float32, got {dt_dtype}. Regenerate preprocessing outputs.")

        if int(g_shape[0]) != n or int(g_shape[1]) != G:
            raise ValueError(f"Shard {p} globals shape is {g_shape}; expected [{n},{G}].")
        if int(dt_shape[0]) != n or int(dt_shape[1]) != T - 1:
            raise ValueError(f"Shard {p} dt_norm_mat shape is {dt_shape}; expected [{n},{T - 1}].")

        if n <= 0:
            raise ValueError(f"Shard {p} has no trajectories.")
        shards.append(ShardIndex(path=p, n=n))
    return shards, T, S, G


def preload_nbytes(processed_dir: Union[str, Path], split: str) -> Tuple[int, int]:
    """(trajectories, bytes) that preloading `split` allocates: [N,T,S] + [N,T-1] + [N,G] float32."""
    split_dir = Path(processed_dir).expanduser().resolve() / split
    shards, T, S, G = _scan_shards(split_dir)
    n = int(sum(si.n for si in shards))
    return n, n * (T * S + (T - 1) + G) * _STORAGE_DTYPE.itemsize


def raw_store_path(processed_dir: Union[str, Path]) -> Path:
    """The raw-trajectory store's array file, <processed_dir>/raw_trajectories/train.npz."""
    return Path(processed_dir).expanduser().resolve() / _RAW_STORE_DIR / _RAW_STORE_FILE


def raw_store_nbytes(processed_dir: Union[str, Path]) -> Tuple[int, int]:
    """(trajectories, bytes) that RawTrajectories allocates, from the .npy headers alone:
    logy [N,T,S] float64 + end [N] int64 + globals [N,G] float32."""
    path = raw_store_path(processed_dir)
    logy_shape, _ = _npz_read_array_header(path, key="logy")
    g_shape, _ = _npz_read_array_header(path, key="globals")
    if len(logy_shape) != 3 or len(g_shape) != 2:
        raise ValueError(f"{path}: logy must be [N,T,S] and globals [N,G], got {logy_shape} and {g_shape}")
    n, t, s = logy_shape
    nbytes = n * (t * s * torch.float64.itemsize + torch.int64.itemsize + g_shape[1] * torch.float32.itemsize)
    return n, nbytes


class FlowMapRolloutDataset:
    """
    One split of trajectory shards, preloaded into contiguous tensors on `device`.

    create_train_dataloader() / create_val_dataloader() wrap it in the batch streams; a window is
    K+1 consecutive states (K transitions) of one trajectory, starting at any index in
    [0, max_start].
    """

    def __init__(
        self,
        processed_dir: Union[str, Path],
        split: str,
        *,
        total_steps: int,
        device: torch.device,
    ) -> None:
        """
        Args:
            processed_dir: Root processed data directory containing the split subdirs.
            split: "train" | "validation" | "test" (or any subdir name under processed_dir).
            total_steps: K. Windows hold K+1 states (y) and K dt values.
            device: Preload device; all shards are loaded into float32 tensors there.
        """
        processed_dir = Path(processed_dir).expanduser().resolve()
        split_dir = processed_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split dir not found: {split_dir}")

        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")

        self.processed_dir = processed_dir
        self.split = str(split)
        self.total_steps = int(total_steps)
        self.device = device

        # Build the shard index (trajectory counts per shard) from header-only reads.
        self.shards, self.T, self.S, self.G = _scan_shards(split_dir)

        self.max_start = int(self.T - (self.total_steps + 1))
        if self.max_start < 0:
            raise ValueError(f"Trajectories too short for total_steps={self.total_steps} (T={self.T}).")

        self._total_traj = int(sum(si.n for si in self.shards))

        self._preload_all()

    # -------------------------------------------------------------------------
    # Preload
    # -------------------------------------------------------------------------

    def _preload_all(self) -> None:
        """Load all shards into contiguous [N,T,S] / [N,G] / [N,T-1] tensors on *device*."""
        total = int(self._total_traj)
        if total <= 0:
            raise RuntimeError("no trajectories to preload")

        # Allocate contiguous buffers on the target device.
        y_all = torch.empty((total, self.T, self.S), device=self.device, dtype=_STORAGE_DTYPE)
        g_all = torch.empty((total, self.G), device=self.device, dtype=_STORAGE_DTYPE)
        dt_all = torch.empty((total, self.T - 1), device=self.device, dtype=_STORAGE_DTYPE)

        # Read each shard and copy it into the contiguous buffers.
        cursor = 0
        for si in self.shards:
            with np.load(si.path) as z:
                # Shards are float32 (checked from the headers); asarray does not copy them.
                y_np = np.asarray(z["y_mat"], dtype=np.float32)
                g_np = np.asarray(z["globals"], dtype=np.float32)
                dt_np = np.asarray(z["dt_norm_mat"], dtype=np.float32)

            n = int(y_np.shape[0])
            if n != int(si.n):
                raise ValueError(f"Shard {si.path} count mismatch: index says {si.n}, file has {n}.")
            if cursor + n > total:
                raise RuntimeError("preload overflow (bug)")

            y_all[cursor : cursor + n].copy_(torch.from_numpy(y_np), non_blocking=False)
            g_all[cursor : cursor + n].copy_(torch.from_numpy(g_np), non_blocking=False)
            dt_all[cursor : cursor + n].copy_(torch.from_numpy(dt_np), non_blocking=False)
            cursor += n

        if cursor != total:
            raise RuntimeError(f"preload size mismatch: filled {cursor}, expected {total}")

        self._preloaded_all = (y_all, g_all, dt_all)


class RawTrajectories:
    """The raw-trajectory store, loaded whole onto `device`: the source of the long-jump pairs.

    logy [N, T, S] float64 (log10 y on the raw time grid t [T], seconds), end [N] int64 (last valid
    row), g [N, G] float32 (globals normalized as in the shards), and the manifest's mu, sd (float64)
    and dt normalization range. logy_at interpolates log10 y linearly in log10 t between raw rows
    1..end: on the production grid row 0 is t = 0, which has no log time, so row 0 is never read.
    """

    def __init__(
        self,
        processed_dir: Union[str, Path],
        *,
        normalization_manifest: Mapping[str, Any],
        species_variables: Sequence[str],
        device: torch.device,
    ) -> None:
        path = raw_store_path(processed_dir)
        with np.load(path, allow_pickle=False) as z:
            logy, end, g = z["logy"], z["end"], z["globals"]
        t = np.load(path.parent / _RAW_STORE_GRID, allow_pickle=False)

        dtypes = (logy.dtype, end.dtype, g.dtype, t.dtype)
        if dtypes != (np.float64, np.int16, np.float32, np.float64):
            raise ValueError(
                f"{path}: logy, end, globals and {_RAW_STORE_GRID} must be float64, int16, float32, float64, "
                f"got {dtypes}; regenerate it with processing/preprocessing.py"
            )
        n, S = (int(end.shape[0]) if end.ndim == 1 else 0), len(species_variables)
        if n < 1 or t.ndim != 1 or logy.shape != (n, t.shape[0], S) or g.ndim != 2 or g.shape[0] != n:
            raise ValueError(
                f"{path}: expected logy [N, T, {S}], end [N], globals [N, G] and a [T] time grid with N >= 1, got "
                f"{logy.shape}, {end.shape}, {g.shape}, {t.shape}"
            )
        # logy_at needs log10 t increasing over rows 1..T-1 and at least two valid rows (1 and 2).
        if not (t.shape[0] >= 3 and t[1] > 0.0 and np.all(np.diff(t[1:]) > 0.0)):
            raise ValueError(f"{path.parent / _RAW_STORE_GRID}: rows 1..T-1 must be positive and increasing")
        if int(end.min()) < 2 or int(end.max()) > t.shape[0] - 1:
            raise ValueError(f"{path}: end rows {int(end.min())}..{int(end.max())} outside 2..{t.shape[0] - 1}")

        stats = normalization_manifest["per_key_stats"]
        self.n = n
        self.device = device
        self.logy = torch.from_numpy(logy).to(device)
        self.end = torch.from_numpy(end.astype(np.int64)).to(device)
        self.g = torch.from_numpy(g).to(device)
        self.t = torch.from_numpy(t).to(device)
        self.logt = torch.log10(self.t[1:])  # positive rows 1..T-1
        self.mu = torch.tensor([stats[s]["log_mean"] for s in species_variables], dtype=torch.float64, device=device)
        self.sd = torch.tensor([stats[s]["log_std"] for s in species_variables], dtype=torch.float64, device=device)
        self.dt_log10_min = float(normalization_manifest["dt"]["log_min"])
        self.dt_log10_max = float(normalization_manifest["dt"]["log_max"])

    def logy_at(self, idx: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Log-log interpolant: idx [B], times [B, M] (s, float64) -> log10 y [B, M, S] float64.

        Times past a trajectory's end row are held at that row's value (w is clamped to [0, 1]).
        """
        lt = torch.log10(times)
        n_valid = self.end[idx]                       # valid rows are 1..end (n_valid of them)
        j = torch.searchsorted(self.logt.expand(lt.shape[0], -1).contiguous(), lt.contiguous(), side="left")
        j = torch.minimum(torch.clamp(j, min=1), (n_valid - 1).reshape(-1, 1))
        i0 = j            # valid index j-1 -> raw row j
        i1 = j + 1        # valid index j   -> raw row j+1
        t0 = self.logt[j - 1]
        t1 = self.logt[j]
        w = ((lt - t0) / (t1 - t0)).clamp(0.0, 1.0)
        b = idx.reshape(-1, 1).expand_as(j)
        y0 = self.logy[b, i0]
        y1 = self.logy[b, i1]
        return (1.0 - w).unsqueeze(-1) * y0 + w.unsqueeze(-1) * y1

    def z_of(self, logy: torch.Tensor) -> torch.Tensor:
        """log10 y -> z (float64)."""
        return (logy - self.mu) / self.sd


def sample_jump_pairs(
    raw: RawTrajectories,
    B: int,
    generator: torch.Generator,
    d_min: float,
    d_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """B pairs (z(t), z(t + D)) with D log-uniform in [d_min, d_max] s and t log-uniform in [t[1], t_end - D].

    About 2B candidates are drawn and the ones that do not fit are moved to the end by a stable
    sort; the first B are kept. The draws come from the CPU `generator` and are moved to raw.device,
    so the pairs do not depend on the device. Returns z0, z1 [B, S] float64, the dt_norm of D [B]
    float32, not clamped (it exceeds 1 for D > dt_max; the relaxation head's network never sees
    dt), and g [B, G] float32.
    """
    dev = raw.device
    m = 2 * B + 256
    idx = torch.randint(0, raw.n, (m,), generator=generator).to(dev)
    lD = np.log10(d_min) + (np.log10(d_max) - np.log10(d_min)) * torch.rand(m, generator=generator, dtype=torch.float64).to(dev)
    Dt = torch.pow(10.0, lD)
    t_lo = raw.t[1]
    t_hi = raw.t[raw.end[idx]] - Dt
    ok = t_hi > t_lo
    u = torch.rand(m, generator=generator, dtype=torch.float64).to(dev)
    lo, hi = torch.log10(t_lo), torch.log10(torch.clamp(t_hi, min=t_lo * (1 + 1e-12)))
    t0 = torch.pow(10.0, lo + u * (hi - lo))
    order = torch.sort(torch.logical_not(ok).to(torch.int8), stable=True).indices[:B]
    idx, t0, Dt = idx[order], t0[order], Dt[order]
    zz = raw.z_of(raw.logy_at(idx, torch.stack([t0, t0 + Dt], 1)))
    dtn = ((torch.log10(Dt) - raw.dt_log10_min) / (raw.dt_log10_max - raw.dt_log10_min)).float()
    return zz[:, 0], zz[:, 1], dtn, raw.g[idx]


@dataclass(frozen=True)
class JumpSampler:
    """Long-jump pairs added to every training batch: batch_size pairs from `raw`, D in [min_seconds, max_seconds]."""
    raw: RawTrajectories
    batch_size: int
    min_seconds: float
    max_seconds: float


class _PreloadedWindowStream(IterableDataset):
    """Shared part of the two batch streams: views of the preload buffers and the window gather.

    The window indices are computed on the CPU and moved to the preload device, so the windows
    do not depend on where the data is preloaded. When the preload device is CPU, DataLoader
    pinning can be used on the yielded batches for efficient host-to-device transfer.
    """

    def __init__(self, base: FlowMapRolloutDataset, *, batch_size: int) -> None:
        super().__init__()

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.base = base
        self.batch_size = int(batch_size)

        self.device = base.device

        self.K = int(base.total_steps)
        self.S = int(base.S)
        self.T = int(base.T)
        self.T_dt = int(base.T - 1)

        self._y_all, self._g_all, self._dt_all = base._preloaded_all
        self.N = int(self._y_all.shape[0])

        self._t_y = torch.arange(self.K + 1, device=self.device, dtype=torch.long)
        self._t_dt = torch.arange(self.K, device=self.device, dtype=torch.long)

    def _gather(self, traj: torch.Tensor, start: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Batch of the windows (traj[b], start[b]) without materializing [B, T, S] intermediates."""
        B = int(traj.shape[0])
        y_flat = self._y_all.reshape(-1, self.S)  # [N*T, S] (view)
        dt_flat = self._dt_all.reshape(-1)        # [N*(T-1)] (view)

        t_y = start[:, None] + self._t_y[None, :]  # [B, K+1]
        lin_y = (traj[:, None] * self.T + t_y).reshape(-1)
        y_win = y_flat.index_select(0, lin_y).reshape(B, self.K + 1, self.S)

        t_dt = start[:, None] + self._t_dt[None, :]  # [B, K]
        lin_dt = (traj[:, None] * self.T_dt + t_dt).reshape(-1)
        dt_win = dt_flat.index_select(0, lin_dt).reshape(B, self.K)

        g = self._g_all.index_select(0, traj)

        return {"y": y_win, "dt": dt_win, "g": g}


class _PreloadedStochasticBatchStream(_PreloadedWindowStream):
    """
    Training stream: `batches_per_epoch` full batches per epoch, windows drawn uniformly at
    random with replacement (SGD-style) from a CPU generator.

    The generator is reseeded at the start of every epoch from the base seed and the epoch set by
    set_epoch(), so epoch e draws the same windows whether or not the run was resumed. With a
    JumpSampler, each batch also gets jump.batch_size long-jump pairs (sample_jump_pairs) from a
    second CPU generator, reseeded the same way under its own stream id.
    """

    def __init__(
        self,
        base: FlowMapRolloutDataset,
        *,
        batch_size: int,
        batches_per_epoch: int,
        seed: int,
        jump: Optional[JumpSampler],
    ) -> None:
        super().__init__(base, batch_size=batch_size)
        if batches_per_epoch <= 0:
            raise ValueError(f"batches_per_epoch must be positive, got {batches_per_epoch}")
        if jump is not None and jump.batch_size <= 0:
            raise ValueError(f"jump batch_size must be positive, got {jump.batch_size}")
        self.batches_per_epoch = int(batches_per_epoch)
        self._seed = int(seed)
        self._epoch = 0
        self._generator = torch.Generator(device="cpu")
        self._jump = jump
        self._jump_generator = torch.Generator(device="cpu")

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch whose windows the next iteration draws."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # The body of this generator runs at the first next(), i.e. after Lightning's
        # on_train_epoch_start has called set_epoch (Lightning builds the iterator earlier).
        state = np.random.SeedSequence([self._seed, _TRAIN_STREAM_ID, self._epoch]).generate_state(1, dtype=np.uint64)
        self._generator.manual_seed(int(state[0]))
        jump = self._jump
        if jump is not None:
            state = np.random.SeedSequence([self._seed, _JUMP_STREAM_ID, self._epoch]).generate_state(1, dtype=np.uint64)
            self._jump_generator.manual_seed(int(state[0]))

        B = self.batch_size
        for _ in range(self.batches_per_epoch):
            traj = torch.randint(0, self.N, (B,), generator=self._generator, dtype=torch.long).to(self.device)
            start = torch.randint(
                0,
                self.base.max_start + 1,
                (B,),
                generator=self._generator,
                dtype=torch.long,
            ).to(self.device)
            batch = self._gather(traj, start)
            if jump is not None:
                z0, z1, dtn, g = sample_jump_pairs(
                    jump.raw, jump.batch_size, self._jump_generator, jump.min_seconds, jump.max_seconds
                )
                batch.update(jump_z0=z0, jump_z1=z1, jump_dt_norm=dtn, jump_g=g)
            yield batch


class _PreloadedValidationStream(_PreloadedWindowStream):
    """
    Validation stream: a fixed, evenly spaced subset of the enumerated windows.

    Window i of the N * n_starts enumerated windows (n_starts = max_start + 1) is trajectory
    i // n_starts at start i % n_starts. The scored set is i_k = (k * total) // count for
    k in [0, count), count = min(n_windows, total): every window when n_windows >= total, an even
    stride otherwise. Batches are consecutive B-sized slices of k (the last one ragged), computed
    per batch, so no index table is materialized.
    """

    def __init__(self, base: FlowMapRolloutDataset, *, batch_size: int, n_windows: int) -> None:
        super().__init__(base, batch_size=batch_size)
        if n_windows <= 0:
            raise ValueError(f"n_windows must be positive, got {n_windows}")
        self.n_starts = int(base.max_start + 1)
        self.total = int(self.N * self.n_starts)
        self.count = int(min(n_windows, self.total))
        # k * total is formed in int64 for k < count.
        if (self.count - 1) * self.total >= _INT64_LIMIT:
            raise ValueError(
                f"validation window index overflows int64 (count={self.count}, total={self.total})"
            )

    def __len__(self) -> int:
        return (self.count + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for b in range(len(self)):
            k = torch.arange(b * self.batch_size, min((b + 1) * self.batch_size, self.count), dtype=torch.long)
            i = (k * self.total) // self.count
            traj = i // self.n_starts
            start = i % self.n_starts
            yield self._gather(traj.to(self.device), start.to(self.device))


def _wrap_stream(stream: IterableDataset, *, pin_memory: bool) -> DataLoader:
    """DataLoader over an already-batching stream, in the main process (num_workers=0)."""
    # main.py rejects pin_memory=true with a CUDA preload device before any data is loaded.
    return DataLoader(
        dataset=stream,
        batch_size=None,  # stream yields already-batched tensors
        shuffle=False,
        num_workers=0,
        pin_memory=bool(pin_memory),
        persistent_workers=False,
    )


def create_train_dataloader(
    dataset: FlowMapRolloutDataset,
    *,
    batch_size: int,
    batches_per_epoch: int,
    seed: int,
    pin_memory: bool,
    jump: Optional[JumpSampler],
) -> DataLoader:
    """Training loader: `batches_per_epoch` full random batches per epoch, plus long-jump pairs
    when `jump` is given (see the stream class)."""
    stream = _PreloadedStochasticBatchStream(
        dataset, batch_size=batch_size, batches_per_epoch=batches_per_epoch, seed=seed, jump=jump
    )
    return _wrap_stream(stream, pin_memory=pin_memory)


def create_val_dataloader(
    dataset: FlowMapRolloutDataset,
    *,
    batch_size: int,
    n_windows: int,
    pin_memory: bool,
) -> DataLoader:
    """Validation loader: the fixed window set of size min(n_windows, N * n_starts)."""
    stream = _PreloadedValidationStream(dataset, batch_size=batch_size, n_windows=n_windows)
    return _wrap_stream(stream, pin_memory=pin_memory)
