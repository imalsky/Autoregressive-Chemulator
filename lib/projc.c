/* (lib copy of fixes-phase checks/solver_speed/projc.c, 2026-10-12; driver lib/projc.py)
   Structured element-ratio projection on the CPU (fixes phase 2026-10-12; author: "can this be delegated to CPUs").
   Same equations and solver as struct_lm.StructLM: species grouped by their (C, N, O) atom counts, He exact,
   Levenberg-Marquardt on the 3 C/N/O residuals with a scalar step cap, a step taken only if |F|^2 falls, fixed iterations.
   One cell per loop iteration, OpenMP across cells, fp64. stop > 0: a cell stops once max |F| < stop (converged).
   gcc -O3 -march=native -fopenmp -fPIC -shared projc.c -o projc.so -lm */
#include <math.h>
#include <string.h>

#define KMAX 16
#define SMAX 32

static void totals(int K, const double *G, const double *Hg, const double *P, const double *lam,
                   double *E, double *H, double dE[3][3], double dH[3]) {
    for (int e = 0; e < 3; e++) { E[e] = 0.0; dH[e] = 0.0; for (int f = 0; f < 3; f++) dE[e][f] = 0.0; }
    *H = 0.0;
    for (int k = 0; k < K; k++) {
        const double *p = P + 3 * k;
        double m = exp(p[0] * lam[0] + p[1] * lam[1] + p[2] * lam[2]);
        double gm = G[k] * m, hm = Hg[k] * m;
        *H += hm;
        for (int f = 0; f < 3; f++) dH[f] += p[f] * hm;
        for (int e = 0; e < 3; e++) {
            if (p[e] == 0.0) continue;
            double v = p[e] * gm;
            E[e] += v;
            for (int f = 0; f < 3; f++) dE[e][f] += p[f] * v;
        }
    }
}

static int solve3(double A[3][3], const double *b, double *x) {   /* Cholesky, A SPD */
    double l00 = sqrt(fmax(A[0][0], 1e-300));
    double l10 = A[1][0] / l00, l20 = A[2][0] / l00;
    double l11 = sqrt(fmax(A[1][1] - l10 * l10, 1e-300));
    double l21 = (A[2][1] - l20 * l10) / l11;
    double l22 = sqrt(fmax(A[2][2] - l20 * l20 - l21 * l21, 1e-300));
    double y0 = b[0] / l00, y1 = (b[1] - l10 * y0) / l11, y2 = (b[2] - l20 * y0 - l21 * y1) / l22;
    x[2] = y2 / l22; x[1] = (y1 - l21 * x[2]) / l11; x[0] = (y0 - l10 * x[1] - l20 * x[2]) / l00;
    return isfinite(x[0]) && isfinite(x[1]) && isfinite(x[2]);
}

/* lt [B,S] relaxed ln y; lr [B,4] target ln ratios to H of (C, N, O, He); grp [S] group index (-1: He); P [K,3];
   hs [S] H atoms; as [S,4] (C, N, O, He) atoms; mol [S] 1 if counted in the molecular sum. out [B,S] ln y'; res [B]. */
void project(int B, int S, int K, const double *lt, const double *lr, const int *grp, const double *P,
             const double *hs, const double *as, const int *mol, int he, int iters, double cap, double stop,
             double *out, double *res) {
#pragma omp parallel for schedule(static)
    for (int c = 0; c < B; c++) {
        const double *l = lt + (size_t)c * S, *r = lr + (size_t)c * 4;
        double sh = -INFINITY;
        for (int s = 0; s < S; s++) if (s != he && l[s] > sh) sh = l[s];
        double G[KMAX] = {0}, Hg[KMAX] = {0};
        for (int s = 0; s < S; s++) {
            if (grp[s] < 0) continue;
            double b = exp(l[s] - sh);
            G[grp[s]] += b;
            Hg[grp[s]] += hs[s] * b;
        }
        double lam[3] = {0, 0, 0}, damp = 1e-3, E[3], H, dE[3][3], dH[3], F[3];
        totals(K, G, Hg, P, lam, E, &H, dE, dH);
        for (int e = 0; e < 3; e++) F[e] = log(E[e]) - log(H) - r[e];
        for (int it = 0; it < iters; it++) {
            if (stop > 0.0 && fmax(fabs(F[0]), fmax(fabs(F[1]), fabs(F[2]))) < stop) break;  /* converged: done */
            double J[3][3], A[3][3], g[3], st[3];
            for (int e = 0; e < 3; e++)
                for (int f = 0; f < 3; f++) J[e][f] = dE[e][f] / E[e] - dH[f] / H;
            for (int i = 0; i < 3; i++) {
                g[i] = 0.0;
                for (int e = 0; e < 3; e++) g[i] += J[e][i] * F[e];
                for (int j = 0; j < 3; j++) {
                    double v = 0.0;
                    for (int e = 0; e < 3; e++) v += J[e][i] * J[e][j];
                    A[i][j] = v + (i == j ? damp : 0.0);
                }
            }
            if (!solve3(A, g, st)) { st[0] = st[1] = st[2] = 0.0; }
            double mx = fmax(fabs(st[0]), fmax(fabs(st[1]), fabs(st[2])));
            double sc = mx > cap ? cap / mx : 1.0;
            double lam2[3] = {lam[0] - sc * st[0], lam[1] - sc * st[1], lam[2] - sc * st[2]};
            double E2[3], H2, dE2[3][3], dH2[3], F2[3];
            totals(K, G, Hg, P, lam2, E2, &H2, dE2, dH2);
            double n1 = 0.0, n2 = 0.0;
            for (int e = 0; e < 3; e++) { F2[e] = log(E2[e]) - log(H2) - r[e]; n1 += F[e] * F[e]; n2 += F2[e] * F2[e]; }
            if (n2 < n1) {
                memcpy(lam, lam2, sizeof lam); memcpy(E, E2, sizeof E); memcpy(F, F2, sizeof F);
                memcpy(dE, dE2, sizeof dE); memcpy(dH, dH2, sizeof dH); H = H2;
                damp = fmax(damp * 0.3, 1e-12);
            } else {
                damp = fmin(damp * 10.0, 1e12);
            }
        }
        double lhe = r[3] + log(H) + sh - l[he];
        double *w = out + (size_t)c * S, mx = -INFINITY;
        for (int s = 0; s < S; s++) {
            const double *a = as + 4 * s;
            w[s] = l[s] + a[0] * lam[0] + a[1] * lam[1] + a[2] * lam[2] + a[3] * lhe;
            if (mol[s] && w[s] > mx) mx = w[s];
        }
        double sm = 0.0;
        for (int s = 0; s < S; s++) if (mol[s]) sm += exp(w[s] - mx);
        double ln = mx + log(sm);
        for (int s = 0; s < S; s++) w[s] -= ln;
        /* residual from the output itself (as elemproj.log_ratios) */
        double tH = 0.0, tE[4] = {0, 0, 0, 0};
        for (int s = 0; s < S; s++) {
            double y = exp(w[s]);
            tH += hs[s] * y;
            for (int e = 0; e < 4; e++) tE[e] += as[4 * s + e] * y;
        }
        double rm = 0.0;
        for (int e = 0; e < 4; e++) rm = fmax(rm, fabs(log(tE[e]) - log(tH) - r[e]));
        res[c] = isfinite(rm) ? rm : INFINITY;
    }
}
