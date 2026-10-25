/*
 * Xorcist per-bin kernels - the part that belongs on a GPU.
 *
 * Every FFT bin carries its OWN order-p recursion, fitted from its own
 * exponentially forgotten normal equations.  At order 8 and six channels
 * that is 2049 * 6 = 12294 independent p x p systems per period, each a
 * few hundred flops with no cross-talk whatsoever - the shape a GPU is
 * built for and the shape a CPU core has to walk one at a time.  The
 * scaling is the argument: the solve is O(p^3) plus O(p^2) per
 * Durand-Kerner step, so going from order 4 to 20 is ~125x the work in
 * that stage, while the useful order is material-dependent and on dense
 * clusters demonstrably high (order 4 -5.4 dB, order 8 -32.2, order 12
 * -46.3 on a twelve-note cluster with eight partials each).
 *
 * DUAL MODE.  This file is valid OpenCL C and, with XO_HOST_C defined,
 * valid C99.  That is deliberate: it makes the kernels testable against
 * the reference implementation on a machine with no OpenCL device at
 * all, and it removes the classic failure mode where the CPU and GPU
 * paths drift apart because they are two texts.  There is one text.
 *
 * Precision: float throughout for the spectra and coefficients (measured
 * to cost nothing), double for the accumulators (measured to cost up to
 * 1.1 dB in float).  A GPU without cl_khr_fp64 therefore needs the
 * accumulator arrays in float and will pay that dB - checked at runtime
 * by the host, not silently.
 *
 * Copyright (C) 2026 Sebion7125
 *
 * Dual-licensed: use under the GNU General Public License v3.0 or
 * later (see LICENSE), or under a separate commercial license for
 * proprietary use without the GPL's obligations - contact
 * sebion7125@gmail.com for commercial licensing terms.
 */

#ifdef XO_HOST_C
/* --- C99 mode: emulate the OpenCL address-space and vector spellings -- */
#include <math.h>
#include <stddef.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define __kernel
#define __global
#define __constant const
#define XO_GID(d) (xo_gid)
static int xo_gid; /* set by the host-side test driver per invocation */
#else
/* --- OpenCL mode ------------------------------------------------------ */
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#define XO_GID(d) ((int)get_global_id(d))
#endif

/* Private-memory budget.  The solve needs an order x (2*order+2) scratch
 * matrix per work-item, so sizing it for the LARGEST order the host might
 * ever ask for is the classic way to wreck a GPU port: at 20 that is
 * 6.7 kB per work-item, which on NVIDIA spills straight out of the
 * register file into local (= global) memory and gives back everything
 * the parallelism won.  The host therefore compiles the program with
 * -DXO_ORDER_MAX=<the order actually in use> - kernel builds are cached
 * and the order changes at most when a user turns a knob.  The default
 * below is only the fallback for a hand-compiled kernel. */
#ifndef XO_ORDER_MAX
#define XO_ORDER_MAX 8
#endif

#ifndef XO_SCHUR
#define XO_SCHUR 0
#endif

/* Working precision, PER STAGE - because the stages are not equally
 * sensitive and a consumer GPU makes double expensive (an RTX 3070 runs
 * FP64 at 1/64 of its FP32 rate, 0.317 against 20.3 TFLOPS).
 *
 *   XO_FP32_STORE   the normal equations P and q in memory
 *   XO_FP32_SOLVE   the Gauss-Jordan scratch matrix and its arithmetic
 *   XO_FP32_ROOT    Durand-Kerner, the clamp and the coefficient rebuild
 *
 * XO_FP32 sets all three.  MEASURED, and the result is the opposite of
 * what one would guess: the elimination is fine in float (max angular
 * root deviation 0.0000 degrees at order 8 AND 12), while DURAND-KERNER
 * is not - in float it converges on a DIFFERENT root set in a handful of
 * bins, up to 174.6 degrees away, which is not an inaccuracy but another
 * root.  So the recommended mix is
 *
 *     -DXO_FP32_STORE=1 -DXO_FP32_SOLVE=1     (root finding left double)
 *
 * which measures 0.0000 degrees, coefficients within 1e-07, and is the
 * fastest of all combinations tried.  The intuition that failed: the
 * elimination handles large nearly cancelling numbers, so it looked like
 * the fragile stage - but it is a FINITE sequence of operations, whereas
 * Durand-Kerner is an iteration whose fixed point can move to a
 * neighbouring root once the steps stop separating them.
 *
 * What matters downstream is not the coefficient but the ROOT SET, and
 * of a root only its ANGLE survives unfiltered: the angle is the
 * component's frequency, the modulus is its growth, and the clamp
 * normalises the modulus anyway for every root above 0.9.  A precision
 * error therefore only hurts to the extent that it moves angles - which
 * is why the benchmark reports angular deviation next to the
 * coefficient distance.  Coefficient distance alone says nothing. */
#ifdef XO_FP32
#ifndef XO_FP32_STORE
#define XO_FP32_STORE 1
#endif
#ifndef XO_FP32_SOLVE
#define XO_FP32_SOLVE 1
#endif
#ifndef XO_FP32_ROOT
#define XO_FP32_ROOT 1
#endif
#endif

#ifndef XO_FP32_STORE
#define XO_FP32_STORE 0
#endif
#ifndef XO_FP32_SOLVE
#define XO_FP32_SOLVE 0
#endif
#ifndef XO_FP32_ROOT
#define XO_FP32_ROOT 0
#endif

#if XO_FP32_STORE
typedef float xo_store;
#else
typedef double xo_store;
#endif
#if XO_FP32_SOLVE
typedef float xo_solv;
#else
typedef double xo_solv;
#endif
#if XO_FP32_ROOT
typedef float xo_root;
#else
typedef double xo_root;
#endif
typedef xo_solv xo_real; /* legacy spelling inside the RLS kernel */

/* ---------------------------------------------------------------------
 * Kernel 1: recursive least squares update, one work-item per bin.
 *
 * P[k][l] += X_k conj(X_l),  q[k] += X_0 conj(X_k),  all scaled by lam
 * first.  X_0 is this grid point's frame, X_k the one k*arHop earlier.
 * Layout is [bin][k*order + l] complex-interleaved so that consecutive
 * work-items touch consecutive memory: on a GPU that is the difference
 * between coalesced and not.
 * ------------------------------------------------------------------- */
__kernel void xo_rls_update(
  __global xo_store *P,       /* [nbins][order*order][2] */
  __global xo_store *q,       /* [nbins][order][2]       */
  __global const float *spec, /* [order+1][nbins][2] - index 0 = newest */
  const int nbins,
  const int order,
  const xo_real lam) {
  const int bin = XO_GID(0);

  if (bin >= nbins)
    return;

  const xo_real x0r = spec[(0 * nbins + bin) * 2];
  const xo_real x0i = spec[(0 * nbins + bin) * 2 + 1];
  __global xo_store *Pb = P + (size_t)bin * order * order * 2;
  __global xo_store *qb = q + (size_t)bin * order * 2;

  for (int k = 0; k < order; k++) {
    const xo_real xkr = spec[((k + 1) * nbins + bin) * 2];
    const xo_real xki = spec[((k + 1) * nbins + bin) * 2 + 1];

    qb[2 * k] = lam * qb[2 * k] + x0r * xkr + x0i * xki;
    qb[2 * k + 1] = lam * qb[2 * k + 1] + x0i * xkr - x0r * xki;
    for (int l = 0; l < order; l++) {
      const xo_real xlr = spec[((l + 1) * nbins + bin) * 2];
      const xo_real xli = spec[((l + 1) * nbins + bin) * 2 + 1];
      const int o = 2 * (k * order + l);

      Pb[o] = lam * Pb[o] + xkr * xlr + xki * xli;
      Pb[o + 1] = lam * Pb[o + 1] + xki * xlr - xkr * xli;
    }
  }
}

/* ---------------------------------------------------------------------
 * Schur-Cohn stabilisation, root-free.
 *
 * The coefficient vector and the reflection coefficients k are two
 * equivalent parametrisations of the same model, convertible in O(p^2)
 * both ways (Schur descent 6.3, Levinson ascent 6.4 in the derivation).
 * In the k-parametrisation stability is a BOX condition - every
 * |k_l| < 1 - so it can be enforced by clamping instead of searched for.
 *
 * Why this matters here: the only fp32-fragile stage measured was
 * Durand-Kerner, at 174.6 degrees of angular root deviation while
 * elimination and storage sat at 0.0000.  That is not rounding, it is a
 * fixed-point iteration landing on a NEIGHBOURING root.  Descent and
 * ascent are finite operation sequences - the same category as the
 * elimination, which showed no precision problem at all - so this
 * removes the iteration rather than tightening it.
 *
 * k_l also has meaning: |k_l| < 1 says the prediction's residual power
 * stays positive (sigma_l^2 = sigma_{l-1}^2 (1 - |k_l|^2)), and in the
 * tube picture it is the reflection at a bore transition.  For an organ
 * pipe that is not an analogy but the model itself.
 *
 * Caveat, and it is the reason this is measured against the root clamp
 * rather than assumed better: clamping k_l changes EVERY mode that
 * depends on it, not just the unstable one.  The selective root clamp
 * touches only roots outside the circle.
 * ------------------------------------------------------------------- */
static void xo_schur_clamp(
  xo_root *ar, xo_root *ai, int order, xo_root delta) {
  xo_root kr[XO_ORDER_MAX], ki[XO_ORDER_MAX];
  xo_root ur[XO_ORDER_MAX + 1], ui[XO_ORDER_MAX + 1];
  int ok = 1;

  /* alpha convention here: a_i are the RECURSION coefficients, so the
   * characteristic polynomial is z^p - a_1 z^(p-1) - ... - a_p and the
   * Schur vector is alpha_i = -a_i. */
  ur[0] = (xo_root)1;
  ui[0] = (xo_root)0;
  for (int i = 1; i <= order; i++) {
    ur[i] = -ar[i - 1];
    ui[i] = -ai[i - 1];
  }
  /* --- descent: alpha -> k, degree by degree ------------------------ */
  for (int l = order; l >= 1; l--) {
    kr[l - 1] = ur[l];
    ki[l - 1] = ui[l];

    const xo_root km = kr[l - 1] * kr[l - 1] + ki[l - 1] * ki[l - 1];
    /* 1 - |k|^2 is NEGATIVE whenever |k| > 1 - which is precisely the
     * case the clamp exists to repair, so it must not be rejected.  Only
     * |k| ~ 1 needs protection: there the denominator vanishes, and that
     * is the normal state of a sustained tone (a root exactly ON the
     * circle), not an edge case.
     *
     * Guarding with `den > eps` instead of `fabs(den) > eps` was the bug:
     * every bin with an unstable root bailed out and was returned
     * UNCHANGED, so the invariant failed by a hair (max |z| 1.0028 at
     * order 8) instead of failing loudly. */
    const xo_root den = (xo_root)1 - km;

    if (!(fabs(den) > (xo_root)1e-6)) {
      ok = 0;
      break;
    }
    /* Two passes: step i reads alpha_{l-i}, which an in-place write has
     * already replaced for i > l/2.  Verified against the derivation's
     * reference trace, which is what caught it. */
    {
      xo_root nr[XO_ORDER_MAX + 1], ni[XO_ORDER_MAX + 1];

      for (int i = 1; i < l; i++) {
        nr[i] = (ur[i] - (kr[l - 1] * ur[l - i] + ki[l - 1] * ui[l - i])) / den;
        ni[i] = (ui[i] - (ki[l - 1] * ur[l - i] - kr[l - 1] * ui[l - i])) / den;
      }
      for (int i = 1; i < l; i++) {
        ur[i] = nr[i];
        ui[i] = ni[i];
      }
    }
  }
  if (!ok)
    return; /* leave the caller's coefficients alone */

  /* --- clamp: the box condition ------------------------------------- */
  for (int l = 0; l < order; l++) {
    const xo_root m2 = kr[l] * kr[l] + ki[l] * ki[l];

    if (m2 >= (xo_root)1) {
      const xo_root sc = ((xo_root)1 - delta) / sqrt(m2);

      kr[l] *= sc;
      ki[l] *= sc;
    }
  }
  /* --- ascent: k -> alpha (Levinson) -------------------------------- */
  ur[0] = (xo_root)1;
  ui[0] = (xo_root)0;
  for (int l = 1; l <= order; l++) {
    xo_root nr[XO_ORDER_MAX + 1], ni[XO_ORDER_MAX + 1];

    for (int i = 1; i < l; i++) {
      nr[i] = ur[i] + (kr[l - 1] * ur[l - i] + ki[l - 1] * ui[l - i]);
      ni[i] = ui[i] + (ki[l - 1] * ur[l - i] - kr[l - 1] * ui[l - i]);
    }
    for (int i = 1; i < l; i++) {
      ur[i] = nr[i];
      ui[i] = ni[i];
    }
    ur[l] = kr[l - 1];
    ui[l] = ki[l - 1];
  }
  for (int i = 1; i <= order; i++) {
    ar[i - 1] = -ur[i];
    ai[i - 1] = -ui[i];
  }
}

/* rho TEST (1:1 match of Xorcist::RhoTest in xorcist.cpp).  Schur
 * descent run as a pure TEST: bails out at the first violation, no
 * clamping, no re-ascent.  Returns 1 = all roots strictly inside rho.
 *
 * This gives the kernel the DETECTOR MODE that until now only the CPU
 * path had - previously the host locked the GPU out entirely whenever
 * detectRho > 0, so the "GPU" checkbox had no effect at the default
 * rho = 1.02.  Side effect: 96% of bins pass the test and skip
 * Durand-Kerner entirely - the test costs O(p^2), the iteration
 * O(40 p^2). */
static int xo_rho_test(
  const xo_solv *ar, const xo_solv *ai, const int order, const xo_solv rho) {
  xo_solv ur[XO_ORDER_MAX + 1], ui[XO_ORDER_MAX + 1];
  xo_solv f = (xo_solv)1;

  ur[0] = (xo_solv)1;
  ui[0] = (xo_solv)0;
  for (int i = 1; i <= order; i++) {
    f /= rho;
    ur[i] = -ar[i - 1] * f;
    ui[i] = -ai[i - 1] * f;
  }
  for (int l = order; l >= 1; l--) {
    const xo_solv kr = ur[l], ki = ui[l];
    const xo_solv km = kr * kr + ki * ki;

    if (km >= (xo_solv)1)
      return 0; /* box condition violated: root outside rho */

    const xo_solv den = (xo_solv)1 - km;
    xo_solv nr[XO_ORDER_MAX + 1], ni[XO_ORDER_MAX + 1];

    for (int i = 1; i < l; i++) {
      nr[i] = (ur[i] - (kr * ur[l - i] + ki * ui[l - i])) / den;
      ni[i] = (ui[i] - (ki * ur[l - i] - kr * ui[l - i])) / den;
    }
    for (int i = 1; i < l; i++) {
      ur[i] = nr[i];
      ui[i] = ni[i];
    }
  }
  return 1;
}

/* ---------------------------------------------------------------------
 * Kernel 2: solve the p x p system and turn it into stable coefficients.
 *
 * Gaussian elimination with partial pivoting, then Durand-Kerner on the
 * companion polynomial, warm-started from the bin's previous roots.
 * Only roots ALREADY near the unit circle are snapped onto it: a bin
 * holding a single partial yields {z, -z/2, ...}, and those extra roots
 * are the fit's shadow - they must be left to decay over the horizon.
 * Forcing them all onto the circle installs permanent parasitic
 * oscillation (measured +4.1 dB on a pure sine where the plain rotator
 * reaches -75).
 *
 * This is the kernel the whole exercise is about: it is O(p^3) and it is
 * where the CPU runs out of room.
 *
 * SOLVE TAP (optional, tap != 0): one row per bin for observability,
 *
 *     [ max|z| , branch , Re z0 , Im z0 , Re z1 , Im z1 , ... ]
 *
 * i.e. 2 + 2*order floats.  Everything is captured BEFORE clamping:
 * with the threshold drawn as a color break or a circle, the display
 * shows exactly the poles that ran up against the clamp; AFTER
 * clamping they would all sit exactly on the threshold and be
 * indistinguishable from the ones just below it.  Where no roots are
 * produced at all (detector passed, or the Gaussian elimination was
 * singular), the radius is -1 and the pole is NAN - a distinct
 * category, not a faked number (a zero would read as a pole at the
 * origin, i.e. a false claim).
 * ------------------------------------------------------------------- */
#define XO_TAP_SINGULAR 0.0f   /* Gaussian elimination non-finite, coefficients zeroed */
#define XO_TAP_DETECTOR 1.0f   /* rho test passed, raw coefficients passed through */
#define XO_TAP_ROTATOR  2.0f   /* order 1 */
#define XO_TAP_QUADRAT  3.0f   /* order 2, closed form */
#define XO_TAP_DK       4.0f   /* Durand-Kerner */
#define XO_TAP_SCHUR    5.0f   /* root-free path (dead in the GPU build) */
#define XO_TAP_NONFIN   6.0f   /* at least one root non-finite */

/* Row width depends on the order, which the kernel already knows. */
#define XO_TAP_ROW (tap + (size_t)bin * (2 + 2 * order))

#define XO_TAP(mz_, path_)                                              \
  do {                                                                  \
    if (tap != 0) {                                                     \
      __global float *tr_ = XO_TAP_ROW;                                 \
                                                                        \
      tr_[0] = (float)(mz_);                                            \
      tr_[1] = (path_);                                                 \
    }                                                                   \
  } while (0)

/* Pole k, BEFORE clamping.  Must be called BEFORE any change to z -
   the whole point of the display is to see where the fit WANTED to
   go, before the clamp pulls it back. */
#define XO_TAP_ROOT(k_, re_, im_)                                       \
  do {                                                                  \
    if (tap != 0) {                                                     \
      __global float *tr_ = XO_TAP_ROW;                                 \
                                                                        \
      tr_[2 + 2 * (k_)] = (float)(re_);                                 \
      tr_[3 + 2 * (k_)] = (float)(im_);                                 \
    }                                                                   \
  } while (0)

__kernel void xo_solve(
  __global const xo_store *P,
  __global const xo_store *q,
  __global float *coef,      /* [nbins][order][2] */
  __global float *roots,     /* [nbins][order][2] - warm start, updated */
  const int nbins,
  const int order,
  const int binFirst,        /* stripe start - a rolling slice per call */
  const int binLast,
  const float clampT,        /* root-clamp trigger; engine: 0.9 default */
  const float detRho,        /* > 0 = detector mode, 0 = classic */
  __global float *tap) {     /* [nbins][2] solve tap, 0 = off */
  const int idx = XO_GID(0);
  const int bin = binFirst + idx;

  if (bin >= binLast || bin >= nbins)
    return;

  /* Clear the pole slots preemptively: the row survives across calls,
     and a branch that computes no roots would otherwise leave the
     PREVIOUS call's roots standing - which would look like a result. */
  if (tap != 0)
    for (int k = 0; k < order; k++) {
      XO_TAP_ROOT(k, NAN, NAN);
    }

  xo_solv m[XO_ORDER_MAX][2 * XO_ORDER_MAX + 2];
  __global const xo_store *Pb = P + (size_t)bin * order * order * 2;
  __global const xo_store *qb = q + (size_t)bin * order * 2;

  for (int k = 0; k < order; k++) {
    for (int l = 0; l < order; l++) {
      /* row k of the transposed system: the coefficient of a_l is P[l][k] */
      m[k][2 * l] = Pb[2 * (l * order + k)];
      m[k][2 * l + 1] = Pb[2 * (l * order + k) + 1];
    }
    m[k][2 * order] = qb[2 * k];
    m[k][2 * order + 1] = qb[2 * k + 1];
    m[k][2 * k] += (xo_real)1e-30; /* never exactly singular */
  }
  for (int c = 0; c < order; c++) {
    int piv = c;
    xo_real best = m[c][2 * c] * m[c][2 * c] + m[c][2 * c + 1] * m[c][2 * c + 1];

    for (int r = c + 1; r < order; r++) {
      const xo_real v
        = m[r][2 * c] * m[r][2 * c] + m[r][2 * c + 1] * m[r][2 * c + 1];

      if (v > best) {
        best = v;
        piv = r;
      }
    }
    /* Engine: 1e-300.  In float this literal underflows to 0, so the
     * bail-out never triggers there - exactly what a float engine
     * would do. */
    if (best < (xo_real)1e-300) {
      for (int k = 0; k < order; k++) {
        coef[(size_t)bin * order * 2 + 2 * k] = 0.0f;
        coef[(size_t)bin * order * 2 + 2 * k + 1] = 0.0f;
      }
      return;
    }
    if (piv != c)
      for (int j = 0; j <= 2 * order + 1; j++) {
        const xo_real t = m[c][j];

        m[c][j] = m[piv][j];
        m[piv][j] = t;
      }

    const xo_real dr = m[c][2 * c], di = m[c][2 * c + 1];
    const xo_real inv = 1.0 / (dr * dr + di * di);

    for (int j = c; j <= order; j++) {
      const xo_real xr = m[c][2 * j], xi = m[c][2 * j + 1];

      m[c][2 * j] = (xr * dr + xi * di) * inv;
      m[c][2 * j + 1] = (xi * dr - xr * di) * inv;
    }
    for (int r = 0; r < order; r++) {
      if (r == c)
        continue;

      const xo_real fr = m[r][2 * c], fi = m[r][2 * c + 1];

      if (fr == 0.0 && fi == 0.0)
        continue;
      for (int j = c; j <= order; j++) {
        const xo_real xr = m[c][2 * j], xi = m[c][2 * j + 1];

        m[r][2 * j] -= fr * xr - fi * xi;
        m[r][2 * j + 1] -= fr * xi + fi * xr;
      }
    }
  }

  xo_solv ar[XO_ORDER_MAX], ai[XO_ORDER_MAX];

  for (int k = 0; k < order; k++) {
    ar[k] = m[k][2 * order];
    ai[k] = m[k][2 * order + 1];
    if (!isfinite(ar[k]) || !isfinite(ai[k])) {
      for (int j = 0; j < order; j++) {
        coef[(size_t)bin * order * 2 + 2 * j] = 0.0f;
        coef[(size_t)bin * order * 2 + 2 * j + 1] = 0.0f;
      }
      XO_TAP(-1.0f, XO_TAP_SINGULAR);
      return;
    }
  }

  /* DETECTOR MODE.  Order exactly as in SolveBin: first the rho test
   * on the RAW Gaussian-elimination coefficients, then order 1, then
   * Schur.  Passed -> pass the coefficients through unchanged, so no
   * Durand-Kerner and no rebuild: each mode's fitted envelope is
   * preserved.  Order 1 doesn't participate in this (the rotator has
   * only one root and its own normalization). */
  if (detRho > 0.0f && order >= 2
      && xo_rho_test(ar, ai, order, (xo_solv)detRho)) {
    for (int k = 0; k < order; k++) {
      coef[(size_t)bin * order * 2 + 2 * k] = (float)ar[k];
      coef[(size_t)bin * order * 2 + 2 * k + 1] = (float)ai[k];
    }
    /* The test only PROVES "all roots inside rho", it doesn't compute
     * any - there's no radius to report here. */
    XO_TAP(-1.0f, XO_TAP_DETECTOR);
    return;
  }

#if XO_SCHUR
  /* Root-free path: no iteration, therefore no fp32 fragility. */
  {
    xo_root sr[XO_ORDER_MAX], si[XO_ORDER_MAX];

    for (int k = 0; k < order; k++) {
      sr[k] = (xo_root)ar[k];
      si[k] = (xo_root)ai[k];
    }
    xo_schur_clamp(sr, si, order, (xo_root)1e-3);
    for (int k = 0; k < order; k++) {
      coef[(size_t)bin * order * 2 + 2 * k] = (float)sr[k];
      coef[(size_t)bin * order * 2 + 2 * k + 1] = (float)si[k];
    }
    XO_TAP(-1.0f, XO_TAP_SCHUR);
    return;
  }
#endif

  xo_root zr[XO_ORDER_MAX], zi[XO_ORDER_MAX];

  if (order == 1) {
    const xo_real mm = sqrt(ar[0] * ar[0] + ai[0] * ai[0]);

    XO_TAP_ROOT(0, ar[0], ai[0]);   // before the normalization right below
    if (mm > (xo_real)clampT) {
      ar[0] /= mm;
      ai[0] /= mm;
    }
    coef[(size_t)bin * order * 2] = (float)ar[0];
    coef[(size_t)bin * order * 2 + 1] = (float)ai[0];
    /* For the rotator, the coefficient magnitude IS the pole radius. */
    XO_TAP(mm, XO_TAP_ROTATOR);
    return;
  }
  float tapPath = XO_TAP_DK;

  if (order == 2) {
    tapPath = XO_TAP_QUADRAT;
    /* closed form - iterating on a quadratic is pure waste */
    const xo_root dr2 = ar[0] * ar[0] - ai[0] * ai[0] + 4.0 * ar[1];
    const xo_root di2 = 2.0 * ar[0] * ai[0] + 4.0 * ai[1];
    const xo_root dm = sqrt(dr2 * dr2 + di2 * di2);
    const xo_root sr2 = sqrt(fmax(0.0, 0.5 * (dm + dr2)));
    xo_root si2 = sqrt(fmax(0.0, 0.5 * (dm - dr2)));

    if (di2 < 0.0)
      si2 = -si2;
    zr[0] = 0.5 * (ar[0] + sr2);
    zi[0] = 0.5 * (ai[0] + si2);
    zr[1] = 0.5 * (ar[0] - sr2);
    zi[1] = 0.5 * (ai[0] - si2);
  } else {
    /* warm start from the previous roots; fall back to the circle seed
     * on the first solve or when two stored roots (nearly) coincide -
     * Durand-Kerner needs pairwise distinct start values */
    int warm = (roots != 0);
    xo_root norm = 0.0;

    if (warm) {
      for (int k = 0; k < order; k++) {
        zr[k] = roots[(size_t)bin * order * 2 + 2 * k];
        zi[k] = roots[(size_t)bin * order * 2 + 2 * k + 1];
        norm += zr[k] * zr[k] + zi[k] * zi[k];
        if (!isfinite(zr[k]) || !isfinite(zi[k]))
          warm = 0;
      }
      if (norm < 1e-4)
        warm = 0;
      for (int i = 0; i < order && warm; i++)
        for (int j = i + 1; j < order; j++) {
          const xo_root d2 = (zr[i] - zr[j]) * (zr[i] - zr[j])
            + (zi[i] - zi[j]) * (zi[i] - zi[j]);

          if (d2 < 1e-8) {
            warm = 0;
            break;
          }
        }
    }
    if (!warm)
      for (int k = 0; k < order; k++) {
        const xo_root ang = 2.0 * M_PI * k / order + 0.35;

        zr[k] = 0.9 * cos(ang);
        zi[k] = 0.9 * sin(ang);
      }
    for (int it = 0; it < 40; it++) {
      xo_root moved = 0.0;

      for (int k = 0; k < order; k++) {
        xo_root vr = 1.0, vi = 0.0;

        for (int j = 0; j < order; j++) {
          const xo_root tr = vr * zr[k] - vi * zi[k];
          const xo_root ti = vr * zi[k] + vi * zr[k];

          vr = tr - ar[j];
          vi = ti - ai[j];
        }

        xo_root dr3 = 1.0, di3 = 0.0;

        for (int j = 0; j < order; j++) {
          if (j == k)
            continue;

          const xo_root er = zr[k] - zr[j], ei = zi[k] - zi[j];
          const xo_root tr = dr3 * er - di3 * ei;

          di3 = dr3 * ei + di3 * er;
          dr3 = tr;
        }

        const xo_root den = dr3 * dr3 + di3 * di3;

        if (den < (xo_root)1e-300)   /* Engine: 1e-300 */
          continue;

        const xo_root qr = (vr * dr3 + vi * di3) / den;
        const xo_root qi = (vi * dr3 - vr * di3) / den;

        zr[k] -= qr;
        zi[k] -= qi;
        moved += qr * qr + qi * qi;
      }
      /* Convergence threshold must match the WORKING PRECISION, not the
       * algorithm's wish.  `moved` is a sum of SQUARED corrections, so
       * 1e-20 means corrections of 1e-10 - unreachable in float, whose
       * resolution stops near 1e-7.  With that threshold the float
       * iteration never exits early: measured 17.9 ms against 2.2 ms for
       * the full double version, i.e. SLOWER while being wrong, because
       * it burned all 40 iterations and let pairs of roots collapse onto
       * each other (max angular deviation 174.6 degrees, which is not an
       * inaccuracy but a different root). */
      if (moved < (xo_root)(sizeof(xo_root) == 4 ? 1e-12 : 1e-20))
        break;
    }
  }
  float tapMax = 0.0f;

  for (int k = 0; k < order; k++) {
    const xo_root mm = sqrt(zr[k] * zr[k] + zi[k] * zi[k]);

    XO_TAP_ROOT(k, zr[k], zi[k]);   // ungedeckelt, siehe Kopfkommentar
    if (!isfinite(mm)) {
      zr[k] = 0.0;
      zi[k] = 0.0;
      tapPath = XO_TAP_NONFIN;
    } else if (
      detRho > 0.0f ? (mm > (xo_root)detRho) : (mm > (xo_root)clampT)) {
      /* Detector mode: CLAMP to rho (budget); classic mode: to 1.0. */
      const xo_root tgt = detRho > 0.0f ? (xo_root)detRho : (xo_root)1;

      zr[k] *= tgt / mm;
      zi[k] *= tgt / mm;
    }
    /* Measured BEFORE clamping - see the header comment above. */
    if (isfinite(mm) && (float)mm > tapMax)
      tapMax = (float)mm;
    if (roots != 0) {
      roots[(size_t)bin * order * 2 + 2 * k] = (float)zr[k];
      roots[(size_t)bin * order * 2 + 2 * k + 1] = (float)zi[k];
    }
  }
  XO_TAP(tapMax, tapPath);

  /* coefficients back from the clamped roots: prod(z - z_k) expands to
   * sum c_j z^(order-j) with c_0 = 1, and a_j = -c_j */
  xo_root cr[XO_ORDER_MAX + 1], ci[XO_ORDER_MAX + 1];

  cr[0] = 1.0;
  ci[0] = 0.0;
  for (int k = 1; k <= order; k++) {
    cr[k] = 0.0;
    ci[k] = 0.0;
  }
  for (int k = 0; k < order; k++)
    for (int j = k + 1; j >= 1; j--) {
      cr[j] -= zr[k] * cr[j - 1] - zi[k] * ci[j - 1];
      ci[j] -= zr[k] * ci[j - 1] + zi[k] * cr[j - 1];
    }
  for (int k = 0; k < order; k++) {
    coef[(size_t)bin * order * 2 + 2 * k] = (float)(-cr[k + 1]);
    coef[(size_t)bin * order * 2 + 2 * k + 1] = (float)(-ci[k + 1]);
  }
}

/* ---------------------------------------------------------------------
 * Kernel 3: one recursion step for every bin.
 *
 * state[k] holds X(n - k*arHop), newest at k = 0; the step computes
 * sum_k a_k state[k] and shifts.  Layout is [k][bin] (structure of
 * arrays) so the bin index is the fastest-varying one.
 * ------------------------------------------------------------------- */
__kernel void xo_advance(
  __global const float *coef, /* [nbins][order][2] */
  __global float *state,      /* [order][nbins][2] */
  const int nbins,
  const int order) {
  const int bin = XO_GID(0);

  if (bin >= nbins)
    return;

  float re = 0.0f, im = 0.0f;

  for (int k = 0; k < order; k++) {
    const float aR = coef[(size_t)bin * order * 2 + 2 * k];
    const float aI = coef[(size_t)bin * order * 2 + 2 * k + 1];
    const float sR = state[((size_t)k * nbins + bin) * 2];
    const float sI = state[((size_t)k * nbins + bin) * 2 + 1];

    re += aR * sR - aI * sI;
    im += aR * sI + aI * sR;
  }
  for (int k = order - 1; k > 0; k--) {
    state[((size_t)k * nbins + bin) * 2]
      = state[((size_t)(k - 1) * nbins + bin) * 2];
    state[((size_t)k * nbins + bin) * 2 + 1]
      = state[((size_t)(k - 1) * nbins + bin) * 2 + 1];
  }
  state[(size_t)bin * 2] = re;
  state[(size_t)bin * 2 + 1] = im;
}
