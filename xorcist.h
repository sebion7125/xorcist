/*
 * Xorcist - it casts out the xruns.
 *
 * Deadline-miss concealment for real-time audio: when a period's mix is
 * not finished in time, send a PREDICTION of it instead of a dropout.
 *
 * The prediction is a phase-vocoder spectral freeze - every STFT bin is
 * continued with the frequency implied by its phase advance over the two
 * most recent analysis frames, so hundreds of partials and the noise
 * floor's envelope carry on at once, with no autoregressive order limit.
 * Two refinements decide the quality: the ENTRY SEAM is repaired either
 * by anchoring the first sample to a Burg one-step prediction (sparse,
 * sustained material) or by crossfading a short Burg free-run over it
 * (dense material, where the freeze cannot identify per-bin frequencies)
 * - chosen per period from the spectral similarity of the two frames -
 * and the overall LEVEL trend between those frames is continued too, so
 * a swell or a decay keeps moving through the gap.
 *
 * Copyright (C) 2026 Sebion7125
 *
 * Dual-licensed: use under the GNU General Public License v3.0 or
 * later (see LICENSE), or under a separate commercial license for
 * proprietary use without the GPL's obligations - contact
 * sebion7125@gmail.com for commercial licensing terms.
 */

#ifndef XORCIST_H
#define XORCIST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque instance - one per output device/stream. Not thread safe by
 * itself; see the threading contract on the functions below. */
typedef struct Xorcist Xorcist;

/** Concealment method. OFF makes Xorcist_conceal() emit hard silence,
 * which is the honest baseline of what an unconcealed miss sounds like.
 * The PV modes name the analysis frame size: 4096 is the sweet spot on
 * dense chords (2048's 23 Hz bins merge neighbouring partials, 8192's
 * 170 ms frame goes stale). */
enum {
  XORCIST_OFF = 0,
  XORCIST_PV_2K = 1,
  XORCIST_PV_4K = 2,
  XORCIST_PV_8K = 3,
  /* Appended rather than reordered: the three values above are baked
     into saved host settings.  1024 samples = 21 ms frame, 47 Hz bins -
     too coarse to separate a chord's partials, but four times the time
     resolution of PV 4k.  Exists to test whether a higher Prony order
     can make up for the missing frequency resolution.  Needs a period
     size of 128: the analysis hop must divide nfft/4 = 256. */
  XORCIST_PV_1K = 4
};

/** Seam repair. ADAPTIVE picks per period from the spectral density and
 * is what a host should ship; the two fixed settings exist because they
 * are each optimal on one kind of material and an A/B tool needs to
 * reach them:
 *   ANCHOR - pin the first sample to Burg's one-step prediction and
 *            leave the freeze alone.  Wins on sustained, sparse spectra
 *            (a held organ tone: -1.4 -> -17.2 dB excess NMR).
 *   BRIDGE - crossfade a short Burg free run over the seam, REPLACING
 *            the freeze there.  Wins on dense spectra, where the freeze
 *            cannot identify a frequency per bin.
 * Deliberately absent: everything that lost its listening test - a
 * per-bin level trend (follows beating), an envelope "sustain"
 * correction, replay/history-match. */
enum {
  XORCIST_SEAM_ADAPTIVE = 0,
  XORCIST_SEAM_ANCHOR = 1,
  XORCIST_SEAM_BRIDGE = 2
};

/** Switches for A/B work. Zero-initialise and the defaults apply. */
typedef struct {
  int seam;      /* XORCIST_SEAM_*; default ADAPTIVE                  */
  int noTrend;   /* 1 = freeze the level instead of continuing it     */
  int anaHop;    /* analysis hop override in frames; 0 = one period   */
  /* Damping of the per-bin FREQUENCY estimate, 0..1: each analysis moves
   * the rotator (1 - damping) of the way towards its fresh value.
   * 0 = replace it outright (classic), 1 = never move it again.
   *
   * Expressed as DAMPING rather than inertia so that the whole range is
   * usable: with "inertia" the meaningful extreme - freeze the frequency
   * completely - collided with the zero value that a zero-initialised
   * struct must map to "leave the defaults alone".
   *
   * Its purpose is the positive feedback that appears once the history
   * contains the predictor's OWN output: re-analysing predicted audio
   * re-fits the model to its own error, and the loop settles at a
   * plateau instead of at the truth.  Averaging the frequency makes the
   * model sluggish, which is exactly what breaks that loop - measured on
   * a held chord at 50 % concealment: 1.0 -> -10.7 dB, 0.2 -> -12.0,
   * 0.05 -> -12.5.  Only the rotator is damped (unit vectors, so the
   * blend is a proper interpolation); magnitudes stay fresh, and the
   * complex spectrum is never blended - a fractional power of a wrapped
   * phase has no unique value. */
  float freqDamping; /* 0 = off (classic), 1 = frequency frozen */
  /* PRECISION NOTE for GPU ports: the normal equations are accumulated
   * in double here, but that is not a requirement - measured against a
   * float-accumulator build across four materials and orders 2/4/8, the
   * worst loss is 0.1 dB.  It matters because consumer GPUs throttle
   * FP64 hard (an RTX 3070 runs it at 1/64 of its FP32 rate), so a port
   * should simply use float and not reach for double-float emulation.
   *
   * Order of the per-bin continuation.  0 or 1 = the classic rotator,
   * X(n) = z X(n-1) with |z| = 1, which is EXACT for one partial per bin
   * and structurally wrong for two.  2 = Prony per bin: fit
   * X(n) = a X(n-1) + b X(n-2), whose roots are the two partials' per-hop
   * rotations, from exponentially forgotten normal equations.
   *
   * Order 2 runs on its OWN analysis hop of nfft/8 (512 at 4096), which
   * is not a tuning choice but an alias limit: the two roots sit
   * delta-omega * hop apart on the unit circle, so a bigger hop is a
   * longer lever arm - until the separation wraps at pi, i.e. above
   * delta-f = fs / (2 hop).  Setting that equal to the widest separation
   * a bin can hold (the Hann main lobe, 4 fs / N) gives hop = N / 8.
   *
   * Measured IN THE ENGINE at PV 4k, 600 misses/min:
   *
   *                       order 1  order 2  order 3  order 4
   *   single organ tone     -15.7    -24.2    -25.4    -25.3
   *   chord + attacks       -13.8    -22.3    -24.6    -25.2
   *   full D major chord     -4.9    -14.0    -16.1    -16.5
   *   dense Vierne           -2.4     -3.5     -3.6     -3.5
   *
   * Two sinusoids 6 Hz apart inside ONE bin go from -2.1 to -95.8 dB in
   * the offline chain - a case the rotator cannot represent at all.  It
   * does not break at note onsets (measured 5..120 ms after a
   * spectral-flux onset: no loss against the control) and it keeps its
   * advantage at 27 % concealment.
   *
   * COST.  Not deferrable to the misses: Predict() runs every period,
   * because the shadow has to stand ready BEFORE a deadline is missed.
   * Per channel at PV 4k and 128 frames, against a 2667 us period budget:
   * order 1 64 us, order 2 112, order 3 166, order 4 214.  As a share of
   * the budget:
   *
   *   channels   order 1  order 2  order 3  order 4
   *          2      5.0 %    8.7 %   12.8 %   16.3 %
   *          8     19.9 %   36.8 %   52.9 %   70.4 %
   *
   * Only the p x p solve and its root finding are deferred (once per
   * AR_SOLVE_EVERY grid points); the ring write, the forgotten normal
   * equations and the p-fold advance arithmetic are per period by
   * construction. */
  int order; /* 1 = rotator (default), 2..8 = per-bin Prony */
  /* 1 = stabilise via the reflection coefficients (Schur descent, box
   * condition, Levinson ascent) instead of finding and clamping the
   * roots.  Finite O(p^2) both ways, no iteration, and measured 4.8x to
   * 8.4x faster on an RTX 3070.  NOT the same result: it moves every
   * mode depending on a violated k_l, not only the unstable one - which
   * is exactly what still has to be measured on real material. */
  int schur;
  /* Safety cap: the shadow may not peak above safetyCap x the loudest
   * of the last 2048 history samples; it SCALES, it does not clip.  0
   * turns the cap off entirely, < 0 keeps the built-in 1.2.
   *
   * A knob because it was under suspicion (06.08.2026) of being what
   * breaks the higher orders: the peak is taken over the WHOLE shadow
   * (m_ShadowLen, ~32 ms) while only the played 128 samples are ever
   * heard, so more modes near |z| = 1 should mean more spurious hits -
   * and each hit ratchets, because the scaled samples enter the ring and
   * shrink the next histPeak.  Measured on the D major chord, 619 gaps,
   * anchor seam, no trend, the hit rate does NOT scale with order
   * (14.1 / 14.9 / 14.6 / 14.3 % at orders 4 / 8 / 12 / 16) and
   * disabling the cap moves nothing (order 4: -6.30 -> -6.28). It fires
   * often but with a gain so close to 1 that it costs nothing here. */
  float safetyCap;
  /* 1 = skip AR fit updates whose tuple overlaps the predictor's own
   * output (the clean-seed gate one level up).  Under test since
   * 2026-08-06: at 21 % concealment the library falls 4.45 dB behind a
   * numpy prototype whose rectangular window forgets contamination
   * after its span, while these normal equations forget it only
   * exponentially.  Beware that the tuple spans P*arStride + nfft/buf
   * periods - 48 at order 4 and 4k - so at high density the gate may
   * simply starve the fit; Xorcist_gateStats() says how often it
   * fired. */
  int gateFit;
  /* Periods past the last concealment during which the shadow is NOT
   * rebuilt - the running extrapolation is kept instead.  0 = only
   * within a miss streak, which is what the engine has always done.
   * Motivation: the analysis window is nfft samples (32 periods at 4k),
   * so predicted material lingers in it long after the miss is over,
   * and every re-analysis there measures phase advances on the
   * predictor's own output.  An oracle that feeds the ring truth while
   * still playing the prediction puts that feedback at 3.6 dB (2.5 %
   * concealment) to 10.2 dB (23.7 %). */
  int staleWindow;
  /* 1 = recompute a missing AR seed from the history instead of dropping
   * to the order-1 rotator.  The spectrum ring is filled only when a
   * shadow is built, and none is built during a miss streak, so a
   * cluster tears a hole; every later grid point landing in it silently
   * ran order 1.  Measured: 11.1 % of builds at order 4 / 600 per
   * minute, 30.6 % at order 12, and 71.2 % / 97.5 % at 6000 - so the
   * high orders were mostly not running at all. */
  int backfill;
  /* 0 = exponentially forgetting normal equations (the default).  J > 0
   * = rebuild them at each solve from the last J equally weighted
   * tuples, which is what the offline prototype does - it forgets
   * ABRUPTLY after J frames while exponential forgetting never fully
   * lets go.  Capped by the spectrum ring to (ringLen-1)/stride + 1 - P
   * tuples, i.e. 17 at order 4 and 4k. */
  int fitWindow;
  /* 0 = the striped solve (each bin renewed every AR_SOLVE_EVERY *
   * arStride = 16 periods, i.e. 43 ms).  1 = every bin every period,
   * which is what the offline prototype does at every gap.  Measured at
   * the user's note onsets: 1.22 dB of the 1.52 dB the prototype was
   * ahead.  The stripe exists because the point solve measured 10.8 ms
   * serial at order 4 over six channels. */
  int solveEvery;
  /* Periods between refits of the anchor's Burg model.  0 = the built-in
   * BURG_REFIT_PERIODS = 8 (staggered per channel, so a value can be up
   * to 1024 samples old); 1 = refit every period, as the offline
   * prototype does at every gap.  The anchor injects
   * (shadow[0] - v0) * exp(-i/tau) with tau = 1/3 ms - broadband, so an
   * error in v0 IS a click. */
  int burgEvery;
  /* 1 = run the chain as the offline numpy prototype does, for a
   * like-for-like comparison: rectangular fit over 20 tuples rebuilt
   * every solve, every bin solved every period, Durand-Kerner started
   * cold, the anchor's Burg refitted every period and applied over the
   * whole shadow, and the safety cap applied AFTER the anchor instead
   * of before it.  Not a recommendation - the cap order in particular
   * is worse this way; it exists to make the two comparable. */
  int replica;
  /* 1 = do NOT write the re-entry crossfade into the history ring.  The
   * ring then holds a hard splice prediction -> truth at every re-entry,
   * whose step is broadband and poisons the next 32 analysis windows up
   * to 24 kHz.  That was the behaviour until 2026-08-06 and the cause of
   * the vertical stripes; 0 (the default) is the fix.  Output is
   * identical either way - only the history changes. */
  int noFadeInRing;
  /* Samples of the shadow the safety cap measures its peak over.  0 =
   * the whole shadow (1664 samples).  Anything past one gap plus the
   * re-entry crossfade is replaced by a fresh shadow before it can be
   * heard, so measuring it only lets a transient in the inaudible tail
   * trip the cap and scale the PLAYED samples down with it. */
  int capSpan;
  /* Re-entry crossfade length in frames; 0 keeps the built-in
   * min(period, 3 ms).  Can only be SHORTENED - the buffers are sized at
   * construction.  Exists so the lab's fade setting reaches the library
   * at all, which it never did before. */
  int reentryFade;
  /* Time constant of the anchor's decaying exponential, in samples.
   * 0 = the built-in sampleRate/3000 (16 at 48 kHz, i.e. 1/3 ms).  The
   * anchor subtracts (shadow[0] - v0) * exp(-i/tau): broadband by
   * nature, so tau trades bandwidth against duration at equal step
   * height. */
  int anchorTau;
  /* Trigger threshold of the root clamp; 0 = the built-in 0.9.  The
   * TARGET is always the unit circle - the threshold only decides which
   * roots get snapped onto it.  Raising it toward 1.0 lets transient
   * modes keep their fitted decay instead of being frozen into eternal
   * oscillators; only genuinely divergent roots are still pulled back. */
  float rootClamp;
  /* 1 = run the per-bin SOLVE on an OpenCL device (first platform, first
   * device - NVIDIA/AMD/Intel/pocl, whatever the ICD loader finds).  The
   * RLS update stays on the CPU so its accumulators remain a warm
   * fallback; on any device error the library reverts to the CPU solve
   * for the rest of the instance's life.  Kernels are the one text in
   * xorcist_kernels.cl, verified bit-identical against SolveBin.  No
   * OpenCL runtime installed = silent CPU operation. */
  int gpu;
  /* > 1 = detector mode: raw post-Gauss coefficients are tested for
   * all-roots-inside-detectRho (Schur descent, O(p^2), finite).  Pass
   * (96 % of bins at 1.02 on the D major chord) = coefficients used
   * RAW - no Durand-Kerner, no rebuild, fitted per-mode envelopes kept.
   * Fail = warm double DK, then moduli CAPPED AT detectRho instead of
   * being lifted to 1.0: growth is cut to budget, decay survives,
   * nothing is eternalised.  0 = the classic path.  Process-wide like
   * schur/rootClamp.  The GPU solve path disengages in this mode until
   * the kernel learns it. */
  float detectRho;
  /* 1 = transient patching: when the frame similarity collapses (onset)
   * and then holds for 6 periods (stabilised), the zone around the
   * onset is overwritten IN THE ANALYSIS VIEW ONLY by the backward-
   * rotated continuation of the stabilised state - phase-exact at its
   * right edge, amplitude-morphed into the old tone at its left, and
   * static (windows grow back to pure truth by themselves).  The ring
   * stays untouched.  Guards: no concealment in the last 16 periods,
   * 4096-sample debounce.  Transparency on stationary material:
   * -26 to -39 dB. */
  int patchTransients;
  /* 1 = two-chain hybrid mode: a short secondary 1k chain runs
   * continuously alongside the main one, kept warm.  When the onset
   * detector fires, the affected ~94 Hz bands get fresh normal
   * equations in BOTH chains, so their memory never reaches further
   * back than the onset, and the order climbs as real tuples become
   * available (a ladder).  While that happens, the short chain plays
   * those bands; once the LONG chain's fresh fit reaches hybHandOrder,
   * control is handed back (smoothed over ~3 periods).  Needs a period
   * size that divides 256, and an analysis frame bigger than 1k. */
  int hybrid;
  /* Cap of the short chain's order ladder; 0 = the built-in 6.  Order p
   * is used once p + 2 periods have passed since the reset. */
  int hybShortCap;
  /* Order the long chain's fresh fit must reach before hand-back; 0 =
   * the built-in 6.  At PV 2k, (6 + 2) grid points lands exactly on the
   * moment no analysis frame still contains pre-onset signal
   * (42.7 ms). */
  int hybHandOrder;
  /* 1 = reset the accumulator on note onset (the original design:
   * memory should never reach back past the onset).  MEASURED HARMFUL
   * as a default - costs roughly +6.5 dB NMR on its own, from the
   * damage this does to the long chain's state.  0 (the default) =
   * both chains keep running continuously; the detector only switches
   * which one is heard per band. */
  int hybReset;
  /* 1 = on an onset, switch ALL bands to the short chain instead of
   * only the rising ones.  Measured: switching only a subset tears
   * partials apart between the two chains - worst case +16.7 dB NMR
   * over a 200 ms held-note test with partial selection, vs. +7.05 dB
   * when all bands switch together. */
  int hybAllBands;
  /* 1 = residual-based chain chooser (experiment, measured to fail):
   * each band plays whichever chain currently has the smaller relative
   * one-step prediction error.  Problem: that metric measures TRACKING,
   * not EXTRAPOLATION across the concealment horizon - the short chain
   * wins it even on plateaus, where it merely follows the beating - so
   * the chooser kept switching far from actual onsets (measured 3935
   * spurious switches, +6.4 dB NMR).  0 = the default, detector-driven
   * switching instead. */
  int hybChooser;
  /* 1 = 512-sample pre-stage: on a 512-sample frame (13-29 ms after the
   * physical onset), rising bands switch to the short chain
   * PROVISIONALLY; if the 2k detector has not confirmed the onset
   * within XO_PRE_TTL (16) periods, the switch is withdrawn and the
   * long chain - never actually disturbed - resumes immediately at
   * full quality.  The trigger threshold (XO_PRE_ON, 0.41) is tuned
   * very finely, which is why this safety net exists. */
  int hybPre;
  /* 1 = simplified detector mode for A/B listening: exactly the first,
   * minimal onset-detection logic with none of the later refinements -
   * 2k-wide candidate test, switch latch, splice veto, and a 48-period
   * re-entry lockout.  No 512-sample pre-stage, no chain chooser, no
   * per-band verdict, no level gate, no withdrawal. */
  int hybSimple;
  /* Order the short chain idles at while warm (default 2): only the
   * START order runs continuously (1/7th the update cost of order 6);
   * once switched on, a fresh accumulator climbs to the full cap order
   * as real tuples arrive.  0 = run it at full cap order even while
   * idle, as before. */
  int hybWarmOrder;
  /* Residual gate for the simplified detector, appended here rather
   * than reordered (the field order above is baked into the lab's
   * ctypes layout).  0 = off.
   * A band only counts as rising if its own prediction error exceeds
   * this threshold - the onset width is built from the bands that pass
   * this test.  It separates what level alone cannot: a beat pattern
   * keeps the long chain going, a real onset does not.  Measured on a
   * D-major-chord test signal at 6000 misses/min: 0.02 removes all
   * three plateau false triggers seen without it, costs 2 of 6 onsets,
   * and still improves every aggregate measure (mean -18.08 ->
   * -18.28 dB, worst case 15.61 -> 15.05, at onsets -13.66 -> -13.81) -
   * a false trigger on a plateau costs far more than a missed onset.
   * decay: per-period decay rate of each band's running maximum (0 =
   * default 0.85, roughly eight periods of memory); the instantaneous
   * value alone misses weak onsets whose residual only crosses the
   * gate after the fact. */
  float detResGate;
  float detResDecay;
  /* 1 = derive the seam anchor from hindcast: the offset is measured
   * directly as shadow[-1] against the last real sample (both chains
   * keep their pre-boundary sample), then the decaying exponential is
   * advanced one sample forward.  This removes the seam path's Burg
   * fit entirely.  0 = the default Burg v0 path.  Measured: equivalent
   * to the default without hybrid mode active; with hybrid mode,
   * onsets are identical and the rest of the signal differs by about
   * 0.14 dB; the Burg path itself costs about 4.3 us per round. */
  int v0Hindcast;
  /* 1 = solve trace recording (lab/diagnostic use only, requires
   * gpu=1): the kernel records, per bin, what it did - the largest
   * pole radius before clamping, which branch was taken, and every
   * pole itself - and after each complete pass over channel 0's bins,
   * one snapshot moves into a ring buffer.  Retrieve it with
   * Xorcist_solveTapRead.
   *
   * COSTS TIME: the trace must be read back from the device, blocking,
   * after every kernel launch, which serialises the queue.  Any timing
   * measurement taken with this flag set is meaningless.  The result
   * itself (the concealment output) is unaffected - the tap only
   * records alongside it. */
  int solveTap;
} XorcistTuning;

void Xorcist_setTuning(Xorcist *x, const XorcistTuning *t);
void Xorcist_getTuning(const Xorcist *x, XorcistTuning *t);

/**
 * Create an instance.
 *
 * @param channels      interleaved channel count of the audio handed in
 * @param framesPerPeriod  one period (= the buffer the device asks for)
 * @param sampleRate    in Hz
 * @param wisdomPath    file the measured FFTW plans are cached in, or
 *                      NULL to disable.  FFTW_MEASURE times the codelets
 *                      on THIS machine and cannot be precomputed at build
 *                      time, but its result persists: ~0.9 s of planning
 *                      once per machine instead of once per start.
 * @return NULL on allocation failure.
 */
Xorcist *Xorcist_new(
  unsigned channels,
  unsigned framesPerPeriod,
  unsigned sampleRate,
  const char *wisdomPath);

void Xorcist_free(Xorcist *x);

/**
 * Split form of Xorcist_predict() for hosts with spare worker threads.
 *
 * The channels are fully independent - own FFT workspace, own normal
 * equations, own recursion - so they may be computed CONCURRENTLY:
 *
 *     if (Xorcist_predictBegin(x) == 1) {
 *         parallel_for (ch)  Xorcist_predictChannel(x, ch);
 *         Xorcist_predictEnd(x);
 *     }
 *
 * Begin returns what Xorcist_predict() returns; only on 1 may the channel
 * calls and End follow, and End must run exactly once after the last one.
 * Serially the cost scales linearly with the channel count - at eight
 * channels that is 20 % of a 128-frame period for the plain rotator and
 * 70 % for order 4, on ONE core.
 */
int Xorcist_predictBegin(Xorcist *x);
void Xorcist_predictChannel(Xorcist *x, unsigned ch);
void Xorcist_predictEnd(Xorcist *x);
/** Release the compute slot without publishing - for the case where not
 * every channel of an opened prediction actually ran.  Without it the
 * slot stays taken and all later predictions skip. */
void Xorcist_predictAbort(Xorcist *x);
unsigned Xorcist_channels(const Xorcist *x);

/** Select the method (see the enum). Cheap, may be called at any time. */
void Xorcist_setMode(Xorcist *x, int mode);
int Xorcist_getMode(const Xorcist *x);

/**
 * Append one finished period of audio (interleaved) to the history.
 *
 * Call from whichever thread produced the mix, while it still owns the
 * buffer.  This is a plain ring append - no locks, no allocation.
 */
void Xorcist_push(Xorcist *x, const float *interleaved);

/**
 * Append @p periods finished periods at once.
 *
 * Purely a convenience for offline tools and scripting bindings, where
 * the per-call overhead of a language boundary would otherwise dominate
 * the ring append it wraps.  A real-time host has one period at a time
 * and uses Xorcist_push().
 */
void Xorcist_pushMany(Xorcist *x, const float *interleaved, unsigned periods);

/**
 * Overwrite the period most recently appended by Xorcist_conceal() with
 * the real audio for that slot.
 *
 * For hosts that let a late mix run to completion: its output is useless
 * for playback (its slot is gone) but it is the TRUTH for the history,
 * and replacing the prediction there removes the feedback at its source.
 * Measured on a held chord at 50 % concealment: -10.8 dB with the
 * prediction fed back, -15.1 dB with the truth - and feeding truth for
 * only every second missed period already reaches -12.6 dB.
 *
 * Does nothing if the last append did not come from Xorcist_conceal().
 *
 * Threading caveat: this rewinds the ring, so it must not run while
 * another thread is inside Xorcist_conceal() or Xorcist_push().  A host
 * whose late mix completes on a different thread than the callback has
 * to serialise the two - the period-tag check catches the common
 * mismatch, but the rewind itself is not atomic.
 */
void Xorcist_replaceLast(Xorcist *x, const float *interleaved);

/**
 * Compute the prediction from the history.
 *
 * This is the expensive call (tens of microseconds); run it OFF the
 * critical path - typically as an idle-tail task after the period's real
 * work is done.  It is safe to skip individual periods: predictions are
 * tagged with the history position they were made at, so a late or
 * missed computation resolves to "use a later slice" instead of a phase
 * error.
 *
 * @return 1 if a prediction was produced, 0 if it was skipped (mode off,
 *         not enough history yet, or nothing new to anchor on),
 *         2 if the running prediction was deliberately KEPT because a
 *         miss streak is in progress - re-analysing would fit a spectrum
 *         to already-predicted audio and restart the extrapolation,
 *         seaming every period.  Costs nothing, which is convenient
 *         precisely when the machine has no time to spare.
 *         3 if another thread's round is still in flight (busy guard) -
 *         behaviourally identical to 0 for the caller, but the host's
 *         late-slice diagnosis needs the two apart.
 */
int Xorcist_predict(Xorcist *x);

/* Analysis WITHOUT synthesis: updates the spectrum cache, the rotator,
   the routing ring, the AR spectrum ring and the normal equations -
   exactly the state Predict() would leave - but builds no shadow.  For
   offline hosts that know no miss can be near: the shadow is the
   expensive part, and at low miss rates almost every one is discarded
   unheard. */
void Xorcist_absorb(Xorcist *x);

/* Transient patch (analysis overlay; the ring itself stays truth).
   patchForce builds the backward-rotated continuation of the LAST
   analysed state over [zoneStart, zoneStart+len) absolute samples -
   the bite-test entry.  patchRead returns the patch content (reading
   does NOT clear), patchClear drops all patches. */
int Xorcist_patchForce(
  Xorcist *x, unsigned ch, uint64_t zoneStart, unsigned len);
unsigned Xorcist_patchRead(
  Xorcist *x, unsigned ch, uint64_t *pStart, float *out, unsigned maxLen);
void Xorcist_patchClear(Xorcist *x);

/**
 * A period was missed: write the concealed audio into @p out.
 *
 * Real-time safe - a copy, a few multiplies, no locks or allocation.
 * Consecutive misses degrade honestly, on a SAMPLE clock rather than a
 * period count (validity is elapsed time, not how many device buffers
 * happen to slice it): full level for ~25 ms - the freeze's audibility
 * horizon - then a generous ~150 ms fade, long enough that a returning
 * real signal can cross-fade back in from a still-audible level.  What
 * was actually played is appended to the history, so the next prediction
 * stays time-aligned.
 *
 * @return 1 if a prediction was played, 0 if silence had to be emitted.
 */
int Xorcist_conceal(Xorcist *x, float *out);

/**
 * The period arrived in time: copy @p real into @p out, applying the
 * re-entry crossfade if the previous period was concealed.  @p out and
 * @p real may be the same pointer.
 */
void Xorcist_pass(Xorcist *x, float *out, const float *real);

/**
 * Copy the CURRENT prediction into @p out (interleaved) without playing
 * it - the miss streak, the re-entry state and the history stay
 * untouched.  For offline tools: an A/B lab can score the very samples
 * the engine would have emitted, instead of reimplementing the algorithm
 * and slowly drifting away from it.
 *
 * @return frames actually written (0 if no prediction is available).
 */
unsigned Xorcist_peek(const Xorcist *x, float *out, unsigned frames);

/** Frames the prediction spans - the upper bound for Xorcist_peek(). */
unsigned Xorcist_horizonFrames(const Xorcist *x);

/** Periods concealed so far (both prediction and silence). */
uint64_t Xorcist_concealedCount(const Xorcist *x);
/** Predictions skipped because a computation was already running. */
uint64_t Xorcist_skippedCount(const Xorcist *x);

/* Safety-cap diagnostics: shadow builds that were scaled down, out of
   how many were built.  Reading RESETS both, so a caller can bracket a
   single run.  XO_SAFETY_CAP overrides the factor (0 = cap off). */
void Xorcist_capStats(uint64_t *pHits, uint64_t *pCalls);
float Xorcist_safetyCap(const Xorcist *x);

/* Fit-gate diagnostics: grid points skipped, out of how many tested.
   Reading RESETS both. */
void Xorcist_gateStats(uint64_t *pSkips, uint64_t *pTests);

/* How often the AR path fell back to the order-1 rotator because a seed
   was missing from the spectrum ring, out of how many times it was
   wanted.  Reading RESETS both. */
void Xorcist_arDropStats(uint64_t *pDrops, uint64_t *pTries);

/* Seeds recomputed from history since the last call (reading RESETS). */
uint64_t Xorcist_backfillCount(void);

/* rho-Schur detector counters (measurement only, armed by the
   XO_COUNT_RHO env var): bins whose RAW post-Gauss coefficients fail
   the all-roots-inside-rho test, out of bins tested.  Reading RESETS. */
void Xorcist_rhoStats(uint64_t *pFails, uint64_t *pTests);

/* Solve rounds since the last call, split by device vs. CPU (reading
   RESETS both).  A nonzero GPU count proves nothing on its own: the
   replica path and Schur mode always stay on the CPU, and a device
   that fails once falls back to the CPU permanently.  A host that
   exposes the gpu setting should also be able to show whether it is
   actually having any effect. */
void Xorcist_gpuStats(uint64_t *pOnDevice, uint64_t *pOnCpu);

/* Concurrency diagnostics for hosts that run predictBegin/
   predictChannel/predictEnd on their OWN worker threads (reading
   RESETS both).  A single-threaded caller - such as an offline tool -
   will always read 0/0 here.

   Both counters are diagnostic only, not an indication of lost
   quality: they report how much a host's scheduling overlaps with
   Xorcist's own round window, not how much audio quality that costs.

   staleDiscards: rounds during which the history ring was appended to
     while the round was still in flight.  They are still completed and
     published against their own frozen analysis window rather than
     discarded.
   lateSlices: concealments drawn from a slice index > 0 without the
     previous period itself having been concealed.  This means the
     fresh round was not yet finished when a bail-out happened, so the
     previous round's result was played instead - correct output, but
     a sign that the round's compute time is pushing against the
     period length. */
void Xorcist_raceStats(uint64_t *pStaleDiscards, uint64_t *pLateSlices);

/* Root-cause breakdown of the late slices above (reading RESETS both).
   afterSkip: how many of the lateSlices above occurred within two
     periods of predictBegin() finding a round already in progress (a
     "busy" result).  afterSkip == lateSlices confirms the chain "open
     call was skipped -> the anchor fell behind"; afterSkip much
     smaller than lateSlices means the anchor is falling behind for
     some OTHER reason.
   maxSlice: the largest slice index actually played.  1 = exactly one
     round was missed; higher values mean rounds are falling behind
     repeatedly. */
void Xorcist_lateStats(uint64_t *pAfterSkip, uint64_t *pMaxSlice);

/* ---- Stage timing -----------------------------------------------------
   Where does a round's compute time go?  OFF by default (the clock
   itself costs ~20 ns per query, and the solve runs per bin);
   XO_PROFILE=1 in the environment, or Xorcist_setProfiling(1), turns it
   on.

   Stages, in this order: analysis, 4k detector, 1k hybrid, AR
   accumulation, AR solve (which INCLUDES Durand-Kerner, not a sibling
   stage - it only runs where order >= 3, the rho test fails, and Schur
   mode is off, so only on a fraction of bins), synthesis, cap + seam.

   Xorcist_stageStats reports nanoseconds and call counts and RESETS
   both on read, so a host can bracket exactly one time window. */
/* Environment diagnostic overrides currently in effect, as
   "NAME=VALUE NAME=VALUE" - empty if everything is at its default.
   Lets a host DISPLAY what is actually active, instead of having to
   guess from an A/B test whether an environment variable was picked up
   at all.  The returned pointer stays valid; the environment is read
   once, on the first call. */
const char *Xorcist_diagOverrides(void);

void Xorcist_setProfiling(int on);
int Xorcist_profiling(void);
unsigned Xorcist_stageCount(void);
const char *Xorcist_stageName(unsigned index);
void Xorcist_stageStats(uint64_t *pNs, uint64_t *pCalls, unsigned maxStages);

/* Transient-detector TRIGGERS since the last call (reading RESETS).  On
   held material this should read zero: the detection criterion is a
   pure ratio (a band's energy against its own level eight periods
   earlier), so it is level-independent by construction - verified
   across a 100 dB range, same count either way.  What CAN still fool
   the detector at a high miss rate is its own RE-ENTRIES: the splice
   veto only looks back 3 periods, while its 4k analysis window spans
   32.
   `tentative` counts the provisional triggers from the 512-sample
   pre-stage that can still be withdrawn (not used by GrandOrgue). */
/* PER CHANNEL: return value = how many of the 256 routing bands
   currently carry anything (within 60 dB of that channel's loudest
   band); pOnsets = onset triggers for this channel since start (does
   NOT reset on read).

   Why this is the number that matters: the detector's criterion is a
   ratio, and in an empty band a factor of two is free - there is
   nothing there but the predictor's own error.  A broadband channel
   tolerates that without issue; a sparsely occupied one keeps
   triggering spuriously during steady playback (measured: 86 occupied
   bands, 0 spurious triggers; 3 occupied bands, 228).  A stereo
   downmix can never show this problem, because summing channels fills
   in bands that would otherwise be empty. */
unsigned Xorcist_detectorPerChannel(unsigned ch, uint64_t *pOnsets);

void Xorcist_onsetStats(uint64_t *pOnsets, uint64_t *pTentative);

/* Sample positions of the most recent onset triggers, oldest first, so
   a host can PLOT them against its own timeline.  Reading does NOT
   reset the log (so a display can redraw at any time); the ring holds
   the last 256 events.  Return value: how many were written. */
unsigned Xorcist_onsetLog(uint64_t *pOut, unsigned maxEvents);

/* Hybrid-mode EFFECT since the last call (reading RESETS all three):
   periods during which the second chain was active, periods where the
   flag was set but it did nothing (base frame is 1k, or the period
   size does not divide 256), and the total band-periods the short
   chain actually played. */
/* Hybrid episodes as triples (startSample, endSample,
   band | order<<16 | nfft<<32), channel 0 only, newest last, up to
   4096 kept.  For a spectrogram display: WHICH bands were computed by
   the short chain, and WHEN.  Reading does not clear the log;
   hybLogClear does. */
unsigned Xorcist_hybLog(uint64_t *pOut, unsigned maxEvents);
void Xorcist_hybLogClear(void);

void Xorcist_hybStats(
  uint64_t *pActive, uint64_t *pInert, uint64_t *pBandPeriods);

/* Transient patches built since the last call (reading RESETS). */
uint64_t Xorcist_patchCount(void);

/* Recent patch events as triples (onsetSample, zoneStart, zoneLen) -
   newest last, up to 512 kept.  For hosts that want to SHOW where the
   detector fired.  Reading does not clear; patchLogClear does. */
unsigned Xorcist_patchLog(uint64_t *pOut, unsigned maxEvents);
void Xorcist_patchLogClear(void);

/** Name of the FFTW wisdom file, so a host can whitelist it in its own
 * cache-directory housekeeping. */
const char *Xorcist_wisdomFileName(void);

/** Library version, e.g. "0.1.0". */
const char *Xorcist_version(void);

/* Device name the solve would run on with gpu=1, or NULL when no OpenCL
   runtime/device is available (the library then computes on the CPU). */
const char *Xorcist_gpuDevice(void);

/* ---- Solve trace (tuning field solveTap) ------------------------------
   One frame per complete pass over channel 0's bins.  The ring is
   capped by MEMORY (48 MB, 16..512 frames), because a frame grows with
   the order: it carries, per bin, the largest pole radius, which
   branch was taken, and ALL poles - each before clamping.

   The info call reports how many frames are available to read, the
   geometry they were written with, and how many have been written in
   total.  The first two numbers are needed to size the buffer passed
   to Read; pTotal is meant for watching a live run (the difference
   from the last call is the number of new frames).  If the bin count
   or order changes, the ring starts over - frames from two different
   geometries could not be represented together.

   Read writes the MOST RECENT maxFrames frames in chronological order.
   Any pointer may be 0: pZ and pPath each need maxFrames*nbins floats,
   pRoots needs maxFrames*nbins*order*2 (re/im, NAN where that branch
   computed no root), pPos the sample position of each frame.  Return
   value = frames actually written.  Reading does not clear the ring;
   solveTapClear does. */
void Xorcist_solveTapInfo(
  unsigned *pFrames, unsigned *pNbins, unsigned *pOrder, uint64_t *pTotal);
unsigned Xorcist_solveTapRead(
  float *pZ, float *pPath, float *pRoots, uint64_t *pPos,
  unsigned maxFrames);
void Xorcist_solveTapClear(void);

/* The analysis geometry this instance is ACTUALLY using - so a display
   does not have to reconstruct it itself and risk getting it wrong.

   nfft   Frame length of the current mode (1024/2048/4096/8192).
   anaHop STFT hop size.  Normally ONE audio period, because this
          period's frame is exactly the older frame the next one
          needs; only an unusual period size falls back to nfft/4.
          Reflects the anaHop tuning override if one is set.
   arHop  Spacing of the grid points the AR fit sits on
          (max(nfft/8, period size)) - this is the number meant by
          "hop" in strategy names, NOT the STFT hop above.
   Any pointer may be 0. */
void Xorcist_analysisGeometry(
  const Xorcist *x, unsigned *pNfft, unsigned *pAnaHop, unsigned *pArHop);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XORCIST_H */
