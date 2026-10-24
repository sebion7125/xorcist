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
  /* ANGEHAENGT, nicht einsortiert: die drei Werte oben stecken in
     gespeicherten GO-Einstellungen.  1024 Samples = 21 ms Fenster,
     47 Hz Bins - zu grob, um Teiltoene eines Akkords zu trennen, aber
     mit der vierfachen Zeitaufloesung von 4k.  Zum Ausmessen der Frage,
     ob eine hoehere Prony-Ordnung die fehlende Frequenzaufloesung
     zurueckholt (Nutzer, 07.08.).  Braucht Puffer 128: der Analyse-Hop
     muss nfft/4 = 256 teilen. */
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
  /* 1 = ZWEI-KETTEN-HYBRID (Nutzer-Entwurf 07.08.): eine kurze
   * 1k-Zweitkette laeuft dauerhaft warm mit.  Feuert der
   * Einsatz-Detektor, bekommen die betroffenen 94-Hz-Baender in BEIDEN
   * Ketten frische Normalgleichungen - ihr Gedaechtnis reicht damit nie
   * weiter zurueck als der Anschlag, und die Ordnung klettert mit den
   * verfuegbaren Tupeln (Leiter).  Waehrenddessen spielt die kurze
   * Kette diese Baender; sobald der frische Fit der LANGEN Kette
   * hybHandOrder traegt, wird zurueckgegeben (weich ueber ~3 Perioden).
   * Braucht eine Puffergroesse, die 256 teilt, und einen Rahmen > 1k. */
  int hybrid;
  /* Deckel der 1k-Ordnungsleiter, 0 = eingebaute 6.  Ordnung p wird
   * genutzt, sobald p + 2 Perioden seit dem Reset vergangen sind. */
  int hybShortCap;
  /* Uebergabe an die lange Kette, wenn deren frischer Fit diese Ordnung
   * traegt; 0 = eingebaute 6.  Bei 2k faellt (6+2) Rasterpunkte exakt
   * mit dem Moment zusammen, ab dem kein Analyserahmen mehr Signal von
   * vor dem Anschlag enthaelt (42,7 ms). */
  int hybHandOrder;
  /* 1 = Akkumulator-Reset beim Anschlag (urspruenglicher Entwurf:
   * Gedaechtnis reicht nie hinter den Anschlag).  GEMESSEN SCHAEDLICH
   * als Vorgabe: +6.5 dB NMR allein durch den Schaden an der langen
   * Kette.  0 (Vorgabe) = beide Ketten laufen kontinuierlich, der
   * Detektor schaltet nur die Bandmaske. */
  int hybReset;
  /* 1 = beim Einsatz schalten ALLE Baender auf die kurze Kette statt
   * nur der steigenden.  Gemessen 08.08.: die Teilauswahl reisst
   * Teiltoene zwischen den Ketten auseinander - 200-ms-Haltetest max
   * +16.7 dB NMR mit Auswahl, +7.05 mit allen Baendern. */
  int hybAllBands;
  /* 1 = RESIDUEN-WAEHLER (Experiment, GEMESSEN GESCHEITERT 08.08.):
   * jedes Band spielt die Kette mit kleinerem relativem
   * Ein-Schritt-Vorhersagefehler.  Der misst aber NACHFUEHRUNG, nicht
   * EXTRAPOLATION ueber den Verdeckungshorizont - die kurze Kette
   * gewinnt ihn auch auf Plateaus (folgt der Schwebung) und der
   * Waehler schaltete 3935x fern der Einsaetze (+6.4 dB NMR).  0 =
   * Vorgabe, detektor-getriebene Umschaltung. */
  int hybChooser;
  /* 1 = 512er-VORSTUFE: die Breite auf 512er-Rahmen (13-29 ms nach dem
   * physischen Anschlag, Augen-Paritaet) schaltet die steigenden
   * Baender VORLAEUFIG auf die kurze Kette; bestaetigt der 2k-Detektor
   * nicht binnen XO_PRE_TTL (16) Perioden, wird zurueckgezogen - die
   * lange Kette wurde nie angetastet und spielt sofort wieder voll.
   * Schwelle XO_PRE_ON (0.41), hauchduenn kalibriert - deshalb das
   * Rueckzugsnetz. */
  int hybPre;
  /* 1 = SIMPLER Detektor-Modus zum A/B-Hoeren: exakt die erste
   * Hybrid-Erkennung ohne allen spaeteren Schnickschnack - 2k-Breite
   * mit Kandidat, Schaltlatte und Spleiss-Veto, Wiedereintritts-Sperre
   * 48 Perioden.  KEINE 512er-Vorstufe, kein Wechselwunsch, kein
   * Per-Band-Urteil, kein Lautstaerke-Tor, kein Rueckzug. */
  int hybSimple;
  /* Warmhalte-Ordnung der 1k-Kette (Vorgabe 2): warm laeuft nur die
   * STARTordnung (1/7 der Update-Kosten von Ordnung 6); ab der
   * Schaltung akkumuliert ein frischer Satz bei voller Deckelordnung
   * und die Leiter klettert mit echten Tupeln.  0 = Vollbetrieb wie
   * bisher (warm bei Deckelordnung). */
  int hybWarmOrder;
  /* RESIDUEN-TOR des Simpel-Detektors, ANGEHAENGT (die Reihenfolge darueber
     steckt im ctypes-Layout des Labors).  0 = aus.
     Ein Band zaehlt nur als steigend, wenn sein eigener Vorhersagefehler
     ueber diesem Wert liegt - die Breite wird aus den so bestaetigten
     Baendern gebildet.  Trennt, was der Pegel nicht trennt: eine Schwebung
     setzt die lange Kette fort, ein Anschlag nicht.  Gemessen am D-Dur-
     Material bei 6000 Luecken/min: 0.02 raeumt alle drei Plateau-
     Fehlzuendungen ab, kostet zwei von sechs Einsaetzen und verbessert
     trotzdem jedes Mass (Mittel -18.08 -> -18.28 dB, schlimmster Fall
     15.61 -> 15.05, an den Einsaetzen -13.66 -> -13.81) - eine
     Fehlzuendung im Plateau kostet eben viel mehr als ein verpasster
     Anschlag.  decay: Abklingrate des Band-Maximums je Periode
     (0 = Vorgabe 0.85, rund acht Perioden Gedaechtnis); der Momentanwert
     allein verpasst schwache Einsaetze, deren Residuum erst nach der
     Breitenschwelle kreuzt. */
  float detResGate;
  float detResDecay;
  /* 1 = Naht-Anker aus dem HINDCAST: Versatz direkt gemessen als
     Schatten[-1] gegen das letzte echte Sample (beide Ketten sammeln ihr
     Vor-Grenz-Sample ein), Exponentiale ein Sample nach vorn.  Der
     Burg-Fit des Nahtpfads entfaellt dann komplett.  0 = Burg-v0
     (Vorgabe).  Gemessen 12.08.2026 (D-Dur, 6000/min, 2k/O8): ohne
     Hybrid gleichwertig (-17,99 = -17,99), mit Hybrid Einsaetze
     identisch, Rest 0,14 dB Plateau; Burg-Pfad kostete 4,3 us je Runde.
     Umgebungsstart XO_V0_SHADOW=1 setzt nur die Vorgabe (Harness). */
  int v0Hindcast;
  /* 1 = SOLVE-MITSCHRIFT (nur Labor, nur zusammen mit gpu=1): der Kernel
     legt je Bin ab, was er getan hat - groesster Polradius VOR dem
     Deckeln, genommener Zweig, und alle Pole selbst -, und je
     vollstaendigem Bin-Durchlauf von Kanal 0 wandert eine Momentaufnahme
     in einen Ring.  Abzuholen mit Xorcist_solveTapRead.

     KOSTET ZEIT: der Streifen muss nach jedem Kernelstart blockierend
     zurueckgelesen werden, was die Warteschlange serialisiert.  Jede
     Zeitmessung mit gesetztem Haken ist wertlos.  Das Ergebnis selbst
     bleibt unveraendert - der Tap schreibt nur mit. */
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

/* Solve-Streifen seit dem letzten Aufruf, getrennt nach Geraet und CPU
   (Lesen SETZT ZURUECK).  Ein gesetztes GPU-Haekchen beweist nichts: der
   Replica-Pfad und Schur bleiben CPU, ein gescheitertes Geraet faellt
   dauerhaft zurueck.  Wer die Einstellung anzeigt, soll die Wirkung
   anzeigen koennen. */
void Xorcist_gpuStats(uint64_t *pOnDevice, uint64_t *pOnCpu);

/* Nebenlaeufigkeits-Messung fuer Wirte, die predictBegin/predictChannel/
   predictEnd auf EIGENEN Threads fahren (Lesen SETZT ZURUECK).  Ein
   einthreadiger Aufrufer - das Labor - liest hier zwingend 0/0.

   Beide Zahlen sind seit dem Einfrieren des Rundenfensters (10.08.2026)
   DIAGNOSE, kein Schaden mehr: sie sagen, wie stark der Wirt verschraenkt,
   nicht wie viel er verliert.

   staleDiscards: Runden, waehrend derer der Ring weitergeschrieben wurde.
     Bis zum Einfrieren wurden sie verworfen (daher der Name) - heute
     werden sie mit ihrem eigenen, festgehaltenen Fenster zu Ende gerechnet
     und veroeffentlicht.
   lateSlices:    Verdeckungen aus einer Scheibe > 0, OHNE dass die
     Vorperiode selbst verdeckt war.  Heisst: die frische Runde war beim
     Bail noch nicht fertig, gespielt wurde die vorige - inhaltlich
     korrekt, aber ein Hinweis darauf, dass die Rundenlaufzeit an die
     Periodendauer stoesst. */
void Xorcist_raceStats(uint64_t *pStaleDiscards, uint64_t *pLateSlices);

/* Ursachen-Zuordnung der spaeten Scheiben (Lesen SETZT ZURUECK).
   afterSkip: wie viele der obigen lateSlices binnen zwei Perioden nach
     einem Besetzt-Fall des Oeffnungswaechters lagen (predictBegin traf
     eine laufende Runde).  afterSkip == lateSlices beweist die Kette
     "Oeffnung verpasst -> Anker blieb zurueck"; afterSkip << lateSlices
     heisst, der Anker bleibt aus einem ANDEREN Grund zurueck.
   maxSlice: groesste dabei gespielte Scheibennummer.  1 = genau eine
     Runde verpasst, mehr = Serienausfall der Oeffnungen. */
void Xorcist_lateStats(uint64_t *pAfterSkip, uint64_t *pMaxSlice);

/* ---- Stufenmessung ---------------------------------------------------
   Wohin geht die Rundenlaufzeit?  AUS per Vorgabe (die Uhr kostet ~20 ns
   je Abfrage, und der Solve laeuft pro Bin); XO_PROFILE=1 in der Umgebung
   oder Xorcist_setProfiling(1) schaltet scharf.

   Stufen in dieser Reihenfolge: Analyse, Detektor 4k, Hybrid 1k,
   AR-Akkumulation, AR-Solve, davon Durand-Kerner, Synthese, Cap + Naht.
   "davon Durand-Kerner" ist in "AR-Solve" ENTHALTEN, nicht daneben - er
   laeuft nur, wo Ordnung >= 3 ist, der rho-Test durchfaellt und Schur aus
   ist, also in einem Bruchteil der Bins.

   Xorcist_stageStats liefert Nanosekunden und Aufrufzahlen und SETZT
   BEIM LESEN ZURUECK, damit ein Wirt genau ein Zeitfenster einrahmen
   kann. */
/* Diagnoseschalter der Umgebung, die GERADE abweichen, als
   "NAME=WERT NAME=WERT" - leer, wenn alles auf Vorgabe steht.  Damit ein
   Wirt ANZEIGEN kann, was wirkt, statt dass nach einem A/B geraten werden
   muss, ob die Variable ueberhaupt angekommen ist.  Der Zeiger bleibt
   gueltig, die Umgebung wird einmalig beim ersten Aufruf gelesen. */
const char *Xorcist_diagOverrides(void);

void Xorcist_setProfiling(int on);
int Xorcist_profiling(void);
unsigned Xorcist_stageCount(void);
const char *Xorcist_stageName(unsigned index);
void Xorcist_stageStats(uint64_t *pNs, uint64_t *pCalls, unsigned maxStages);

/* ZUENDUNGEN des Transientendetektors seit dem letzten Aufruf (Lesen
   SETZT ZURUECK).  Auf gehaltenem Klang gehoert hier eine Null hin: das
   Kriterium ist ein reines Verhaeltnis (Bandenergie gegen dieselbe acht
   Perioden frueher) und damit pegelunabhaengig - nachgemessen ueber 100
   dB, gleiche Zahl.  Was den Detektor bei hoher Lueckendichte trotzdem
   taeuscht, sind die eigenen WIEDEREINSTIEGE; das Spleiss-Veto reicht
   nur 3 Perioden zurueck, sein 4k-Fenster aber 32.
   `tentative` zaehlt die vorlaeufigen Zuendungen der 512er-Vorstufe,
   die widerrufen werden koennen (GO faehrt sie nicht). */
/* JE KANAL: Rueckgabe = wie viele der 256 Routing-Baender gerade etwas
   tragen (mehr als -60 dB unter dem lautesten Band des Kanals), pOnsets =
   Zuendungen dieses Kanals seit Start (NICHT zuruecksetzend).

   Warum das die entscheidende Zahl ist: das Kriterium des Detektors ist
   ein Verhaeltnis, und in einem leeren Band ist Faktor zwei geschenkt -
   dort steht nur der eigene Vorhersagefehler.  Ein breitbandiger Kanal
   traegt das problemlos, ein duenn besetzter zuendet im Dauerlauf
   (gemessen: 86 belegte Baender 0 Zuendungen, 3 belegte 228).  Ein
   Stereo-Downmix kann das nie zeigen, weil Summieren Baender fuellt. */
unsigned Xorcist_detectorPerChannel(unsigned ch, uint64_t *pOnsets);

void Xorcist_onsetStats(uint64_t *pOnsets, uint64_t *pTentative);

/* Abtastwert-Positionen der letzten Zuendungen, aelteste zuerst, damit
   ein Wirt sie in seinen Verlauf ZEICHNEN kann.  Lesen setzt NICHT
   zurueck (ein Zeichner darf jederzeit neu malen); der Ring haelt die
   letzten 256 Ereignisse.  Rueckgabe: wie viele geschrieben wurden. */
unsigned Xorcist_onsetLog(uint64_t *pOut, unsigned maxEvents);

/* Hybrid-WIRKUNG seit dem letzten Aufruf (Lesen SETZT ZURUECK):
   Perioden mit aktiver Zweitkette, Perioden in denen der Haken gesetzt
   war aber NICHTS tat (Basisrahmen 1k, oder Puffergroesse teilt 256
   nicht), und die Summe der Band-Perioden, die die kurze Kette wirklich
   gespielt hat. */
/* Hybrid-Episoden als Tripel (startSample, endSample,
   band | ordnung<<16 | nfft<<32), Kanal 0, neueste zuletzt, bis 4096.
   Fuer die Spektrogramm-Anzeige: WELCHE Baender WANN mit der kurzen
   Kette gerechnet wurden.  Lesen loescht nicht; hybLogClear schon. */
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

/* ---- Solve-Mitschrift (Tuning-Feld solveTap) -------------------------
   Ein Bild je vollstaendigem Bin-Durchlauf von Kanal 0.  Der Ring ist
   nach SPEICHER gedeckelt (48 MB, 16..512 Bilder), weil ein Bild mit der
   Ordnung waechst: es traegt je Bin den groessten Polradius, den Zweig
   und ALLE Pole - jeweils vor dem Deckeln.

   Info liefert, wie viele Bilder abrufbar sind, mit welcher Geometrie
   sie geschrieben wurden, und wie viele insgesamt geschrieben wurden.
   Ohne die ersten Zahlen laesst sich der Puffer fuer Read nicht
   bemessen; pTotal ist fuer den Blick WAEHREND eines Laufs da (Differenz
   zum letzten Blick = Zahl der neuen Bilder).  Aendern sich Bin-Zahl
   oder Ordnung, faengt der Ring neu an - halbe Bilder zweier Geometrien
   waeren nicht darstellbar.

   Read schreibt die JUENGSTEN maxFrames Bilder in zeitlicher Reihenfolge.
   Jeder Zeiger darf 0 sein: pZ und pPath je maxFrames*nbins Floats,
   pRoots maxFrames*nbins*order*2 (Re/Im, NAN wo der Zweig keine Wurzel
   berechnet hat), pPos die Sampleposition je Bild.  Rueckgabe = wirklich
   geschriebene Bilder.  Lesen loescht nicht; solveTapClear schon. */
void Xorcist_solveTapInfo(
  unsigned *pFrames, unsigned *pNbins, unsigned *pOrder, uint64_t *pTotal);
unsigned Xorcist_solveTapRead(
  float *pZ, float *pPath, float *pRoots, uint64_t *pPos,
  unsigned maxFrames);
void Xorcist_solveTapClear(void);

/* Die Analyse-Geometrie, die diese Instanz WIRKLICH rechnet - damit eine
   Anzeige sie nicht nachbauen muss und dabei falsch liegt.

   nfft   Rahmenlaenge des aktuellen Modus (1024/2048/4096/8192).
   anaHop Sprung der STFT.  In der Regel EINE Audio-Periode, weil der
          Rahmen dieser Periode genau der aeltere Rahmen ist, den die
          naechste braucht; nur bei krummen Puffergroessen nfft/4.
          Beruecksichtigt den Tuning-Ueberschreiber anaHop.
   arHop  Abstand der Stuetzstellen, auf denen der AR-Fit sitzt
          (max(nfft/8, Puffer)) - das ist die Zahl, die in den
          Strategienamen "hop" heisst, NICHT der STFT-Sprung.
   Jeder Zeiger darf 0 sein. */
void Xorcist_analysisGeometry(
  const Xorcist *x, unsigned *pNfft, unsigned *pAnaHop, unsigned *pArHop);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XORCIST_H */
