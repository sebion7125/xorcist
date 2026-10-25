# Spectral-continuation concealment of real-time audio deadline misses

A technical disclosure published to establish prior art and foreclose
patenting of the technique described below. Written to be
understandable and reproducible by a person skilled in digital signal
processing / real-time audio engineering, independent of any particular
host application.

## Abstract

A technique for concealing missed processing deadlines
("dropouts"/"xruns") in real-time digital audio output by synthesizing a
plausible continuation of the signal, computed speculatively, instead of
silence or a glitch. A short-time Fourier analysis runs continuously
over the recent signal history; when concealment is needed, each
analysis bin is continued independently by extrapolating its phase and
amplitude and, for bins containing two sinusoidal partials, a fitted
two-pole autoregressive model - generalizing the classical
single-rotator phase-vocoder freeze, exact only for one partial per bin.
A derived aliasing limit constrains the analysis hop usable for the
two-partial model. A cheap closed-form test checks the per-bin
recursion's stability before falling back to explicit root-finding and
clamping only when needed. The continuation is spliced onto the real
signal using one of two seam-repair strategies - a broadband exponential
correction anchored to a short linear predictor, or a short linear
predictor substituted at the seam - chosen per occurrence from a
spectral-similarity measurement, each optimal for a different class of
material. A frequency-damping term suppresses a positive-feedback
failure mode arising when previously concealed audio is itself
re-analyzed. An optional second, short-window instance of the same model
runs in parallel and takes over newly-attacked content, because a long
window's history remains contaminated by pre-attack content long after a
new note begins. Measured on real sustained musical material at high
simulated miss rates, each mechanism measurably improved concealment
quality, for the content it specifically targets, over omitting it.

## Field

Concealment of missed processing deadlines ("dropouts", or "xruns" in
the terminology of several audio stacks — used here only as a widely
recognized name for the general phenomenon, not to scope this
disclosure to any one host) in real-time digital audio output, where a
periodic callback occasionally fails to produce its output buffer in
time and something must be played instead of silence or a hard glitch.

## Background

The simplest fallback when a real-time audio pipeline misses its
deadline is silence, a repeated previous buffer, or a short linear fade
to silence — audibly obvious as a dropout. **Signal-continuation
concealment itself is an established line of work, not introduced
here**: packet-loss concealment (PLC) for speech/audio transport
(e.g. ITU-T G.711 Appendix I's pitch-based regeneration) and dedicated
audio-extrapolation techniques — sinusoidal-model extrapolation with a
linear-prediction-based seam (closely related to parts 1 and 2 below;
see e.g. Malackani & Anido, "Packet loss concealment based on sinusoidal
extrapolation," ICASSP 2002, and Kauppinen & Roth, "Audio Signal
Extrapolation — Theory and Applications," DAFx-02, 2002) and frequency-
domain phase-evolution approaches for PLC (reported in later patent
filings on "phase evolution" / "Phase ECU"-style concealment) — already
disclose the general idea of continuing a signal's own recent spectral
content through a gap rather than fading to silence. (These were
identified via a sample search, not an exhaustive clearance search; a
proper prior-art search before relying on this disclosure for its
defensive-publication purpose should verify and extend this list rather
than treat it as complete.) What this disclosure contributes on top of
that established line of work is specific: computing the continuation
**speculatively every period**, before any deadline miss is known to be
occurring (real-time deadline-miss concealment has no advance warning,
unlike packet-loss concealment's typically-known loss events), a
per-bin **two-pole** continuation model with a derived hop constraint
(part 1) rather than the single-rotator freeze or whole-signal
sinusoidal models in the cited work, the specific **dual seam-repair
strategy with automatic per-event selection** (part 2), the
**frequency-damping mitigation** for the self-feedback failure mode of
re-analyzing one's own prior output (part 3), and the **dual
time-scale** attack-handling refinement (part 4) — each described in
detail below, scoped against this established background rather than
as if no continuation-based concealment existed before it.

## Summary of the technique

For each output channel, a short-time Fourier transform (STFT) analysis
is kept running continuously over the recent audio history. When a gap
must be concealed, every analysis bin is continued independently for the
duration of the gap by extrapolating its own recent phase and amplitude
behaviour — a phase-vocoder "spectral freeze", generalized beyond the
classical single-rotator freeze by also fitting a two-pole
autoregressive (Prony) model per bin, which correctly continues two
musical partials that happen to fall inside one analysis bin (the
classical single-rotator freeze is exact only for one partial per bin
and is measurably wrong for two). The synthesized continuation is
spliced back onto the real signal at both ends of the gap using one of
two alternative seam-repair strategies, chosen automatically per
occurrence from a cheap spectral-similarity measurement of the two
frames bounding the gap. An explicit damping term on each bin's fitted
frequency estimate suppresses a specific positive-feedback failure mode
that otherwise appears when concealed audio is later re-analyzed as if
it were real signal. An optional secondary, shorter-analysis-window
model runs concurrently and takes over newly-attacked frequency content
immediately, handing back once the primary model's own fit has caught
up — because a long analysis window's history necessarily still
contains pre-attack information for some time after a new note begins,
which measurably degrades the continuation of the attack transient
specifically.

## Detailed description

### 1. Per-bin continuation model

The signal history is analyzed with a sliding STFT of window length
`N` (a practical range is 1024–8192 samples depending on the acceptable
analysis latency vs. frequency resolution trade-off; shorter windows
resolve attack transients better but merge closely-spaced partials).
For concealment, the two most recent complete analysis frames are
compared per bin:

- **Order 0/1 (classical rotator):** each bin's complex value is
  continued as `X(n) = z · X(n-1)`, where `z` is a unit-magnitude
  complex number whose angle is the bin's phase advance between the two
  most recent frames (i.e. the implied steady rotation frequency of that
  bin) and whose magnitude continuation follows the measured amplitude
  trend between the frames (see part 3). This is exact when exactly one
  sinusoidal partial occupies the bin, and is the degenerate case of the
  general model below.
- **Order 2 (per-bin Prony/AR continuation):** a two-tap linear
  recursion `X(n) = a·X(n-1) + b·X(n-2)` is fit to each bin from a short
  history of that bin's own complex values by least squares, via
  **conjugate** (Hermitian, `Aᴴ`, not plain-transpose) normal equations
  — the correct estimator for a complex-valued least-squares fit, as
  opposed to the real-valued form a plain-transpose formulation would
  give — accumulated incrementally across STFT frames via
  exponentially-forgetting or windowed normal equations, not re-derived
  from scratch each time, whose two
  characteristic roots are the per-hop complex rotations of the two
  partials sharing that bin, when two are present. This correctly
  separates and continues two partials within one bin, which the
  order-0/1 rotator cannot represent at all.

  The analysis hop used specifically for the order-2 fit is constrained
  by a derivable **coincidence limit**, not an arbitrary tuning choice,
  though the exact threshold needs stating carefully. The two roots sit
  on the unit circle separated by `Δθ = Δω · hop` radians per hop, where
  `Δω` is their angular frequency separation; this is a *two-root*
  separation, not a single rotator's own per-hop phase, so the relevant
  bound is where the **two distinct roots coincide** (`Δθ` wraps a full
  `2π`, landing both roots on the same point — a genuine rank-deficient
  fit, not merely a directional ambiguity), not the single-rotator
  Nyquist bound of `π` used for the order-0/1 case elsewhere in this
  disclosure. That a window's main-lobe width is the quantity governing
  how closely two sinusoids can be resolved at all is itself standard
  spectral-analysis background (e.g. J.O. Smith III, *Spectral Audio
  Signal Processing*, "Sinusoidal Peak Interpolation"/window-width
  discussion) — the contribution of this part is the specific
  coincidence-limit derivation below for the two-partials-per-bin AR(2)
  case, not the general premise that main-lobe width bounds resolvable
  separation. Setting the maximum separation a single analysis bin can
  plausibly hold (approximately the main-lobe width of the analysis
  window, `Δf_max = 4·fs/N` for a Hann window, i.e. `Δω_max = 8π/N`) to
  reach exactly `2π` per hop gives a hard coincidence limit of
  `hop_max = N/4` samples. The disclosed implementation's actual hop,
  `N/8`, is **half of that hard limit** — a deliberate safety margin
  below the coincidence point, not the coincidence point itself; this
  matches a direct, derivable consequence of the two-partials-per-bin
  assumption and the window's main-lobe width, with `N/8` chosen for
  comfortable conditioning rather than being the boundary value itself.
  (Whatever coarse frame hop the order-0/1 model itself uses — `N`
  samples at zero frame overlap, less at any nonzero overlap — the
  order-2 fit's `N/8` hop is correspondingly that many times finer; this
  disclosure does not fix the order-0/1 model's own overlap, so the
  exact multiplier between the two models' time resolutions depends on
  that choice, not on a fixed "8×" figure.)

  **Evidence for the N/4 coincidence limit, as a conditioning figure.**
  An independent numerical check (an exact two-partial fit, `N = 2048`,
  separation equal to one full main lobe) of the AR(2) design matrix's
  conditioning (ratio of smallest to largest singular value — closer to
  1 is better-conditioned, closer to 0 is near-singular) across hop
  values confirms the derivation above and is reported here as the
  evidentiary figure this paragraph's claim needs:

  | hop | hop as a fraction of N | conditioning (sv_min/sv_max) |
  |---|---|---|
  | 64 | N/32 | 0.13 |
  | 256 | N/8 (the disclosed operating value) | 0.60 (best-conditioned of the three) |
  | 512 | N/4 (the derived hard limit) | 4.4e-14 (numerically singular) |

  Conditioning is good at the disclosed `N/8` operating hop, remains
  good up to just below `N/4`, and collapses sharply at `N/4` itself —
  consistent with `N/4` being the point where the two roots coincide
  (a genuine rank deficiency) rather than a smoothly-approached
  degradation, and consistent with `N/8` being a comfortable margin
  below that point rather than itself marginal. (This table reports a
  check performed during external review of this disclosure, not an
  in-house measurement; it is included because it is the correct kind
  of evidence for this specific claim, not because this disclosure's
  own authors generated it.)

  When the fitted recursion's characteristic roots lie outside the unit
  circle (an unstable/diverging fit, which can occur because the fit is
  least-squares on a short, noisy history), the continuation would
  otherwise grow without bound. Two alternative stabilization strategies
  are disclosed; which one applies is a configuration choice, not a
  fixed pipeline, and they differ in where they place a corrected root,
  not only in cost:

  - **Direct clamp-to-boundary.** The actual roots are computed (e.g.
    via Durand–Kerner iteration) and compared against a threshold
    `rho_clamp` (a practical default is `0.9`); any root whose magnitude
    exceeds the threshold is normalized to lie exactly on the unit
    circle (`|root'| = 1`, phase unchanged), while roots at or below the
    threshold are left untouched (preserving a transient's genuine
    decay). The threshold only decides *which* roots get corrected, not
    where they land — every corrected root ends up on the unit circle
    (an undamped, eternally-sustained oscillation at that root's
    frequency). This is the simpler, always-applied baseline strategy.
  - **Gated clamp-below-threshold.** A cheap, closed-form stability test
    is applied first, testing directly against the *same* threshold
    `rho_detect` the clamp below uses (a practical operating value is
    just above 1.0, e.g. `1.02`), not against the plain unit circle —
    evaluated by substituting `z → z / rho_detect` into the fitted
    recursion's characteristic polynomial before testing, so "passes"
    means "every root already lies within radius `rho_detect`," leaving
    no gap between what the test accepts and what the clamp below would
    otherwise correct. **The coefficients here are complex** (the fit is
    on complex STFT values), so the standard real-coefficient
    Schur-Cohn/Jury test for a 2nd-order recursion — stable iff `|b| < 1`
    and `|a| < 1 - b` — does not apply as written and measurably
    misclassifies complex cases; the complex form is `|b| < 1` and
    `|a + conj(a)·b| < 1 - |b|²` (which reduces to the familiar real
    form when `a, b` are real), evaluated on the `z/rho_detect`-scaled
    coefficients for the threshold-at-`rho_detect` version. If the test
    passes, the raw fitted coefficients are used directly with no
    further processing, avoiding root-finding in the common case. Only
    if the test fails are the actual roots computed — for this order-2
    case, directly via the quadratic formula (no iterative root-finder
    such as Durand–Kerner is needed at order 2; that iterative approach
    is reserved for the higher-order, order-4+ recursions this library
    also supports, where no closed form exists) — and clamped:

    ```
    root' = root * min(1, rho_detect / |root|)
    ```

    i.e. a failing root's magnitude is capped at exactly `rho_detect`,
    its phase left unchanged — **not** normalized all the way to the
    unit circle. This caps a detected instability's growth without
    turning it into an artificial eternal pure tone, trading a slightly
    more complex pipeline (a stability test, occasionally followed by
    root-finding) for a materially different, less aggressive
    correction than the first strategy's unconditional snap-to-unit-circle.

  Both strategies solve the same underlying problem (bound a possibly
  divergent least-squares fit) via the same two levers — a threshold
  deciding which roots to touch, and a target for where a touched root
  is placed — but make different choices on the second lever, which a
  host selects between as a configuration option.

### 2. Dual seam-repair strategy, chosen per occurrence

The spliced-in continuation must match the real signal at both the entry
and (if the gap ends before the continuation would naturally be
replaced by real audio again) the exit boundary, or an audible click
results at the seam. Two alternative repair strategies are disclosed,
each optimal on a different class of material, with automatic per-event
selection between them based on a cheap spectral-similarity measurement
of the two frames bounding the gap (a dense, highly time-varying
spectrum vs. a sparse, stationary one):

- **Anchor:** a broadband, exponentially-decaying correction term is
  added to the spectral continuation's own output, sized to make its
  first sample match a short-order linear (Burg-method) one-step-ahead
  prediction computed directly from the raw time-domain history
  immediately preceding the gap:

  ```
  x'(i) = x_spectral(i) + (x_burg(0) - x_spectral(0)) * exp(-i / tau)
  ```

  for `i = 0, 1, 2, ...` samples from the start of the gap, where
  `x_spectral` is the unmodified spectral continuation, `x_burg(0)` is
  the Burg predictor's one-step prediction, and `tau` is a decay time
  constant (a practical value is on the order of `sampleRate / 3000`,
  i.e. roughly 1/3 ms at 48 kHz) chosen to resolve the correction within
  a fraction of a millisecond while keeping it broadband rather than
  introducing a new, audible narrow-band component. (This exact sign
  convention is confirmed by the boundary condition the mechanism must
  satisfy — `x'(0)` must equal `x_burg(0)` exactly, which only this sign
  arrangement produces — rather than by a literal, unambiguous
  sign-convention statement in the implementation's own short-hand
  description of the mechanism.) This performs best
  on sustained, spectrally sparse material (e.g. a single sustained
  tone), where a single accurately-placed sample is sufficient to
  eliminate the discontinuity.
- **Bridge:** a short linear (Burg-method) free-running prediction
  entirely *replaces* the spectral continuation for a brief crossfaded
  interval at the seam. This performs best on spectrally dense material,
  where the per-bin spectral model cannot reliably identify a single
  dominant frequency per bin to anchor against, but a short-horizon
  linear time-domain predictor remains locally accurate regardless of
  spectral density.

Choosing one fixed strategy for all material was measured to underperform
this adaptive, per-occurrence choice; the two strategies are complementary
rather than one dominating the other.

### 3. Level-trend continuation and frequency damping

This explicit linear magnitude extrapolation is specifically how the
**order-0/1 rotator** of part 1 gets a non-constant amplitude trend at
all: that recursion's own dynamics hold `|X(n)|` exactly constant
(`|z| = 1` by construction), so without this separate mechanism a
crescendo or decay in progress would flatten for the duration of the
gap. Each bin's magnitude trend between the two most recent analysis
frames is extrapolated forward through the gap instead:

```
|X(n)| = |X_1| + n * (|X_1| - |X_0|) / H
```

where `X_0` and `X_1` are the bin's two most recent analysis frames,
`H` is the hop (in samples) between them, and `n` is the number of
samples into the gap — a plain linear extrapolation of the amplitude
trend. (A floor clamp at zero exists as a correctness safeguard against
a pathologically steep decay estimate; at the decay rates typically
observed on real musical material, the extrapolated value does not
reach zero within any realistic gap length, so this clamp is a
boundary-case safeguard, not a mechanism that is regularly exercised.)
The **order-2 per-bin model**, by contrast, already carries its own
amplitude evolution in the fitted recursion's root magnitudes (a root
with `|root| < 1` decays geometrically on its own, one at exactly
`|root| = rho_detect` after clamping sustains without further
correction); this disclosure does not separately re-apply the linear
extrapolation above on top of that recursion's own dynamics, though the
exact combination rule where both an order-2 fit and a longer-term
amplitude trend might disagree is not separately specified in the
material available for this disclosure.

A specific positive-feedback failure mode arises whenever concealed
(synthesized) audio can later be fed back into the same analysis history
that drives future concealment — which is unavoidable in a continuously
running analysis, since the history buffer cannot distinguish real audio
from a previous concealment. Re-analyzing the predictor's own output
re-fits each bin's frequency estimate to whatever error the predictor
itself introduced, and this loop was measured to settle into a stable,
self-reinforcing but *wrong* plateau rather than converging back to the
truth once real audio resumes feeding it. The disclosed mitigation damps
only the **frequency** (phase-advance) estimate's update — each new
measurement moves the bin's rotation unit vector only a fraction of the
way toward the freshly measured value, rather than replacing it outright:

```
z_new = d * z_old + (1 - d) * z_measured,  then renormalized to |z_new| = 1
```

where `z_old` and `z_measured` are unit-magnitude complex numbers (the
bin's previously held rotator and the one freshly measured this frame)
and `d ∈ [0, 1]` is the damping factor (`d = 0`: replace outright, the
classical freeze; `d = 1`: the frequency estimate never updates again).
Magnitudes and the underlying complex spectrum itself are left undamped
— only this already-unit-magnitude rotator is blended (interpolating a
wrapped phase angle linearly has no single well-defined value, whereas
blending two unit vectors and renormalizing is a proper interpolation);
the complex spectrum used for the actual synthesis is not touched by
this damping at all.
This was measured to materially reduce the self-feedback error at
sustained high concealment rates, at negligible cost to ordinary,
low-rate concealment accuracy.

### 4. Dual time-scale model for attack transients

Running separate, differently-sized analysis windows for transient
versus harmonic/sustained content is itself a known idea in the
phase-vocoder literature generally (identified via a sample search, not
independently verified full-text by this disclosure's authors, and not
treated as exhaustive). What this part contributes specifically is the
rest of the mechanism below: concurrent, always-warm operation of both
windows (never switched on cold), the per-band onset detector deciding
which window's output is audible, and the explicit reasoning for never
resetting either model's state at a detected attack.

Because the primary analysis window spans many periods (tens of
milliseconds), its history necessarily still contains **pre-attack**
signal content for a materially long time after a new note's onset —
degrading the model's continuation specifically during and shortly after
attack transients, independent of window length chosen (a shorter window
reduces but does not eliminate this, and trades away frequency
resolution for sustained material in exchange). The disclosed refinement
runs a **second, independent instance of the same continuation model**
at a shorter analysis window length, continuously, in parallel with the
primary (long-window) model — not switched on only when needed, because
its own internal accumulators would otherwise start cold exactly when
first needed.

A lightweight onset detector (comparing the current short-term energy
of each of a bank of frequency bands against its own value several
analysis frames earlier, thresholded with hysteresis to avoid chatter)
decides, per analysis band, when that band's content looks like a new
attack rather than a continuation of already-sustained material. Bands
judged to be a new attack are, for a limited time, synthesized from the
short-window (fast-adapting, attack-appropriate) model instead of the
long-window one; control is handed back to the long-window model once
its own fit — now informed only by post-attack history — has
accumulated enough data to be trusted again, with a short crossfade to
avoid an audible switch. Both models' internal state is maintained
continuously regardless of which one is currently audible for a given
band, specifically because resetting either model's accumulated state at
the moment of a detected attack was measured to be acoustically harmful
(it discards history the *other*, not-currently-switching bands still
need).

### Worked numerical example

One concrete, fully-dimensioned configuration, to make every symbol
above a specific number at once (values are a representative operating
point, not the only usable one):

- Sample rate `fs = 48000` Hz.
- Order-0/1 model: analysis window `N = 4096` samples (23.3 Hz bins);
  at a coarse frame hop equal to `N` (zero overlap, for concreteness —
  see part 1's caveat that this choice sets the exact "times finer"
  multiplier for the order-2 model below), a new frame every
  `4096/48000 ≈ 85.3` ms.
- Order-2 model's main-lobe separation bound:
  `Δf_max = 4·fs/N = 4·48000/4096 ≈ 46.9` Hz.
- Order-2 hard coincidence limit (part 1): `hop_max = N/4 = 1024`
  samples (`≈ 21.3` ms) — the point where two maximally-close partials'
  roots coincide.
- Order-2 disclosed operating hop: `N/8 = 512` samples (`≈ 10.7` ms) —
  half of the hard limit, i.e. `8×` finer than the order-0/1 model's
  85.3 ms frame rate in this zero-overlap configuration specifically
  (a different order-0/1 overlap choice would change this multiplier,
  not the `512`-sample order-2 hop itself).
- Stability threshold: `rho_detect = 1.02`.
- Anchor seam time constant: `tau = fs/3000 = 16` samples (`≈ 0.33` ms).
- Frequency-damping factor: `d = 0.2` (one of the three values measured
  in finding (c) below).

### Pseudocode

Per-bin state carried between frames: the bin's two most recent complex
STFT values (for the order-0/1 rotator and the magnitude-trend
extrapolation), and its exponentially-forgetting complex normal-equation
accumulators (for the order-2 fit). Per concealment event:

```
for each bin:
    # --- Part 1: continuation ---
    if order == 0 or 1:
        z = (bin.X1 / bin.X0) normalized to |z| = 1   # measured rotation
        continuation(n) = bin.X1 * z^n                 # part 1
        continuation(n) *= magnitude_trend(n)           # part 3, below

    else:  # order == 2
        (a, b) = solve bin's accumulated normal equations       # part 1
        if NOT stable_test(a, b, rho_detect):            # cheap, see part 1
            (r1, r2) = quadratic_roots(a, b)              # closed form, order 2
            r1 = clamp(r1, rho_detect); r2 = clamp(r2, rho_detect)
            (a, b) = coeffs_from_roots(r1, r2)
        continuation(n) = a * continuation(n-1) + b * continuation(n-2)
                          # seeded from bin.X1, bin.X0

    # --- Part 2: seam repair, chosen once per event, applied to every bin ---
    if spectral_similarity(frame_before_gap, frame_after_gap) indicates
       SPARSE material:
        continuation(n) += (burg_predict(history, 1) - continuation(0))
                            * exp(-n / tau)              # Anchor
    else:
        continuation(0 .. bridge_len) = burg_free_run(history, bridge_len)
                            crossfaded with continuation  # Bridge

# --- Part 3: magnitude trend, order-0/1 bins only ---
def magnitude_trend(n):
    return (|X1| + n * (|X1| - |X0|) / H) / |X1|          # linear, part 3

# --- Part 4: dual time-scale, applied before the per-bin loop above ---
for each frequency band:
    if onset_detector(band) fires:
        use the SHORT-WINDOW model's continuation for this band
    else:
        use the LONG-WINDOW model's continuation for this band
    # both models' own accumulators keep updating every frame regardless

# --- Part 3: frequency damping, applied AFTER synthesis, before the
#     next frame's analysis reads this bin's history ---
z_measured = phase advance measured on this frame's OWN real input
             (not on the just-synthesized continuation)
bin.z = normalize(d * bin.z + (1 - d) * z_measured)
```

The frequency-damping step's placement is the mechanism that defeats
the self-feedback failure mode of part 3: the *measurement* always
comes from real input when real input is available, and is blended
only fractionally into the carried rotator state even then, so a run
of concealed frames (where `z_measured` would otherwise be derived from
the technique's own prior output) drifts the carried state only slowly
rather than snapping to a value the technique generated about itself.

### Signal-flow diagram

```
                 continuous STFT analysis of real signal history
                              |
                   (gap must be concealed)
                              |
        +---------------------------------------------+
        |     PER BIN (part 1), per frequency band,    |
        |     short/long window per part 4's detector  |
        |                                               |
        |  order 0/1: rotator z, exact for 1 partial    |
        |  order 2:   AR(2) fit -> stability test ->    |
        |             (pass: use raw) or (fail: roots,  |
        |             quadratic formula, clamp)         |
        +---------------------------------------------+
                              |
                   continuation(n), per bin
                              |
              +---------------------------+
              |  part 3: magnitude trend   |  (order-0/1 bins only;
              |  linear extrapolation      |   order-2 carries its own
              +---------------------------+   amplitude in its roots)
                              |
              +---------------------------+
              |  part 2: seam repair       |
              |  Anchor (sparse) or        |
              |  Bridge (dense), chosen    |
              |  once per event            |
              +---------------------------+
                              |
                    inverse STFT / OLA
                              |
                      played as output  ---> re-enters history,
                                              closing the loop
                                              back to "STFT analysis"
                                              (part 3's damping term
                                              sits exactly on this
                                              feedback path)
```

## Measured results

**NMR metric, defined precisely.** All dB figures below use a
noise-to-mask-ratio (NMR) perceptual error metric computed as follows:
around each concealment event, the candidate and reference signals'
STFT magnitudes are compared frame-by-frame; the squared *excess*
energy where the candidate exceeds the reference, `max(candidate -
reference, 0)^2` per bin, is divided by a combined masking threshold
(a simultaneous-masking term derived from the reference's own spectrum
via a Bark-spreading model, with an exponentially-decaying post-masking
memory from preceding frames, plus an absolute-hearing-threshold floor
curve); the result is smoothed with a 150 ms moving average over time,
and the reported NMR is `10*log10(max of that ratio + epsilon)` — i.e.
a **worst-case-over-the-evaluation-window** figure, in dB (more negative
= further below the masking threshold = less audible; >0 dB = audible
by this metric). The metric is deliberately **one-sided**: it only
scores excess content the candidate adds that the reference lacks
(e.g. clicks, spurious partials), not the reverse (content the
candidate is missing, e.g. a partial that silently died during the
gap) — a known, explicitly documented limitation of this specific
metric, not of the concealment technique itself; a second, swapped
computation (reported as a separate figure, not used below) exists to
score the missing-content direction instead. "Mean / worst case over
the test material" below means the mean and the maximum of this
per-event figure taken across multiple concealment events in one test
run, not a different aggregation of the same event. (The reference
implementation of this exact metric is `audibility_metric` /
`concealment_metrics` in the reproducibility harness referenced under
"Reference implementation" below — that implementation, not this
prose, is the precise, executable specification; the description here
is a paraphrase of it.)

On real sustained-chord organ-pipe material at a simulated miss rate of
6000 concealment events per minute (a stress-test rate far above
ordinary expected real-world deadline-miss rates, chosen to make the
relevant failure modes measurable and quantifiable), the described
techniques were evaluated against an oracle upper bound (replacing
concealed bands with the actual, true missing audio, i.e. the best any
architecture built on this general approach could achieve) using the
NMR metric just defined (mean / worst case over the test material).
Finding (c) below uses a different, not directly convertible dropout-
density expression ("50% concealment", a time fraction, measured in a
separate test run rather than at "6000 events/minute") — reported here
exactly as found, not rescaled to the same units, because no documented
conversion between the two exists in the material available for this
disclosure.

(a) **The achievable ceiling is almost entirely a low-frequency
problem — reported with two open caveats, not as a clean result.**
Against the oracle, on a simulated miss rate of 6000 concealment events
per minute (the exact test buffer size behind this rate is not
confirmed in the material for this disclosure; at a commonly-used
128-sample buffer at 48 kHz this would be roughly 27% of all periods,
a plausible, non-impossible density, which is noted here only to show
the rate is physically achievable at a typical buffer size, not as a
confirmed configuration for this specific measurement): content above
12 kHz measured a perfect 0.00 / 0.00 dB, the 6–12 kHz band measured
1.43 / 0.00 dB, and the content *below* 6 kHz measured 4.99 / 3.14 dB
(mean / worst case). **Caveat 1:** a worst-case figure lower than the
corresponding mean (the 6–12 kHz band's 1.43 mean vs. 0.00 worst) is
internally inconsistent for a mean/worst-case pair and is reported
here exactly as found in the source material, unresolved — this
specific pair of numbers needs to be re-derived from a fresh
measurement run before publication, not taken at face value. **Caveat
2:** a perfect 0.00/0.00 above 12 kHz is consistent with genuine
near-perfect concealment, but equally consistent with that band simply
carrying little energy in organ-pipe material and/or not being
distinctly processed by the technique at all (making the 0.00 figure a
property of the test material or an artifact of the metric rather than
an earned result) — per-band signal energy was not included in the
measurement available for this disclosure, so this alternative
explanation cannot be ruled out. With both caveats open, the claim that
is actually supported is weaker than "confirmed... to be worth
essentially nothing": the *available* measurement is consistent with
most achievable error sitting below 6 kHz, but re-measurement
addressing both caveats is needed before relying on this as a firm
engineering conclusion. (This is also a single test run, not a
repeated-trial statistic — see the general caveat on (c) and (d)
below, which applies here too.)

(b) The per-bin order-2 continuation (part 1) is intended to recover
quality that a shorter, more time-resolved analysis window loses when
limited to the classical order-0/1 rotator; this was consistently
observed on real material, though the exact magnitude is bin- and
material-dependent and was not reduced to one isolated figure in the
measurements available for this disclosure.

(c) **Frequency-estimate damping (part 3)** was measured on a held
chord at 50% concealment, varying only the damping factor `d`: `d=1.0`
(no damping) measured **−10.7 dB**, `d=0.2` measured **−12.0 dB**, and
`d=0.05` measured **−12.5 dB** — i.e. increasing the damping (slowing
the frequency estimate's adaptation, suppressing the self-feedback loop
of part 3) monotonically improved the result by up to **1.8 dB** on
this material, with no measured cost at ordinary, non-sustained
concealment rates. (As with (a), this is a single test-run figure on
one held chord, not a repeated-trial statistic.)

(d) **The dual time-scale model (part 4)** was measured, on real
material with distinct note attacks, to beat the single long-window
model by **1.3 dB specifically at onsets** (the content it targets) in
one directly-compared configuration; it did not surpass a
longer-window, higher-order single-model configuration on *overall*
mean/worst-case error in that same comparison — consistent with the
technique's stated purpose (part 4) of specifically improving attack
continuation, not of being a general substitute for a well-tuned
single-window model. Its additional computational cost is on the order
of the short-window model's own analysis cost alone (substantially less
than a second full-order instance of the long-window model), because
the short-window model runs at a low "warm" order until an attack is
detected, stepping up to full order only once genuinely needed. (As
with (a) and (c), this is a single directly-compared configuration, not
a repeated-trial statistic.)

## Variations considered (disclosed for completeness)

- **GPU offload of the per-bin linear solve:** because each analysis
  bin's model fit is an independent small dense linear system, the solve
  step (though not the continuously-accumulating history update, which
  remains on the host CPU as a live fallback) can be dispatched to a
  general-purpose GPU compute kernel when one is available at runtime,
  with automatic, silent fallback to CPU-only operation if no compatible
  device is present or if a device error occurs during the instance's
  lifetime. Measured to reduce host CPU cost materially on hardware with
  a suitable device, at no change to the numerical result (kernel output
  verified bit-identical to the host computation on the same input).
- **Single- vs. double-precision history accumulation:** measured, on
  the test material and model orders used, to cost at most a fraction of
  a dB versus a double-precision accumulator — relevant because some
  compute devices execute double-precision arithmetic at a small
  fraction of their single-precision throughput, so a single-precision
  accumulator is the appropriate choice for a GPU-oriented
  implementation rather than emulating double precision there.

## Reference implementation and reproducibility

An open-source reference implementation of this technique — the
per-bin continuation models of part 1, the dual seam-repair strategy of
part 2, the level-trend and frequency-damping mechanisms of part 3, and
the dual time-scale model of part 4 — is published alongside this
disclosure:

- Source repository: `https://github.com/sebion7125/xorcist` (private
  until this disclosure's own publication; the URL itself does not
  change when visibility is flipped to public)

That repository also carries the stability-test unit tests, the
measurement harness (including the exact, executable definition of the
NMR metric used in "Measured results" above), and GPU-offload timing
benchmarks, so that every mechanism and every reported figure in this
disclosure can be built and run directly rather than taken on faith.

## Scope note

This disclosure describes the spectral-continuation concealment
technique, its dual seam-repair strategy, the frequency-damping
mitigation for analysis-of-own-output feedback, and the dual-time-scale
attack model, as general real-time audio signal-processing techniques,
independent of any specific host application, operating system, or
hardware vendor. It does not disclose, and makes no claim regarding, any
particular host application's thread-scheduling, buffering, or
real-time task-management architecture around the concealment
computation itself — only the signal-processing technique.
