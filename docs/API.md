# Xorcist API reference

Full reference for every public type, constant, and function in
`xorcist.h`. For what the library is and a five-line usage example,
see the [README](../README.md); this document goes deeper, in the
spirit of a man page, on everything a host integrating the library
could need.

## Quick reference

**Types and constants**
[`Xorcist`](#xorcist-type) ·
[`XORCIST_OFF`/`XORCIST_PV_2K`/`XORCIST_PV_4K`/`XORCIST_PV_8K`/`XORCIST_PV_1K`](#concealment-mode) ·
[`XORCIST_SEAM_ADAPTIVE`/`ANCHOR`/`BRIDGE`](#seam-repair-mode) ·
[`XorcistTuning`](#xorcisttuning-reference)

**Lifecycle**
[`Xorcist_new`](#xorcist_new) ·
[`Xorcist_free`](#xorcist_free) ·
[`Xorcist_channels`](#xorcist_channels) ·
[`Xorcist_version`](#xorcist_version)

**Core loop**
[`Xorcist_push`](#xorcist_push) ·
[`Xorcist_predict`](#xorcist_predict) ·
[`Xorcist_conceal`](#xorcist_conceal) ·
[`Xorcist_pass`](#xorcist_pass)

**Split (multi-threaded) prediction**
[`Xorcist_predictBegin` / `Xorcist_predictChannel` / `Xorcist_predictEnd`](#split-prediction) ·
[`Xorcist_predictAbort`](#xorcist_predictabort)

**Mode and tuning**
[`Xorcist_setMode` / `Xorcist_getMode`](#xorcist_setmode--xorcist_getmode) ·
[`Xorcist_setTuning` / `Xorcist_getTuning`](#xorcist_settuning--xorcist_gettuning)

**Buffering edge cases**
[`Xorcist_pushMany`](#xorcist_pushmany) ·
[`Xorcist_replaceLast`](#xorcist_replacelast) ·
[`Xorcist_absorb`](#xorcist_absorb)

**Inspecting a pending prediction**
[`Xorcist_peek`](#xorcist_peek) ·
[`Xorcist_horizonFrames`](#xorcist_horizonframes) ·
[`Xorcist_analysisGeometry`](#xorcist_analysisgeometry)

**Transient-patch testing**
[`Xorcist_patchForce` / `Xorcist_patchRead` / `Xorcist_patchClear`](#transient-patch-testing)

**Counters and diagnostics**
[`Xorcist_concealedCount` / `Xorcist_skippedCount`](#basic-counters) ·
[`Xorcist_capStats` / `Xorcist_safetyCap`](#xorcist_capstats--xorcist_safetycap) ·
[`Xorcist_gateStats`](#xorcist_gatestats) ·
[`Xorcist_arDropStats`](#xorcist_ardropstats) ·
[`Xorcist_backfillCount`](#xorcist_backfillcount) ·
[`Xorcist_rhoStats`](#xorcist_rhostats) ·
[`Xorcist_gpuStats`](#xorcist_gpustats) ·
[`Xorcist_raceStats` / `Xorcist_lateStats`](#xorcist_racestats--xorcist_latestats)

**Stage timing and environment overrides**
[`Xorcist_setProfiling` / `Xorcist_profiling`](#stage-timing) ·
[`Xorcist_stageCount` / `Xorcist_stageName` / `Xorcist_stageStats`](#stage-timing) ·
[`Xorcist_diagOverrides`](#xorcist_diagoverrides)

**Onset and hybrid-mode event logs**
[`Xorcist_detectorPerChannel`](#xorcist_detectorperchannel) ·
[`Xorcist_onsetStats` / `Xorcist_onsetLog`](#onset-logging) ·
[`Xorcist_hybStats` / `Xorcist_hybLog` / `Xorcist_hybLogClear`](#hybrid-mode-logging) ·
[`Xorcist_patchCount` / `Xorcist_patchLog` / `Xorcist_patchLogClear`](#patch-event-log)

**Solve trace**
[`Xorcist_solveTapInfo` / `Xorcist_solveTapRead` / `Xorcist_solveTapClear`](#solve-trace)

**Device and file info**
[`Xorcist_wisdomFileName`](#xorcist_wisdomfilename) ·
[`Xorcist_gpuDevice`](#xorcist_gpudevice)

**Guides**
[XorcistTuning reference](#xorcisttuning-reference) ·
[Typical usage patterns](#typical-usage-patterns)

---

## Types and constants

### `Xorcist` type

```c
typedef struct Xorcist Xorcist;
```

Opaque instance handle - one per output device/stream. Not thread-safe
by itself; each function below states which thread(s) it is meant to
be called from.

### Concealment mode

```c
enum {
  XORCIST_OFF     = 0,
  XORCIST_PV_2K   = 1,
  XORCIST_PV_4K   = 2,
  XORCIST_PV_8K   = 3,
  XORCIST_PV_1K   = 4
};
```

`XORCIST_OFF` makes [`Xorcist_conceal`](#xorcist_conceal) emit hard
silence - the honest baseline of what an unconcealed miss sounds like,
and a useful A/B reference.

The `PV_*` values name the STFT analysis frame size. `PV_4K` is the
general-purpose default: 2048's 23 Hz bins merge neighbouring partials
on dense chords, and 8192's 170 ms frame goes stale. `PV_1K` (1024
samples, 21 Hz frame, 47 Hz bins) trades frequency resolution for four
times the time resolution of `PV_4K`; it exists to test whether a
higher Prony order ([`XorcistTuning.order`](#order)) can make up for
the coarser frequency resolution, and needs a period size of 128 (the
analysis hop must divide `nfft/4 = 256`). The values are appended
rather than reordered because `0`-`3` are baked into saved host
settings - do not renumber them.

Set and read with [`Xorcist_setMode`](#xorcist_setmode--xorcist_getmode).

### Seam repair mode

```c
enum {
  XORCIST_SEAM_ADAPTIVE = 0,
  XORCIST_SEAM_ANCHOR   = 1,
  XORCIST_SEAM_BRIDGE   = 2
};
```

Controls how the re-entry point at the start of a concealed period is
repaired. `ADAPTIVE` (the field's default) picks per period from the
spectral density of the material, and is what a host should ship. The
two fixed settings exist because each is optimal for one kind of
material and an A/B tool needs to reach them directly:

- `ANCHOR` - pin the first sample to a Burg one-step prediction and
  leave the frequency freeze alone. Wins on sustained, sparse spectra
  (a held organ tone: measured -1.4 dB -> -17.2 dB excess NMR).
- `BRIDGE` - crossfade a short Burg free-run over the seam, replacing
  the freeze there. Wins on dense spectra, where the freeze cannot
  identify a single frequency per bin.

Set via the [`seam`](#seam) field of `XorcistTuning`.

---

## Lifecycle

### `Xorcist_new`

```c
Xorcist *Xorcist_new(
  unsigned channels,
  unsigned framesPerPeriod,
  unsigned sampleRate,
  const char *wisdomPath);
```

Creates one instance.

- `channels` - interleaved channel count of the audio handed to
  [`Xorcist_push`](#xorcist_push) and friends.
- `framesPerPeriod` - one period, i.e. the buffer size the audio
  device asks for. Fixed for the instance's lifetime.
- `sampleRate` - in Hz.
- `wisdomPath` - file the measured FFTW plans are cached in, or `NULL`
  to disable caching. FFTW's `FFTW_MEASURE` times the actual codelets
  on this machine and cannot be precomputed at build time, but the
  result persists across runs: roughly 0.9 s of planning once per
  machine instead of once per process start. See also
  [`Xorcist_wisdomFileName`](#xorcist_wisdomfilename).

Returns `NULL` on allocation failure.

### `Xorcist_free`

```c
void Xorcist_free(Xorcist *x);
```

Destroys an instance created by `Xorcist_new`.

### `Xorcist_channels`

```c
unsigned Xorcist_channels(const Xorcist *x);
```

Returns the channel count the instance was created with.

### `Xorcist_version`

```c
const char *Xorcist_version(void);
```

Returns the library version string, e.g. `"0.1.0"`.

---

## Core loop

The minimal real-time usage pattern, every period, is: `Xorcist_push`
the audio that was just produced, call `Xorcist_predict` to prepare
the *next* period's fallback, and on a miss call `Xorcist_conceal`
instead of emitting silence. See
[Typical usage patterns](#typical-usage-patterns) for a complete
example.

### `Xorcist_push`

```c
void Xorcist_push(Xorcist *x, const float *interleaved);
```

Appends one finished period of audio (interleaved) to the history
ring. Call this from whichever thread produced the mix, while it still
owns the buffer. A plain ring append - no locks, no allocation, real-time
safe.

### `Xorcist_predict`

```c
int Xorcist_predict(Xorcist *x);
```

Computes the prediction for the *next* miss from the current history.
This is the expensive call (tens of microseconds) - run it off the
critical path, typically as an idle-tail task after the period's real
work is done. It is safe to skip individual periods: predictions are
tagged with the history position they were made at, so a late or
missed computation resolves to "use a later slice of an older
prediction" rather than a phase error.

Returns:

| Value | Meaning |
| --- | --- |
| `1` | A prediction was produced. |
| `0` | Skipped - mode off, not enough history yet, or nothing new to anchor on. |
| `2` | The running prediction was deliberately *kept* because a miss streak is in progress. Re-analysing would fit a spectrum to already-predicted audio and restart the extrapolation, seaming every period. Costs nothing, which is convenient exactly when the machine has no time to spare. |
| `3` | Another thread's round is still in flight (busy guard) - behaviourally identical to `0` for the caller, but a host's late-slice diagnosis needs the two distinguished. |

See also the [split form](#split-prediction) for hosts with spare
worker threads, and [`Xorcist_absorb`](#xorcist_absorb) for analysis
without building a shadow.

### `Xorcist_conceal`

```c
int Xorcist_conceal(Xorcist *x, float *out);
```

Call this when a period was missed, instead of emitting silence.
Writes the concealed audio into `out`. Real-time safe - a copy, a few
multiplies, no locks or allocation.

Consecutive misses degrade honestly, on a *sample clock* rather than a
period count (validity is elapsed time, not how many device-buffer
periods happen to slice it): full level for roughly the first 25 ms -
the freeze's audibility horizon - then a generous ~150 ms fade, long
enough that a returning real signal can cross-fade back in from a
still-audible level.

What was actually played is appended to the history automatically, so
the next prediction stays time-aligned - do not also call
`Xorcist_push` for a concealed period.

Returns `1` if a prediction was played, `0` if silence had to be
emitted (nothing was available to play - e.g. the very first period,
or mode off).

### `Xorcist_pass`

```c
void Xorcist_pass(Xorcist *x, float *out, const float *real);
```

Call this when the period arrived in time (the normal case): copies
`real` into `out`, applying the re-entry crossfade if the *previous*
period was concealed. `out` and `real` may be the same pointer (an
in-place call).

---

## Split prediction

```c
int  Xorcist_predictBegin(Xorcist *x);
void Xorcist_predictChannel(Xorcist *x, unsigned ch);
void Xorcist_predictEnd(Xorcist *x);
void Xorcist_predictAbort(Xorcist *x);
```

Split form of [`Xorcist_predict`](#xorcist_predict) for hosts with
spare worker threads. Channels are fully independent - own FFT
workspace, own normal equations, own recursion - so they may be
computed *concurrently*:

```c
if (Xorcist_predictBegin(x) == 1) {
    parallel_for (ch)  Xorcist_predictChannel(x, ch);
    Xorcist_predictEnd(x);
}
```

`Xorcist_predictBegin` returns exactly what `Xorcist_predict` returns
(see the table [above](#xorcist_predict)); only on `1` may the channel
calls and `Xorcist_predictEnd` follow, and `Xorcist_predictEnd` must
run exactly once after the last channel call. Computed serially, the
cost scales linearly with the channel count - at eight channels that
is roughly 20% of a 128-frame period for the plain rotator and 70% for
Prony order 4, on one core (see [`order`](#order) for the full
per-order cost table).

### `Xorcist_predictAbort`

```c
void Xorcist_predictAbort(Xorcist *x);
```

Releases the compute slot opened by `Xorcist_predictBegin` *without*
publishing a result - for the case where not every channel of an
opened prediction actually ran (e.g. a worker pool that gave up
partway through). Without calling this, the slot stays taken and all
later predictions are skipped.

---

## `Xorcist_setMode` / `Xorcist_getMode`

```c
void Xorcist_setMode(Xorcist *x, int mode);
int  Xorcist_getMode(const Xorcist *x);
```

Selects the [concealment mode](#concealment-mode) (one of the
`XORCIST_*` enum values). Cheap; may be called at any time, including
from a different thread than the real-time loop, though a host doing
that should apply its own serialisation if it also mutates tuning
concurrently.

---

## `Xorcist_setTuning` / `Xorcist_getTuning`

```c
void Xorcist_setTuning(Xorcist *x, const XorcistTuning *t);
void Xorcist_getTuning(const Xorcist *x, XorcistTuning *t);
```

Reads or replaces the full tuning surface. See the
[XorcistTuning reference](#xorcisttuning-reference) below for every
field, and the [README](../README.md#configuring-the-tuning-surface)
for a short usage example. A zero-initialised `XorcistTuning` reproduces
the library's built-in default behaviour field by field.

---

## Buffering edge cases

### `Xorcist_pushMany`

```c
void Xorcist_pushMany(Xorcist *x, const float *interleaved, unsigned periods);
```

Appends `periods` finished periods at once. Purely a convenience for
offline tools and scripting bindings, where the per-call overhead of a
language boundary would otherwise dominate the ring append it wraps. A
real-time host has one period at a time and uses
[`Xorcist_push`](#xorcist_push).

### `Xorcist_replaceLast`

```c
void Xorcist_replaceLast(Xorcist *x, const float *interleaved);
```

Overwrites the period most recently appended by
[`Xorcist_conceal`](#xorcist_conceal) with the real audio for that
slot. For hosts that let a late mix run to completion: its output is
useless for playback (the slot it belonged to is already gone) but it
*is* the truth for the history, and replacing the prediction there
removes the feedback loop at its source. Measured on a held chord at
50% concealment: -10.8 dB with the prediction fed back into history,
-15.1 dB with the truth fed back instead - and feeding the truth for
only every second missed period already reaches -12.6 dB.

Does nothing if the last append did not come from `Xorcist_conceal`.

**Threading caveat:** this rewinds the ring, so it must not run while
another thread is inside `Xorcist_conceal` or `Xorcist_push`. A host
whose late mix completes on a different thread than the real-time
callback must serialise the two itself - the period-tag check catches
the common mismatch, but the rewind itself is not atomic.

### `Xorcist_absorb`

```c
void Xorcist_absorb(Xorcist *x);
```

Analysis *without* synthesis: updates the spectrum cache, the rotator,
the routing ring, the AR spectrum ring and the normal equations -
exactly the state `Xorcist_predict` would leave - but builds no
shadow buffer. For offline hosts that know no miss can be near: the
shadow is the expensive part of a prediction, and at low miss rates
almost every shadow built by `Xorcist_predict` is discarded unheard.

---

## Inspecting a pending prediction

### `Xorcist_peek`

```c
unsigned Xorcist_peek(const Xorcist *x, float *out, unsigned frames);
```

Copies the *current* prediction into `out` (interleaved) without
playing it - the miss streak, the re-entry state and the history all
stay untouched. For offline tools: an A/B lab can score the exact
samples the engine would have emitted on a real miss, instead of
reimplementing the algorithm separately and slowly drifting away from
it.

Returns the number of frames actually written (`0` if no prediction is
currently available).

### `Xorcist_horizonFrames`

```c
unsigned Xorcist_horizonFrames(const Xorcist *x);
```

Frames the current prediction spans - the upper bound to pass as
`frames` to `Xorcist_peek`.

### `Xorcist_analysisGeometry`

```c
void Xorcist_analysisGeometry(
  const Xorcist *x, unsigned *pNfft, unsigned *pAnaHop, unsigned *pArHop);
```

Reports the analysis geometry this instance is *actually* using, so a
display does not have to reconstruct it itself (and risk getting it
wrong). Any pointer may be `0`.

- `nfft` - frame length of the current mode (1024/2048/4096/8192).
- `anaHop` - STFT hop size. Normally one audio period, because this
  period's frame is exactly the older frame the next one needs; only
  an unusual period size falls back to `nfft/4`. Reflects the
  [`anaHop`](#anahop) tuning override if one is set.
- `arHop` - spacing of the grid points the AR fit sits on
  (`max(nfft/8, period size)`) - this is what "hop" means in the
  tuning-field comments and the strategy names above, *not* the STFT
  hop `anaHop`.

---

## Transient-patch testing

```c
int      Xorcist_patchForce(Xorcist *x, unsigned ch, uint64_t zoneStart, unsigned len);
unsigned Xorcist_patchRead(Xorcist *x, unsigned ch, uint64_t *pStart, float *out, unsigned maxLen);
void     Xorcist_patchClear(Xorcist *x);
```

Manual entry points for the transient-patch mechanism behind the
[`patchTransients`](#patchtransients) tuning field (an *analysis
overlay only* - the history ring itself always stays truth).

`Xorcist_patchForce` builds the backward-rotated continuation of the
last analysed state over `[zoneStart, zoneStart + len)` absolute
samples on channel `ch`, on demand - the entry point a test harness
uses to force a patch at an arbitrary point instead of waiting for the
real onset detector to fire. Returns `1` on success, `0` if `ch` is
out of range or the patch could not be built.

`Xorcist_patchRead` returns the patch content written by a forced (or
real) patch; reading does *not* clear it. `pStart` receives the
absolute sample position the patch starts at. Returns the number of
samples actually copied into `out` (capped at `maxLen`).

`Xorcist_patchClear` drops all pending patches on every channel.

See also [`Xorcist_patchCount`](#patch-event-log) and
`Xorcist_patchLog` for counting/logging patches built by the real
detector during normal operation.

---

## Counters and diagnostics

All counters below that say "reading RESETS" return the count *since
the previous call* and zero their internal accumulator - a host can
bracket exactly one measurement window by reading before and after it.
Counters that do not reset are noted explicitly.

### Basic counters

```c
uint64_t Xorcist_concealedCount(const Xorcist *x);
uint64_t Xorcist_skippedCount(const Xorcist *x);
```

`Xorcist_concealedCount` - periods concealed so far in total (both an
actual prediction and emitted silence count). Does not reset.

`Xorcist_skippedCount` - predictions skipped because a computation was
already running (see `Xorcist_predict`'s return value `3`). Does not
reset.

### `Xorcist_capStats` / `Xorcist_safetyCap`

```c
void  Xorcist_capStats(uint64_t *pHits, uint64_t *pCalls);
float Xorcist_safetyCap(const Xorcist *x);
```

`Xorcist_capStats` - shadow builds that were scaled down by the
[`safetyCap`](#safetycap) mechanism, out of how many shadows were
built in total. Reading resets both counters.

`Xorcist_safetyCap` - the cap factor currently in effect (the
`XO_SAFETY_CAP` environment override takes precedence over the tuning
field if both are set; `0` means the cap is off).

### `Xorcist_gateStats`

```c
void Xorcist_gateStats(uint64_t *pSkips, uint64_t *pTests);
```

Grid points skipped by the [`gateFit`](#gatefit) mechanism, out of how
many were tested. Reading resets both.

### `Xorcist_arDropStats`

```c
void Xorcist_arDropStats(uint64_t *pDrops, uint64_t *pTries);
```

How often the AR path fell back to the order-1 rotator because a seed
was missing from the spectrum ring, out of how many times an AR
continuation was wanted. See [`backfill`](#backfill) - this is the
counter that shows whether `backfill` is needed. Reading resets both.

### `Xorcist_backfillCount`

```c
uint64_t Xorcist_backfillCount(void);
```

Seeds recomputed from history since the last call, by the
[`backfill`](#backfill) mechanism. Reading resets it. Note: process-wide,
not per-instance, matching the `backfill` field's own scope.

### `Xorcist_rhoStats`

```c
void Xorcist_rhoStats(uint64_t *pFails, uint64_t *pTests);
```

Bins whose raw post-Gauss coefficients fail the all-roots-inside-rho
test, out of bins tested - see [`detectRho`](#detectrho). Measurement
only; armed by the `XO_COUNT_RHO` environment variable. Reading resets
both.

### `Xorcist_gpuStats`

```c
void Xorcist_gpuStats(uint64_t *pOnDevice, uint64_t *pOnCpu);
```

Solve rounds since the last call, split by whether they ran on the
OpenCL device or fell back to the CPU. A nonzero device count proves
nothing on its own: the [`replica`](#replica) path and
[`schur`](#schur) mode always stay on the CPU regardless of
[`gpu`](#gpu), and a device that fails once falls back to the CPU
permanently for the rest of the instance's life. A host that exposes
the `gpu` setting to its user should also be able to show, via this
call, whether it is actually having any effect. Reading resets both.

### `Xorcist_raceStats` / `Xorcist_lateStats`

```c
void Xorcist_raceStats(uint64_t *pStaleDiscards, uint64_t *pLateSlices);
void Xorcist_lateStats(uint64_t *pAfterSkip, uint64_t *pMaxSlice);
```

Concurrency diagnostics for hosts running
[split prediction](#split-prediction) on their own worker threads.
A single-threaded caller (such as an offline tool) always reads `0/0`
here. Both are diagnostic only - they report how much a host's
scheduling overlaps with Xorcist's own round window, not how much
audio quality that costs. Reading resets all four counters.

`Xorcist_raceStats`:

- `staleDiscards` - rounds during which the history ring was appended
  to while the round was still in flight. They are still completed
  and published against their own frozen analysis window rather than
  discarded.
- `lateSlices` - concealments drawn from a slice index greater than 0
  without the previous period itself having been concealed. This
  means the fresh round was not yet finished when a bail-out happened,
  so the previous round's result was played instead - correct output,
  but a sign that the round's compute time is pushing against the
  period length.

`Xorcist_lateStats` - root-cause breakdown of the late slices above:

- `afterSkip` - how many of `lateSlices` occurred within two periods of
  `Xorcist_predictBegin` finding a round already in progress (a "busy"
  result). `afterSkip == lateSlices` confirms the chain "the open call
  was skipped, so the anchor fell behind"; `afterSkip` much smaller
  than `lateSlices` means the anchor is falling behind for some other
  reason.
- `maxSlice` - the largest slice index actually played. `1` means
  exactly one round was missed; higher values mean rounds are falling
  behind repeatedly.

---

## Stage timing

```c
void         Xorcist_setProfiling(int on);
int          Xorcist_profiling(void);
unsigned     Xorcist_stageCount(void);
const char  *Xorcist_stageName(unsigned index);
void         Xorcist_stageStats(uint64_t *pNs, uint64_t *pCalls, unsigned maxStages);
```

Where does a round's compute time go? Off by default - the clock
itself costs roughly 20 ns per query, and the solve runs per bin, so
profiling has a real cost. Enable it with `Xorcist_setProfiling(1)` or
the `XO_PROFILE=1` environment variable; `Xorcist_profiling` reads the
current state.

Stages, in order: analysis, 4k detector, 1k hybrid, AR accumulation,
AR solve (which *includes* Durand-Kerner root-finding, not a sibling
stage - it only runs where [`order`](#order) >= 3, the rho test fails,
and [`schur`](#schur) mode is off, so only on a fraction of bins),
synthesis, cap + seam.

`Xorcist_stageCount` - number of stages. `Xorcist_stageName` - name of
stage `index`. `Xorcist_stageStats` - fills `pNs` (nanoseconds) and
`pCalls` (call count) for up to `maxStages` stages, and resets both on
read, so a host can bracket exactly one time window.

### `Xorcist_diagOverrides`

```c
const char *Xorcist_diagOverrides(void);
```

Environment diagnostic overrides currently in effect, formatted as
`"NAME=VALUE NAME=VALUE"` - empty if everything is at its default.
Lets a host *display* what is actually active, instead of a user
having to guess from an A/B test whether an environment variable was
picked up at all. The returned pointer stays valid for the process's
lifetime; the environment is read once, on the first call.

---

## Onset logging

### `Xorcist_detectorPerChannel`

```c
unsigned Xorcist_detectorPerChannel(unsigned ch, uint64_t *pOnsets);
```

Per channel: the return value is how many of the 256 routing bands
currently carry anything (within 60 dB of that channel's loudest
band). `pOnsets` receives onset triggers for this channel since start
(does *not* reset on read).

Why the occupied-band count is the number that matters: the onset
detector's criterion is a ratio, and in an empty band a factor of two
is free - there is nothing there but the predictor's own error. A
broadband channel tolerates that without issue; a sparsely occupied
one keeps triggering spuriously during steady playback (measured: 86
occupied bands, 0 spurious triggers; 3 occupied bands, 228). A stereo
downmix can never show this problem, because summing channels fills in
bands that would otherwise be empty.

### `Xorcist_onsetStats` / `Xorcist_onsetLog`

```c
void     Xorcist_onsetStats(uint64_t *pOnsets, uint64_t *pTentative);
unsigned Xorcist_onsetLog(uint64_t *pOut, unsigned maxEvents);
```

`Xorcist_onsetStats` - transient-detector triggers since the last call
(reading resets it). On held material this should read zero: the
detection criterion is a pure ratio (a band's energy against its own
level eight periods earlier), so it is level-independent by
construction - verified across a 100 dB range, same count either way.
What can still fool the detector at a high miss rate is its own
re-entries: the splice veto only looks back 3 periods, while its 4k
analysis window spans 32. `pTentative` counts the provisional triggers
from the 512-sample pre-stage ([`hybPre`](#hybpre)) that can still be
withdrawn - not used by GrandOrgue, but available for a host that
wants it.

`Xorcist_onsetLog` - sample positions of the most recent onset
triggers, oldest first, so a host can plot them against its own
timeline. Reading does *not* reset the log (so a display can redraw at
any time); the ring holds the last 256 events. Returns how many were
written into `pOut`.

## Hybrid-mode logging

```c
void     Xorcist_hybStats(uint64_t *pActive, uint64_t *pInert, uint64_t *pBandPeriods);
unsigned Xorcist_hybLog(uint64_t *pOut, unsigned maxEvents);
void     Xorcist_hybLogClear(void);
```

`Xorcist_hybStats` - hybrid-mode effect since the last call (reading
resets all three): periods during which the second (short) chain was
active, periods where [`hybrid`](#hybrid) was set but it did nothing
(base frame is already 1k, or the period size does not divide 256),
and the total band-periods the short chain actually played.

`Xorcist_hybLog` - hybrid episodes as triples
`(startSample, endSample, band | order<<16 | nfft<<32)`, channel 0
only, newest last, up to 4096 kept. For a spectrogram-style display:
which bands were computed by the short chain, and when. Reading does
*not* clear the log; `Xorcist_hybLogClear` does.

## Patch event log

```c
uint64_t Xorcist_patchCount(void);
unsigned Xorcist_patchLog(uint64_t *pOut, unsigned maxEvents);
void     Xorcist_patchLogClear(void);
```

`Xorcist_patchCount` - transient patches built since the last call
(reading resets it).

`Xorcist_patchLog` - recent patch events as triples
`(onsetSample, zoneStart, zoneLen)`, newest last, up to 512 kept. For
hosts that want to show where the detector fired. Reading does *not*
clear the log; `Xorcist_patchLogClear` does.

See also [Transient-patch testing](#transient-patch-testing) for
forcing a patch manually.

---

## Solve trace

```c
void     Xorcist_solveTapInfo(
           unsigned *pFrames, unsigned *pNbins, unsigned *pOrder, uint64_t *pTotal);
unsigned Xorcist_solveTapRead(
           float *pZ, float *pPath, float *pRoots, uint64_t *pPos,
           unsigned maxFrames);
void     Xorcist_solveTapClear(void);
```

Diagnostic trace behind the [`solveTap`](#solvetap) tuning field -
**lab/diagnostic use only**, requires `gpu=1`, and costs real time (see
`solveTap`'s own description). One frame is recorded per complete pass
over channel 0's bins. The ring is capped by memory (48 MB, 16 to 512
frames), because a frame's size grows with the order: it carries, per
bin, the largest pole radius before clamping, which branch was taken,
and every pole itself.

`Xorcist_solveTapInfo` reports how many frames are available to read,
the geometry they were written with, and how many have been written in
total. The first two numbers are needed to size the buffers passed to
`Xorcist_solveTapRead`; `pTotal` is meant for watching a live run (the
difference from the last call is the number of new frames). If the bin
count or order changes, the ring starts over - frames from two
different geometries could not be represented together.

`Xorcist_solveTapRead` writes the most recent `maxFrames` frames in
chronological order. Any pointer may be `0`: `pZ` and `pPath` each need
`maxFrames * nbins` floats, `pRoots` needs
`maxFrames * nbins * order * 2` floats (re/im pairs, `NAN` where that
branch computed no root), `pPos` the sample position of each frame.
Returns the number of frames actually written. Reading does *not*
clear the ring; `Xorcist_solveTapClear` does.

---

## Device and file info

### `Xorcist_wisdomFileName`

```c
const char *Xorcist_wisdomFileName(void);
```

Name of the FFTW wisdom file (see [`Xorcist_new`](#xorcist_new)'s
`wisdomPath`), so a host can whitelist it in its own cache-directory
housekeeping.

### `Xorcist_gpuDevice`

```c
const char *Xorcist_gpuDevice(void);
```

Name of the device the per-bin solve would run on with
[`gpu=1`](#gpu), or `NULL` when no OpenCL runtime or device is
available (the library then computes on the CPU silently).

---

## XorcistTuning reference

```c
void Xorcist_setTuning(Xorcist *x, const XorcistTuning *t);
void Xorcist_getTuning(const Xorcist *x, XorcistTuning *t);
```

A zero-initialised `XorcistTuning` reproduces the library's built-in
default behaviour field by field - see the
[README](../README.md#configuring-the-tuning-surface) for the
get/modify/set usage pattern. The authoritative, full-length
explanation of every field (including every measured number) lives as
a comment directly above its declaration in `xorcist.h`; this section
gives each field a short description and groups them the same way the
README does, so the two stay consistent.

### Fields relevant to a typical integration

#### `order`

`int`, default `1`. The main quality/cost knob. `1` is the classic
single-rotator continuation (`X(n) = z·X(n-1)`, `|z|=1`) - exact for
one partial per bin, structurally wrong for two. `2`-`8` fit a per-bin
Prony model instead, which can represent two partials per bin and
measurably improves concealment on dense material, at a real,
documented CPU cost per order - see the field's full comment in
`xorcist.h` for the complete dB-vs-cost table across orders 1-4.

#### `gpu`

`int`, default `0`. `1` = run the per-bin solve on an OpenCL device
(first platform, first device found). The RLS update itself always
stays on the CPU. On any device error the library reverts to CPU-only
operation for the rest of the instance's life. No OpenCL runtime
installed means silent CPU operation - this field is always safe to
set. See [`Xorcist_gpuDevice`](#xorcist_gpudevice) and
[`Xorcist_gpuStats`](#xorcist_gpustats) to check whether it is actually
engaging a device.

#### `seam`

`int`, one of the [`XORCIST_SEAM_*`](#seam-repair-mode) constants,
default `XORCIST_SEAM_ADAPTIVE`. Picks the seam-repair strategy at the
start of a concealed period.

#### `noTrend`

`int`, default `0`. `1` = freeze the level at the point of the miss
instead of continuing its trend (a swell or decay) through the gap.

#### `detectRho`

`float`, default `0` (classic path). Above `0`: detector mode - tests
raw post-Gauss coefficients for all-roots-inside-`detectRho` before
falling back to explicit root-finding. See the field's full comment
for the pass/fail criteria and measured pass rate. Process-wide, like
[`schur`](#schur) and [`rootClamp`](#rootclamp).

#### `freqDamping`

`float`, `0`-`1`, default `0` (off/classic). Damps the per-bin
frequency estimate to break a positive-feedback loop that otherwise
appears once the analysis history contains the predictor's *own*
output. `1` = frequency frozen after the first estimate.

#### `rootClamp`

`float`, default `0` (built-in `0.9`). Trigger threshold of the root
clamp; the *target* is always the unit circle, this only decides which
roots get snapped onto it.

#### `safetyCap`

`float`, default `0` (built-in `1.2`); negative keeps the built-in
value explicitly. Caps the shadow's peak at `safetyCap` times the
loudest sample in the last 2048 history samples - it scales, it does
not clip. `0` turns the cap off entirely.

### Fields from this library's own research

The rest of the struct documents experiments, A/B comparisons against
an offline reference implementation, and lab/diagnostic hooks built
during this library's own development. They are documented in full in
`xorcist.h` for reproducibility and for anyone doing the same kind of
tuning work - not because a typical integration is expected to touch
them. Several say so explicitly in their own comment (`solveTap`:
"lab/diagnostic use only"; `replica`: "Not a recommendation"); where a
field does not say so, its default (`0` unless noted) is the
recommended starting point. One-line pointers, in header order:

- **`anaHop`** - analysis hop override, in frames; `0` = one period.
- **`order`-adjacent precision note** - see `order` above; the comment
  block also documents a GPU-port precision finding (float accumulator
  loses at most 0.1 dB vs. the double accumulator used here).
- **`schur`** - alternate stabilisation via reflection coefficients
  instead of root-finding; faster, not bit-identical to the default.
- **`gateFit`** - skips AR fit updates whose input overlaps the
  predictor's own prior output; see `Xorcist_gateStats`.
- **`staleWindow`** - periods after a miss streak during which the
  shadow is not rebuilt.
- **`backfill`** - recomputes a missing AR seed from history instead
  of silently dropping to the order-1 rotator; see
  `Xorcist_arDropStats` and `Xorcist_backfillCount`.
- **`fitWindow`** - switches the normal equations from exponential
  forgetting to a fixed-length rebuilt window.
- **`solveEvery`** - how often each bin's AR fit is actually resolved
  (the "striped" solve) versus every period.
- **`burgEvery`** - how often the seam anchor's Burg model is refit.
- **`replica`** - runs the chain to match an offline numpy prototype
  for like-for-like comparison; explicitly not a recommended setting.
- **`noFadeInRing`** - whether the re-entry crossfade is written into
  the history ring (default: it is, which is the fix for a feedback
  artefact found during development).
- **`capSpan`** - how many shadow samples the safety cap measures its
  peak over.
- **`reentryFade`** - re-entry crossfade length override, in frames
  (can only shorten the built-in default).
- **`anchorTau`** - time constant of the seam anchor's decaying
  exponential, in samples.
- **`patchTransients`** - enables the transient-patching analysis
  overlay; see [Transient-patch testing](#transient-patch-testing).
- **`hybrid`** and the whole `hyb*` family (`hybShortCap`,
  `hybHandOrder`, `hybReset`, `hybAllBands`, `hybChooser`, `hybPre`,
  `hybSimple`, `hybWarmOrder`) - the experimental two-chain hybrid
  mode; see `Xorcist_hybStats`/`Xorcist_hybLog` for its runtime effect.
- **`detResGate`, `detResDecay`** - residual-based gate used by the
  simplified hybrid detector (`hybSimple`).
- **`v0Hindcast`** - alternate seam-anchor computation that skips the
  Burg fit entirely; `0` (default) uses the Burg v0 path.
- **`solveTap`** - lab/diagnostic solve-trace recording; see
  [Solve trace](#solve-trace).

---

## Typical usage patterns

### Single-threaded real-time loop

```c
#include "xorcist.h"

Xorcist *x = Xorcist_new(channels, framesPerPeriod, sampleRate, "/var/cache/xorcist.wisdom");
Xorcist_setMode(x, XORCIST_PV_4K);

/* every period: */
if (period_completed_in_time) {
    Xorcist_pass(x, output_buffer, mixed_audio);
    Xorcist_push(x, output_buffer);
} else {
    if (!Xorcist_conceal(x, output_buffer)) {
        /* nothing to play - emit real silence this once */
    }
    /* Xorcist_conceal already appended the played audio to history */
}

/* off the critical path, e.g. as an idle-tail task: */
Xorcist_predict(x);

Xorcist_free(x);
```

### Multi-threaded prediction

For a host that already has a worker pool and wants the per-channel
prediction cost spread across it instead of paid serially on one core:

```c
if (Xorcist_predictBegin(x) == 1) {
    unsigned n = Xorcist_channels(x);
    /* submit one task per channel to the worker pool: */
    for (unsigned ch = 0; ch < n; ch++)
        pool_submit(^{ Xorcist_predictChannel(x, ch); });
    pool_wait_all();
    Xorcist_predictEnd(x);
} else {
    /* Begin returned 0, 2 or 3 - nothing to do, same as Xorcist_predict()
     * returning that value. If a worker pool fails partway through
     * instead of completing every channel, call Xorcist_predictAbort(x)
     * rather than Xorcist_predictEnd(x). */
}
```

### Instrumentation for a diagnostics display

```c
Xorcist_setProfiling(1);

/* periodically, e.g. once a second: */
unsigned n = Xorcist_stageCount();
uint64_t ns[16], calls[16];  /* size >= Xorcist_stageCount() */
Xorcist_stageStats(ns, calls, n);
for (unsigned i = 0; i < n; i++)
    printf("%-16s %8llu ns / %6llu calls\n",
           Xorcist_stageName(i), (unsigned long long)ns[i],
           (unsigned long long)calls[i]);

uint64_t hits, builds;
Xorcist_capStats(&hits, &builds);
printf("safety cap: %llu/%llu shadows scaled down\n",
       (unsigned long long)hits, (unsigned long long)builds);
```

See [Stage timing](#stage-timing) and
[Counters and diagnostics](#counters-and-diagnostics) for the complete
set of available counters.
