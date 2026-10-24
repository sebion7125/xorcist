/*
 * Xorcist - it casts out the xruns.  See xorcist.h for the contract.
 *
 * The DSP was developed and listening-tested inside GrandOrgue
 * (GOSoundConcealment) and extracted verbatim, so the engine, this
 * library and the offline lab all run ONE implementation - lab results
 * that do not transfer to the product were a real and costly bug class
 * before the split.
 *
 * Copyright (C) 2026 Sebion7125
 *
 * Dual-licensed: use under the GNU General Public License v3.0 or
 * later (see LICENSE), or under a separate commercial license for
 * proprietary use without the GPL's obligations - contact
 * sebion7125@gmail.com for commercial licensing terms.
 */

#include "xorcist.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "fftw3.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define XORCIST_VERSION "0.1.0"

// ---- Stufenmessung ----------------------------------------------------
// Wohin geht die Rundenlaufzeit wirklich?  Bis 10.08.2026 gab es dazu nur
// zwei Zahlen (absorb gegen predict), und die Kostenmodelle der Messreihen
// stammten von absorb - also OHNE Solve, Extrapolation, Synthese, Cap und
// Naht, zusammen 77 % der Arbeit.  Hier wird jede Stufe einzeln gestoppt,
// Durand-Kerner ausdruecklich getrennt, weil er nur in einem Teil der Bins
// ueberhaupt laeuft (Ordnung >= 3, rho-Test durchgefallen, Schur aus).
//
// AUS per Vorgabe: die Uhr kostet ~20 ns je Abfrage, und SolveBin laeuft
// pro Bin.  XO_PROFILE=1 oder Xorcist_setProfiling(1) schaltet scharf.
enum XoStage {
  XO_ST_VORLAUF = 0,   // Aufmass vor dem Detektor - fast nichts
  XO_ST_DETEKTOR,      // eigene 4k-Analyse des Simpel-Detektors
  XO_ST_HYBRID,        // kurze 1k-Kette: Analyse, Tick, Synthese
  XO_ST_ANALYSE,       // Vorwaerts-FFT, Freeze, Backfill, Naht-Routing
  XO_ST_AR_AKKU,       // Normalgleichungen fortschreiben
  XO_ST_AR_SOLVE,      // SolveBin gesamt (inkl. Durand-Kerner)
  XO_ST_DURAND,        // NUR die Wurzeliteration
  XO_ST_SYNTHESE,      // Extrapolation + iFFT + Ueberlappungsaddition
  XO_ST_CAP_NAHT,      // Sicherheitsdeckel, Naht, Burg-Bruecke
  XO_ST_COUNT
};

// Namen sagen, was die Uhr WIRKLICH umschliesst.  Der erste Anlauf hiess
// hier "Analyse" und mass 0.2 us - die FFT lag naemlich im naechsten
// Block.  Eine Messung mit falscher Beschriftung ist schlimmer als keine.
static const char *const s_StageNames[XO_ST_COUNT] = {
  "Vorlauf", "Detektor 4k", "Hybrid 1k", "Analyse (FFT/Freeze)",
  "AR-Akkumulation", "AR-Solve", "davon Durand-Kerner", "Synthese",
  "Cap + Naht"};

static std::atomic<int> s_Profile{[] {
  const char *e = getenv("XO_PROFILE");

  return e && atoi(e) ? 1 : 0;
}()};
static std::atomic<uint64_t> s_StageNs[XO_ST_COUNT];
static std::atomic<uint64_t> s_StageCalls[XO_ST_COUNT];

// Kein RAII: mehrere Stufen liegen ineinander (Durand-Kerner in AR-Solve),
// und ein Bereichsobjekt je Bin waere hier teurer als der Messwert.
struct XoClock {
  std::chrono::steady_clock::time_point t0;
  bool on;

  explicit XoClock(bool armed)
    : t0(), on(armed) {
    if (on)
      t0 = std::chrono::steady_clock::now();
  }
  void Stop(XoStage s) {
    if (!on)
      return;
    s_StageNs[s].fetch_add(
      (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0)
        .count(),
      std::memory_order_relaxed);
    s_StageCalls[s].fetch_add(1, std::memory_order_relaxed);
  }
};

// Diagnoseschalter: XO_NO_FREEZE=1 nimmt das Einfrieren des Rundenfensters
// KOMPLETT zurueck - Schreibkopf und Periodennummer werden wieder live
// gelesen, UND PredictEnd verwirft wieder, wenn der Ring waehrenddessen
// weitergelaufen ist.  Das ist der Zustand vor dem 10.08.2026, auf Knopfdruck
// und in derselben Binaerdatei, damit ein A/B im laufenden Wirt moeglich ist
// statt einer Bisektion ueber Commits.  Ohne das Verwerfen waere es ein
// dritter Zustand, den es nie gab.
static const bool s_NoFreeze = [] {
  const char *e = getenv("XO_NO_FREEZE");

  return e && atoi(e) != 0;
}();

// Diagnoseschalter: XO_V0_LAST=1 ersetzt den Anker-Bezugswert v0 (Burgs
// Einschritt-Vorhersage des NAECHSTEN Samples) durch das LETZTE echte
// Sample vor der Luecke.  Nutzerfrage 12.08.2026: reicht das klanglich?
// Der Unterschied ist der natuerliche Sample-zu-Sample-Fortschritt des
// Signals - wer ihn in den Sprung einrechnet, injiziert ihn als
// breitbandige Exponentiale in jede Naht.  Hier als Messknopf, nicht als
// Meinung.
static const bool s_V0Last = [] {
  const char *e = getenv("XO_V0_LAST");

  return e && atoi(e) != 0;
}();

// Diagnoseschalter: XO_NO_SEAM=1 nimmt die GESAMTE Burg-Naht heraus -
// kein Fit, kein v0, kein Anker-Abklingen, keine Bruecke.  Nutzerfrage
// 12.08.2026: braucht die Naht-Maschinerie ab Spektral-Ordnung 2
// ueberhaupt noch jemand (beide Kalibrierungen stammen aus der
// Ordnung-1-Zeit)?  Der Schalter macht die Frage messbar, bevor
// irgendwer Code abreisst.
static const bool s_NoSeam = [] {
  const char *e = getenv("XO_NO_SEAM");

  return e && atoi(e) != 0;
}();

// Hindcast-Anker (Nutzeridee 12.08.2026): Naht am gemessenen Versatz
// verankern statt an der Vorhersage - seit dem Monitor-Schalter ein
// TUNING-Feld (XorcistTuning::v0Hindcast, dort die volle Beschreibung);
// XO_V0_SHADOW setzt nur noch den Startwert (Harness-Skripte).
//
// XO_V0_ORDER=n (1..32) laesst den Burg-Fit dagegen nur bis Ordnung n
// laufen - die "kleines Modell fuer EIN Sample"-Variante derselben Idee.
static const unsigned s_V0Order = [] {
  const char *e = getenv("XO_V0_ORDER");
  const int v = e ? atoi(e) : 0;

  return v > 0 ? (unsigned)v : 0u;
}();

static inline bool XoProfiling() {
  return s_Profile.load(std::memory_order_relaxed) != 0;
}

// ---- AVX2-Kern der Extrapolation --------------------------------------
// Dasselbe Muster wie GOSoundFader/GOSoundResampleBatch im Wirt: der
// Kernel traegt target("avx2"), der Rest der Uebersetzungseinheit bleibt
// auf der Grundarchitektur (GO liefert generisches x86-64 aus), und ein
// Laufzeit-Tor entscheidet.
//
// FMA wird BEWUSST nicht benutzt, und fp-contract steht aus.  Getrennte
// mul/sub/add machen jedes Element bit-identisch zum Skalarpfad - sonst
// rechnete dieselbe Bibliothek auf einer AVX2-Maschine anders als auf
// einer ohne, und Lab und GO waeren nicht mehr vergleichbar.  Genau die
// Vergleichbarkeit ist der Grund, warum die Bibliothek ueberhaupt
// gebunden statt nachgebaut wird.
#if (defined(__GNUC__) || defined(__clang__))                                  \
  && (defined(__x86_64__) || defined(__i386__))
#define XO_AVX2 1
#include <immintrin.h>
#endif

#ifdef XO_AVX2
#ifndef __clang__
#define XO_AVX2_FN __attribute__((target("avx2"), optimize("fp-contract=off")))
#else
#define XO_AVX2_FN __attribute__((target("avx2")))
#endif

static inline bool XoAvx2() {
  // XO_NO_AVX2=1 erzwingt den Skalarpfad - Notausgang und, wichtiger,
  // die Gegenprobe auf Bit-Gleichheit auf EINER Maschine.
  static const bool has
    = __builtin_cpu_supports("avx2") && !getenv("XO_NO_AVX2");

  return has;
}

// acc = a * s  (komplex), acht Bins je Durchgang.
XO_AVX2_FN static void XoAdvFirst(
  unsigned n, const float *aRe, const float *aIm, const float *sRe,
  const float *sIm, float *accRe, float *accIm) {
  unsigned i = 0;

  for (; i + 8 <= n; i += 8) {
    const __m256 ar = _mm256_loadu_ps(aRe + i);
    const __m256 ai = _mm256_loadu_ps(aIm + i);
    const __m256 sr = _mm256_loadu_ps(sRe + i);
    const __m256 si = _mm256_loadu_ps(sIm + i);

    _mm256_storeu_ps(
      accRe + i,
      _mm256_sub_ps(_mm256_mul_ps(ar, sr), _mm256_mul_ps(ai, si)));
    _mm256_storeu_ps(
      accIm + i,
      _mm256_add_ps(_mm256_mul_ps(ar, si), _mm256_mul_ps(ai, sr)));
  }
  for (; i < n; i++) {
    accRe[i] = aRe[i] * sRe[i] - aIm[i] * sIm[i];
    accIm[i] = aRe[i] * sIm[i] + aIm[i] * sRe[i];
  }
}

// acc += a * s  (komplex).
XO_AVX2_FN static void XoAdvAcc(
  unsigned n, const float *aRe, const float *aIm, const float *sRe,
  const float *sIm, float *accRe, float *accIm) {
  unsigned i = 0;

  for (; i + 8 <= n; i += 8) {
    const __m256 ar = _mm256_loadu_ps(aRe + i);
    const __m256 ai = _mm256_loadu_ps(aIm + i);
    const __m256 sr = _mm256_loadu_ps(sRe + i);
    const __m256 si = _mm256_loadu_ps(sIm + i);

    _mm256_storeu_ps(
      accRe + i,
      _mm256_add_ps(
        _mm256_loadu_ps(accRe + i),
        _mm256_sub_ps(_mm256_mul_ps(ar, sr), _mm256_mul_ps(ai, si))));
    _mm256_storeu_ps(
      accIm + i,
      _mm256_add_ps(
        _mm256_loadu_ps(accIm + i),
        _mm256_add_ps(_mm256_mul_ps(ar, si), _mm256_mul_ps(ai, sr))));
  }
  for (; i < n; i++) {
    accRe[i] += aRe[i] * sRe[i] - aIm[i] * sIm[i];
    accIm[i] += aRe[i] * sIm[i] + aIm[i] * sRe[i];
  }
}
#endif

// ---- endlich? ---------------------------------------------------------
// std::isfinite() ist in diesem Uebersetzungsmodul WERTLOS: GrandOrgue
// baut mit -ffast-math, das -ffinite-math-only einschliesst, und darunter
// faltet GCC jedes isfinite() zur Konstanten `true` (nachgeprueft am
// Assembler: `movl $1, %eax; ret`).  Saemtliche NaN-Netze in SolveBin
// waren damit toter Code - gemessene Folge: nach Stille + Einsatz
// divergiert Durand-Kerner, die NaN-Wurzeln landen im Warmstart-Speicher
// des Bins und werden von dort aus EWIG neu gesaet (Reproduktion
// 2026-08-09: 140288 NaN-Abtastwerte in drei Sekunden eines voellig
// normalen gehaltenen Tons, lange nach dem ausloesenden Ereignis).
//
// Der Bitmustertest kennt keine Rechenregel und ueberlebt daher jede
// Optimierungsstufe und jedes Mathematik-Flag.  Der Uebersetzungsschalter
// -fno-finite-math-only in src/xorcist/CMakeLists.txt repariert dasselbe
// noch einmal von aussen; beide Guertel bleiben, weil genau diese Klasse
// von stiller Wirkungslosigkeit hier schon mehrfach Wochen gekostet hat.
static inline bool XoFinite(double v) {
  uint64_t b;

  std::memcpy(&b, &v, sizeof b);
  return (b & 0x7FF0000000000000ull) != 0x7FF0000000000000ull;
}

static inline bool XoFinite(float v) {
  uint32_t b;

  std::memcpy(&b, &v, sizeof b);
  return (b & 0x7F800000u) != 0x7F800000u;
}

// Wisdom is a process-wide FFTW resource: import once, no matter
// how many instances a host creates.
static std::atomic<bool> s_WisdomTried{false};

// ---- OpenCL host layer (optional) -------------------------------------
// dlopen'd, never linked: the same .so must run in the VM without any
// OpenCL and on the host with an NVIDIA/AMD/Intel/pocl runtime.  The
// pattern is clbench's, lifted into the library; the kernels are the ONE
// TEXT in xorcist_kernels.cl, verified bit-identical against SolveBin
// (fb6e9bfaa).  v1 scope: only the SOLVE runs on the device - it is the
// O(p^3) stage - while the RLS update stays on the CPU so the
// accumulators remain a warm fallback if the device dies mid-run.
#ifndef XO_AR_ORDER_MAX
#define XO_AR_ORDER_MAX 8
#endif
#ifndef XO_NO_OPENCL
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <dlfcn.h>
#include <mutex>
#include <string>

namespace {

struct XoCl {
  bool ok = false;
  cl_context ctx = nullptr;
  cl_command_queue q = nullptr;
  cl_kernel kSolve = nullptr;
  char device[160] = {0};

  decltype(&clGetPlatformIDs) GetPlatformIDs = nullptr;
  decltype(&clGetDeviceIDs) GetDeviceIDs = nullptr;
  decltype(&clGetDeviceInfo) GetDeviceInfo = nullptr;
  decltype(&clCreateContext) CreateContext = nullptr;
  decltype(&clCreateCommandQueueWithProperties) CreateQueue = nullptr;
  decltype(&clCreateProgramWithSource) CreateProgram = nullptr;
  decltype(&clBuildProgram) BuildProgram = nullptr;
  decltype(&clGetProgramBuildInfo) GetBuildInfo = nullptr;
  decltype(&clCreateKernel) CreateKernel = nullptr;
  decltype(&clCreateBuffer) CreateBuffer = nullptr;
  decltype(&clReleaseMemObject) ReleaseMem = nullptr;
  decltype(&clEnqueueWriteBuffer) Write = nullptr;
  decltype(&clEnqueueReadBuffer) Read = nullptr;
  decltype(&clEnqueueFillBuffer) Fill = nullptr;
  decltype(&clSetKernelArg) SetArg = nullptr;
  decltype(&clEnqueueNDRangeKernel) Launch = nullptr;
  decltype(&clFinish) Finish = nullptr;

  static XoCl &get() {
    static XoCl g;
    static std::once_flag once;

    std::call_once(once, [] { g.Init(); });
    return g;
  }

  static char *Slurp(const char *path, size_t *pLen) {
    FILE *f = fopen(path, "rb");

    if (!f)
      return nullptr;
    fseek(f, 0, SEEK_END);
    *pLen = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);

    char *b = (char *)malloc(*pLen + 1);
    const bool okRead = b && fread(b, 1, *pLen, f) == *pLen;

    fclose(f);
    if (!okRead) {
      free(b);
      return nullptr;
    }
    b[*pLen] = 0;
    return b;
  }

  void Init() {
    void *so = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);

    if (!so)
      so = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!so)
      return; // keine OpenCL-Laufzeit - stiller CPU-Betrieb

#define XO_SYM(f, n) \
  do { \
    f = (decltype(f))dlsym(so, n); \
    if (!f) \
      return; \
  } while (0)
    XO_SYM(GetPlatformIDs, "clGetPlatformIDs");
    XO_SYM(GetDeviceIDs, "clGetDeviceIDs");
    XO_SYM(GetDeviceInfo, "clGetDeviceInfo");
    XO_SYM(CreateContext, "clCreateContext");
    XO_SYM(CreateQueue, "clCreateCommandQueueWithProperties");
    XO_SYM(CreateProgram, "clCreateProgramWithSource");
    XO_SYM(BuildProgram, "clBuildProgram");
    XO_SYM(GetBuildInfo, "clGetProgramBuildInfo");
    XO_SYM(CreateKernel, "clCreateKernel");
    XO_SYM(CreateBuffer, "clCreateBuffer");
    XO_SYM(ReleaseMem, "clReleaseMemObject");
    XO_SYM(Write, "clEnqueueWriteBuffer");
    XO_SYM(Read, "clEnqueueReadBuffer");
    XO_SYM(Fill, "clEnqueueFillBuffer");
    XO_SYM(SetArg, "clSetKernelArg");
    XO_SYM(Launch, "clEnqueueNDRangeKernel");
    XO_SYM(Finish, "clFinish");
#undef XO_SYM

    cl_platform_id plat;
    cl_device_id dev;

    if (GetPlatformIDs(1, &plat, nullptr) != CL_SUCCESS)
      return;
    if (GetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, nullptr) != CL_SUCCESS)
      return;
    GetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(device) - 1, device, nullptr);

    cl_int err = CL_SUCCESS;

    ctx = CreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
      return;
    q = CreateQueue(ctx, dev, nullptr, &err);
    if (err != CL_SUCCESS)
      return;

    // Kernelquelle: Umgebung > neben der .so > Repo-Pfad.
    size_t slen = 0;
    char *src = nullptr;
    const char *envSrc = getenv("XO_KERNEL_SRC");

    if (envSrc)
      src = Slurp(envSrc, &slen);
    if (!src) {
      Dl_info di;
      static int here = 0;

      if (dladdr((void *)&here, &di) && di.dli_fname) {
        std::string p(di.dli_fname);
        const size_t cut = p.find_last_of('/');

        if (cut != std::string::npos) {
          p = p.substr(0, cut + 1) + "xorcist_kernels.cl";
          src = Slurp(p.c_str(), &slen);
        }
      }
    }
    if (!src)
      src = Slurp("src/xorcist/xorcist_kernels.cl", &slen);
    if (!src)
      return;

    cl_program prog = CreateProgram(ctx, 1, (const char **)&src, &slen, &err);

    free(src);
    if (err != CL_SUCCESS)
      return;

    char opts[96];

    snprintf(
      opts, sizeof(opts), "-cl-std=CL1.2 -DXO_ORDER_MAX=%d",
      (int)XO_AR_ORDER_MAX);
    if (BuildProgram(prog, 1, &dev, opts, nullptr, nullptr) != CL_SUCCESS) {
      size_t n = 0;

      GetBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);

      char *log = (char *)malloc(n + 1);

      GetBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n, log, nullptr);
      log[n] = 0;
      fprintf(stderr, "xorcist: Kernel-Build fehlgeschlagen:\n%s\n", log);
      free(log);
      return;
    }
    kSolve = CreateKernel(prog, "xo_solve", &err);
    if (err != CL_SUCCESS)
      return;
    ok = true;
  }
};

} // namespace
#endif // XO_NO_OPENCL


struct Xorcist {
  Xorcist(
    unsigned nChannels,
    unsigned samplesPerBuffer,
    unsigned sampleRate,
    const char *wisdomPath);
  ~Xorcist();

  Xorcist(const Xorcist &) = delete;
  Xorcist &operator=(const Xorcist &) = delete;

  void Push(const float *pInterleaved);
  float DegradeGain(uint64_t missedSamples) const;
  int Predict();
  void Absorb();
  bool BuildBackPatch(unsigned ch, uint64_t zoneStart, unsigned len);
  int Conceal(float *pOutBuf);
  void Pass(float *pOutBuf, const float *pReal);

  static constexpr const char *WISDOM_FILE_NAME = "fftw-wisdomf.dat";

  // Method and telemetry: per instance here, so a host can run several
  // devices independently.  (They were process-wide statics while this
  // lived in the engine, driven by one dialog.)
  std::atomic<int> m_Mode{XORCIST_PV_4K};
  std::atomic<int> m_SeamMode{XORCIST_SEAM_ADAPTIVE};
  std::atomic<int> m_NoTrend{0};
  std::atomic<int> m_AnaHopOverride{0};
  std::atomic<float> m_FreqDamping{0.0f};
  std::vector<float> m_RotKeepRe, m_RotKeepIm; // damped rotator state
  std::vector<char> m_RotKeepValid; // [ch] - was, until the channels
                                    // could run concurrently, one flag
                                    // set by the LAST channel.
  std::atomic<uint64_t> m_ConcealCount{0};
  // History position whose ring append came from Conceal() itself.  When
  // it equals the current position the last period was concealed, i.e. a
  // streak is running and the ring holds PREDICTED, not real, audio.
  std::atomic<uint64_t> m_ConcealedThrough{~0ull};
  // Bitmask of the most recent ring appends, LSB = newest: 1 = the
  // period came from Conceal()'s self-feed, 0 = real audio.  Read by the
  // seam decision: the Burg bridge must not run from a PREDICTED seed.
  std::atomic<uint32_t> m_RecentPredMask{0};
  // Coarse band-energy snapshots for the seam ROUTING, one ring of
  // (1024/buf + 1) slots per channel.  Routing needs a ~21 ms baseline:
  // the short (one-period) similarity was calibrated by scaling with
  // hop^2, which is only valid for deterministic partial drift - the
  // noise/reverb floor of real recordings does NOT shrink with the hop,
  // and x64 pushed it over the bridge threshold even on a clean held
  // tone (measured wBridge 0.84 where the anchor was 4.5 dB better).
  // 256 bands keep the memory traffic negligible (1 KB/period) and
  // still separate the corpus cleanly: held tones <= 0.003, dense
  // moving material >= 0.02.
  static constexpr unsigned ROUTE_BANDS = 256;
  unsigned m_RouteSlots; // ring length = 1024/buf + 1
  std::vector<float> m_RouteBandE;        // [ch][slot][ROUTE_BANDS]
  std::vector<uint64_t> m_RoutePeriod;    // [ch][slot] period tag
  std::vector<unsigned> m_RouteNfft;      // [ch][slot] frame-size tag
  // ---- per-bin AR(2) continuation (Prony), tuning.order == 2 ----------
  // The rotator is an order-1 recursion and cannot represent two partials
  // in one bin.  Order 2 fits X(n) = a X(n-1) + b X(n-2) per bin from
  // exponentially forgotten normal equations (recursive least squares),
  // its two roots being the partials' per-hop rotations.
  //
  // Own analysis hop, nfft/8: the roots separate by delta-omega * hop on
  // the unit circle, so a longer hop conditions the 2x2 system better -
  // until the separation wraps at pi.  hop = N/8 is exactly the largest
  // hop that still resolves the widest pair a bin can hold (Hann main
  // lobe = 4 bins).  Measured against the closed form, on the Hertz.
  //
  // The spectrum ring exists so the recursion can START from a fresh
  // pair X(n), X(n - arHop) even though the FIT only advances on the
  // absolute arHop grid: fitting from stale frames costs nothing, but
  // extrapolating from them costs 0.5 dB per period of staleness.
  std::atomic<int> m_Order{1};
  inline static std::atomic<int> s_UseSchur{0};
  // Ausloeseschwelle der Wurzelkappung; 0 = eingebaute 0.9.  Das ZIEL
  // ist immer der Einheitskreis - die Schwelle entscheidet nur, WER
  // geschnappt wird.  Statisch wie s_UseSchur, weil SolveBin statisch
  // ist; fuer das Lab (eine Instanz pro Lauf) ausreichend.
  inline static std::atomic<float> s_RootClamp{0.0f};
  // Detektor-Modus (06.08.2026, Nutzer-Entwurf): > 1 = rho-Schur-Test
  // auf den ROHEN Gauss-Koeffizienten.  Bestanden (gemessen 96 % der
  // Bins bei 1.02) -> Koeffizienten unveraendert, KEIN Durand-Kerner,
  // kein Rebuild - die gefittete Huellkurve jeder Mode bleibt erhalten.
  // Durchgefallen -> warmer DK in double, dann DECKELN: |z| > rho wird
  // auf rho begrenzt statt auf 1.0 gehoben.  Wachstum wird aufs Budget
  // gestutzt, Abklingen bleibt, nichts wird verewigt.
  inline static std::atomic<float> s_DetectRho{0.0f};
  static double RootClampThresh() {
    const float t = s_RootClamp.load(std::memory_order_relaxed);

    return t > 0.0f ? (double)t : 0.9;
  }
  unsigned m_ArStride = 0;  // periods between AR frames, at the LARGEST hop
  unsigned m_ArRingLen = 0; // 2 * stride + 1
  std::vector<float> m_ArRingRe, m_ArRingIm;  // [ch][slot][NBINS_MAX]
  std::vector<uint64_t> m_ArRingPeriod;       // [ch][slot], ~0 = empty
  // Normal equations, exponentially forgotten.  double on purpose: the
  // float32 SPECTRA cost nothing (measured 0.0 dB), but float32
  // ACCUMULATION over ~14 terms costs up to 1.1 dB.
  //
  // Laid out as [ch][bin][k][l] for P and [ch][bin][k] for q, with the
  // FULL p x p stored (not just the Hermitian half) so the solve is a
  // plain dense elimination without index gymnastics.  AR_ORDER_MAX = 4:
  // order 6 was measured and collapses into overfitting.
  // Overridable at build time so an offline tool can explore orders the
  // engine would not allocate for: the p x p normal equations are sized
  // by this, so six channels cost ~44 MB at 8 and ~158 MB at 20.  The
  // lab builds its own libxorcist with -DXO_AR_ORDER_MAX=20.
#ifndef XO_AR_ORDER_MAX
#define XO_AR_ORDER_MAX 8
#endif
  static constexpr unsigned AR_ORDER_MAX = XO_AR_ORDER_MAX;
  // ---- Normalgleichungen: planares HERMITESCHES DREIECK ---------------
  // P[k][l] = sum X_k * conj(X_l) ist hermitesch - und zwar EXAKT, nicht
  // naeherungsweise: die Rekursion startet bei null, und
  // lam*hermitesch + hermitesches Aussenprodukt bleibt hermitesch.  Der
  // Realteil ist symmetrisch, der Imaginaerteil vorzeichenverkehrt, beides
  // bitgenau (Gleitkomma-Multiplikation ist kommutativ).  Gespeichert wird
  // deshalb nur k <= l: 36 statt 64 Eintraege bei Ordnung 8.
  //
  // PLANAR, getrennte Re-/Im-Ebenen, Bin als schnellster Index.  Die alte
  // Ablage [bin][k][l] mit verschachtelten Re/Im war in BEIDE Richtungen
  // unvektorisierbar: ueber l springen die Eingangswerte um NBINS_MAX,
  // ueber die Bins springt P um 128 Doubles.  Planar laeuft die innere
  // Schleife zusammenhaengend ueber alle Bins - und dasselbe Layout ist
  // auf der GPU der Unterschied zwischen koaleszierten und gestreuten
  // Zugriffen benachbarter Arbeitsobjekte.
  //
  // Dreieck und Vektorisierung widersprechen sich NICHT, solange ueber die
  // Bins vektorisiert wird: das Dreieck kuerzt die AEUSSERE Paarschleife
  // von 64 auf 36, die innere bleibt in beiden Faellen nbins lang.  (Ueber
  // l vektorisiert waere es umgekehrt - variable, kurze Zaehllaengen.)
  static constexpr unsigned AR_PAIRS
    = AR_ORDER_MAX * (AR_ORDER_MAX + 1) / 2;

  /** Index des Paares (k, l) im Dreieck, NUR fuer k <= l. */
  static constexpr unsigned PairOf(unsigned k, unsigned l) {
    return k * AR_ORDER_MAX - k * (k - 1) / 2 + (l - k);
  }

  std::vector<double> m_ArPRe, m_ArPIm; // [ch][paar][bin]
  std::vector<double> m_ArQRe, m_ArQIm; // [ch][k][bin]

  /** Ein Bin aus dem Dreieck in die DICHTE Form auffalten, wie SolveBin,
   * der Leiter-Solve und der GPU-Upload sie erwarten: P verschachtelt
   * [k*AR_ORDER_MAX + l][re,im], Q [k][re,im].  Fuer k > l wird gespiegelt
   * und der Imaginaerteil negiert - das ist die Hermitezitaet, exakt.
   *
   * Kostet 36 gestreute Lesevorgaenge je geloestem Bin, und geloest wird
   * nur ein Sechzehntel der Bins je Periode; gegen die Ersparnis in der
   * Akkumulation, die JEDE Periode ALLE Bins anfasst, ist das nichts. */
  void GatherBin(
    unsigned ch, unsigned bin, unsigned order, double *dstP,
    double *dstQ) const {
    const size_t pb = (size_t)ch * AR_PAIRS * NBINS_MAX + bin;
    const size_t qb = (size_t)ch * AR_ORDER_MAX * NBINS_MAX + bin;

    for (unsigned k = 0; k < order; k++) {
      dstQ[2 * k] = m_ArQRe[qb + (size_t)k * NBINS_MAX];
      dstQ[2 * k + 1] = m_ArQIm[qb + (size_t)k * NBINS_MAX];
      for (unsigned l = 0; l < order; l++) {
        const bool up = k <= l;
        const size_t o = pb
          + (size_t)(up ? PairOf(k, l) : PairOf(l, k)) * NBINS_MAX;
        const size_t d = 2 * ((size_t)k * AR_ORDER_MAX + l);

        dstP[d] = m_ArPRe[o];
        dstP[d + 1] = up ? m_ArPIm[o] : -m_ArPIm[o];
      }
    }
  }

  /** Alle Eintraege EINES Bins auf null - beim Hybrid-Reset und beim
   * Ordnungswechsel.  Gestreut, aber selten. */
  void ClearBin(unsigned ch, unsigned bin) {
    const size_t pb = (size_t)ch * AR_PAIRS * NBINS_MAX + bin;
    const size_t qb = (size_t)ch * AR_ORDER_MAX * NBINS_MAX + bin;

    for (unsigned i = 0; i < AR_PAIRS; i++)
      m_ArPRe[pb + (size_t)i * NBINS_MAX]
        = m_ArPIm[pb + (size_t)i * NBINS_MAX] = 0.0;
    for (unsigned k = 0; k < AR_ORDER_MAX; k++)
      m_ArQRe[qb + (size_t)k * NBINS_MAX]
        = m_ArQIm[qb + (size_t)k * NBINS_MAX] = 0.0;
  }
  std::vector<uint64_t> m_ArFitPeriod; // [ch] last accumulator update
  std::vector<unsigned> m_ArNfft;      // [ch] nfft they belong to
  // Warm-up: a single update leaves the 2x2 system exactly rank 1, so the
  // solve is meaningless until a few triples have accumulated.  This is a
  // cold-start guard, not a per-bin order gate - those were measured and
  // change nothing.
  std::vector<unsigned> m_ArFitCount;  // [ch] updates since the last reset
  static constexpr unsigned AR_WARMUP = 4;
  // Grid steps between 2x2 solves.  The coefficients come out of a
  // ~14-frame memory, so re-solving at every grid point is wasted work;
  // the STATE stays fresh every period regardless, which is what the
  // extrapolation is actually sensitive to.
  static constexpr unsigned AR_SOLVE_EVERY = 4;
  // The solved recursion, kept PER CHANNEL and reused.  The coefficients
  // come out of a 14-frame memory and only change when the accumulators
  // do, i.e. once per arStride periods - but Predict() runs every period,
  // because the shadow has to stand ready before a miss happens.  Solving
  // per period cost 2.45x the order-1 prediction; solving per grid point
  // is the same numbers for a fraction of that.
  std::vector<float> m_ArCoefRe, m_ArCoefIm; // [ch][bin][k], k = 0..p-1
  // PLANARER SPIEGEL der Koeffizienten, [ch][k][bin] statt [ch][bin][k].
  // Grund: AdvanceBins laeuft ueber die BINS, und der Zustand liegt schon
  // planar - nur die Koeffizienten lagen verschachtelt, was die Schleife
  // unvektorisierbar machte (der Uebersetzer haette je Bin sammeln
  // muessen).  Umkopieren kostet nbins*P Werte je Kanal und Periode, also
  // rund 6 % dessen, was die Schleife danach vektorisiert erledigt.
  // Die Ablage oben bleibt, wie sie ist - SolveBin und der Hybrid-Pfad
  // schreiben und lesen sie weiter per Bin.
  std::vector<float> m_ArCoefReP, m_ArCoefImP;
  // Kratzspeicher der Extrapolation, je Kanal ein Bin-Vektor.
  std::vector<float> m_AdvAccRe, m_AdvAccIm;
  // Last solved roots per bin, the WARM START for Durand-Kerner: on
  // quasi-stationary material the roots barely move between solves, so
  // seeding with them converges in 2-3 iterations instead of ~15.
  std::vector<float> m_ArRootRe, m_ArRootIm; // [ch][bin][k]
  std::vector<uint64_t> m_ArSolvedPeriod;    // [ch]
  std::vector<int> m_ArSolvedOrder;          // [ch]
  // Running state of the recursion, [k][bin] with k = 0 the newest.
  std::vector<float> m_ArStRe, m_ArStIm;

  // 1 - 1/lambda ~ 14 fit frames of memory.  Measured as a plateau: 0.86
  // to 0.96 differ by less than 0.1 dB, because the 16384-sample history
  // ring only holds 24 frames at hop 512 and the truncation dominates.
  // Exponential forgetting beat a rectangular 14-frame block fit by
  // 0.4-0.5 dB, including on attacks - the concern that it would lag
  // there did not materialise.
  // Effective memory 1/(1-lambda), in FIT FRAMES, and it has to grow with
  // the order: order 2 is best around 14 frames, order 4 wants 20 (in the
  // offline prototype it still improved out to 40, but GO's 16384-sample
  // history ring only holds 24 frames at hop 512, so 20 is the ceiling
  // here).  4p + 6 reproduces 0.9286 at order 2 exactly and reaches
  // 0.9545 at order 4.
  static double ArLambda(unsigned order) {
    return 1.0 - 1.0 / (double)(4 * order + 6);
  }
  std::atomic<uint64_t> m_SkipCount{0};

  static std::atomic<uint64_t> s_ForcedCount;   // bails delivered
  static std::atomic<uint64_t> s_SkipCount;     // shadow updates skipped (busy)
  static std::atomic<uint64_t> s_LateRealCount; // real mix arrived after wait
  static std::atomic<int64_t> s_MaxWaitUs;      // longest delivery wait

  // Safety cap: how often it fires, and how hard it is set.  Both are
  // diagnostics for the 06.08.2026 suspicion that the cap - measured
  // over the whole 32 ms shadow but audible only over the played 128
  // samples - is what breaks the higher orders.  XO_SAFETY_CAP raises
  // the factor; 0 disables the cap outright.
  // Newest history period whose content came from the predictor.  The
  // 32-bit m_RecentPredMask cannot serve here: one AR tuple spans
  // P*arStride + nfft/buf periods, which is 48 at order 4 and 4k.
  std::atomic<uint64_t> m_LastConcealedPeriod{0};
  std::atomic<int> m_GateFit{0};
  std::atomic<int> m_StaleWindow{0};
  // Wie oft faellt der AR-Pfad auf den Ordnung-1-Rotator zurueck, weil
  // eine Stuetzstelle im Spektrenring fehlt?  Der Ring wird NUR beim
  // Schattenbau beschrieben, und der laeuft waehrend einer Lueckenkette
  // nicht - also reisst dort ein Loch, das spaetere Gitterpunkte trifft.
  std::atomic<int> m_Backfill{0};
  std::atomic<int> m_FitWindow{0};
  std::atomic<int> m_SolveEvery{0};
  std::atomic<int> m_BurgEvery{0};
  std::atomic<int> m_Replica{0};
  std::atomic<int> m_FadeInRing{1};   // Vorgabe AN - siehe Push()
  std::atomic<int> m_CapSpan{0};      // 0 = ganzer Schatten
  std::atomic<int> m_ReentryOverride{0};
  std::atomic<int> m_AnchorTau{0};   // Samples; 0 = rate/3000
  // Eigenes Flag, weil Pass() und Push() in verschiedenen Reihenfolgen
  // gerufen werden: die Engine pusht im Mix-Epilog und blendet erst
  // spaeter beim Ausliefern, das Lab umgekehrt.  Zwei Flags, jedes von
  // genau einem Verbraucher geleert, sind gegen beide Reihenfolgen
  // immun.
  bool m_RingBlendPending = false;
  // Absorb(): Analyse ohne Synthese - siehe die Sollbruchstelle in
  // ComputeChannelShadow.  Nur vom (sequenziellen) Lab benutzt; die
  // Engine setzt es nie.
  bool m_AnalyzeOnly = false;

  // ---- GPU (v1: nur der Solve) ---------------------------------------
  std::atomic<int> m_Gpu{[] {
    const char *e = getenv("XO_GPU");

    return e ? atoi(e) : 0;
  }()};
  // Solve-Film: Mitschrift dessen, was der Kernel je Bin GETAN hat.
  // Nur Labor - GO setzt das Feld nie, und der Ring wird erst beim
  // ersten Bild belegt.
  std::atomic<int> m_SolveTap{0};
#ifndef XO_NO_OPENCL
  bool m_GpuBad = false;      // einmal gescheitert -> CPU bis zum Ende
  unsigned m_GpuOrder = 0, m_GpuNbins = 0;
  std::vector<cl_mem> m_dP, m_dQ, m_dCoef, m_dRoots, m_dTap; // [ch]
  std::vector<double> m_GpuPack;
  std::vector<float> m_GpuCoefDl;
  // Laufendes Feld von Kanal 0.  Ein Kernelstart deckt nur EINEN
  // Streifen ab; das Bild entsteht erst, wenn die Streifen den Bereich
  // einmal gekachelt haben - bis dahin steht in den noch nicht
  // besuchten Bins -1 ("noch nicht gerechnet"), keine erfundene Null.
  std::vector<float> m_TapDl, m_TapZ, m_TapPath, m_TapRoots;

  void GpuRelease() {
    XoCl &cl = XoCl::get();

    if (!cl.ok)
      return;
    for (auto &v : {&m_dP, &m_dQ, &m_dCoef, &m_dRoots, &m_dTap})
      for (cl_mem m : *v)
        if (m)
          cl.ReleaseMem(m);
    m_dP.clear();
    m_dQ.clear();
    m_dCoef.clear();
    m_dRoots.clear();
    m_dTap.clear();
    m_GpuOrder = m_GpuNbins = 0;
  }

  // Loest den Streifen [b0,b1) von Kanal ch auf dem Geraet.  true =
  // erledigt, false = CPU soll uebernehmen (und m_GpuBad haelt es dann
  // dauerhaft dort - die CPU-Akkumulatoren sind ja weitergelaufen).
  bool GpuSolveRange(
    unsigned ch, unsigned b0, unsigned b1, unsigned order, unsigned nbins,
    bool resetRoots) {
    XoCl &cl = XoCl::get();

    if (!cl.ok) {
      m_GpuBad = true;
      return false;
    }

    cl_int err = CL_SUCCESS;
    const bool tap = m_SolveTap.load(std::memory_order_relaxed) != 0;

    if (m_GpuOrder != order || m_GpuNbins != nbins) {
      GpuRelease();
      m_dP.assign(m_NChannels, nullptr);
      m_dQ.assign(m_NChannels, nullptr);
      m_dCoef.assign(m_NChannels, nullptr);
      m_dRoots.assign(m_NChannels, nullptr);
      m_dTap.assign(m_NChannels, nullptr);
      for (unsigned c = 0; c < m_NChannels && err == CL_SUCCESS; c++) {
        m_dP[c] = cl.CreateBuffer(
          cl.ctx, CL_MEM_READ_ONLY,
          (size_t)nbins * order * order * 2 * sizeof(double), nullptr, &err);
        if (err == CL_SUCCESS)
          m_dQ[c] = cl.CreateBuffer(
            cl.ctx, CL_MEM_READ_ONLY,
            (size_t)nbins * order * 2 * sizeof(double), nullptr, &err);
        if (err == CL_SUCCESS)
          m_dCoef[c] = cl.CreateBuffer(
            cl.ctx, CL_MEM_WRITE_ONLY,
            (size_t)nbins * order * 2 * sizeof(float), nullptr, &err);
        if (err == CL_SUCCESS)
          m_dRoots[c] = cl.CreateBuffer(
            cl.ctx, CL_MEM_READ_WRITE,
            (size_t)nbins * order * 2 * sizeof(float), nullptr, &err);
        // Immer belegt, auch bei ausgeschaltetem Film: 16 KB je Kanal
        // gegen eine Sonderbehandlung im Neubau-Pfad ist kein Handel.
        // Der Kernel bekommt bei ausgeschaltetem Film NULL uebergeben
        // und fasst den Puffer dann nicht an.
        if (err == CL_SUCCESS)
          m_dTap[c] = cl.CreateBuffer(
            cl.ctx, CL_MEM_WRITE_ONLY,
            (size_t)nbins * TapStride(order) * sizeof(float), nullptr, &err);
      }
      if (err != CL_SUCCESS) {
        m_GpuBad = true;
        return false;
      }
      m_GpuOrder = order;
      m_GpuNbins = nbins;
      resetRoots = true;
    }
    if (resetRoots) {
      const float z = 0.0f;

      err = cl.Fill(
        cl.q, m_dRoots[ch], &z, sizeof(z), 0,
        (size_t)nbins * order * 2 * sizeof(float), 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        m_GpuBad = true;
        return false;
      }
    }

    // Streifen packen: Engine-Layout (Stride AR_ORDER_MAX) -> dicht.
    const unsigned n = b1 - b0;
    // Das Geraet bekommt weiter die DICHTE Form - der Kernel bleibt damit
    // Wort fuer Wort derselbe.  Das planare Dreieck waere fuer die GPU
    // sogar besser (benachbarte Arbeitsobjekte laesen dann benachbarte
    // Adressen statt 128 Doubles auseinander), aber das ist ein eigener
    // Schritt in beiden Texten und nicht dieser hier.
    m_GpuPack.resize((size_t)n * order * order * 2 + (size_t)n * order * 2);

    double *tP = m_GpuPack.data();
    double *tQ = tP + (size_t)n * order * order * 2;
    double gP[AR_ORDER_MAX * AR_ORDER_MAX * 2], gQ[AR_ORDER_MAX * 2];

    for (unsigned bin = 0; bin < n; bin++) {
      double *dP = tP + (size_t)bin * order * order * 2;
      double *dQ = tQ + (size_t)bin * order * 2;

      GatherBin(ch, b0 + bin, order, gP, gQ);
      for (unsigned k = 0; k < order; k++) {
        dQ[2 * k] = gQ[2 * k];
        dQ[2 * k + 1] = gQ[2 * k + 1];
        for (unsigned l = 0; l < order; l++) {
          dP[2 * (k * order + l)] = gP[2 * (k * AR_ORDER_MAX + l)];
          dP[2 * (k * order + l) + 1] = gP[2 * (k * AR_ORDER_MAX + l) + 1];
        }
      }
    }
    err = cl.Write(
      cl.q, m_dP[ch], CL_FALSE, (size_t)b0 * order * order * 2 * sizeof(double),
      (size_t)n * order * order * 2 * sizeof(double), tP, 0, nullptr, nullptr);
    if (err == CL_SUCCESS)
      err = cl.Write(
        cl.q, m_dQ[ch], CL_FALSE, (size_t)b0 * order * 2 * sizeof(double),
        (size_t)n * order * 2 * sizeof(double), tQ, 0, nullptr, nullptr);

    const int iN = (int)nbins, iO = (int)order, iF = (int)b0, iL = (int)b1;
    const float clampT = (float)RootClampThresh();

    if (err == CL_SUCCESS) {
      cl.SetArg(cl.kSolve, 0, sizeof(cl_mem), &m_dP[ch]);
      cl.SetArg(cl.kSolve, 1, sizeof(cl_mem), &m_dQ[ch]);
      cl.SetArg(cl.kSolve, 2, sizeof(cl_mem), &m_dCoef[ch]);
      cl.SetArg(cl.kSolve, 3, sizeof(cl_mem), &m_dRoots[ch]);
      cl.SetArg(cl.kSolve, 4, sizeof(int), &iN);
      cl.SetArg(cl.kSolve, 5, sizeof(int), &iO);
      cl.SetArg(cl.kSolve, 6, sizeof(int), &iF);
      cl.SetArg(cl.kSolve, 7, sizeof(int), &iL);
      err = cl.SetArg(cl.kSolve, 8, sizeof(float), &clampT);
      if (err == CL_SUCCESS) {
        const float detRho = s_DetectRho.load(std::memory_order_relaxed);

        err = cl.SetArg(cl.kSolve, 9, sizeof(float), &detRho);
      }
      if (err == CL_SUCCESS) {
        // Ein NULL-Pufferobjekt ist die vorgesehene Art, einem
        // __global-Zeiger nichts zu geben - der Kernel prueft darauf.
        // Nur Kanal 0 wird gelesen, also soll auch nur er schreiben.
        const cl_mem tapArg = (tap && ch == 0) ? m_dTap[ch] : nullptr;

        err = cl.SetArg(cl.kSolve, 10, sizeof(cl_mem), &tapArg);
      }
    }

    size_t gws = n;

    if (err == CL_SUCCESS)
      err = cl.Launch(cl.q, cl.kSolve, 1, nullptr, &gws, nullptr, 0, nullptr,
                      nullptr);

    m_GpuCoefDl.resize((size_t)n * order * 2);
    if (err == CL_SUCCESS)
      err = cl.Read(
        cl.q, m_dCoef[ch], CL_TRUE, (size_t)b0 * order * 2 * sizeof(float),
        (size_t)n * order * 2 * sizeof(float), m_GpuCoefDl.data(), 0, nullptr,
        nullptr);
    if (err != CL_SUCCESS) {
      m_GpuBad = true;
      return false;
    }

    // Solve-Film abholen.  Blockierend wie der Koeffizienten-Rueckweg
    // direkt darueber, also KEIN zusaetzlicher Stillstand der
    // Warteschlange - aber ein zweiter Transfer, und der kostet Zeit.
    if (tap && ch == 0) {
      const size_t stride = TapStride(order);

      m_TapDl.resize((size_t)n * stride);
      err = cl.Read(
        cl.q, m_dTap[ch], CL_TRUE, (size_t)b0 * stride * sizeof(float),
        (size_t)n * stride * sizeof(float), m_TapDl.data(), 0, nullptr,
        nullptr);
      if (err != CL_SUCCESS) {
        m_GpuBad = true;
        return false;
      }
      if (m_TapZ.size() != (size_t)nbins
          || m_TapRoots.size() != (size_t)nbins * order * 2) {
        m_TapZ.assign(nbins, -1.0f);
        m_TapPath.assign(nbins, -1.0f);
        m_TapRoots.assign((size_t)nbins * order * 2, NAN);
      }
      for (unsigned bin = 0; bin < n; bin++) {
        const float *row = m_TapDl.data() + (size_t)bin * stride;

        m_TapZ[b0 + bin] = row[0];
        m_TapPath[b0 + bin] = row[1];
        memcpy(
          &m_TapRoots[(size_t)(b0 + bin) * order * 2], row + 2,
          (size_t)order * 2 * sizeof(float));
      }
      // Der Streifen, der den Bereich schliesst, gibt das Bild frei.
      // b1 == nbins trifft genau einmal je Zyklus zu (der letzte
      // Streifen rundet auf nbins auf) und immer beim Vollsolve.
      if (b1 == nbins)
        TapPush(
          m_TapZ.data(), m_TapPath.data(), m_TapRoots.data(), nbins, order,
          RoundPeriod() * m_SamplesPerBuffer);
    }

    // dicht -> Engine-Layout der Koeffizienten
    float *cRe = m_ArCoefRe.data() + (size_t)ch * NBINS_MAX * AR_ORDER_MAX;
    float *cIm = m_ArCoefIm.data() + (size_t)ch * NBINS_MAX * AR_ORDER_MAX;

    for (unsigned bin = 0; bin < n; bin++)
      for (unsigned k = 0; k < order; k++) {
        cRe[(size_t)(b0 + bin) * AR_ORDER_MAX + k]
          = m_GpuCoefDl[(size_t)bin * order * 2 + 2 * k];
        cIm[(size_t)(b0 + bin) * AR_ORDER_MAX + k]
          = m_GpuCoefDl[(size_t)bin * order * 2 + 2 * k + 1];
      }
    return true;
  }
#endif // XO_NO_OPENCL
  // Wirksame Fade-Laenge, zur Laufzeit verkuerzbar.  m_ReentryFade
  // dimensioniert die Puffer und bleibt konstant; hiermit laesst sich
  // eine KUERZERE Rampe waehlen, ohne neu zu allozieren.  Der Nutzer
  // will eine Messreihe bis hinab zu 0,2 ms fahren.
  std::atomic<unsigned> m_ReentryUse{0};   // 0 = m_ReentryFade

  unsigned ReentryLen() const {
    const unsigned u = m_ReentryUse.load(std::memory_order_relaxed);

    return (u == 0 || u > m_ReentryFade) ? m_ReentryFade : u;
  }
  void SetReentryFade(int frames) {
    m_ReentryUse.store(
      frames > 0 ? (unsigned)frames : 0u, std::memory_order_relaxed);
  }
  static float ReentryCurve(unsigned i, unsigned n) {
    return 0.5f + 0.5f * cosf((float)M_PI * (float)i / (float)n);
  }
  inline static std::atomic<uint64_t> s_Backfills{0};
  // rho-Schur-DETEKTOR, nur Zaehlung (XO_COUNT_RHO=<rho>): wie viele
  // Bins wuerden den Wachstumsbudget-Test verfehlen?  Die eine Zahl,
  // an der haengt, ob der Lazy-DK-Umbau lohnt.
  inline static std::atomic<uint64_t> s_PatchCount{0};
  // Kleiner Ereignisring, damit das Lab ZEIGEN kann, wo gepatcht wurde -
  // ohne Anzeige ist ein Feature nicht beurteilbar.
  static constexpr unsigned PATCH_LOG = 512;
  inline static std::atomic<uint64_t> s_PatchLogN{0};
  inline static uint64_t s_PatchLog[PATCH_LOG][3] = {{0}};
  // Hybrid-Episoden fuers Spektrogramm: (startSample, endSample,
  // band | ordnung<<16 | nfft<<32) - eine Episode = ein Band, das die
  // kurze Kette gespielt hat, Kanal 0.  Ring wie das Patch-Log.
  // 16384: mit 'alle Baender' erzeugt JEDES Ereignis 256 Episoden -
  // bei 4096 fielen die aeltesten aus dem Ring, und die ersten vier
  // Einsaetze der D-Dur-Datei hatten keine Marks (Nutzer, 08.08.).
  static constexpr unsigned HYB_LOG = 16384;
  inline static std::atomic<uint64_t> s_HybLogN{0};
  inline static uint64_t s_HybLog[HYB_LOG][3] = {{0}};

  // ---- Solve-Film ------------------------------------------------------
  // Ein Bild je vollstaendigem Bin-Durchlauf von Kanal 0.  Anders als die
  // Ereignisringe darueber ist ein Bild kein Tripel, sondern nbins Werte;
  // deshalb HEAP und erst bei Bedarf belegt (bei 4k und 512 Bildern sind
  // es 16 MB, die kein Wirt bezahlen soll, der nie zuschaut).  Der Mutex
  // ist billig, weil er nur bei eingeschaltetem Film ueberhaupt angefasst
  // wird - und ein halb ueberschriebenes Bild waere als Bild wertlos.
  // Eine Bin-Zeile des Geraets: [max|z|, Zweig, Re z0, Im z0, ...].
  static constexpr unsigned TapStride(unsigned order) {
    return 2 + 2 * order;
  }
  // Nicht die BILDZAHL wird gedeckelt, sondern der Speicher: eine Zeile
  // waechst mit der Ordnung (bei 4k/O20 ist ein Bild 21x so gross wie
  // bei O1), und ein fester Ringlaenge hiesse, dass ausgerechnet die
  // teuren Laeufe am meisten belegen.
  static constexpr size_t TAP_BUDGET = 48u << 20;   // 48 MB
  static constexpr unsigned TAP_FRAMES_MIN = 16, TAP_FRAMES_MAX = 512;
  inline static std::mutex s_TapMx;
  inline static std::vector<float> s_TapZ, s_TapPath, s_TapRoots;
  inline static std::vector<uint64_t> s_TapPos;
  inline static unsigned s_TapNbins = 0, s_TapOrder = 0, s_TapFrames = 0;
  inline static uint64_t s_TapN = 0;   // insgesamt geschriebene Bilder

  static void TapPush(
    const float *z, const float *path, const float *roots, unsigned nbins,
    unsigned order, uint64_t pos) {
    std::lock_guard<std::mutex> g(s_TapMx);

    // Moduswechsel: Bilder zweier Geometrien liessen sich nicht in einer
    // Anzeige zeigen, also faengt der Ring neu an.
    if (s_TapNbins != nbins || s_TapOrder != order || s_TapZ.empty()) {
      const size_t perFrame
        = (size_t)nbins * (2 + (size_t)order * 2) * sizeof(float);
      unsigned nf = (unsigned)(TAP_BUDGET / (perFrame ? perFrame : 1));

      if (nf < TAP_FRAMES_MIN)
        nf = TAP_FRAMES_MIN;
      if (nf > TAP_FRAMES_MAX)
        nf = TAP_FRAMES_MAX;
      s_TapFrames = nf;
      s_TapZ.assign((size_t)nf * nbins, -1.0f);
      s_TapPath.assign((size_t)nf * nbins, -1.0f);
      s_TapRoots.assign((size_t)nf * nbins * order * 2, NAN);
      s_TapPos.assign(nf, 0);
      s_TapNbins = nbins;
      s_TapOrder = order;
      s_TapN = 0;
    }

    const size_t slot = (size_t)(s_TapN % s_TapFrames);

    memcpy(&s_TapZ[slot * nbins], z, (size_t)nbins * sizeof(float));
    memcpy(&s_TapPath[slot * nbins], path, (size_t)nbins * sizeof(float));
    memcpy(
      &s_TapRoots[slot * nbins * order * 2], roots,
      (size_t)nbins * order * 2 * sizeof(float));
    s_TapPos[slot] = pos;
    s_TapN++;
  }

  inline static std::atomic<uint64_t> s_RhoFails{0};
  inline static std::atomic<uint64_t> s_RhoTests{0};
  // Lief das Geraet wirklich?  Ein gesetztes Haekchen beweist nichts:
  // Rechteckfenster (replica) und Schur bleiben CPU, ein gescheitertes
  // Geraet faellt dauerhaft zurueck - und bis zum 07.08. sperrte auch
  // der Detektor-Modus die GPU stillschweigend aus.  Wer die
  // Einstellung anzeigt, soll die WIRKUNG anzeigen koennen.
  // Wirkt der Hybrid-Haken ueberhaupt?  Er faellt still aus, wenn der
  // Basisrahmen 1k IST oder die Puffergroesse 256 nicht teilt (das Lab
  // koppelt den Puffer an die Luecken-Breite!).  Nach dem GPU-Vorfall
  // vom 07.08. gilt: Wirkung anzeigen, nicht Zustand.
  inline static std::atomic<uint64_t> s_HybOnPeriods{0};
  inline static std::atomic<uint64_t> s_HybOffPeriods{0};
  inline static std::atomic<uint64_t> s_HybBandPeriods{0};
  // ZUENDUNGEN des Transientendetektors, je Kanal und Ereignis.  Auf
  // gehaltenem Klang gehoert hier eine Null hin; gemessen 10.08.2026
  // steht dort bei 6000 Luecken/min eine 25 - der Detektor zuendet auf
  // den eigenen Wiedereinstiegen.  Deshalb sichtbar statt vermutet.
  inline static std::atomic<uint64_t> s_HybOnsets{0};
  // JE KANAL, damit ein Wirt sagen kann WELCHER Kanal zuendet und wie
  // dicht dessen Spektrum ueberhaupt besetzt ist.  Der summierende
  // Gesamtzaehler verbirgt genau das - und ein Stereo-Downmix, wie ihn
  // ein Rekorder schreibt, kann es prinzipiell nicht zeigen: Summieren
  // fuellt Baender, die im Einzelkanal leer sind.
  static constexpr unsigned DET_CH_MAX = 16;
  inline static std::atomic<uint64_t> s_OnsetPerCh[DET_CH_MAX]{};
  inline static std::atomic<unsigned> s_OccPerCh[DET_CH_MAX]{};
  inline static std::atomic<uint64_t> s_HybOnsetsTent{0};
  // Zeitstempel der letzten Zuendungen (Abtastwert-Zaehler), damit ein
  // Wirt sie in seinen Verlauf ZEICHNEN kann statt nur zu zaehlen.
  static constexpr unsigned ONSET_LOG = 256;
  inline static std::atomic<uint64_t> s_OnsetLog[ONSET_LOG]{};
  inline static std::atomic<uint32_t> s_OnsetSeq{0};
  inline static std::atomic<uint64_t> s_GpuRanges{0};
  inline static std::atomic<uint64_t> s_CpuRanges{0};

  // true = alle Wurzeln strikt innerhalb von rho (Schur-Abstieg als
  // reiner TEST: fruehester Abbruch bei der ersten Verletzung, kein
  // Kappen, kein Wiederaufstieg).  O(p^2), endlich, eliminationsartig.
  static bool RhoTest(
    const double *ar, const double *ai, unsigned order, double rho) {
    double ur[AR_ORDER_MAX + 1], ui[AR_ORDER_MAX + 1];
    double nr[AR_ORDER_MAX + 1], ni[AR_ORDER_MAX + 1];
    double f = 1.0;

    ur[0] = 1.0;
    ui[0] = 0.0;
    for (unsigned i = 1; i <= order; i++) {
      f /= rho;
      ur[i] = -ar[i - 1] * f;
      ui[i] = -ai[i - 1] * f;
    }
    for (unsigned l = order; l >= 1; l--) {
      const double kr = ur[l], ki = ui[l];
      const double km = kr * kr + ki * ki;

      if (km >= 1.0)
        return false; // Box-Bedingung verletzt: Wurzel ausserhalb rho
      const double den = 1.0 - km;

      for (unsigned i = 1; i < l; i++) {
        nr[i] = (ur[i] - (kr * ur[l - i] + ki * ui[l - i])) / den;
        ni[i] = (ui[i] - (ki * ur[l - i] - kr * ui[l - i])) / den;
      }
      for (unsigned i = 1; i < l; i++) {
        ur[i] = nr[i];
        ui[i] = ni[i];
      }
    }
    return true;
  }
  inline static std::atomic<uint64_t> s_ArDrops{0};
  inline static std::atomic<uint64_t> s_ArTries{0};
  inline static std::atomic<uint64_t> s_GateSkips{0};
  inline static std::atomic<uint64_t> s_GateTests{0};
  inline static std::atomic<uint64_t> s_CapHits{0};
  inline static std::atomic<uint64_t> s_CapCalls{0};
  // Rennen-Messung (2026-08-09).  Beide Zaehler sind fuer den Wirt
  // gedacht, der die Vorhersage NEBENLAEUFIG rechnet - das Labor ist
  // einthreadig und muss hier zwingend 0/0 liefern.
  //   s_StaleDiscards: PredictEnd hat eine fertige Runde WEGGEWORFEN,
  //     weil der Ring waehrenddessen weitergeschrieben wurde.  Genau
  //     das ist "der Rueckruf pfuscht in die Schattenberechnung".
  //   s_LateSlices: Conceal hat eine Scheibe > 0 gespielt, OHNE dass die
  //     Vorperiode selbst verdeckt war - also nicht die legitime
  //     Fehlschlagkette, sondern ein Anker, der zurueckgeblieben ist.
  //     Das ist die Groesse, die wirklich weh tut (Nahtreparatur sitzt
  //     nur auf pShadow[0]).
  inline static std::atomic<uint64_t> s_StaleDiscards{0};
  inline static std::atomic<uint64_t> s_LateSlices{0};
  // Ursachen-Zuordnung der spaeten Scheiben (11.08.2026, reine Messung):
  // die Hypothese ist, dass der Anker zurueckblieb, WEIL kurz zuvor eine
  // Oeffnung am Besetzt-Waechter scheiterte (m_ComputeBusy in Predict).
  // s_LateAfterSkip zaehlt spaete Scheiben, deren letzter Besetzt-Fall
  // hoechstens zwei Perioden zurueckliegt.  Laufen s_LateSlices und
  // s_LateAfterSkip im Gleichschritt, ist die Kette bewiesen; klafft es,
  // gibt es (auch) eine andere Ursache.  s_LateSliceMax haelt die
  // groesste gespielte Scheibennummer fest - Scheibe 1 heisst "eine
  // Runde verpasst", mehr heisst Serienausfall der Oeffnungen.
  static constexpr uint64_t NO_SKIP_YET = ~0ull;
  std::atomic<uint64_t> m_LastSkipPeriod{NO_SKIP_YET};
  inline static std::atomic<uint64_t> s_LateAfterSkip{0};
  inline static std::atomic<uint64_t> s_LateSliceMax{0};
  // Per instance so the lab can A/B it inside one process; XO_SAFETY_CAP
  // only supplies the starting value.  0 = cap off.
  std::atomic<float> m_SafetyCap{[] {
    const char *e = getenv("XO_SAFETY_CAP");

    return e ? (float)atof(e) : 1.2f;
  }()};

  static constexpr unsigned NFFT_MAX = 8192;
  static constexpr unsigned NBINS_MAX = NFFT_MAX / 2 + 1;
  // Padded so that every channel's slice starts 64-byte aligned.
  static constexpr unsigned FREQ_STRIDE = (NBINS_MAX + 15) / 16 * 16;
  // 4 * NFFT_MAX = 32768 samples (683 ms), doubled from 2x when the ring
  // turned out to be what capped the higher orders: it buys order 6 its
  // +2.9 dB and order 8 its +4.2 on a dense chord, for 1.6 MB at six
  // channels.  The real chord saturates here (49152 adds 0.7 dB, 98304
  // nothing); only dense CLUSTERS keep improving beyond it.
  static constexpr unsigned HIST = 4 * NFFT_MAX; // per-channel history kept

  // One prepared analysis/synthesis kit per selectable NFFT.
  struct PvKit {
    unsigned nfft;
    // Distance between the two analysis frames.  Set to one audio period
    // whenever that divides the synthesis hop, so the frame computed this
    // period IS the older frame the next period needs - one forward FFT
    // instead of two.  Measured better in quality as well: a 21 ms
    // baseline averages over beating that a 2.7 ms one resolves.
    unsigned anaHop;
    std::vector<float> win;
    // Reciprocal of the overlap-add window-square sum.  It depends only on
    // the window and the frame grid, never on the data, so it is built
    // once here instead of being accumulated (and divided by) on every
    // prediction - that was half of the OLA memory traffic.
    std::vector<float> olaRecip;
    fftwf_plan fwd, inv;
  };
  PvKit m_Pv[4];
  // Modus -> Kit-Index.  1k haengt hinten an, damit die gespeicherten
  // GO-Werte 1/2/3 unveraendert bleiben.
  static bool ValidMode(int mode) {
    return (mode >= XORCIST_PV_2K && mode <= XORCIST_PV_8K)
      || mode == XORCIST_PV_1K;
  }
  static unsigned KitIndex(int mode) {
    return mode == XORCIST_PV_1K ? 3u : (unsigned)(mode - XORCIST_PV_2K);
  }

  // Entry-bridge fit: the bridge spans ~2 ms, so the model only has to
  // describe the LOCAL waveform - 8 samples per coefficient is ample for
  // Burg (which is designed for short records), and a shorter window
  // tracks the current waveform more closely than one averaging 21 ms.
  // The fit cost is O(order * window), so this is also 4x cheaper than
  // the 1024 it replaced.
  static constexpr unsigned BURG_ORDER = 32;
  static constexpr unsigned BURG_WINDOW = 512;

  // Bounds of the per-bin magnitude trend (see ComputeChannelShadow).
  // Per analysis hop, so 1.10 is +0.83 dB / 21 ms = 39 dB/s - faster than
  // any swell pedal, while the compounded growth over one shadow stays
  // below +4.1 dB.
  static constexpr float TREND_MIN = 0.5f;
  static constexpr float TREND_MAX = 1.10f;


  const unsigned m_NChannels;
  const unsigned m_SamplesPerBuffer;
  const unsigned m_SampleRate;
  const unsigned m_ReentryFade; // frames of shadow->real crossfade (~3 ms)
  // Entry anchor corrector: the seam jump (shadow[0] minus Burg's one-step
  // prediction) is subtracted as a decaying exponential with tau ~ 1/3 ms.
  // Replaced the 2 ms Burg-recursion crossfade, whose order-starved
  // free-run was itself the dominant seam error on aliquot-rich material.
  const unsigned m_AnchorLen;  // frames to apply the correction (10 tau)
  const double m_AnchorDecay;  // exp(-1/tau), applied as one multiply/frame
  // On a dense complex spectrum the freeze cannot identify dominant
  // phases per bin and leaves the true waveform within samples - there a
  // ~2 ms Burg free-run REPLACES the seam content instead (weighted by
  // the frame-similarity discriminator, see ComputeChannelShadow).
  const unsigned m_BridgeLen;  // frames of the weighted Burg bridge

  // The shadow covers several periods, and its published anchor (the
  // period index of the history it was fitted on) tells the delivery
  // which slice belongs to the period being concealed.  This makes the
  // prediction robust against WHEN it was computed - late, early or
  // skipped for a period under load all resolve to "use a further slice"
  // instead of a phase error - and lets the gap task run only every other
  // period and in the idle tail after the mix.
  // Slices ONE analysis covers.  Sized to span the degradation
  // envelope's hold phase, so a whole short outage is bridged by a
  // single coherent extrapolation instead of a chain of re-analyses.
  // Nearly free: the frame count is (shadowLen + nfft)/hop, and nfft
  // dominates - covering 32 ms instead of 13 costs ONE extra iFFT.
  const unsigned m_HorizonPeriods;
  const unsigned m_ShadowLen;      // m_HorizonPeriods * buf + m_ReentryFade

  // Per-channel history as a MIRRORED ring: each channel owns 2*HIST
  // floats, every sample is written at w and w+HIST, so any window of up
  // to HIST samples ending at the write position is contiguous and all
  // readers stay linear.  (The previous "shift everything down" layout
  // memmoved 128 KB per period - cheap in microseconds, but it evicted
  // that much L2 every 2.7 ms, which a bandwidth-bound 24k-voice render
  // feels.)
  std::vector<float> m_History;
  // Atomar, weil Push() von ZWEI Threads kommt: dem Mix-Epilog (Schnappschuss)
  // und dem Audio-Rueckruf (Selbstfuetterung in Conceal).  Relaxed reicht -
  // die Datenabhaengigkeit klaert der eingefrorene Kopf unten, nicht diese
  // Zelle.
  std::atomic<unsigned> m_HistWrite; // next write position within [0, HIST)
  std::atomic<unsigned> m_HistFill;

  // ---- eingefrorener Schreibkopf fuer die Dauer einer Runde --------------
  // HistWindow liest RELATIV ZUM SCHREIBKOPF.  Solange der live gelesen
  // wurde, verschob ein Push mitten in der Runde das Fenster unter den
  // spaeteren Kanaelen weg - eine verdorbene Analyse mit gueltigem
  // Ankeretikett.  PredictEnd hat solche Runden deshalb WEGGEWORFEN, was
  // den Anker zurueckfallen liess und die naechste Verdeckung auf eine
  // Scheibe > 0 schickte, wo keine Nahtreparatur sitzt (im GO gemessen,
  // 10.08.2026).
  //
  // Jetzt merkt sich die Runde ihren Kopf und liest bis zum Ende daraus.
  // Das ist gefahrlos, weil der Ring GESPIEGELT ist: ein gleichzeitiger
  // Push schreibt bei head..head+buf und head+HIST..head+HIST+buf, das
  // eingefrorene Fenster endet bei head+HIST-1 - die beiden beruehren sich
  // nicht einmal.  Bis der Schreiber das Fensterende wirklich einholt,
  // muesste er HIST-len Abtastwerte weit laufen (>= 128 Perioden); eine
  // Runde dauert weniger als eine.
  //
  // Eingefroren wird BEIDES: der Schreibkopf UND die Periodennummer.  Der
  // Kopf allein reicht nicht - jeder Zustandsring in diesem Modul wird mit
  // nowP etikettiert (AR-Streifen nowP % cycle, Spektren-Ring, Hybrid-
  // Episoden, das 9er-Fach des 4k-Detektors).  Laesse man nowP live, saehe
  // Kanal 0 die Periode P und Kanal 1 nach einem zwischenzeitlichen Push
  // P+1: zwei Kanaele, die dasselbe Fenster lesen, es aber verschieden
  // beschriften.  Genau das hat die alte Verwerfung mitverdeckt, und ohne
  // sie kostet es 47 dB (gemessen 10.08.2026, erster Anlauf dieses Fixes).
  unsigned m_RoundHead = 0;
  uint32_t m_RoundMask = 0;
  std::atomic<bool> m_RoundFrozen{false};

  /** Verdeckungsmaske, wie sie zur Runde gehoert.
   *
   * Push() SCHIEBT diese Maske, und Push() kommt von zwei Threads: aus dem
   * Mix-Epilog und aus Conceal auf dem Audio-Thread.  Live gelesen konnte
   * ein Spleiss, den das Veto sperren muss, waehrend der Runde aus dessen
   * Drei-Bit-Fenster herauswandern - die Zuendung wurde dann durchgelassen,
   * obwohl sie unmittelbar hinter einem Wiedereinstieg sass.  Dasselbe
   * Muster wie beim Schreibkopf und beim Periodenetikett, nur an der
   * Stelle, die ueber Zuenden oder Nichtzuenden entscheidet.  Im
   * einthreadigen Labor gibt es das nicht. */
  uint32_t RoundMask() const {
    return m_RoundFrozen.load(std::memory_order_acquire)
      ? m_RoundMask
      : m_RecentPredMask.load(std::memory_order_relaxed);
  }

  /** Periodennummer, unter der die laufende Runde alles etikettiert. */
  uint64_t RoundPeriod() const {
    return m_RoundFrozen.load(std::memory_order_acquire)
      ? m_PendingPeriod
      : m_HistPeriod.load(std::memory_order_relaxed);
  }

  // Nach dem Stagen eines Analysefensters (Ende bei winEndAbs, Laenge
  // len, bereits gefenstert in dst) die Patch-Ueberlappung einblenden.
  void OverlayPatch(
    unsigned ch, uint64_t winEndAbs, unsigned len, const float *win,
    float *dst) const {
    const uint64_t ps = m_PatchStart[ch], pe = m_PatchEnd[ch];

    if (pe <= ps || winEndAbs < len)
      return;

    const uint64_t ws = winEndAbs - len;
    const uint64_t lo = ws > ps ? ws : ps;
    const uint64_t hi = winEndAbs < pe ? winEndAbs : pe;

    if (lo >= hi)
      return;

    const float *pb = m_PatchBuf.data() + (size_t)ch * PATCH_MAX;

    for (uint64_t a = lo; a < hi; a++) {
      const unsigned i = (unsigned)(a - ws);

      dst[i] = pb[a - ps] * win[i];
    }
  }

  /** Base pointer of the newest `len` samples of a channel (len <= HIST). */
  const float *HistWindow(unsigned ch, unsigned len) const {
    const unsigned head = m_RoundFrozen.load(std::memory_order_acquire)
      ? m_RoundHead
      : m_HistWrite.load(std::memory_order_relaxed);

    return m_History.data() + ch * 2 * HIST + head + HIST - len;
  }
  // Periods appended so far (epilog snapshots and bail self-feeds both
  // count) - the anchor/tag clock for the shadow slices.
  std::atomic<uint64_t> m_HistPeriod;

  // Double-buffered shadow, planar [ch][m_ShadowLen]; the writer fills the
  // inactive half and publishes (anchor << 1) | index atomically.
  static constexpr uint64_t SHADOW_INVALID = ~0ull;
  std::vector<float> m_Shadow[2];
  std::atomic<uint64_t> m_ShadowPub;
  std::atomic<bool> m_ComputeBusy;

  // FFT workspace (fftwf_malloc'd for alignment), sized for NFFT_MAX and
  // shared by all kits (plans on smaller sizes just use the buffer head).
  float *m_TimeBuf;
  fftwf_complex *m_FreqBuf;
  std::vector<float> m_SpecRe, m_SpecIm; // current frame spectrum
  std::vector<float> m_RotRe, m_RotIm;   // per-bin phase advance rotator
  // Last period's newest frame, PER CHANNEL - this is what makes the
  // second forward FFT unnecessary, and it doubles as the "previous
  // spectrum" the rotator reads, so it costs no copy at all.  Tagged
  // with the period, frame size and hop it was written for: a skipped
  // period or a mode change simply falls back to computing the frame.
  std::vector<float> m_SpecCacheRe, m_SpecCacheIm; // [ch][NBINS_MAX]
  std::vector<uint64_t> m_SpecCachePeriod;
  std::vector<unsigned> m_SpecCacheNfft, m_SpecCacheHop;
  std::vector<float> m_Ola;              // overlap-add accumulator

  // Burg workspace (double precision - the fit is numerically fragile in
  // float, see mixer-dsp.c / gap_lab.py)
  // Two buffer pairs (ping-pong): the lattice error update is written
  // carry-free into the spare pair so -O3/-ffast-math can vectorize it.
  std::vector<double> m_BurgF, m_BurgB, m_BurgF2, m_BurgB2; // [ch][n]
  // Coefficients per channel, plus the period they were fitted in.  The
  // fit is 36% of the whole prediction, while its spectral model barely
  // moves in 20 ms - so refit only every BURG_REFIT_PERIODS periods and
  // re-run the (trivial) one-step anchor prediction each time with FRESH
  // seed samples, which is what actually guarantees continuity at the
  // gap entry.
  static constexpr uint64_t BURG_REFIT_PERIODS = 8;
  // Laengstes Rechteck-Fitfenster, das der Spektrenring tragen muss -
  // der Prototyp benutzt nfit = 20 Tupel.
  static constexpr unsigned FIT_WINDOW_MAX = 24;
  std::vector<double> m_BurgA; // [ch][BURG_ORDER]
  std::vector<uint64_t> m_BurgFitPeriod;
  std::vector<double> m_BurgState; // recursion state of the bridge branch
  std::vector<float> m_Bridge;     // bridge output, m_BridgeLen frames
  std::vector<float> m_BridgeWin;  // raised cosine 0->1 (freeze weight)

  // Delivery-side state, only ever touched by this device's callback thread
  std::vector<float> m_PendingTail; // shadow tail for the re-entry fade
  std::vector<float> m_FadeReentry; // cos 1->0, m_ReentryFade frames
  std::vector<float> m_ReentryMix;  // Scratch fuer den Ring-Fade

  // ---- Transienten-Patch (07.08.2026, Nutzer-Entwurf) ------------------
  // Ein statisches Overlay ueber der Analyse-Historie: im Bereich
  // [m_PatchStart, m_PatchEnd) liest jede Fensterung aus m_PatchBuf
  // statt aus dem Ring.  Der Ring selbst bleibt Wahrheit (Straggler,
  // Backfill-Basis, Wiedereinstieg unberuehrt).  Inhalt: die
  // RUECKWAERTS rotierte Fortsetzung des stabilisierten Zustands -
  // phasenexakt an der rechten Kante, weil sie dem realen Zustand
  // entspringt; links Amplitudenmorph in den Alt-Ton.  Statisch heisst:
  // einmal geschrieben, nie nachgefuehrt - mit fortschreitender Zeit
  // enthaelt jedes Fenster von selbst immer mehr Wahrheit (der
  // staleWindow-Fehlmodus ist konstruktiv ausgeschlossen).
  static constexpr unsigned PATCH_MAX = 2048;
  std::vector<float> m_PatchBuf;             // [ch][PATCH_MAX]
  std::vector<uint64_t> m_PatchStart, m_PatchEnd; // abs. Samples; ==: leer
  std::atomic<int> m_PatchMode{0};           // 1 = sim-Trigger scharf
  std::vector<uint64_t> m_OnsetPeriod;       // [ch]; ~0 = kein Kandidat
  std::vector<unsigned> m_StableCount;       // [ch] Perioden ueber Schwelle
  std::vector<uint64_t> m_LastPatchSample;   // [ch] Entprellung
  std::vector<unsigned> m_QuietRun;          // [ch] ruhige Perioden am Stueck

  // ======== HYBRID: kurze 1k-Zweitkette (Nutzer-Entwurf 07.08.) ========
  // Grundsatz: nach einem Anschlag darf das Gedaechtnis des Modells nie
  // weiter zurueckreichen als der Anschlag selbst.  Die kurze Kette
  // (1k-Rahmen, Raster = eine Periode) laeuft dauerhaft warm mit (eine
  // FFT pro Periode); beim Einsatz bekommen die betroffenen Baender in
  // BEIDEN Ketten frische Normalgleichungen.  Das wirksame Gedaechtnis
  // ist damit min(Alter, Zeitkonstante) und waechst STETIG; die Ordnung
  // klettert mit den Tupeln (p nutzbar ab p+2 Rasterpunkten, Deckel
  // hybShortCap).  Uebergabe an die lange Kette, sobald deren frischer
  // Fit hybHandOrder traegt - bei 2k exakt der Moment, ab dem kein
  // Rahmen mehr Signal von vor dem Anschlag enthaelt (42,7 ms).  Beide
  // Ketten synthetisieren per Ueberlappungsaddition, der Schatten ist
  // die Summe; die Bandgewichte ergaenzen sich punktweise zu eins (hart
  // an der 94-Hz-Bandgrenze, weich in der Zeit, ~3 Perioden Rampe).
  //
  // EHRLICHE LUECKEN v1: (1) die ersten Tupel nach dem Reset enthalten
  // Lags von VOR dem Anschlag (der Spektrenring wird nicht geleert) -
  // die Leiter begrenzt nur die GENUTZTE Ordnung; (2) Bandkanten hart;
  // (3) kein Pegeltrend auf der kurzen Kette; (4) kein Backfill fuer
  // ihren Ring (Stempel-Luecken lassen Tupel aus statt zu reparieren).
  static constexpr unsigned HYB_NFFT = 1024;
  static constexpr unsigned HYB_NBINS = HYB_NFFT / 2 + 1;
  static constexpr unsigned HYB_RING = 16;
  static constexpr uint16_t HYB_DORMANT = 0xFFFF;
  std::atomic<int> m_Hybrid{0};
  // 1 = Akkumulator-Reset beim Anschlag (der urspruengliche Entwurf).
  // GEMESSEN SCHAEDLICH als Vorgabe: das Nullen wirft der langen Kette
  // pro Einsatz in ~100 Baendern 200 ms Fit weg - Schaden weit ueber
  // das Transientenfenster hinaus (+6.5 dB NMR ohne B-Beteiligung,
  // XO_HYB_NO_B).  Die KONTINUIERLICHE kurze Kette klingt am Einsatz
  // dagegen ungestoert (Hoerbefund des Nutzers, 1k/2 und 1k/5) - ihr
  // 80-ms-Gedaechtnis waescht den Alt-Klang schnell genug aus.  Bleibt
  // als Knopf fuer Messungen.
  std::atomic<int> m_HybReset{0};
  std::atomic<int> m_HybAllBands{0};
  std::atomic<int> m_HybShortCap{6};
  // 0 = VOLLBETRIEB als Vorgabe.  Wichtig zum Verstaendnis: im
  // Normalmodus gibt es KEINE Ordnungsleiter - die kurze Kette spielt
  // ab dem Wechsel sofort cap aus ihrem WARMEN Akkumulator.  Der
  // Sparbetrieb fuehrt eine Leiter ein (warm bis der frische Satz
  // traegt), und die verliert monoton mit sinkender Warm-Ordnung
  // (6000/min, Mittel/max/Einsatz): voll -2.70/7.10/+4.75, @5
  // -2.02/6.97/+4.29, @4 -1.94/6.93/+5.21, @3 -1.53/7.77/+5.30, @2
  // -1.36/9.88/+7.70.  Grund - zum sechsten Mal diese Woche:
  // VERUNREINIGTE DATEN SCHLAGEN FEHLENDE.  Ein frischer Akkumulator
  // traegt 20 ms nach dem Anschlag nur Ordnung 2-3; der warme ist halb
  // kontaminiert, aber voll bestimmt fuer Ordnung 6.  Der Knopf bleibt
  // fuer Rechenzeit-Not (voll 326 us, @4 283, @2 240 us pro Periode
  // bei 2 Kanaelen).
  std::atomic<int> m_HybWarmOrder{0};
  std::atomic<int> m_HybHandOrder{6};
  std::vector<float> m_HybRingRe, m_HybRingIm; // [ch][slot][HYB_NBINS]
  std::vector<uint64_t> m_HybRingPeriod;       // [ch][slot], ~0 = leer
  std::vector<double> m_HybP, m_HybQ;          // [ch][bin][k][l] bzw. [k]
  // SPARBETRIEB (Nutzer-Frage 08.08.: 'muss die 1k-Kette mit Ordnung 6
  // warmlaufen?'): nein.  Warm laeuft nur die STARTordnung
  // (hybWarmOrder, 2x2 statt 6x6 = 1/7 der Update-Kosten); ab der
  // Schaltung akkumuliert ein ZWEITER, frischer Satz bei voller
  // Deckelordnung, und die Leiter klettert mit echten Tupeln (p ab
  // p+2).  Zwei Saetze statt Mischmatrix: eine Gram-Matrix aus zwei
  // Historien waere keine.
  std::vector<double> m_HybP2, m_HybQ2;        // frisch, nur junge Baender
  std::vector<float> m_HybCoefRe, m_HybCoefIm; // [ch][bin][AR_ORDER_MAX]
  std::vector<float> m_HybRootRe, m_HybRootIm;
  std::vector<uint16_t> m_HybAge; // [ch][ROUTE_BANDS] Perioden seit Reset
  // ECHTE Tupel seit dem Reset, pro Kette.  Alter zaehlt Perioden, aber
  // waehrend einer Ausfallkette laeuft keine Analyse - eine Leiter nach
  // Alter loeste dann unterbestimmt (gemessen +7 dB NMR statt Gewinn).
  std::vector<uint16_t> m_HybTupA, m_HybTupB; // [ch][ROUTE_BANDS]
  std::vector<float> m_HybW;      // [ch][ROUTE_BANDS] Anteil kurze Kette
  std::vector<uint8_t> m_HybAny;  // [ch] irgendein Band aktiv?
  std::vector<uint64_t> m_HybEpStart; // [ch][band] Periode des Wechsels, ~0=zu
  // RESIDUEN-WAEHLER (Nutzer-Go 08.08.): pro Band der geglaettete
  // RELATIVE Ein-Schritt-Vorhersagefehler jeder Kette in ihrer eigenen
  // Domaene, Sum|X - Xdach|^2 / Sum|X|^2 - relative Fehler sind ueber
  // Rahmengroessen vergleichbar.  Gespielt wird, wer besser vorhersagt.
  // GEMESSEN GESCHEITERT als Vorgabe (08.08., zwei Iterationen samt
  // Energie-Tor): der EIN-Schritt-Fehler misst NACHFUEHRUNG, die
  // Verdeckung braucht EXTRAPOLATION ueber den 32-ms-Horizont.  Die
  // kurze Kette ist auf Plateaus der bessere Nachfuehrer (80 ms
  // Gedaechtnis folgt der Schwebung), aber der schlechtere
  // Extrapolator - der Waehler schaltete 3935x fern der Einsaetze
  // (global +6.4 dB NMR statt -2.4).  Bleibt als Messknopf; ein
  // MEHR-Schritt-Residuum waere der naechste Kandidat.
  std::atomic<int> m_HybChooser{0};
  // ---- 512er-VORSTUFE (zweistufiges Schalten, Nutzer-Go 08.08.) ----
  // Die Breite auf 512er-Rahmen kreuzt 13-29 ms nach dem physischen
  // Anschlag - Augen-Paritaet (Nutzer-Messung: +24/+18 ms) und 10-15 ms
  // vor dem 2k-Detektor.  Ihre Trennreserve ist hauchduenn (Stoermax
  // 0.401, Einsatz-Min 0.424), deshalb schaltet sie nur VORLAEUFIG:
  // bestaetigt der 2k-Detektor nicht binnen TENT_TTL Perioden, wird
  // zurueckgezogen - die lange Kette wurde nie angetastet und spielt
  // sofort wieder in voller Ordnung (Idee des Nutzers: Fehlalarme sind
  // billig, wenn man sie rueckwirkend erkennt).
  static constexpr unsigned PRE_NFFT = 512;
  static unsigned preVeto() {
    static const unsigned v = [] {
      const char *e = getenv("XO_PRE_VETO");

      return e ? (unsigned)atoi(e) : 5u;
    }();

    return v;
  }
  static constexpr unsigned PRE_RING = 9; // aktuelle + 8 Baseline
  std::atomic<int> m_HybPre{0};
  std::atomic<int> m_HybSimple{0};
  // Der ERSTE Detektor sah durch die 4k-ANALYSE (Schwellen 0.25/0.12,
  // gehalten p99 0.217) - mit der 2k-Hybrid-Basis erbte die Erkennung
  // das halb so lange, doppelt so zappelige Fenster (p99 0.268).  Der
  // Simpel-Modus bekommt seine 4k-Augen zurueck: eine eigene
  // 4k-Analyse NUR fuer die Erkennung, eine FFT pro Periode,
  // unabhaengig vom Verdeckungsrahmen.
  std::vector<float> m_D4Band;    // [ch][9][ROUTE_BANDS]
  std::vector<uint64_t> m_D4Per;  // [ch][9]
  std::vector<float> m_D4Wide;    // [ch], -1 = ungueltig
  std::vector<uint8_t> m_D4Rise;  // [ch][ROUTE_BANDS]
  std::vector<uint8_t> m_D4Conf;  // [ch][ROUTE_BANDS] residuen-bestaetigt
  // Abklingendes MAXIMUM des Residuums je Band.  Der Momentanwert reicht
  // nicht: das Residuum eines schwachen Einsatzes kreuzt erst ein paar
  // Perioden nach der Breitenschwelle (in den Vorstufen-Messungen der
  // Einsatz bei 7,26 s).  Die Vorstufe fuehrte deshalb ein Maximum ueber
  // ihr Prueffenster mit; hier tut es ein Abkling-Maximum, das kein
  // Fenster verwalten muss.
  std::vector<float> m_D4ResMax; // [ch][ROUTE_BANDS]
  // Live regelbar (Regler im Wirt); die Umgebungsvariablen setzen nur den
  // Startwert, damit ein Skript sie weiter vorgeben kann.
  std::atomic<float> m_D4ResGate{[] {
    const char *e = getenv("XO_D4_RES");

    return e ? (float)atof(e) : 0.0f;
  }()};
  std::atomic<float> m_D4ResDecay{[] {
    const char *e = getenv("XO_D4_DECAY");

    return e ? (float)atof(e) : 0.85f;
  }()};
  // Hindcast-Anker (siehe XorcistTuning::v0Hindcast) - Schalter im Wirt,
  // XO_V0_SHADOW liefert nur den Startwert fuer Harness-Skripte.
  std::atomic<int> m_V0Hindcast{[] {
    const char *e = getenv("XO_V0_SHADOW");

    return e && atoi(e) != 0 ? 1 : 0;
  }()};
  fftwf_plan m_PreFwd = nullptr;
  std::vector<float> m_PreWin;
  std::vector<float> m_PreBandE;     // [ch][PRE_RING][ROUTE_BANDS]
  std::vector<uint64_t> m_PrePeriod; // [ch][PRE_RING], ~0 = leer
  std::vector<uint8_t> m_HybTent;    // [ch][band] Countdown, 0 = fest
  // WIEDERHOLEN STATT VERWERFEN (wie beim Patch-Waechter): steht die
  // Breite ueber der Schaltlatte, aber das Spleiss-Veto blockiert,
  // bleibt der Wechselwunsch 8 Perioden bestehen und feuert bei der
  // naechsten veto-freien Gelegenheit.  Ohne das verhungerte der
  // schwaechste Einsatz (7,30 s) bei 6000/min komplett: 4 Perioden
  // ueber der Latte, alle mit Luecke in den letzten 3 Perioden.
  std::vector<uint8_t> m_HybRetr;     // [ch][band] Episode zurueckgezogen?
  // PERSISTENZ-CHECK der Vorstufe (Nutzer-Richtung 08.08., statt
  // Gesamtenergie - die trennt doppelt gemessen falsch herum: der
  // schwaechste Einsatz hebt sie kaum, die Schwebungsflanke schon).
  // Stoerer sind DUENN: ihre gestiegenen Baender fallen binnen 1-2
  // Perioden zurueck, ein neuer Ton BLEIBT oben.  4 Perioden nach der
  // vorlaeufigen Schaltung wird gegen die EINGEFRORENE Vor-Schalt-
  // Baseline geprueft: haelt weniger als die Haelfte der Rise-Menge,
  // sofortiger Rueckzug (~11 ms statt 43); haelt sie, ist die
  // Schaltung fest, ohne auf den 2k-Detektor zu warten.
  // Kandidaten-Disziplin der ERSTEN Detektor-Implementation (Nutzer:
  // "die war zuverlaessiger"), auf die 512er portiert: ein Kandidat
  // entsteht an der ON-Kreuzung, und bis er sich unter OFF beruhigt
  // hat (6 Perioden) oder verfaellt (TTL 96), entsteht KEIN zweiter -
  // ein sauberes Ereignis pro Anschlag statt Schwellen-Geflatter.
  std::vector<uint64_t> m_PreCandP;   // [ch] Kandidaten-Periode, ~0 = frei
  std::vector<uint8_t> m_PreStable;   // [ch] Perioden unter OFF am Stueck
  std::vector<float> m_PreEmax;       // [ch] laufendes Energiemaximum
  std::vector<uint8_t> m_PreChk;      // [ch] Countdown bis zum Check
  std::vector<float> m_PreResMax;     // [ch] max resA-Mittel seit Zuendung
  std::vector<float> m_PreResMaxB;    // [ch][band] dito pro Band
  std::vector<float> m_PreBase;       // [ch][band] eingefrorene Baseline
  std::vector<uint8_t> m_PreRise;     // [ch][band] urspruengliche Rises
  std::vector<uint8_t> m_HybPend;     // [ch] Countdown Wechselwunsch
  std::vector<uint8_t> m_HybPendRise; // [ch][band] Vereinigung der Rises

  std::vector<float> m_HybResA, m_HybResB; // [ch][ROUTE_BANDS], EMA
  std::vector<float> m_HybEneB;            // [ch][ROUTE_BANDS], EMA-Energie

  bool HybBandYoung(unsigned ch, unsigned band) const {
    if (band >= ROUTE_BANDS)
      band = ROUTE_BANDS - 1;
    return m_HybAge[(size_t)ch * ROUTE_BANDS + band] != HYB_DORMANT;
  }

  // Eine Periode 1k-Analyse: ein Rahmen, ein Ring-Slot, RLS nur fuer
  // junge Baender (ausserhalb wird die kurze Kette nie gehoert, und der
  // Anschlag setzt ohnehin zurueck).  Laeuft auch im Absorb-Pfad.
  void HybridAnalyze(unsigned ch, uint64_t nowP) {
    if (m_HistFill.load(std::memory_order_relaxed) < HYB_NFFT)
      return;

    const PvKit &kit = m_Pv[3];
    float *pTime = m_TimeBuf + (size_t)ch * NFFT_MAX;
    fftwf_complex *pFreq = m_FreqBuf + (size_t)ch * FREQ_STRIDE;
    const float *pHist = HistWindow(ch, HYB_NFFT);
    const float *win = kit.win.data();

    for (unsigned i = 0; i < HYB_NFFT; i++)
      pTime[i] = pHist[i] * win[i];
    fftwf_execute_dft_r2c(kit.fwd, pTime, pFreq);

    const unsigned sl = (unsigned)(nowP % HYB_RING);
    float *fRe = m_HybRingRe.data() + ((size_t)ch * HYB_RING + sl) * HYB_NBINS;
    float *fIm = m_HybRingIm.data() + ((size_t)ch * HYB_RING + sl) * HYB_NBINS;

    for (unsigned bin = 0; bin < HYB_NBINS; bin++) {
      fRe[bin] = pFreq[bin][0];
      fIm[bin] = pFreq[bin][1];
    }
    m_HybRingPeriod[(size_t)ch * HYB_RING + sl] = nowP;

    const bool chooser
      = m_HybChooser.load(std::memory_order_relaxed) != 0;
    const unsigned cap = (unsigned)std::min(
      (int)AR_ORDER_MAX,
      std::max(1, m_HybShortCap.load(std::memory_order_relaxed) > 0
                    ? m_HybShortCap.load(std::memory_order_relaxed)
                    : 6));

    if (nowP < cap)
      return;

    // B-RESIDUUM: wie gut haetten die AKTUELLEN B-Koeffizienten diesen
    // Rahmen aus den Ring-Lags vorhergesagt?  VOR dem Akkumulieren und
    // VOR dem Solve - der Praediktor wird an Daten gemessen, die er
    // noch nicht gesehen hat.
    if (chooser) {
      bool lagOk2 = true;
      const float *lRe[AR_ORDER_MAX], *lIm[AR_ORDER_MAX];

      for (unsigned k = 1; k <= cap; k++) {
        const uint64_t want = nowP - k;
        const unsigned ks = (unsigned)(want % HYB_RING);

        if (m_HybRingPeriod[(size_t)ch * HYB_RING + ks] != want) {
          lagOk2 = false;
          break;
        }
        lRe[k - 1] = m_HybRingRe.data()
          + ((size_t)ch * HYB_RING + ks) * HYB_NBINS;
        lIm[k - 1] = m_HybRingIm.data()
          + ((size_t)ch * HYB_RING + ks) * HYB_NBINS;
      }
      if (lagOk2) {
        float *res = m_HybResB.data() + (size_t)ch * ROUTE_BANDS;

        for (unsigned b = 0; b < ROUTE_BANDS; b++) {
          const unsigned lo = b << 1;
          const unsigned hi = std::min(HYB_NBINS, (b + 1u) << 1);
          double err = 0.0, ene = 0.0;

          for (unsigned bin = lo; bin < hi; bin++) {
            const float *cR = m_HybCoefRe.data()
              + ((size_t)ch * HYB_NBINS + bin) * AR_ORDER_MAX;
            const float *cI = m_HybCoefIm.data()
              + ((size_t)ch * HYB_NBINS + bin) * AR_ORDER_MAX;
            float pr = 0.0f, pi = 0.0f;

            for (unsigned k = 0; k < cap; k++) {
              pr += cR[k] * lRe[k][bin] - cI[k] * lIm[k][bin];
              pi += cR[k] * lIm[k][bin] + cI[k] * lRe[k][bin];
            }

            const float dr = fRe[bin] - pr, di = fIm[bin] - pi;

            err += (double)dr * dr + (double)di * di;
            ene += (double)fRe[bin] * fRe[bin]
              + (double)fIm[bin] * fIm[bin];
          }

          const float r = (float)(err / (ene + 1e-20));

          res[b] = 0.7f * res[b] + 0.3f * std::min(r, 10.0f);

          float *eb = m_HybEneB.data() + (size_t)ch * ROUTE_BANDS;

          eb[b] = 0.7f * eb[b] + 0.3f * (float)ene;
        }
      }
    }

    // BACKFILL wie bei der langen Kette: waehrend einer Ausfallkette
    // laeuft keine Analyse, der Ring bekommt Loecher - die SAMPLES sind
    // aber da (Selbstfuetterung).  Ohne Reparatur baute die Synthese
    // ihren Zustand aus Nullen und die kurze Kette spielte STILLE in
    // genau die Baender, die sie tragen soll - gemessen als +13 dB NMR
    // statt Gewinn, die exakte Wiederholung des alten AR-Pfad-Lochs.
    for (unsigned j = 1; j <= cap; j++) {
      if (nowP < j)
        break;

      const uint64_t want = nowP - j;
      const unsigned sl2 = (unsigned)(want % HYB_RING);

      if (m_HybRingPeriod[(size_t)ch * HYB_RING + sl2] == want)
        continue;

      const unsigned back = j * m_SamplesPerBuffer;

      if (
        HYB_NFFT + back > HIST
        || m_HistFill.load(std::memory_order_relaxed) < HYB_NFFT + back)
        continue;

      const float *pW = HistWindow(ch, HYB_NFFT + back);

      for (unsigned i = 0; i < HYB_NFFT; i++)
        pTime[i] = pW[i] * win[i];
      fftwf_execute_dft_r2c(kit.fwd, pTime, pFreq);

      float *bRe = m_HybRingRe.data()
        + ((size_t)ch * HYB_RING + sl2) * HYB_NBINS;
      float *bIm = m_HybRingIm.data()
        + ((size_t)ch * HYB_RING + sl2) * HYB_NBINS;

      for (unsigned bin = 0; bin < HYB_NBINS; bin++) {
        bRe[bin] = pFreq[bin][0];
        bIm[bin] = pFreq[bin][1];
      }
      m_HybRingPeriod[(size_t)ch * HYB_RING + sl2] = want;
    }

    // Akkumuliert wird IMMER bei der Deckelordnung: P[k][l] ist
    // geschachtelt, der linke obere p x p Block IST das Ordnung-p-System
    // - jede Leitersprosse loest nur einen Unterblock.
    const double lam = ArLambda(cap);
    const float *lagRe[AR_ORDER_MAX], *lagIm[AR_ORDER_MAX];
    bool lagOk = true;

    for (unsigned k = 1; k <= cap; k++) {
      const uint64_t want = nowP - k;
      const unsigned ks = (unsigned)(want % HYB_RING);

      if (m_HybRingPeriod[(size_t)ch * HYB_RING + ks] != want) {
        lagOk = false; // Loch im Ring: Tupel dieser Periode auslassen
        break;
      }
      lagRe[k - 1] = m_HybRingRe.data()
        + ((size_t)ch * HYB_RING + ks) * HYB_NBINS;
      lagIm[k - 1] = m_HybRingIm.data()
        + ((size_t)ch * HYB_RING + ks) * HYB_NBINS;
    }
    if (!lagOk)
      return;

    const size_t pStride = (size_t)AR_ORDER_MAX * AR_ORDER_MAX * 2;
    const size_t qStride = (size_t)AR_ORDER_MAX * 2;
    const uint16_t *age = m_HybAge.data() + (size_t)ch * ROUTE_BANDS;

    // SPARBETRIEB: warm laeuft fuer alle Baender nur die STARTordnung
    // (hybWarmOrder, Vorgabe 2 - 2x2 statt 6x6 = 1/7 der Kosten); die
    // volle Deckelordnung akkumuliert ein zweiter, FRISCHER Satz nur in
    // jungen Baendern.  hybWarmOrder 0 = alter Vollbetrieb.  Im
    // Waehler-Modus immer voll (der braucht cap-Residuen ueberall).
    const int woRaw = m_HybWarmOrder.load(std::memory_order_relaxed);
    const unsigned warmOrd = chooser
      ? cap
      : (woRaw <= 0 ? cap : std::min(cap, (unsigned)woRaw));
    const double lamW = ArLambda(warmOrd);
    const uint16_t *ageAcc = m_HybAge.data() + (size_t)ch * ROUTE_BANDS;

    for (unsigned b = 0; b < ROUTE_BANDS; b++) {
      const bool young = ageAcc[b] != HYB_DORMANT;

      if (young && m_HybTupB[(size_t)ch * ROUTE_BANDS + b] < 60000)
        m_HybTupB[(size_t)ch * ROUTE_BANDS + b]++;

      const unsigned lo = b << 1;
      const unsigned hi = std::min(HYB_NBINS, (b + 1u) << 1);

      for (unsigned bin = lo; bin < hi; bin++) {
        const double x0r = fRe[bin], x0i = fIm[bin];
        double *Pb = m_HybP.data()
          + ((size_t)ch * HYB_NBINS + bin) * pStride;
        double *Qb = m_HybQ.data()
          + ((size_t)ch * HYB_NBINS + bin) * qStride;

        for (unsigned k = 0; k < warmOrd; k++) {
          const double xkr = lagRe[k][bin], xki = lagIm[k][bin];

          Qb[2 * k] = lamW * Qb[2 * k] + x0r * xkr + x0i * xki;
          Qb[2 * k + 1] = lamW * Qb[2 * k + 1] + x0i * xkr - x0r * xki;
          for (unsigned l = 0; l < warmOrd; l++) {
            const double xlr = lagRe[l][bin], xli = lagIm[l][bin];
            const size_t o = 2 * ((size_t)k * AR_ORDER_MAX + l);

            Pb[o] = lamW * Pb[o] + xkr * xlr + xki * xli;
            Pb[o + 1] = lamW * Pb[o + 1] + xki * xlr - xkr * xli;
          }
        }
        if (warmOrd >= cap || !young)
          continue;

        // Frischer Satz bei voller Deckelordnung, nur solange jung.
        double *P2 = m_HybP2.data()
          + ((size_t)ch * HYB_NBINS + bin) * pStride;
        double *Q2 = m_HybQ2.data()
          + ((size_t)ch * HYB_NBINS + bin) * qStride;

        for (unsigned k = 0; k < cap; k++) {
          const double xkr = lagRe[k][bin], xki = lagIm[k][bin];

          Q2[2 * k] = lam * Q2[2 * k] + x0r * xkr + x0i * xki;
          Q2[2 * k + 1] = lam * Q2[2 * k + 1] + x0i * xkr - x0r * xki;
          for (unsigned l = 0; l < cap; l++) {
            const double xlr = lagRe[l][bin], xli = lagIm[l][bin];
            const size_t o = 2 * ((size_t)k * AR_ORDER_MAX + l);

            P2[o] = lam * P2[o] + xkr * xlr + xki * xli;
            P2[o + 1] = lam * P2[o + 1] + xki * xlr - xkr * xli;
          }
        }
      }
    }

    // Waehler-Modus: B kontinuierlich LOESEN (gestreift), damit
    // Residuum und Synthese jederzeit aktuelle Koeffizienten sehen.
    if (chooser) {
      const int se = m_SolveEvery.load(std::memory_order_relaxed);
      const unsigned cycle = se > 0 ? (unsigned)se : 16u;
      const unsigned stripe = (unsigned)(nowP % cycle);
      const unsigned b0 = HYB_NBINS * stripe / cycle;
      const unsigned b1 = HYB_NBINS * (stripe + 1) / cycle;

      for (unsigned bin = b0; bin < b1; bin++) {
        float *cR = m_HybCoefRe.data()
          + ((size_t)ch * HYB_NBINS + bin) * AR_ORDER_MAX;
        float *cI = m_HybCoefIm.data()
          + ((size_t)ch * HYB_NBINS + bin) * AR_ORDER_MAX;

        SolveBin(
          m_HybP.data() + ((size_t)ch * HYB_NBINS + bin) * pStride,
          m_HybQ.data() + ((size_t)ch * HYB_NBINS + bin) * qStride, cap,
          cR, cI,
          m_HybRootRe.data() + ((size_t)ch * HYB_NBINS + bin) * AR_ORDER_MAX,
          m_HybRootIm.data()
            + ((size_t)ch * HYB_NBINS + bin) * AR_ORDER_MAX);
        for (unsigned k = cap; k < AR_ORDER_MAX; k++) {
          cR[k] = 0.0f;
          cI[k] = 0.0f;
        }
      }
    }
  }

  // Bandalter und Gewichte fortschreiben, einmal pro Periode.
  void HybridTick(unsigned ch, unsigned strideA, unsigned orderA,
                  uint64_t nowP) {
    // DIAGNOSE: alle Baender dauerhaft auf der kurzen Kette.  Muss
    // dieselben Zahlen liefern wie die pure 1k-Strategie bei
    // Deckel-Ordnung - misst Maske + B-Synthese isoliert.
    static const bool force = getenv("XO_HYB_FORCE") != nullptr;

    if (force) {
      uint16_t *age = m_HybAge.data() + (size_t)ch * ROUTE_BANDS;
      float *w = m_HybW.data() + (size_t)ch * ROUTE_BANDS;

      for (unsigned b = 0; b < ROUTE_BANDS; b++) {
        age[b] = 1;
        w[b] = 1.0f;
        m_HybTupB[(size_t)ch * ROUTE_BANDS + b] = 60000;
      }
      m_HybAny[ch] = 1;
      return;
    }
    const int handRaw = m_HybHandOrder.load(std::memory_order_relaxed);
    // OHNE Klemme auf AR_ORDER_MAX: im Alter-Modus ist der Wert reine
    // ZEITsteuerung ((X+2) Rasterpunkte der langen Kette), und der
    // 200-ms-Haltetest des Nutzers braucht X=36 bei 2k.  Nur der
    // Reset-Modus liest ihn als Ordnung, und dort verhungert ein
    // uebergrosser Wert harmlos in der Schlaf-Bedingung.
    const unsigned hand = (unsigned)std::max(1, handRaw > 0 ? handRaw : 6);
    const unsigned handP = (hand + 2) * strideA;
    // Schlafen erst, wenn auch die lange Kette ihre volle Ordnung wieder
    // traegt - bis dahin braucht ihr Loeser die Leiter.
    const unsigned dormP = std::max(handP, (orderA + 2) * strideA) + 8;
    uint16_t *age = m_HybAge.data() + (size_t)ch * ROUTE_BANDS;
    float *w = m_HybW.data() + (size_t)ch * ROUTE_BANDS;
    uint8_t any = 0;
    const bool chooser
      = m_HybChooser.load(std::memory_order_relaxed) != 0;
    const float *rA = m_HybResA.data() + (size_t)ch * ROUTE_BANDS;
    const float *rB = m_HybResB.data() + (size_t)ch * ROUTE_BANDS;
    // Hysterese: zu B erst bei KLAR besserer Vorhersage (Faktor 0.5),
    // zurueck zu A, sobald A schlicht besser ist.  Dazwischen bleibt
    // der Zustand - kein Flattern auf Rauschen.
    static const float toB = [] {
      const char *e = getenv("XO_HYB_HYST");

      return e ? (float)atof(e) : 0.5f;
    }();

    for (unsigned b = 0; b < ROUTE_BANDS; b++) {
      if (chooser) {
        const bool onNow = w[b] > 0.5f;
        bool want = onNow;
        // ENERGIE-TOR: in leisen Baendern ist der relative Fehler
        // beider Ketten ~1 und verrauscht - die Entscheidung wuerfelte
        // (404 Episoden fern der Einsaetze, teils VOR dem Anschlag).
        // Ohne nennenswerte Energie gibt es nichts zu waehlen: A.
        const float *eb = m_HybEneB.data() + (size_t)ch * ROUTE_BANDS;
        static const float eneFloorRel = [] {
          const char *e = getenv("XO_HYB_ENEFLOOR");

          return e ? (float)atof(e) : 1e-5f; // -50 dB unterm Maximum
        }();
        float eMax = 0.0f;

        for (unsigned bb = 0; bb < ROUTE_BANDS; bb++)
          eMax = std::max(eMax, eb[bb]);
        if (eb[b] < eneFloorRel * eMax)
          want = false;
        else if (!onNow && rB[b] < toB * rA[b])
          want = true;
        else if (onNow && rB[b] > rA[b])
          want = false;
        w[b] = want ? 1.0f : 0.0f;
        if (want) {
          if (age[b] == HYB_DORMANT)
            age[b] = 0;
          else if (age[b] < 60000)
            age[b]++;
        }
        // Episoden-Protokoll (wie unten, gleiche Kodierung).
        {
          uint64_t &ep = m_HybEpStart[(size_t)ch * ROUTE_BANDS + b];

          if (want && ep == ~0ull)
            ep = nowP;
          else if (!want && ep != ~0ull) {
            if (ch == 0) {
              const int capRaw
                = m_HybShortCap.load(std::memory_order_relaxed);
              const uint64_t cap = (uint64_t)std::min(
                (int)AR_ORDER_MAX, std::max(1, capRaw > 0 ? capRaw : 6));
              const uint64_t n = s_HybLogN.load(std::memory_order_relaxed);
              uint64_t *e = s_HybLog[n % HYB_LOG];

              e[0] = ep * m_SamplesPerBuffer;
              e[1] = nowP * m_SamplesPerBuffer;
              e[2] = (uint64_t)b | (cap << 16) | ((uint64_t)HYB_NFFT << 32);
              s_HybLogN.store(n + 1, std::memory_order_relaxed);
            }
            ep = ~0ull;
          }
        }
        if (!want)
          age[b] = HYB_DORMANT;
        else
          any = 1;
        continue;
      }
      if (age[b] == HYB_DORMANT)
        continue;
      if (age[b] < 60000)
        age[b]++;

      // RUECKZUG: vorlaeufig geschaltete Baender, die der 2k-Detektor
      // binnen TTL nicht bestaetigt, gehen sofort zurueck - Ziel 0,
      // die nie angetastete lange Kette spielt wieder voll.
      {
        uint8_t &tn = m_HybTent[(size_t)ch * ROUTE_BANDS + b];

        if (tn && --tn == 0 && age[b] < handP) {
          age[b] = (uint16_t)handP;
          // ZURUECKGEZOGEN: die Vorstufe hat geschaltet, der
          // 2k-Detektor nie bestaetigt - die Episode endet gleich und
          // wird im Log als Rueckzug geflaggt (Bit 15), das Lab malt
          // sie orange statt blau.
          m_HybRetr[(size_t)ch * ROUTE_BANDS + b] = 1;
        }
      }

      // Weich in der ZEIT: ~3 Perioden Rampe, gegenlaeufig in beiden
      // Ketten (dieselbe Gewichtsfunktion, 1 - w drueben).
      // OHNE Reset entscheidet das ALTER (die lange Kette laeuft ja
      // ungestoert weiter, es gibt kein "traegt endlich" zu erwarten);
      // MIT Reset datengetrieben: erst wenn ihr frischer Fit die
      // Uebergabe-Ordnung wirklich TRAEGT (hand+2 echte Tupel - unter
      // Ausfallketten kommen Tupel spaeter als die Uhr).
      const float tgt = m_HybReset.load(std::memory_order_relaxed)
        ? (m_HybTupA[(size_t)ch * ROUTE_BANDS + b] < hand + 2 ? 1.0f : 0.0f)
        : (age[b] < handP ? 1.0f : 0.0f);
      // AUFWAERTS sofort: unter der BINAEREN Maske hat die Rampe nach
      // oben keine Funktion mehr (gehoert wird erst ab w > 0.5) - sie
      // kostete nur 2 Perioden = 5,3 ms zusaetzlichen Verzug, und der
      // Detektor ist ohnehin 27-40 ms hinter dem physischen Anschlag.
      // Abwaerts bleibt die Rampe als Entprellung der Uebergabe.
      const float d = tgt - w[b];

      w[b] += d > 0.0f ? d : (d < -0.34f ? -0.34f : d);
      if (w[b] < 0.0f)
        w[b] = 0.0f;
      if (w[b] > 1.0f)
        w[b] = 1.0f;
      // Episoden-Protokoll: Wechsel der BINAEREN Maske (w kreuzt 0.5)
      // ergibt Start/Ende eines Rechtecks im Spektrogramm des Labs.
      {
        const bool onNow = w[b] > 0.5f;
        uint64_t &ep = m_HybEpStart[(size_t)ch * ROUTE_BANDS + b];

        if (onNow && ep == ~0ull)
          ep = nowP;
        else if (!onNow && ep != ~0ull) {
          if (ch == 0) {
            const int capRaw
              = m_HybShortCap.load(std::memory_order_relaxed);
            const uint64_t cap = (uint64_t)std::min(
              (int)AR_ORDER_MAX, std::max(1, capRaw > 0 ? capRaw : 6));
            const uint64_t n
              = s_HybLogN.load(std::memory_order_relaxed);
            uint64_t *e = s_HybLog[n % HYB_LOG];

            e[0] = ep * m_SamplesPerBuffer;
            e[1] = nowP * m_SamplesPerBuffer;
            e[2] = (uint64_t)b | (cap << 16)
              | ((uint64_t)HYB_NFFT << 32)
              | ((uint64_t)(m_HybRetr[(size_t)ch * ROUTE_BANDS + b] ? 1 : 0)
                 << 15);
            s_HybLogN.store(n + 1, std::memory_order_relaxed);
          }
          m_HybRetr[(size_t)ch * ROUTE_BANDS + b] = 0;
          ep = ~0ull;
        }
      }
      const bool longCarries = m_HybReset.load(std::memory_order_relaxed)
        ? (m_HybTupA[(size_t)ch * ROUTE_BANDS + b] >= orderA + 2
           || age[b] >= 500)
        : age[b] >= dormP;

      if (w[b] <= 0.001f && longCarries) {
        age[b] = HYB_DORMANT;
        w[b] = 0.0f;
      } else
        any = 1;
    }
    m_HybAny[ch] = any;
  }

  // Anschlag: betroffene Baender (um 1 dilatiert - der Einsatz waechst
  // ueber Perioden in die Nachbarbaender) in BEIDEN Ketten zuruecksetzen.
  void HybridOnset(
    unsigned ch, const uint8_t *rise, unsigned nbinsA, unsigned shiftA,
    bool tentative) {
    // Derselbe Anschlag zuendet nicht doppelt - aber die Sperre darf
    // nur die EIGENZEIT eines Anschlags decken (Breite bleibt 8-24
    // Perioden oben), nicht 128 ms: Motor-/Vorlaeufer-Ereignisse 50-130
    // ms vor einem echten Einsatz sperrten dessen Baender mit, der
    // zweite Ton wurde erst nach Ablauf erkannt (+60 ms, als spaeter
    // 2k-Vollblock statt Per-Band - Nutzer-Beobachtung 08.08.).
    static const unsigned rejoinEnv = [] {
      const char *e = getenv("XO_HYB_REJOIN");

      // 16 statt 12: bei 6000/min aequivalent, aber 11 ms mehr Ruhe
      // innerhalb eines Anschlags.  Der beobachtete 60-ms-Fall braucht
      // nur <= 18 (Vorlaeufer 2.79 s -> Einsatz 2.84 s = 19 Perioden).
      return e ? (unsigned)atoi(e) : 16u;
    }();
    const unsigned REJOIN
      = m_HybSimple.load(std::memory_order_relaxed) ? 48u : rejoinEnv;
    // Bestaetigungsfenster: 2k-Erkennung kommt bis ~27 ms nach der
    // Vorstufe; 16 Perioden = 43 ms decken das mit Reserve.
    static const unsigned TENT_TTL = [] {
      const char *e = getenv("XO_PRE_TTL");

      return e ? (unsigned)atoi(e) : 16u;
    }();
    // Diagnose (XO_HYB_ALLBANDS=1): beim Einsatz schalten ALLE Baender
    // auf die kurze Kette, nicht nur die steigenden.  Trennt fuer den
    // Hoertest "falsche Bandauswahl" von "falsches Timing": klingt es
    // damit wie pures 1k am Transienten, war die Auswahl schuld.
    if (m_HybChooser.load(std::memory_order_relaxed))
      return; // der Residuen-Waehler schaltet selbst - Detektor nur Anzeige

    static const bool envAll = getenv("XO_HYB_ALLBANDS") != nullptr;
    // Probephase NUR mit steigenden Baendern wurde gemessen und
    // VERWORFEN (global -1.40 -> -1.11): der Schwebungs-Preis bei
    // Stressdichte kommt nicht von den 11-ms-Probeschlaegen.
    const bool allBands
      = envAll || m_HybAllBands.load(std::memory_order_relaxed) != 0;
    uint16_t *age = m_HybAge.data() + (size_t)ch * ROUTE_BANDS;
    const size_t pStride = (size_t)AR_ORDER_MAX * AR_ORDER_MAX * 2;
    const size_t qStride = (size_t)AR_ORDER_MAX * 2;
    // Zaehlung der ZUENDUNGEN (Nutzerwunsch 10.08.2026): auf gehaltenem
    // Klang darf hier nichts passieren, und wenn doch, will man es sehen
    // statt es zu vermuten.  Vorlaeufige Zuendungen der 512er-Vorstufe
    // getrennt, weil sie widerrufen werden koennen; GO faehrt sie nicht.
    // Gezaehlt wird nur die WIRKSAME Zuendung, also die, bei der wirklich
    // mindestens ein Band auf die kurze Kette wechselt.  Aufrufe, die die
    // REJOIN-Sperre komplett abfaengt, sind kein Ereignis, das man hoert -
    // sie mitzuzaehlen blaeht die Zahl um das Fuenffache auf (124 statt 25
    // im selben Lauf) und macht die Anzeige unbrauchbar.  JE KANAL.
    unsigned armed = 0;

    for (unsigned b = 0; b < ROUTE_BANDS; b++) {
      const bool hit = allBands || rise[b] || (b > 0 && rise[b - 1])
        || (b + 1 < ROUTE_BANDS && rise[b + 1]);

      if (!hit)
        continue;

      uint8_t &tn = m_HybTent[(size_t)ch * ROUTE_BANDS + b];

      // BESTAETIGUNG: der echte Detektor macht aus vorlaeufig fest -
      // auch wenn das Band schon jung ist und nicht neu gesetzt wird.
      if (!tentative)
        tn = 0;
      if (age[b] != HYB_DORMANT && age[b] < REJOIN)
        continue;
      if (tentative)
        tn = (uint8_t)TENT_TTL;
      m_HybRetr[(size_t)ch * ROUTE_BANDS + b] = 0;
      age[b] = 0;
      m_HybAny[ch] = 1;
      armed++;

      // Sparbetrieb: der frische cap-Satz beginnt bei der Schaltung.
      if (m_HybWarmOrder.load(std::memory_order_relaxed) > 0
          && !m_HybChooser.load(std::memory_order_relaxed)) {
        m_HybTupB[(size_t)ch * ROUTE_BANDS + b] = 0;
        for (unsigned bin = b << 1;
             bin < std::min(HYB_NBINS, (b + 1u) << 1); bin++) {
          std::fill_n(
            m_HybP2.begin() + ((size_t)ch * HYB_NBINS + bin) * pStride,
            pStride, 0.0);
          std::fill_n(
            m_HybQ2.begin() + ((size_t)ch * HYB_NBINS + bin) * qStride,
            qStride, 0.0);
        }
      }
      if (!m_HybReset.load(std::memory_order_relaxed))
        continue; // nur Maske schalten - beide Ketten laufen ungestoert

      m_HybTupA[(size_t)ch * ROUTE_BANDS + b] = 0;
      m_HybTupB[(size_t)ch * ROUTE_BANDS + b] = 0;

      // Kurze Kette: Bins b*2 .. b*2+1
      for (unsigned bin = b << 1; bin < std::min(HYB_NBINS, (b + 1u) << 1);
           bin++) {
        std::fill_n(m_HybP.begin() + ((size_t)ch * HYB_NBINS + bin) * pStride,
                    pStride, 0.0);
        std::fill_n(m_HybQ.begin() + ((size_t)ch * HYB_NBINS + bin) * qStride,
                    qStride, 0.0);
        for (unsigned k = 0; k < AR_ORDER_MAX; k++) {
          m_HybRootRe[((size_t)ch * HYB_NBINS + bin) * AR_ORDER_MAX + k] = 0.0f;
          m_HybRootIm[((size_t)ch * HYB_NBINS + bin) * AR_ORDER_MAX + k] = 0.0f;
        }
      }
      // Lange Kette: Bins des Bandes bei deren Aufloesung.  Die alten
      // KOEFFIZIENTEN bleiben absichtlich stehen - sie spielen die
      // 2-3 Rampenperioden, bis die Maske drueben ist; danach schreibt
      // die Leiter frische.
      for (unsigned bin = b << shiftA;
           bin < std::min(nbinsA, (b + 1u) << shiftA); bin++) {
        ClearBin(ch, bin);
        for (unsigned k = 0; k < AR_ORDER_MAX; k++) {
          m_ArRootRe[((size_t)ch * NBINS_MAX + bin) * AR_ORDER_MAX + k] = 0.0f;
          m_ArRootIm[((size_t)ch * NBINS_MAX + bin) * AR_ORDER_MAX + k] = 0.0f;
        }
      }
    }
    if (!armed)
      return;
    if (tentative) {
      s_HybOnsetsTent.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    const uint32_t slot
      = s_OnsetSeq.fetch_add(1, std::memory_order_relaxed) % ONSET_LOG;

    s_HybOnsets.fetch_add(1, std::memory_order_relaxed);
    if (ch < DET_CH_MAX)
      s_OnsetPerCh[ch].fetch_add(1, std::memory_order_relaxed);
    s_OnsetLog[slot].store(
      RoundPeriod() * m_SamplesPerBuffer, std::memory_order_relaxed);
  }

  // Junge Baender der LANGEN Kette: jede Periode frisch loesen, Ordnung
  // nach Alter (Leiter auf dem frischen Akkumulator, Unterblock-Solve).
  void HybridSolveLongYoung(
    unsigned ch, unsigned P, unsigned arStride, unsigned bandShift,
    unsigned nbins) {
    double gP[AR_ORDER_MAX * AR_ORDER_MAX * 2], gQ[AR_ORDER_MAX * 2];
    float *cRe = m_ArCoefRe.data() + (size_t)ch * NBINS_MAX * AR_ORDER_MAX;
    float *cIm = m_ArCoefIm.data() + (size_t)ch * NBINS_MAX * AR_ORDER_MAX;
    float *rRe = m_ArRootRe.data() + (size_t)ch * NBINS_MAX * AR_ORDER_MAX;
    float *rIm = m_ArRootIm.data() + (size_t)ch * NBINS_MAX * AR_ORDER_MAX;
    const uint16_t *age = m_HybAge.data() + (size_t)ch * ROUTE_BANDS;

    for (unsigned b = 0; b < ROUTE_BANDS; b++) {
      if (age[b] == HYB_DORMANT)
        continue;

      const unsigned g = m_HybTupA[(size_t)ch * ROUTE_BANDS + b];
      const unsigned p = std::min(P, g >= 3 ? g - 2 : 0);
      (void)arStride;

      if (p < 1)
        continue; // Maske haelt das Band ohnehin bei der kurzen Kette

      for (unsigned bin = b << bandShift;
           bin < std::min(nbins, (b + 1u) << bandShift); bin++) {
        GatherBin(ch, bin, p, gP, gQ);
        SolveBin(
          gP, gQ, p,
          cRe + (size_t)bin * AR_ORDER_MAX, cIm + (size_t)bin * AR_ORDER_MAX,
          rRe + (size_t)bin * AR_ORDER_MAX, rIm + (size_t)bin * AR_ORDER_MAX);
        // Schwanz nullen: der Vorlauf laeuft einheitlich bei P, und ein
        // AR(p) IST ein AR(P) mit Nullen dahinter.
        for (unsigned k = p; k < P; k++) {
          cRe[(size_t)bin * AR_ORDER_MAX + k] = 0.0f;
          cIm[(size_t)bin * AR_ORDER_MAX + k] = 0.0f;
        }
      }
    }
  }

  // Synthese der kurzen Kette: nur junge Baender, mit w gewichtet, per
  // eigener Ueberlappungsaddition in den fertigen Schatten addiert.
  // pShadowM1: Hindcast-Sammelpunkt der NAHT (XO_V0_SHADOW) - die kurze
  // Kette traegt ihr Sample vor der Grenze bei, sonst misst der Anker den
  // Versatz nur gegen die lange Kette (gemessen: 0,5 dB Verlust, die
  // geschalteten Baender sitzen genau an den Einsaetzen).  nullptr = aus.
  void HybridSynth(
    unsigned ch, uint64_t nowP, float *pShadow, float *ola,
    float *pShadowM1) {
    const PvKit &kit = m_Pv[3];
    const unsigned synHop = HYB_NFFT / 4;
    const unsigned steps = synHop / m_SamplesPerBuffer;
    const int capRaw = m_HybShortCap.load(std::memory_order_relaxed);
    const unsigned cap = (unsigned)std::min(
      (int)AR_ORDER_MAX, std::max(1, capRaw > 0 ? capRaw : 6));
    float *pTime = m_TimeBuf + (size_t)ch * NFFT_MAX;
    fftwf_complex *pFreq = m_FreqBuf + (size_t)ch * FREQ_STRIDE;
    const uint16_t *age = m_HybAge.data() + (size_t)ch * ROUTE_BANDS;
    const float *w = m_HybW.data() + (size_t)ch * ROUTE_BANDS;
    // Wiederverwendeter Scratch: die lange Kette ist mit Zustand und
    // Rotatoren dieser Periode fertig, bevor wir hier ankommen.
    float *stRe = m_ArStRe.data() + (size_t)ch * AR_ORDER_MAX * NBINS_MAX;
    float *stIm = m_ArStIm.data() + (size_t)ch * AR_ORDER_MAX * NBINS_MAX;
    float *rotRe = m_RotRe.data() + (size_t)ch * NBINS_MAX;
    float *rotIm = m_RotIm.data() + (size_t)ch * NBINS_MAX;
    const size_t pStride = (size_t)AR_ORDER_MAX * AR_ORDER_MAX * 2;
    const size_t qStride = (size_t)AR_ORDER_MAX * 2;
    unsigned pOfBand[ROUTE_BANDS];
    float hybM1 = 0.0f, hybM1W = 0.0f; // Hindcast der kurzen Kette (t = -1)

    // 1) Loesen (Leiter) + Rotator-Rueckfall + Zustand einsammeln
    for (unsigned b = 0; b < ROUTE_BANDS; b++) {
      pOfBand[b] = 0;
      if (age[b] == HYB_DORMANT || w[b] <= 0.5f)
        continue;

      const int woRaw = m_HybWarmOrder.load(std::memory_order_relaxed);
      const unsigned warmOrd
        = woRaw > 0 ? std::min(cap, (unsigned)woRaw) : cap;
      // warmOrd == cap MUSS Vollbetrieb sein: der frische Satz wird
      // dann nie befuellt (die Akkumulation ueberspringt ihn korrekt),
      // aber der Tupelzaehler lief weiter - nach ~24 ms Jung-Zeit
      // loeste die Synthese aus dem LEEREN Akkumulator: Muell in den
      // geschalteten Baendern.  Deshalb klang 'warm@6' SCHLECHTER als
      // 'warm@0', obwohl beide dasselbe meinen (Nutzer-Hoerbefund
      // 08.08.; die Messreihe hatte 6 nie getestet).
      const bool spar = woRaw > 0 && warmOrd < cap
        && !m_HybChooser.load(std::memory_order_relaxed);
      const unsigned tupF = m_HybTupB[(size_t)ch * ROUTE_BANDS + b];
      // Sparbetrieb: bis der frische Satz mehr traegt als der
      // Warmstart, loest die Warm-Ordnung; danach klettert die Leiter
      // auf dem frischen Satz (p ab p+2 Tupeln).
      const bool fresh = spar && tupF >= warmOrd + 3;
      const unsigned p = spar
        ? (fresh ? std::min(cap, tupF - 2) : warmOrd)
        : std::min(cap,
                   m_HybReset.load(std::memory_order_relaxed)
                     ? (tupF >= 3 ? tupF - 2 : 0)
                     : cap);
      const unsigned lo = b << 1;
      const unsigned hi = std::min(HYB_NBINS, (b + 1u) << 1);

      pOfBand[b] = p;
      for (unsigned bin = lo; bin < hi; bin++) {
        if (p >= 1) {
          float *cR = m_HybCoefRe.data()
            + ((size_t)ch * HYB_NBINS + bin) * AR_ORDER_MAX;
          float *cI = m_HybCoefIm.data()
            + ((size_t)ch * HYB_NBINS + bin) * AR_ORDER_MAX;
          const std::vector<double> &sP = fresh ? m_HybP2 : m_HybP;
          const std::vector<double> &sQ = fresh ? m_HybQ2 : m_HybQ;

          SolveBin(
            sP.data() + ((size_t)ch * HYB_NBINS + bin) * pStride,
            sQ.data() + ((size_t)ch * HYB_NBINS + bin) * qStride, p,
            cR, cI,
            m_HybRootRe.data() + ((size_t)ch * HYB_NBINS + bin) * AR_ORDER_MAX,
            m_HybRootIm.data()
              + ((size_t)ch * HYB_NBINS + bin) * AR_ORDER_MAX);
          for (unsigned k = p; k < cap; k++) {
            cR[k] = 0.0f;
            cI[k] = 0.0f;
          }
        } else {
          // Leiter traegt noch nicht: Rotator aus den letzten zwei
          // Ringrahmen (Baseline eine Periode).
          const unsigned s0 = (unsigned)(nowP % HYB_RING);
          const unsigned s1 = (unsigned)((nowP - 1) % HYB_RING);
          float rr = 1.0f, ri = 0.0f;

          if (nowP >= 1
              && m_HybRingPeriod[(size_t)ch * HYB_RING + s0] == nowP
              && m_HybRingPeriod[(size_t)ch * HYB_RING + s1] == nowP - 1) {
            const size_t o0 = ((size_t)ch * HYB_RING + s0) * HYB_NBINS + bin;
            const size_t o1 = ((size_t)ch * HYB_RING + s1) * HYB_NBINS + bin;
            const float ar = m_HybRingRe[o0], ai = m_HybRingIm[o0];
            const float br = m_HybRingRe[o1], bi = m_HybRingIm[o1];
            float xr = ar * br + ai * bi, xi = ai * br - ar * bi;
            const float mm = sqrtf(xr * xr + xi * xi);

            if (mm > 1e-30f) {
              rr = xr / mm;
              ri = xi / mm;
            }
          }
          rotRe[bin] = rr;
          rotIm[bin] = ri;
        }
        // Zustand: st[k] = Ring[now - k], Loecher = 0
        for (unsigned k = 0; k < cap; k++) {
          float vr = 0.0f, vi = 0.0f;

          if (nowP >= k) {
            const unsigned ks = (unsigned)((nowP - k) % HYB_RING);

            if (m_HybRingPeriod[(size_t)ch * HYB_RING + ks] == nowP - k) {
              const size_t o = ((size_t)ch * HYB_RING + ks) * HYB_NBINS + bin;

              vr = m_HybRingRe[o];
              vi = m_HybRingIm[o];
            }
          }
          stRe[(size_t)k * NBINS_MAX + bin] = vr;
          stIm[(size_t)k * NBINS_MAX + bin] = vi;
        }
      }
    }

    // 2) Rahmen-Schleife: Vorlauf, Maske w, iFFT, Ueberlappungsaddition
    const unsigned kEnd = (m_ShadowLen + HYB_NFFT + synHop - 1) / synHop;
    const float *win = kit.win.data();

    std::fill(ola, ola + m_ShadowLen, 0.0f);
    for (unsigned k = 1; k < kEnd; k++) {
      for (unsigned s = 0; s < steps; s++)
        for (unsigned b = 0; b < ROUTE_BANDS; b++) {
          if (age[b] == HYB_DORMANT || w[b] <= 0.5f)
            continue;

          const unsigned lo = b << 1;
          const unsigned hi = std::min(HYB_NBINS, (b + 1u) << 1);

          for (unsigned bin = lo; bin < hi; bin++) {
            if (pOfBand[b] >= 1) {
              const float *cR = m_HybCoefRe.data()
                + ((size_t)ch * HYB_NBINS + bin) * AR_ORDER_MAX;
              const float *cI = m_HybCoefIm.data()
                + ((size_t)ch * HYB_NBINS + bin) * AR_ORDER_MAX;
              float xr = 0.0f, xi = 0.0f;

              for (unsigned j = 0; j < cap; j++) {
                const float sr = stRe[(size_t)j * NBINS_MAX + bin];
                const float si = stIm[(size_t)j * NBINS_MAX + bin];

                xr += cR[j] * sr - cI[j] * si;
                xi += cR[j] * si + cI[j] * sr;
              }
              for (unsigned j = cap - 1; j >= 1; j--) {
                stRe[(size_t)j * NBINS_MAX + bin]
                  = stRe[(size_t)(j - 1) * NBINS_MAX + bin];
                stIm[(size_t)j * NBINS_MAX + bin]
                  = stIm[(size_t)(j - 1) * NBINS_MAX + bin];
              }
              stRe[bin] = xr;
              stIm[bin] = xi;
            } else {
              const float sr = stRe[bin], si = stIm[bin];

              stRe[bin] = sr * rotRe[bin] - si * rotIm[bin];
              stIm[bin] = sr * rotIm[bin] + si * rotRe[bin];
            }
          }
        }
      for (unsigned bin = 0; bin < HYB_NBINS; bin++) {
        pFreq[bin][0] = 0.0f;
        pFreq[bin][1] = 0.0f;
      }
      for (unsigned b = 0; b < ROUTE_BANDS; b++) {
        if (age[b] == HYB_DORMANT || w[b] <= 0.5f)
          continue; // binaer - siehe Maske der langen Kette
        for (unsigned bin = b << 1;
             bin < std::min(HYB_NBINS, (b + 1u) << 1); bin++) {
          pFreq[bin][0] = stRe[bin];
          pFreq[bin][1] = stIm[bin];
        }
      }
      fftwf_execute_dft_c2r(kit.inv, pFreq, pTime);

      const int start = (int)(k * synHop) - (int)HYB_NFFT;
      const unsigned a0 = start < 0 ? 0 : (unsigned)start;
      const unsigned a1
        = std::min(m_ShadowLen, (unsigned)(start + (int)HYB_NFFT));
      const float scale = 1.0f / HYB_NFFT;

      if (pShadowM1 && start <= -1) {
        const unsigned j = (unsigned)(-1 - start);

        hybM1 += pTime[j] * win[j] * scale;
        hybM1W += win[j] * win[j];
      }

      for (unsigned t = a0; t < a1; t++)
        ola[t] += pTime[t - start] * win[t - start] * scale;
    }
    for (unsigned i = 0; i < m_ShadowLen; i++)
      pShadow[i] += ola[i] * kit.olaRecip[i];
    // Beitrag der kurzen Kette zum Hindcast - getrennt normiert, exakt
    // wie ihr Beitrag zu pShadow selbst.
    if (pShadowM1 && hybM1W > 1e-9f)
      *pShadowM1 += hybM1 / hybM1W;
  }
  // Degradation envelope over CONSECUTIVE misses, measured in SAMPLES -
  // the prediction's validity is a property of elapsed time, not of how
  // many device periods happen to slice it (3 periods were 8 ms at 128
  // frames but 32 ms at 512).  Full level for ~25 ms (the freeze's
  // audibility horizon), then a generous cosine fade over ~150 ms, long
  // enough that a returning real signal can cross-fade back in from a
  // still-audible level at any point of the decline.
  const unsigned m_DegradeHold; // samples at full level
  const unsigned m_DegradeFade; // samples of the fade that follows
  uint64_t m_MissedSamples;     // concealed so far in the current streak
  bool m_ReentryPending;
  bool m_ReentrySilent;
  uint64_t m_Rng;

  double NextRand(); // xorshift64*, uniform [0,1)
  bool FitBurg(unsigned ch, const float *hist, unsigned n);
  void ComputeChannelShadow(unsigned ch, float *pShadow, const PvKit &kit);
  void PredictChannel(unsigned ch);
  void PredictEnd();
  void PredictAbort();
  unsigned m_PendingSlot = 0;
  uint64_t m_PendingPeriod = 0;
  // One recursion step for every bin, with the order known at COMPILE
  // time: a runtime trip count stopped the vectoriser and cost 25 % of
  // the whole prediction.  State layout is [k][bin] (structure of
  // arrays) so the bin loop is a straight stride-1 sweep.
  // Auch planar - sonst haetten die Ordnungen jenseits der Schablonen
  // stillschweigend die falsche Ablage gelesen.  Genau diese Sorte
  // Halb-Umstellung war hier schon einmal ein -26 dB teurer Fehler.
  static void AdvanceBinsGeneric(
    unsigned P, unsigned nbins, const float *pCoefRe, const float *pCoefIm,
    float *pStRe, float *pStIm) {
    for (unsigned bin = 0; bin < nbins; bin++) {
      float re = 0.0f, im = 0.0f;

      for (unsigned k = 0; k < P; k++) {
        const float sr_ = pStRe[(size_t)k * NBINS_MAX + bin];
        const float si_ = pStIm[(size_t)k * NBINS_MAX + bin];
        const float ar_ = pCoefRe[(size_t)k * NBINS_MAX + bin];
        const float ai_ = pCoefIm[(size_t)k * NBINS_MAX + bin];

        re += ar_ * sr_ - ai_ * si_;
        im += ar_ * si_ + ai_ * sr_;
      }
      for (unsigned k = P - 1; k > 0; k--) {
        pStRe[(size_t)k * NBINS_MAX + bin]
          = pStRe[(size_t)(k - 1) * NBINS_MAX + bin];
        pStIm[(size_t)k * NBINS_MAX + bin]
          = pStIm[(size_t)(k - 1) * NBINS_MAX + bin];
      }
      pStRe[bin] = re;
      pStIm[bin] = im;
    }
  }

  // Planare Fassung: pCoefRe/Im sind [k][bin], genau wie der Zustand.
  // Damit sind ALLE Zugriffe der Bin-Schleife zusammenhaengend, und der
  // Uebersetzer packt acht Bins in ein AVX2-Register.  Die alte Fassung
  // (AdvanceBinsOld, steht als Referenz daneben) las die Koeffizienten mit
  // Schrittweite AR_ORDER_MAX je Bin - ein Gather, den keine
  // Vektorisierung ueberlebt.
  //
  // accRe/accIm sind EIGENER Kratzspeicher, nicht etwa freie Plaetze im
  // Zustandsfeld: das ist exakt AR_ORDER_MAX * NBINS_MAX gross, bei
  // Ordnung == AR_ORDER_MAX gibt es dort keinen Platz mehr.
  template <unsigned P>
  static void AdvanceBins(
    unsigned nbins, const float *pCoefRe, const float *pCoefIm, float *pStRe,
    float *pStIm, float *accRe, float *accIm) {
    for (unsigned k = 0; k < P; k++) {
      const float *aRe = pCoefRe + (size_t)k * NBINS_MAX;
      const float *aIm = pCoefIm + (size_t)k * NBINS_MAX;
      const float *sRe = pStRe + (size_t)k * NBINS_MAX;
      const float *sIm = pStIm + (size_t)k * NBINS_MAX;

#ifdef XO_AVX2
      if (XoAvx2()) {
        if (k == 0)
          XoAdvFirst(nbins, aRe, aIm, sRe, sIm, accRe, accIm);
        else
          XoAdvAcc(nbins, aRe, aIm, sRe, sIm, accRe, accIm);
        continue;
      }
#endif
      if (k == 0)
        for (unsigned bin = 0; bin < nbins; bin++) {
          accRe[bin] = aRe[bin] * sRe[bin] - aIm[bin] * sIm[bin];
          accIm[bin] = aRe[bin] * sIm[bin] + aIm[bin] * sRe[bin];
        }
      else
        for (unsigned bin = 0; bin < nbins; bin++) {
          accRe[bin] += aRe[bin] * sRe[bin] - aIm[bin] * sIm[bin];
          accIm[bin] += aRe[bin] * sIm[bin] + aIm[bin] * sRe[bin];
        }
    }
    // Zustand um einen Schritt schieben, dann das frische Ergebnis nach
    // vorn.  Zusammenhaengende Bloecke - memmove-Geschwindigkeit statt
    // Einzelwerten mit Schrittweite NBINS_MAX.
    for (unsigned k = P - 1; k > 0; k--) {
      std::memcpy(pStRe + (size_t)k * NBINS_MAX,
                  pStRe + (size_t)(k - 1) * NBINS_MAX,
                  sizeof(float) * nbins);
      std::memcpy(pStIm + (size_t)k * NBINS_MAX,
                  pStIm + (size_t)(k - 1) * NBINS_MAX,
                  sizeof(float) * nbins);
    }
    std::memcpy(pStRe, accRe, sizeof(float) * nbins);
    std::memcpy(pStIm, accIm, sizeof(float) * nbins);
  }

  template <unsigned P>
  static void AdvanceBinsOld(
    unsigned nbins, const float *pCoefRe, const float *pCoefIm, float *pStRe,
    float *pStIm) {
    for (unsigned bin = 0; bin < nbins; bin++) {
      const float *aRe = pCoefRe + (size_t)bin * AR_ORDER_MAX;
      const float *aIm = pCoefIm + (size_t)bin * AR_ORDER_MAX;
      float re = 0.0f, im = 0.0f;

      for (unsigned k = 0; k < P; k++) {
        const float sr_ = pStRe[(size_t)k * NBINS_MAX + bin];
        const float si_ = pStIm[(size_t)k * NBINS_MAX + bin];

        re += aRe[k] * sr_ - aIm[k] * si_;
        im += aRe[k] * si_ + aIm[k] * sr_;
      }
      for (unsigned k = P - 1; k > 0; k--) {
        pStRe[(size_t)k * NBINS_MAX + bin]
          = pStRe[(size_t)(k - 1) * NBINS_MAX + bin];
        pStIm[(size_t)k * NBINS_MAX + bin]
          = pStIm[(size_t)(k - 1) * NBINS_MAX + bin];
      }
      pStRe[bin] = re;
      pStIm[bin] = im;
    }
  }

  static void SolveBin(
    const double *pP, const double *pQ, unsigned order, float *pCoefRe,
    float *pCoefIm, float *pRootRe, float *pRootIm);
};

Xorcist::Xorcist(
  unsigned nChannels,
  unsigned samplesPerBuffer,
  unsigned sampleRate,
  const char *wisdomPath)
  : m_NChannels(nChannels),
    m_SamplesPerBuffer(samplesPerBuffer),
    m_SampleRate(sampleRate),
    m_ReentryFade(std::min(
      samplesPerBuffer, std::max(16u, (unsigned)(0.003 * sampleRate)))),
    // Anchor corrector time constant: 1/3 ms (tau-sweep on a real
    // aliquot-rich tone; 8..24 samples measured equivalent at 48 kHz).
    m_AnchorLen(10 * std::max(8u, sampleRate / 3000)),
    m_AnchorDecay(exp(-3000.0 / std::max(24000u, sampleRate))),
    m_BridgeLen(std::min(
      samplesPerBuffer, std::max(16u, (unsigned)(0.002 * sampleRate)))),
    // Horizon: 32 ms of shadow (12 periods at 128/48k, minimum 2 at
    // large buffers) - deliberately BEYOND the PV freeze's ~20-25 ms
    // audibility horizon, because the tail is not meant to sound perfect
    // but to feed the sample-clock degradation fade instead of a hard
    // edge when no fresh shadow arrives for many periods.
    m_HorizonPeriods(std::max(
      2u, (unsigned)((0.032 * sampleRate + samplesPerBuffer - 1)
                     / samplesPerBuffer))),
    m_ShadowLen(m_HorizonPeriods * samplesPerBuffer + m_ReentryFade),
    m_History(nChannels * 2 * HIST, 0.0f),
    m_HistWrite(0),
    m_HistFill(0),
    m_HistPeriod(0),
    m_ShadowPub(SHADOW_INVALID),
    m_ComputeBusy(false),
    m_TimeBuf(nullptr),
    m_FreqBuf(nullptr),
    m_SpecRe((size_t)nChannels * NBINS_MAX),
    m_SpecIm((size_t)nChannels * NBINS_MAX),
    m_RotRe((size_t)nChannels * NBINS_MAX),
    m_RotIm((size_t)nChannels * NBINS_MAX),
    m_RouteSlots(1024 / samplesPerBuffer + 1),
    m_RouteBandE(
      (size_t)nChannels * (1024 / samplesPerBuffer + 1) * ROUTE_BANDS, 0.0f),
    m_RoutePeriod(
      (size_t)nChannels * (1024 / samplesPerBuffer + 1), ~0ull),
    m_RouteNfft((size_t)nChannels * (1024 / samplesPerBuffer + 1), 0),
    // Sized for the LARGEST frame's hop (8192/8 = 1024 samples); a
    // smaller frame simply uses a shorter stride inside the same ring.
    m_ArStride(std::max(1u, (NFFT_MAX / 8) / samplesPerBuffer)),
    m_ArRingLen((AR_ORDER_MAX + FIT_WINDOW_MAX) * m_ArStride + 1),
    m_ArRingRe((size_t)nChannels * m_ArRingLen * NBINS_MAX, 0.0f),
    m_ArRingIm((size_t)nChannels * m_ArRingLen * NBINS_MAX, 0.0f),
    m_ArRingPeriod((size_t)nChannels * m_ArRingLen, ~0ull),
    m_ArPRe((size_t)nChannels * AR_PAIRS * NBINS_MAX, 0.0),
    m_ArPIm((size_t)nChannels * AR_PAIRS * NBINS_MAX, 0.0),
    m_ArQRe((size_t)nChannels * AR_ORDER_MAX * NBINS_MAX, 0.0),
    m_ArQIm((size_t)nChannels * AR_ORDER_MAX * NBINS_MAX, 0.0),
    m_ArFitPeriod(nChannels, ~0ull),
    m_ArNfft(nChannels, 0),
    m_ArFitCount(nChannels, 0),
    m_ArCoefRe((size_t)nChannels * NBINS_MAX * AR_ORDER_MAX, 0.0f),
    m_ArCoefIm((size_t)nChannels * NBINS_MAX * AR_ORDER_MAX, 0.0f),
    m_ArCoefReP((size_t)nChannels * NBINS_MAX * AR_ORDER_MAX, 0.0f),
    m_ArCoefImP((size_t)nChannels * NBINS_MAX * AR_ORDER_MAX, 0.0f),
    m_AdvAccRe((size_t)nChannels * NBINS_MAX, 0.0f),
    m_AdvAccIm((size_t)nChannels * NBINS_MAX, 0.0f),
    m_ArRootRe((size_t)nChannels * NBINS_MAX * AR_ORDER_MAX, 0.0f),
    m_ArRootIm((size_t)nChannels * NBINS_MAX * AR_ORDER_MAX, 0.0f),
    m_ArSolvedPeriod(nChannels, ~0ull),
    m_ArSolvedOrder(nChannels, 0),
    m_ArStRe((size_t)nChannels * AR_ORDER_MAX * NBINS_MAX),
    m_ArStIm((size_t)nChannels * AR_ORDER_MAX * NBINS_MAX),
    m_RotKeepRe((size_t)nChannels * NBINS_MAX, 1.0f),
    m_RotKeepIm((size_t)nChannels * NBINS_MAX, 0.0f),
    m_RotKeepValid(nChannels, 0),
    m_SpecCacheRe((size_t)nChannels * NBINS_MAX, 0.0f),
    m_SpecCacheIm((size_t)nChannels * NBINS_MAX, 0.0f),
    m_SpecCachePeriod(nChannels, ~0ull),
    m_SpecCacheNfft(nChannels, 0),
    m_SpecCacheHop(nChannels, 0),
    m_Ola((size_t)nChannels * (m_ShadowLen + NFFT_MAX)),
    m_BurgF((size_t)nChannels * BURG_WINDOW),
    m_BurgB((size_t)nChannels * BURG_WINDOW),
    m_BurgF2((size_t)nChannels * BURG_WINDOW),
    m_BurgB2((size_t)nChannels * BURG_WINDOW),
    m_BurgA(nChannels * BURG_ORDER, 0.0),
    m_BurgFitPeriod(nChannels, ~0ull),
    m_BurgState((size_t)nChannels * BURG_ORDER),
    m_Bridge((size_t)nChannels * m_BridgeLen),
    m_BridgeWin(m_BridgeLen),
    m_PendingTail(nChannels * m_ReentryFade, 0.0f),
    m_ReentryMix((size_t)nChannels * samplesPerBuffer, 0.0f),
    m_PatchBuf((size_t)nChannels * PATCH_MAX, 0.0f),
    m_PatchStart(nChannels, 0),
    m_PatchEnd(nChannels, 0),
    m_OnsetPeriod(nChannels, ~0ull),
    m_StableCount(nChannels, 0),
    m_LastPatchSample(nChannels, 0),
    m_QuietRun(nChannels, 0),
    m_HybRingRe((size_t)nChannels * HYB_RING * HYB_NBINS, 0.0f),
    m_HybRingIm((size_t)nChannels * HYB_RING * HYB_NBINS, 0.0f),
    m_HybRingPeriod((size_t)nChannels * HYB_RING, ~0ull),
    m_HybP(
      (size_t)nChannels * HYB_NBINS * AR_ORDER_MAX * AR_ORDER_MAX * 2, 0.0),
    m_HybQ((size_t)nChannels * HYB_NBINS * AR_ORDER_MAX * 2, 0.0),
    m_HybP2(
      (size_t)nChannels * HYB_NBINS * AR_ORDER_MAX * AR_ORDER_MAX * 2, 0.0),
    m_HybQ2((size_t)nChannels * HYB_NBINS * AR_ORDER_MAX * 2, 0.0),
    m_HybCoefRe((size_t)nChannels * HYB_NBINS * AR_ORDER_MAX, 0.0f),
    m_HybCoefIm((size_t)nChannels * HYB_NBINS * AR_ORDER_MAX, 0.0f),
    m_HybRootRe((size_t)nChannels * HYB_NBINS * AR_ORDER_MAX, 0.0f),
    m_HybRootIm((size_t)nChannels * HYB_NBINS * AR_ORDER_MAX, 0.0f),
    m_HybAge((size_t)nChannels * ROUTE_BANDS, HYB_DORMANT),
    m_HybTupA((size_t)nChannels * ROUTE_BANDS, 0),
    m_HybTupB((size_t)nChannels * ROUTE_BANDS, 0),
    m_HybW((size_t)nChannels * ROUTE_BANDS, 0.0f),
    m_HybAny(nChannels, 0),
    m_HybEpStart((size_t)nChannels * ROUTE_BANDS, ~0ull),
    m_PreWin(PRE_NFFT),
    m_PreBandE((size_t)nChannels * PRE_RING * ROUTE_BANDS, 0.0f),
    m_PrePeriod((size_t)nChannels * PRE_RING, ~0ull),
    m_HybTent((size_t)nChannels * ROUTE_BANDS, 0),
    m_HybRetr((size_t)nChannels * ROUTE_BANDS, 0),
    m_D4Band((size_t)nChannels * 9 * ROUTE_BANDS, 0.0f),
    m_D4Per((size_t)nChannels * 9, ~0ull),
    m_D4Wide(nChannels, -1.0f),
    m_D4Rise((size_t)nChannels * ROUTE_BANDS, 0),
    m_D4Conf((size_t)nChannels * ROUTE_BANDS, 0),
    m_D4ResMax((size_t)nChannels * ROUTE_BANDS, 0.0f),
    m_PreCandP(nChannels, ~0ull),
    m_PreStable(nChannels, 0),
    m_PreEmax(nChannels, 0.0f),
    m_PreChk(nChannels, 0),
    m_PreResMax(nChannels, 0.0f),
    m_PreResMaxB((size_t)nChannels * ROUTE_BANDS, 0.0f),
    m_PreBase((size_t)nChannels * ROUTE_BANDS, 0.0f),
    m_PreRise((size_t)nChannels * ROUTE_BANDS, 0),
    m_HybPend(nChannels, 0),
    m_HybPendRise((size_t)nChannels * ROUTE_BANDS, 0),
    m_HybResA((size_t)nChannels * ROUTE_BANDS, 0.0f),
    m_HybResB((size_t)nChannels * ROUTE_BANDS, 0.0f),
    m_HybEneB((size_t)nChannels * ROUTE_BANDS, 0.0f),
    m_FadeReentry(m_ReentryFade),
    m_DegradeHold((unsigned)(0.025 * sampleRate)),
    // Die Ausblendung endet dort, wo der Schatten endet.
    //
    // Vorher standen hier 150 ms - eine Kurve, die NIE durchlaufen wurde:
    // sie braucht mit der Haltezeit 175 ms bis null, der Schatten ist aber
    // nach m_HorizonPeriods (32 ms) alle.  Am letzten brauchbaren Wert
    // stand sie bei 0,5 + 0,5*cos(pi*336/7200) = 99,5 %, und danach kam
    // hartes Nichts.  Eine ununterbrochene Aussetzerkette laenger als
    // 32 ms endete also in einem SCHNITT bei fast vollem Pegel, nicht in
    // einem Ausblenden - unbemerkt geblieben, weil die erzwungenen
    // Fehlschlaege (einzeln, paarweise, 1:1) so lange Ketten gar nicht
    // erzeugen; das passiert nur unter echter Ueberlast.
    //
    // Jetzt: was zwischen Haltezeit und Horizont uebrig ist, wird zur
    // Ausblendung - bei 128/48k also 336 Werte = 7 ms.  Sie kommt damit
    // genau dann bei null an, wenn die letzte Scheibe verbraucht ist.
    // In die letzte Scheibe allein gequetscht waeren es 2,7 ms; die
    // vollen 7 ms sind derselbe Endpunkt, nur sanfter.
    m_DegradeFade(
      m_HorizonPeriods * samplesPerBuffer > m_DegradeHold
        ? m_HorizonPeriods * samplesPerBuffer - m_DegradeHold
        : std::max(1u, samplesPerBuffer)),
    m_MissedSamples(0),
    m_ReentryPending(false),
    m_ReentrySilent(false),
    m_Rng(0x9E3779B97F4A7C15ULL ^ (uint64_t)(uintptr_t)this) {
  m_Shadow[0].assign(nChannels * m_ShadowLen, 0.0f);
  m_Shadow[1].assign(nChannels * m_ShadowLen, 0.0f);
  for (unsigned i = 0; i < m_ReentryFade; i++)
    m_FadeReentry[i] = 0.5f + 0.5f * cosf((float)M_PI * i / m_ReentryFade);
  for (unsigned i = 0; i < m_BridgeLen; i++)
    m_BridgeWin[i] = 0.5f - 0.5f * cosf((float)M_PI * i / m_BridgeLen);

  // Plans are created here on the control thread (audio stopped during
  // setup); fftwf plan creation is not thread-safe against itself and must
  // never run from the realtime path.  All kits share the max-sized
  // buffers; a smaller plan simply uses the buffer head.
  // One workspace PER CHANNEL, so the channels can be predicted
  // concurrently.  FREQ_STRIDE pads the complex buffer to a 64-byte
  // multiple: FFTW's new-array execute requires every array to have the
  // SAME alignment as the one the plan was measured on, and a stride of
  // NBINS_MAX (4097 complex = 32776 bytes) would give every second
  // channel a different one.
  m_TimeBuf
    = (float *)fftwf_malloc(sizeof(float) * NFFT_MAX * m_NChannels);
  m_FreqBuf = (fftwf_complex *)fftwf_malloc(
    sizeof(fftwf_complex) * FREQ_STRIDE * m_NChannels);
  const unsigned sizes[4] = {2048, 4096, 8192, 1024};

  // Persisted wisdom: import once per process, so only the very first run
  // on a given machine pays the measuring (~0.9 s).  A file from different
  // hardware is simply ignored by FFTW, which then re-measures.
  const bool mayLearn = wisdomPath && *wisdomPath
    && !s_WisdomTried.exchange(true, std::memory_order_relaxed);

  if (mayLearn)
    fftwf_import_wisdom_from_filename(wisdomPath);

  const auto planStart = std::chrono::steady_clock::now();

  for (unsigned k = 0; k < 4; k++) {
    PvKit &kit = m_Pv[k];

    kit.nfft = sizes[k];
    // Analysis hop = one audio period when that divides the synthesis
    // hop, which lets the next period reuse this period's frame.  The
    // rotator is then raised to the integer power synHop/anaHop (the
    // 2*pi wraps cancel exactly in the exponential, so this is not an
    // approximation).  Odd buffer sizes fall back to the classic
    // two-frame analysis.
    kit.anaHop = (kit.nfft / 4) % samplesPerBuffer == 0
        && samplesPerBuffer <= kit.nfft / 4
      ? samplesPerBuffer
      : kit.nfft / 4;
    // PERIODISCH (Nenner nfft) ist das richtige Fenster fuer die
    // Ueberlappungsaddition - es erfuellt COLA bei Hop N/4.  Der
    // Offline-Prototyp benutzt np.hanning, also SYMMETRISCH (Nenner
    // N-1), mit exakten Nullen an beiden Raendern.  Ein Sample
    // Unterschied, aber es aendert Nebenkeulendaempfung und
    // Rekonstruktion; XO_SYM_WINDOW=1 gleicht es zum Vergleich an.
    static const unsigned symWin = getenv("XO_SYM_WINDOW") ? 1u : 0u;

    kit.win.resize(kit.nfft);
    for (unsigned i = 0; i < kit.nfft; i++)
      kit.win[i] = 0.5f
      - 0.5f * cosf(2.0f * (float)M_PI * i / (float)(kit.nfft - symWin));

    // Constant OLA normalisation for this kit's frame grid (see the
    // member comment): sum of squared window contributions per output
    // sample, stored as its reciprocal so the synthesis multiplies.
    const unsigned hop = kit.nfft / 4;
    const unsigned kEnd = (m_ShadowLen + kit.nfft + hop - 1) / hop;

    kit.olaRecip.assign(m_ShadowLen, 0.0f);
    for (unsigned f = 1; f < kEnd; f++) {
      const int start = (int)(f * hop) - (int)kit.nfft;
      const unsigned a0 = start < 0 ? 0 : (unsigned)start;
      const unsigned a1
        = std::min(m_ShadowLen, (unsigned)std::max(0, start + (int)kit.nfft));

      for (unsigned t = a0; t < a1; t++) {
        const float w = kit.win[t - start];

        kit.olaRecip[t] += w * w;
      }
    }
    for (unsigned t = 0; t < m_ShadowLen; t++)
      kit.olaRecip[t] = 1.0f / std::max(kit.olaRecip[t], 1e-9f);

    // FFTW_MEASURE: 23-29% faster execution than ESTIMATE for these sizes,
    // paid once per process (~0.15 s per plan; FFTW's wisdom makes the
    // plans of any further concealment instance free).  Planning scribbles
    // over the buffers, which is harmless here - nothing is in them yet.
    kit.fwd
      = fftwf_plan_dft_r2c_1d(kit.nfft, m_TimeBuf, m_FreqBuf, FFTW_MEASURE);
    kit.inv
      = fftwf_plan_dft_c2r_1d(kit.nfft, m_FreqBuf, m_TimeBuf, FFTW_MEASURE);
  }

  for (unsigned i = 0; i < PRE_NFFT; i++)
    m_PreWin[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / PRE_NFFT);
  m_PreFwd
    = fftwf_plan_dft_r2c_1d(PRE_NFFT, m_TimeBuf, m_FreqBuf, FFTW_MEASURE);

  // Write the wisdom back only when the planner really had to measure -
  // with a complete wisdom file the loop above finishes in microseconds.
  if (
    mayLearn
    && std::chrono::duration_cast<std::chrono::milliseconds>(
         std::chrono::steady_clock::now() - planStart)
         .count()
      > 50)
    fftwf_export_wisdom_to_filename(wisdomPath);
}

Xorcist::~Xorcist() {
  // k < 4: der 1k-Kit (Index 3) leckte seit seinem Einbau.
  for (unsigned k = 0; k < 4; k++) {
    if (m_Pv[k].fwd)
      fftwf_destroy_plan(m_Pv[k].fwd);
    if (m_Pv[k].inv)
      fftwf_destroy_plan(m_Pv[k].inv);
  }
  if (m_PreFwd)
    fftwf_destroy_plan(m_PreFwd);
  if (m_TimeBuf)
    fftwf_free(m_TimeBuf);
  if (m_FreqBuf)
    fftwf_free(m_FreqBuf);
}

double Xorcist::NextRand() {
  m_Rng ^= m_Rng >> 12;
  m_Rng ^= m_Rng << 25;
  m_Rng ^= m_Rng >> 27;
  return (double)((m_Rng * 2685821657736338717ULL) >> 11)
    * (1.0 / 9007199254740992.0);
}

void Xorcist::Push(const float *pInterleaved) {
  // ---- Wiedereinstiegs-Fade AUCH in die Historie -----------------------
  // Bis 06.08.2026 bekam der Ring nach einer Luecke die ROHE echte
  // Periode, waehrend der Ausgang die ueberblendete Fassung hoerte.  Im
  // Ring stand damit an jedem Kettenende ein harter Spleiss
  // Vorhersage -> Wahrheit, dessen Sprunghoehe der aufgelaufene
  // Vorhersagefehler ist.  Ein Sprung ist breitbandig, das
  // Analysefenster 32 Perioden lang - jeder Spleiss verseuchte also 32
  // folgende Analysen bis hinauf zu 24 kHz.  Das waren die vertikalen
  // Striche im Spektrogramm und der als "Aufbau-Anteil" bezifferte
  // Posten von 2,5 bis 3,3 dB.
  //
  // Der Ausgang aendert sich dadurch NICHT - nur die Historie wird
  // ehrlich und zeigt dem naechsten Fit, was tatsaechlich gespielt wurde.
  if (m_RingBlendPending) {
    m_RingBlendPending = false;
    if (m_FadeInRing.load(std::memory_order_relaxed)) {
      const unsigned n = std::min(ReentryLen(), m_SamplesPerBuffer);
      float *pTmp = m_ReentryMix.data();

      std::memcpy(
        pTmp, pInterleaved,
        sizeof(float) * m_NChannels * m_SamplesPerBuffer);
      for (unsigned i = 0; i < n; i++) {
        const float fo = ReentryCurve(i, n); // 1 -> 0

        for (unsigned ch = 0; ch < m_NChannels; ch++) {
          float v = pTmp[i * m_NChannels + ch] * (1.0f - fo);

          if (!m_ReentrySilent)
            v += m_PendingTail[ch * m_ReentryFade + i] * fo;
          pTmp[i * m_NChannels + ch] = v;
        }
      }
      pInterleaved = pTmp;
    }
  }
  // Real audio by default; Conceal() overwrites the bit after its
  // self-feed, ReplaceLast() clears it again.
  m_RecentPredMask.store(
    m_RecentPredMask.load(std::memory_order_relaxed) << 1,
    std::memory_order_relaxed);
  // Mirrored ring (see m_History): write each new sample twice, once at w
  // and once at w+HIST, so every reader gets a contiguous window.  Only
  // the new period is touched - no shifting of the whole history.
  const unsigned buf = std::min(m_SamplesPerBuffer, HIST);
  const unsigned skip = m_SamplesPerBuffer - buf; // >= HIST is degenerate
  const unsigned w0 = m_HistWrite.load(std::memory_order_relaxed);
  unsigned w = w0;

  for (unsigned ch = 0; ch < m_NChannels; ch++) {
    float *pRing = m_History.data() + ch * 2 * HIST;
    const float *pSrc = pInterleaved + skip * m_NChannels + ch;

    w = w0;
    for (unsigned i = 0; i < buf; i++) {
      const float v = pSrc[i * m_NChannels];

      pRing[w] = v;
      pRing[w + HIST] = v;
      if (++w == HIST)
        w = 0;
    }
  }
  // Erst die Abtastwerte, dann der Kopf: ein Leser ohne eingefrorenen Kopf
  // (Absorb, Patch-Pfad) darf nie einen Kopf sehen, hinter dem noch keine
  // Daten stehen.
  m_HistWrite.store(w, std::memory_order_release);
  m_HistFill.store(
    std::min(HIST, m_HistFill.load(std::memory_order_relaxed) + buf),
    std::memory_order_relaxed);
  m_HistPeriod.fetch_add(1, std::memory_order_release);
}

bool Xorcist::FitBurg(unsigned ch, const float *hist, unsigned n) {
  const unsigned order = s_V0Order && s_V0Order < BURG_ORDER
    ? s_V0Order
    : BURG_ORDER;

  if (n < 2 * order + 8)
    return false;

  // Per-channel slices: FitBurg runs inside the per-channel prediction,
  // so a shared workspace would let concurrent channels overwrite each
  // other's forward/backward errors.
  double *f = m_BurgF.data() + (size_t)ch * BURG_WINDOW;
  double *b = m_BurgB.data() + (size_t)ch * BURG_WINDOW;
  double *f2 = m_BurgF2.data() + (size_t)ch * BURG_WINDOW;
  double *b2 = m_BurgB2.data() + (size_t)ch * BURG_WINDOW;
  double *a = m_BurgA.data() + ch * BURG_ORDER;
  unsigned len = n - 1;

  for (unsigned i = 0; i < len; i++) {
    f[i] = hist[i + 1];
    b[i] = hist[i];
  }
  // Volle Breite nullen: bei geklemmter Ordnung (XO_V0_ORDER) lesen die
  // Verbraucher weiterhin alle BURG_ORDER Taps - Altwerte oben waeren Gift.
  std::fill(a, a + BURG_ORDER, 0.0);

  for (unsigned m = 0; m < order; m++) {
    double den = 0.0, num = 0.0;

    for (unsigned i = 0; i < len; i++) {
      den += f[i] * f[i] + b[i] * b[i];
      num += f[i] * b[i];
    }
    if (den <= 0.0)
      return false;

    const double k = 2.0 * num / den;

    // lattice -> direct form: a_new[m]=k, a_new[j]=a[j]-k*a[m-1-j]
    for (unsigned j = 0; j < m / 2 + (m % 2); j++) {
      const double aj = a[j], amj = a[m - 1 - j];

      a[j] = aj - k * amj;
      if (j != m - 1 - j)
        a[m - 1 - j] = amj - k * aj;
    }
    a[m] = k;

    // f, b = f[1:]-k*b[1:], b[:-1]-k*f[:-1] - written carry-free into the
    // spare pair (then pointer-swapped) so the compiler can vectorize.
    for (unsigned i = 0; i + 1 < len; i++)
      f2[i] = f[i + 1] - k * b[i + 1];
    for (unsigned i = 0; i + 1 < len; i++)
      b2[i] = b[i] - k * f[i];
    std::swap(f, f2);
    std::swap(b, b2);
    len--;
  }
  return true;
}

// Solve one bin's order-p normal equations and stabilise the result.
//
// The system is  sum_l a_l P[l][k] = q[k]  (P[k][l] = <X_k, X_l>), solved
// by dense Gaussian elimination with partial pivoting - p is at most 4,
// so nothing cleverer is warranted.
//
// The roots of  z^p - a1 z^(p-1) - ... - a_p  are then found by
// Durand-Kerner and only those ALREADY near the unit circle are snapped
// onto it; the fit's shadow roots must stay inside and die over the
// horizon.  Forcing all of them onto the circle installs permanent
// parasitic oscillations, and dropping the clamp entirely costs ~3 dB on
// sustained material (measured at order 4: -21.3 vs -24.2 dB on a held
// tone).  A per-bin level ceiling, which would have been nearly free,
// was measured as an alternative and is 7-9 dB worse - it lets an
// unstable root ride at the ceiling instead of dying.
// Root-free stabilisation via the reflection coefficients: Schur descent
// alpha -> k, box condition |k| < 1, Levinson ascent k -> alpha.  Both
// directions are FINITE O(p^2) sequences, so unlike the root finder there
// is no iteration whose fixed point could move to a neighbouring root -
// which is what made Durand-Kerner the only float-fragile stage (174.6
// degrees of angular deviation where elimination and storage sat at
// 0.0000).  Measured on an RTX 3070 it is also 4.8x to 8.4x faster.
//
// Two traps, both of which bit on the first attempt:
//  - 1 - |k|^2 is NEGATIVE for |k| > 1, i.e. exactly the case the clamp
//    exists to repair.  Only |k| ~ 1 needs a floor, and it needs one
//    because a root ON the circle is the normal state of a sustained
//    tone, not an edge case.
//  - The descent must not write alpha in place: step i reads alpha_{l-i},
//    which for i > l/2 has already been replaced.
//
// NOT equivalent to the selective root clamp: this moves every mode that
// depends on a violated k_l, not just the unstable one.
static void SchurClampUnit(double *ar, double *ai, unsigned order,
                           double delta);

// rho: the radius the box condition is tested on (derivation §6.5).
// Scaling a_i by rho^-i maps every root z to z/rho, so the test then
// reads "all roots inside radius rho"; clamping there and scaling back
// guarantees exactly that.  Over the 12-period horizon rho = 1.005
// allows at most ~13 % growth, which the safety cap bounds anyway.
//
// RHO = 1.0 IS NOT AN OPERATING POINT, IT IS THE SINGULARITY.  A
// stationary tone has its roots exactly ON the unit circle; such a
// polynomial is self-inversive, and for those |k_p| = 1 holds exactly,
// not approximately (derivation §6.4.7).  At rho = 1.0 the very first
// descent step therefore hits the clamp on every sustained bin, and
// what comes back has its angles destroyed - measured against roots
// placed at known angles, tools/concealment/schur_rho_test.c:
//
//   order    rho=1.000   rho=1.001   rho=1.002   rho=1.005
//       4    179.88 deg      0.000       0.000       0.000
//       8    165.14 deg      0.000       0.000       0.000
//      12    179.88 deg      0.000       0.000       0.000
//
// 180 degrees is an inverted frequency, and the earlier reading of
// "11-32 dB worse than the selective root clamp" was taken at rho = 1.0
// and therefore measured the singularity rather than the method.
//
// But a thousandth does NOT fix it on real audio, and the synthetic
// table above is exactly why: it places the roots exactly on the
// circle, so an infinitesimal rho suffices to free them.  Real bins are
// scattered, most of them further outside than one thousandth, and rho
// = 1.001 exempts a null set.  Measured on the D major chord, order 8,
// anchor seam, no trend, 619 gaps at 600/min (NMR+, lower is better):
//
//   selective root clamp  -5.32     rho=1.005  -0.42
//   rho=1.000             +0.87     rho=1.008  -1.08
//   rho=1.001             +1.41     rho=1.020  -3.11
//   rho=1.002             +0.86
//
// Monotone in rho all the way out, with no optimum in between - the
// clamp firing at all is what costs, which is precisely why the
// selective root clamp wins: it only touches |z| > 0.9 and leaves
// everything else alone.  SCHUR_RHO_MIN removes the singularity; it
// does not make Schur competitive, and nothing below rho ~ 1.02 does.
//
// Clamping k BEFORE the descent instead of after was tried (it would
// bound the denominator by 2*delta and retire the floor below); it
// lifts rho = 1.0 from 180 to 132-162 degrees, which is no less
// useless.  The clamp itself destroys the information, not the
// division - so the floor stays as the NaN guard it always was, and
// the fix is to keep rho off 1.0.
static constexpr double SCHUR_RHO_MIN = 1.001;

static void SchurClamp(double *ar, double *ai, unsigned order,
                       double delta, double rho) {
  if (rho < SCHUR_RHO_MIN)
    rho = SCHUR_RHO_MIN;
  {
    double f = 1.0;

    for (unsigned i = 0; i < order; i++) {
      f /= rho;
      ar[i] *= f;
      ai[i] *= f;
    }
  }
  SchurClampUnit(ar, ai, order, delta);
  {
    double f = 1.0;

    for (unsigned i = 0; i < order; i++) {
      f *= rho;
      ar[i] *= f;
      ai[i] *= f;
    }
  }
}

static void SchurClampUnit(double *ar, double *ai, unsigned order,
                           double delta) {
  double kr[32], ki[32], ur[33], ui[33], nr[33], ni[33];

  if (order > 31)
    return;
  ur[0] = 1.0;
  ui[0] = 0.0;
  for (unsigned i = 1; i <= order; i++) {
    ur[i] = -ar[i - 1];
    ui[i] = -ai[i - 1];
  }
  for (unsigned l = order; l >= 1; l--) {
    kr[l - 1] = ur[l];
    ki[l - 1] = ui[l];

    double den = 1.0 - (kr[l - 1] * kr[l - 1] + ki[l - 1] * ki[l - 1]);

    // FLOOR the denominator, do not bail: |k| ~ 1 is the NORMAL state of
    // a sustained tone (root on the circle), and bailing out returned
    // exactly those bins UNCHANGED - the ones that needed clamping most.
    // Measured, that turned Schur into "no clamp at all" on stationary
    // material: -25.5 -> -6.3 dB on the chord.  The handover said
    // "Untergrenze noetig"; an early return is not a floor.
    if (std::fabs(den) < 1e-6)
      den = den < 0.0 ? -1e-6 : 1e-6;
    for (unsigned i = 1; i < l; i++) {
      nr[i] = (ur[i] - (kr[l - 1] * ur[l - i] + ki[l - 1] * ui[l - i])) / den;
      ni[i] = (ui[i] - (ki[l - 1] * ur[l - i] - kr[l - 1] * ui[l - i])) / den;
    }
    for (unsigned i = 1; i < l; i++) {
      ur[i] = nr[i];
      ui[i] = ni[i];
    }
  }
  for (unsigned l = 0; l < order; l++) {
    const double m2 = kr[l] * kr[l] + ki[l] * ki[l];

    if (m2 >= 1.0) {
      const double sc = (1.0 - delta) / sqrt(m2);

      kr[l] *= sc;
      ki[l] *= sc;
    }
  }
  ur[0] = 1.0;
  ui[0] = 0.0;
  for (unsigned l = 1; l <= order; l++) {
    for (unsigned i = 1; i < l; i++) {
      nr[i] = ur[i] + (kr[l - 1] * ur[l - i] + ki[l - 1] * ui[l - i]);
      ni[i] = ui[i] + (ki[l - 1] * ur[l - i] - kr[l - 1] * ui[l - i]);
    }
    for (unsigned i = 1; i < l; i++) {
      ur[i] = nr[i];
      ui[i] = ni[i];
    }
    ur[l] = kr[l - 1];
    ui[l] = ki[l - 1];
  }
  for (unsigned i = 1; i <= order; i++) {
    ar[i - 1] = -ur[i];
    ai[i - 1] = -ui[i];
  }
}

void Xorcist::SolveBin(
  const double *pP, const double *pQ, unsigned order, float *pCoefRe,
  float *pCoefIm, float *pRootRe, float *pRootIm) {
  // Nicht als Bereichsobjekt: die Funktion hat mehrere fruehe Ausgaenge,
  // und die sollen alle mitzaehlen - deshalb SolveBinEnd() an jedem.
  XoClock sbClock(XoProfiling());
  struct End {
    XoClock &c;
    ~End() { c.Stop(XO_ST_AR_SOLVE); }
  } sbEnd{sbClock};
  const unsigned N = AR_ORDER_MAX;
  double m[AR_ORDER_MAX][2 * AR_ORDER_MAX + 2];
  const double tiny = 1e-30;

  for (unsigned k = 0; k < order; k++) {
    for (unsigned l = 0; l < order; l++) {
      // Row k of the transposed system: coefficient of a_l is P[l][k].
      m[k][2 * l] = pP[2 * ((size_t)l * N + k)];
      m[k][2 * l + 1] = pP[2 * ((size_t)l * N + k) + 1];
    }
    m[k][2 * order] = pQ[2 * k];
    m[k][2 * order + 1] = pQ[2 * k + 1];
    m[k][2 * k] += tiny; // never exactly singular
  }
  for (unsigned c = 0; c < order; c++) {
    unsigned piv = c;
    double best = m[c][2 * c] * m[c][2 * c] + m[c][2 * c + 1] * m[c][2 * c + 1];

    for (unsigned r = c + 1; r < order; r++) {
      const double v
        = m[r][2 * c] * m[r][2 * c] + m[r][2 * c + 1] * m[r][2 * c + 1];

      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best < 1e-300) {
      for (unsigned k = 0; k < order; k++) {
        pCoefRe[k] = 0.0f;
        pCoefIm[k] = 0.0f;
      }
      return;
    }
    if (piv != c)
      for (unsigned j = 0; j <= 2 * order + 1; j++)
        std::swap(m[c][j], m[piv][j]);

    const double dr = m[c][2 * c], di = m[c][2 * c + 1];
    const double inv = 1.0 / (dr * dr + di * di);

    for (unsigned j = c; j <= order; j++) {
      const double xr = m[c][2 * j], xi = m[c][2 * j + 1];

      m[c][2 * j] = (xr * dr + xi * di) * inv;
      m[c][2 * j + 1] = (xi * dr - xr * di) * inv;
    }
    for (unsigned r = 0; r < order; r++) {
      if (r == c)
        continue;

      const double fr = m[r][2 * c], fi = m[r][2 * c + 1];

      if (fr == 0.0 && fi == 0.0)
        continue;
      for (unsigned j = c; j <= order; j++) {
        const double xr = m[c][2 * j], xi = m[c][2 * j + 1];

        m[r][2 * j] -= fr * xr - fi * xi;
        m[r][2 * j + 1] -= fr * xi + fi * xr;
      }
    }
  }

  double ar[AR_ORDER_MAX], ai[AR_ORDER_MAX];

  for (unsigned k = 0; k < order; k++) {
    ar[k] = m[k][2 * order];
    ai[k] = m[k][2 * order + 1];
    if (!XoFinite(ar[k]) || !XoFinite(ai[k])) {
      for (unsigned j = 0; j < order; j++) {
        pCoefRe[j] = 0.0f;
        pCoefIm[j] = 0.0f;
      }
      return;
    }
  }
  // Messung ohne Verhaltensaenderung: bestuende dieser Bin den
  // rho-Schur-Test auf den ROHEN Gauss-Koeffizienten?
  {
    static const double countRho = [] {
      const char *e = getenv("XO_COUNT_RHO");

      return e ? atof(e) : 0.0;
    }();

    if (countRho > 0.0 && order >= 2) {
      s_RhoTests.fetch_add(1, std::memory_order_relaxed);
      if (!RhoTest(ar, ai, order, countRho))
        s_RhoFails.fetch_add(1, std::memory_order_relaxed);
    }
  }

  const double det = (double)s_DetectRho.load(std::memory_order_relaxed);

  if (det > 0.0 && order >= 2 && RhoTest(ar, ai, order, det)) {
    // Alle Wurzeln innerhalb des Budgets: roh durchreichen.  Das ist
    // der Normalfall - und der erste Pfad seit Bestehen der Kappung,
    // der die Radien (= die gefitteten Huellkurven) unangetastet laesst.
    for (unsigned k = 0; k < order; k++) {
      pCoefRe[k] = (float)ar[k];
      pCoefIm[k] = (float)ai[k];
    }
    return;
  }

  if (order == 1) {
    // Order 1 is the rotator: normalise and be done.
    const double mm = sqrt(ar[0] * ar[0] + ai[0] * ai[0]);
    const double clampT = RootClampThresh();

    if (mm > clampT) {
      ar[0] /= mm;
      ai[0] /= mm;
    }
    pCoefRe[0] = (float)ar[0];
    pCoefIm[0] = (float)ai[0];
    return;
  }

  if (s_UseSchur.load(std::memory_order_relaxed)) {
    // 1.02 rather than 1.0: at 1.0 the clamp fires on every sustained
    // bin and inverts its angle (see SchurClamp), and the sweep is
    // monotone in rho right out to the point where the clamp never
    // fires at all - so 1.02 is the largest radius at which this still
    // BOUNDS anything (1.02^12 = 1.27 over the horizon, which the
    // safety cap covers) while not being the catastrophic end of the
    // scale.  It is still ~2 dB behind the selective root clamp; the
    // switch stays experimental.
    static const double schurRho = [] {
      const char *e = getenv("XO_SCHUR_RHO");

      return e ? atof(e) : 1.02;
    }();

    SchurClamp(ar, ai, order, 1e-3, schurRho);
    for (unsigned k = 0; k < order; k++) {
      pCoefRe[k] = (float)ar[k];
      pCoefIm[k] = (float)ai[k];
    }
    return;
  }

  double zr[AR_ORDER_MAX], zi[AR_ORDER_MAX];
  XoClock dkClock(XoProfiling());

  if (order == 2) {
    // Closed form - Durand-Kerner on a quadratic costs 40 iterations
    // where two square roots do.  Measured: 177 -> 137 us per channel.
    const double dr = ar[0] * ar[0] - ai[0] * ai[0] + 4.0 * ar[1];
    const double di = 2.0 * ar[0] * ai[0] + 4.0 * ai[1];
    const double dm = sqrt(dr * dr + di * di);
    const double sr2 = sqrt(std::max(0.0, 0.5 * (dm + dr)));
    double si2 = sqrt(std::max(0.0, 0.5 * (dm - dr)));

    if (di < 0.0)
      si2 = -si2;
    zr[0] = 0.5 * (ar[0] + sr2);
    zi[0] = 0.5 * (ai[0] + si2);
    zr[1] = 0.5 * (ar[0] - sr2);
    zi[1] = 0.5 * (ai[0] - si2);
  } else {
  // Durand-Kerner on p(z) = z^order - a1 z^(order-1) - ... - a_order.

  // WARM START: seed with the previous solve's roots.  Quasi-stationary
  // material moves them barely at all, so the iteration converges in 2-3
  // steps instead of ~15 - the difference between a 10 ms solve spike
  // (all bins, six channels, serial) and background noise.  Fall back to
  // the circle seed on the first solve or if two stored roots (nearly)
  // coincide - Durand-Kerner needs pairwise distinct start values.
  bool warm = false;

  if (pRootRe) {
    double norm = 0.0;

    warm = true;
    for (unsigned k = 0; k < order; k++) {
      zr[k] = pRootRe[k];
      zi[k] = pRootIm[k];
      norm += zr[k] * zr[k] + zi[k] * zi[k];
      if (!XoFinite(zr[k]) || !XoFinite(zi[k]))
        warm = false;
    }
    if (norm < 1e-4)
      warm = false;
    for (unsigned i = 0; i < order && warm; i++)
      for (unsigned j = i + 1; j < order; j++) {
        const double dr2 = (zr[i] - zr[j]) * (zr[i] - zr[j])
          + (zi[i] - zi[j]) * (zi[i] - zi[j]);

        if (dr2 < 1e-8) {
          warm = false;
          break;
        }
      }
  }
  if (!warm)
    for (unsigned k = 0; k < order; k++) {
      // Spread the start values around a circle; cannot coincide with a
      // real root of a real polynomial.
      const double ang = 2.0 * M_PI * k / order + 0.35;

      zr[k] = 0.9 * cos(ang);
      zi[k] = 0.9 * sin(ang);
    }
  for (unsigned it = 0; it < 40; it++) {
    double moved = 0.0;

    for (unsigned k = 0; k < order; k++) {
      // p(z_k) by Horner, monic leading term.
      double vr = 1.0, vi = 0.0;

      for (unsigned j = 0; j < order; j++) {
        const double tr = vr * zr[k] - vi * zi[k];
        const double ti = vr * zi[k] + vi * zr[k];

        vr = tr - ar[j];
        vi = ti - ai[j];
      }
      // Divide by the product of differences to the other roots.
      double dr = 1.0, di = 0.0;

      for (unsigned j = 0; j < order; j++) {
        if (j == k)
          continue;

        const double er = zr[k] - zr[j], ei = zi[k] - zi[j];
        const double tr = dr * er - di * ei;

        di = dr * ei + di * er;
        dr = tr;
      }

      const double den = dr * dr + di * di;

      if (den < 1e-300)
        continue;

      const double qr = (vr * dr + vi * di) / den;
      const double qi = (vi * dr - vr * di) / den;

      zr[k] -= qr;
      zi[k] -= qi;
      moved += qr * qr + qi * qi;
    }
    // Abbruch bei Divergenz: `den < 1e-300` faengt nur den zu KLEINEN
    // Nenner ab, nicht den ueberlaufenden.  Nach Stille und einem
    // Einsatz ist das Polynom entartet, das Horner-Produkt laeuft nach
    // unendlich, und die Division inf/inf liefert NaN (aufgespuert per
    // FE_INVALID-Falle, xorcist.cpp:2763 der damaligen Zaehlung).  Ohne
    // diesen Ausstieg rechnen 40 Iterationen auf Muell weiter und
    // streuen ihn ueber alle Wurzeln.  Die Kappung hinter der Schleife
    // nullt, was hier kaputtging.
    for (unsigned k = 0; k < order; k++)
      if (!XoFinite(zr[k]) || !XoFinite(zi[k])) {
        moved = 0.0;
        break;
      }
    if (moved < 1e-20)
      break;
  }
  }
  dkClock.Stop(XO_ST_DURAND);

  const double clampT = RootClampThresh();

  for (unsigned k = 0; k < order; k++) {
    const double mm = sqrt(zr[k] * zr[k] + zi[k] * zi[k]);

    if (!XoFinite(mm)) {
      zr[k] = 0.0;
      zi[k] = 0.0;
    } else if (det > 0.0 ? (mm > det) : (mm > clampT)) {
      // Detektor-Modus: auf rho DECKELN (Budget), klassisch: auf 1.0.
      const double tgt = det > 0.0 ? det : 1.0;

      zr[k] *= tgt / mm;
      zi[k] *= tgt / mm;
    }
  }
  if (pRootRe)
    for (unsigned k = 0; k < order; k++) {
      pRootRe[k] = (float)zr[k];
      pRootIm[k] = (float)zi[k];
    }
  // Coefficients back from the clamped roots: prod(z - z_k) expands to
  // sum c_j z^(order-j) with c_0 = 1, and a_j = -c_j.
  double cr[AR_ORDER_MAX + 1] = {1.0}, ci[AR_ORDER_MAX + 1] = {0.0};

  for (unsigned k = 1; k <= order; k++) {
    cr[k] = 0.0;
    ci[k] = 0.0;
  }
  for (unsigned k = 0; k < order; k++)
    for (unsigned j = k + 1; j >= 1; j--) {
      cr[j] -= zr[k] * cr[j - 1] - zi[k] * ci[j - 1];
      ci[j] -= zr[k] * ci[j - 1] + zi[k] * cr[j - 1];
    }
  for (unsigned k = 0; k < order; k++) {
    pCoefRe[k] = (float)(-cr[k + 1]);
    pCoefIm[k] = (float)(-ci[k + 1]);
  }
}

void Xorcist::ComputeChannelShadow(
  unsigned ch, float *pShadow, const PvKit &kit) {
  XoClock analyseClock(XoProfiling());
  const unsigned nfft = kit.nfft;
  const unsigned synHop = nfft / 4;
  const unsigned nbins = nfft / 2 + 1;
  const float *win = kit.win.data();
  // Every scratch buffer below is this CHANNEL's slice, so the channels
  // can be predicted concurrently (one task per channel in the host).
  float *const pTime = m_TimeBuf + (size_t)ch * NFFT_MAX;
  fftwf_complex *const pFreq = m_FreqBuf + (size_t)ch * FREQ_STRIDE;
  float *const specRe = m_SpecRe.data() + (size_t)ch * NBINS_MAX;
  float *const specIm = m_SpecIm.data() + (size_t)ch * NBINS_MAX;
  float *const rotRe = m_RotRe.data() + (size_t)ch * NBINS_MAX;
  float *const rotIm = m_RotIm.data() + (size_t)ch * NBINS_MAX;
  float *const stRe
    = m_ArStRe.data() + (size_t)ch * AR_ORDER_MAX * NBINS_MAX;
  float *const stIm
    = m_ArStIm.data() + (size_t)ch * AR_ORDER_MAX * NBINS_MAX;
  float *const ola = m_Ola.data() + (size_t)ch * (m_ShadowLen + NFFT_MAX);
  float *const bridge = m_Bridge.data() + (size_t)ch * m_BridgeLen;
  double *const burgSt = m_BurgState.data() + (size_t)ch * BURG_ORDER;
  // ANALYSIS hop, decoupled from the synthesis hop: when it equals one
  // audio period, the frame computed THIS period is exactly the older
  // frame the NEXT period needs - so the second forward FFT is replaced
  // by a cache lookup (measured 3.9 us per channel at 4096).  The
  // shorter measurement baseline is not a compromise: it measured
  // BETTER on every material (synthetic chord -9.2 -> -12.0 dB excess
  // NMR, held organ tone and dense orchestral material equal or better),
  // because a 21 ms baseline averages over beating that a 2.7 ms one
  // resolves.
  const unsigned ovr = (unsigned)m_AnaHopOverride.load(std::memory_order_relaxed);
  const unsigned anaHop
    = (ovr && synHop % ovr == 0 && ovr <= synHop) ? ovr : kit.anaHop;
  const unsigned mPow = synHop / anaHop; // rotator exponent, >= 1
  const unsigned len = nfft + anaHop;
  const float *pHist = HistWindow(ch, len);
  // Etikett der RUNDE, nicht des Rings - siehe RoundPeriod().  Muss zum
  // eingefrorenen Fenster passen, das HistWindow oben geliefert hat.
  const uint64_t nowP = RoundPeriod();
  const uint64_t nowSamp = nowP * m_SamplesPerBuffer;

  const unsigned arHop = std::max(nfft / 8, m_SamplesPerBuffer);
  const unsigned arStride = arHop / m_SamplesPerBuffer;
  const int wantOrder = std::min(
    (int)AR_ORDER_MAX, std::max(1, m_Order.load(std::memory_order_relaxed)));
  const unsigned P = (unsigned)wantOrder;
  const bool arUsable = arStride >= 1
    && AR_ORDER_MAX * arStride < m_ArRingLen + 1
    && arHop % m_SamplesPerBuffer == 0 && synHop % arHop == 0;

  // ---- HYBRID: kurze 1k-Zweitkette laeuft immer warm mit ---------------
  // Braucht Puffer, der 256 teilt, und einen groesseren Basisrahmen.
  const bool hybOn = m_Hybrid.load(std::memory_order_relaxed) != 0
    && 256u % m_SamplesPerBuffer == 0 && m_SamplesPerBuffer <= 256
    && nfft > HYB_NFFT;

  if (m_Hybrid.load(std::memory_order_relaxed))
    (hybOn ? s_HybOnPeriods : s_HybOffPeriods)
      .fetch_add(1, std::memory_order_relaxed);
  const bool hybSimple
    = hybOn && m_HybSimple.load(std::memory_order_relaxed) != 0;

  analyseClock.Stop(XO_ST_VORLAUF);

  XoClock detClock(XoProfiling());

  if (hybSimple) {
    m_D4Wide[ch] = -1.0f;
    if (m_HistFill.load(std::memory_order_relaxed) >= 4096) {
      const PvKit &k4 = m_Pv[1]; // 4096
      float *pT = m_TimeBuf + (size_t)ch * NFFT_MAX;
      fftwf_complex *pF = m_FreqBuf + (size_t)ch * FREQ_STRIDE;
      const float *pW = HistWindow(ch, 4096);

      for (unsigned i = 0; i < 4096; i++)
        pT[i] = pW[i] * k4.win[i];
      fftwf_execute_dft_r2c(k4.fwd, pT, pF);

      const unsigned sl = (unsigned)(nowP % 9);
      float *cur4 = m_D4Band.data() + ((size_t)ch * 9 + sl) * ROUTE_BANDS;

      for (unsigned b = 0; b < ROUTE_BANDS; b++)
        cur4[b] = 0.0f;
      for (unsigned bin = 0; bin < 2049; bin++) {
        const unsigned b = std::min(bin >> 3, ROUTE_BANDS - 1);

        cur4[b] += pF[bin][0] * pF[bin][0] + pF[bin][1] * pF[bin][1];
      }
      m_D4Per[(size_t)ch * 9 + sl] = nowP;

      const unsigned s8 = (unsigned)((nowP - 8) % 9);

      if (nowP >= 8 && m_D4Per[(size_t)ch * 9 + s8] == nowP - 8) {
        const float *old4 = m_D4Band.data()
          + ((size_t)ch * 9 + s8) * ROUTE_BANDS;
        uint8_t *r4 = m_D4Rise.data() + (size_t)ch * ROUTE_BANDS;
        unsigned nR = 0;

        // PEGELUNTERGRENZE (XO_D4_FLOOR, Vorgabe 0 = aus).
        //
        // Das Kriterium darueber ist ein reines VERHAELTNIS: ein Band bei
        // -120 dBFS, das sich verdoppelt, zaehlt genauso viel wie der
        // Grundton.  Der Ring enthaelt bei hoher Lueckendichte aber zu
        // einem guten Viertel die eigene Vorhersage, und die ist so
        // gebaut, dass sie UNHOERBAR ist - nicht, dass sie in jedem Band
        // pegelgleich waere.  Zwischen den Teiltoenen, wo nur Rauschgrund
        // und Vorhersagefehler wohnen, ist Faktor zwei deshalb billig zu
        // haben, ohne dass ein Ohr etwas davon merkt.  Genau das war die
        // Beobachtung des Nutzers: der Detektor zuendet auf gehaltenem
        // Klang, wo nichts zu hoeren ist.
        //
        // Bezug ist die MITTLERE Bandenergie des Rahmens, nicht die
        // Spitze: skaleninvariant wie das Verhaeltnis selbst, und die
        // Spitze waere bei einem kraeftigen Grundton so hoch, dass sie
        // jedes echte Nebenband miterschlagen wuerde.
        static const float envFloor = [] {
          const char *e = getenv("XO_D4_FLOOR");

          return e ? (float)atof(e) : 0.0f;
        }();

        float floorAbs = 0.0f;

        if (envFloor > 0.0f) {
          double tot = 0.0;

          for (unsigned b = 0; b < ROUTE_BANDS; b++)
            tot += cur4[b];
          floorAbs = envFloor * (float)(tot / ROUTE_BANDS);
        }
        // XO_D4_DEN=1 legt dieselbe Schwelle an den NENNER statt an den
        // Zaehler: nur Baender zaehlen, die VORHER etwas getragen haben.
        static const bool envDen = getenv("XO_D4_DEN") != nullptr;

        // BELEGTE BAENDER (XO_D4_OCC, Vorgabe 0 = aus).
        //
        // Das Kriterium ist ein Verhaeltnis, und in einem Band, in dem
        // NICHTS steht, ist Faktor zwei geschenkt - dort wohnt nur der
        // eigene Vorhersagefehler.  Solange der Kanal breitbandig ist,
        // faellt das nicht auf; bei einem bandbegrenzten Kanal aber
        // tragen von 257 Baendern noch drei etwas, und die restlichen
        // 250 zuenden im Dauerlauf.  GEMESSEN am gehaltenen Ton des
        // Nutzers bei 6000 Luecken/min: Frontpaar (86 belegte Baender)
        // 0 Zuendungen, Tiefpass 400 Hz (5 Baender) 230, LFE-artig
        // (3 Baender) 228.  Das Labor hat den Fehler nie gesehen, weil
        // GOs Rekorder einen Stereo-Downmix schreibt - der Detektor in
        // GO laeuft dagegen auf JEDEM EINZELKANAL.
        //
        // Nur der ZAEHLER wird beschraenkt, der Nenner bleibt
        // ROUTE_BANDS: normierte man auf die belegten Baender, waere
        // dieselbe Zuendung im Frontpaar dreimal so breit und die
        // Schwellen 0.25/0.12 waeren neu zu kalibrieren.  Folge des
        // Zuschnitts: in einem Kanal mit drei belegten Baendern kann die
        // Breite 3/256 nie die Schwelle erreichen - dort zuendet der
        // Detektor also gar nicht mehr.  Das ist beabsichtigt; ein
        // "breitbandiger Anstieg" ist in einem solchen Kanal kein
        // sinnvoller Begriff.
        static const float envOcc = [] {
          const char *e = getenv("XO_D4_OCC");

          return e ? (float)atof(e) : 0.0f;
        }();
        float occAbs = 0.0f;
        {
          float peak = 0.0f;

          for (unsigned b = 0; b < ROUTE_BANDS; b++)
            peak = std::max(peak, cur4[b]);
          if (envOcc > 0.0f)
            occAbs = envOcc * peak;

          // Diagnose: wie viele Baender tragen ueberhaupt etwas?  Fest
          // bei -60 dB unter dem lautesten Band, unabhaengig vom Regler.
          unsigned nOcc = 0;

          for (unsigned b = 0; b < ROUTE_BANDS; b++)
            nOcc += cur4[b] > peak * 1e-6f;
          if (ch < DET_CH_MAX)
            s_OccPerCh[ch].store(nOcc, std::memory_order_relaxed);
        }
        for (unsigned b = 0; b < ROUTE_BANDS; b++) {
          r4[b] = cur4[b] > 2.0f * old4[b]
            && (envDen ? old4[b] > floorAbs : cur4[b] > floorAbs)
            && cur4[b] > occAbs;
          nR += r4[b];
        }
        m_D4Wide[ch] = (float)nR / ROUTE_BANDS;
      }
    }
  }
  detClock.Stop(XO_ST_DETEKTOR);

  XoClock hybClock(XoProfiling());

  if (hybOn) {
    HybridAnalyze(ch, nowP);
    HybridTick(ch, arStride, P, nowP);

    const float *hw = m_HybW.data() + (size_t)ch * ROUTE_BANDS;
    uint64_t nb = 0;

    for (unsigned b = 0; b < ROUTE_BANDS; b++)
      if (hw[b] > 0.5f)
        nb++;
    if (nb)
      s_HybBandPeriods.fetch_add(nb, std::memory_order_relaxed);
  }
  hybClock.Stop(XO_ST_HYBRID);

  XoClock akkuClock(XoProfiling());

  // ---- backfill: recompute a missing seed instead of dropping the model
  //
  // The spectrum ring is written ONLY when a shadow is built, and no
  // shadow is built during a miss streak - so a cluster tears a hole in
  // it.  Every later grid point whose seeds fall into that hole failed
  // the useAr test below and silently ran the ORDER-1 ROTATOR instead of
  // AR(p).  Measured on the D major chord: 11.1 % of shadow builds at
  // order 4 and 600/min, 30.6 % at order 12, and at 6000/min 71.2 % and
  // 97.5 % - i.e. "order 12" was almost never order 12.  That is the
  // whole of the outlier problem (1.6 % of gaps above 0.20 absolute
  // error against 0.0 % for an offline prototype), and it is why the
  // orders were barely distinguishable by ear in the product.
  //
  // The samples are NOT missing - Conceal() pushes what it played, so
  // the history ring holds those periods in full.  Only the FFT was
  // never taken.  So take it, exactly as the offline prototype does: it
  // keeps no cache and recomputes every frame it needs from the history
  // at each gap, which is precisely why a hole cannot form there.
  //
  // This runs only after a cluster, never during one - so the extra FFT
  // does not land in the moment the machine is short of time.  And it
  // follows the lesson this cost a day to learn: CONTAMINATED DATA BEATS
  // MISSING DATA.  Gating the fit on clean seeds and deferring the
  // re-analysis both made things worse; dropping the whole model because
  // one seed is unavailable is the same mistake in its most extreme
  // form.
  //
  // Placed BEFORE the phase-vocoder freeze because pTime/pFreq are the
  // only scratch, and the freeze's spectrum is still live from here down
  // to the rotator loop.
  if (arUsable && P >= 2 && m_Backfill.load(std::memory_order_relaxed)) {
    const size_t ringBase = (size_t)ch * m_ArRingLen;

    for (unsigned j = 1; j <= P; j++) {
      if (nowP < (uint64_t)j * arStride)
        break;

      const uint64_t want = nowP - (uint64_t)j * arStride;
      const unsigned sl = (unsigned)(want % m_ArRingLen);

      if (m_ArRingPeriod[ringBase + sl] == want)
        continue;

      const unsigned back
        = (unsigned)((uint64_t)j * arStride * m_SamplesPerBuffer);

      if (
        nfft + back > HIST
        || m_HistFill.load(std::memory_order_relaxed) < nfft + back)
        continue;

      const float *pW = HistWindow(ch, nfft + back);
      float *bRe = m_ArRingRe.data() + (ringBase + sl) * NBINS_MAX;
      float *bIm = m_ArRingIm.data() + (ringBase + sl) * NBINS_MAX;

      for (unsigned i = 0; i < nfft; i++)
        pTime[i] = pW[i] * win[i];
      OverlayPatch(ch, nowSamp - back, nfft, win, pTime);
      fftwf_execute_dft_r2c(kit.fwd, pTime, pFreq);
      for (unsigned bin = 0; bin < nbins; bin++) {
        bRe[bin] = pFreq[bin][0];
        bIm[bin] = pFreq[bin][1];
      }
      m_ArRingPeriod[ringBase + sl] = want;
      s_Backfills.fetch_add(1, std::memory_order_relaxed);
    }
  }


  // ---- phase-vocoder freeze -------------------------------------------
  // older frame: [0, nfft), newer frame: [anaHop, anaHop + nfft) - the
  // newer one ends at the latest sample.
  // The cache IS the older spectrum - no copy in or out.  A first
  // attempt that copied cache -> m_PrevRe/Im and m_FreqBuf -> cache made
  // the whole thing SLOWER than the FFT it saved (32 KB of extra traffic
  // per channel in a loop that is already memory-bound).  Instead the
  // rotator loop below reads each cache slot and overwrites it with this
  // period's value in the same iteration - read before write, so it is
  // safe, and it replaces the copy pass the old code needed anyway.
  float *prevRe = m_SpecCacheRe.data() + (size_t)ch * NBINS_MAX;
  float *prevIm = m_SpecCacheIm.data() + (size_t)ch * NBINS_MAX;
  const bool cacheHit = m_SpecCachePeriod[ch] == nowP - 1
    && m_SpecCacheNfft[ch] == nfft && m_SpecCacheHop[ch] == anaHop;

  if (!cacheHit) {
    for (unsigned i = 0; i < nfft; i++)
      pTime[i] = pHist[i] * win[i];
    OverlayPatch(ch, nowSamp - anaHop, nfft, win, pTime);
    fftwf_execute_dft_r2c(kit.fwd, pTime, pFreq);
    for (unsigned bin = 0; bin < nbins; bin++) {
      prevRe[bin] = pFreq[bin][0];
      prevIm[bin] = pFreq[bin][1];
    }
  }

  for (unsigned i = 0; i < nfft; i++)
    pTime[i] = pHist[anaHop + i] * win[i];
  OverlayPatch(ch, nowSamp, nfft, win, pTime);
  fftwf_execute_dft_r2c(kit.fwd, pTime, pFreq);

  m_SpecCachePeriod[ch] = nowP;
  m_SpecCacheNfft[ch] = nfft;
  m_SpecCacheHop[ch] = anaHop;

  akkuClock.Stop(XO_ST_ANALYSE);

  XoClock arClock(XoProfiling());

  // ---- AR(p): store this frame, advance the normal equations -----------
  // Done unconditionally so that changing the order mid-stream finds warm
  // accumulators instead of a cold start.  One ring write plus, on every
  // arStride-th period, the rank-1 update of a p x p system per bin - no
  // extra FFT, because the frame this needs is the one the rotator just
  // computed.
  if (arUsable) {
    const size_t ringBase = (size_t)ch * m_ArRingLen;
    const unsigned slot = (unsigned)(nowP % m_ArRingLen);
    float *pRe = m_ArRingRe.data() + (ringBase + slot) * NBINS_MAX;
    float *pIm = m_ArRingIm.data() + (ringBase + slot) * NBINS_MAX;

    for (unsigned bin = 0; bin < nbins; bin++) {
      pRe[bin] = pFreq[bin][0];
      pIm[bin] = pFreq[bin][1];
    }
    m_ArRingPeriod[ringBase + slot] = nowP;

    // A frame size change invalidates the accumulators: the recursion they
    // describe belongs to the old bin grid AND the old hop.
    if (m_ArNfft[ch] != nfft) {
      // Planar: je Paar bzw. je k einen zusammenhaengenden Bin-Streifen.
      for (unsigned i = 0; i < AR_PAIRS; i++) {
        const size_t o
          = (size_t)ch * AR_PAIRS * NBINS_MAX + (size_t)i * NBINS_MAX;

        std::fill_n(m_ArPRe.begin() + o, nbins, 0.0);
        std::fill_n(m_ArPIm.begin() + o, nbins, 0.0);
      }
      for (unsigned k = 0; k < AR_ORDER_MAX; k++) {
        const size_t o
          = (size_t)ch * AR_ORDER_MAX * NBINS_MAX + (size_t)k * NBINS_MAX;

        std::fill_n(m_ArQRe.begin() + o, nbins, 0.0);
        std::fill_n(m_ArQIm.begin() + o, nbins, 0.0);
      }
      m_ArNfft[ch] = nfft;
      m_ArFitPeriod[ch] = ~0ull;
      m_ArFitCount[ch] = 0;
      m_ArSolvedPeriod[ch] = ~0ull;
    }

    // Fit only on the ABSOLUTE arHop grid, so every tuple the
    // accumulators see is equally spaced - a fit mixing spacings would
    // describe no single recursion at all.
    const uint64_t sampleNow = nowP * (uint64_t)m_SamplesPerBuffer;
    bool haveAll = sampleNow % arHop == 0 && nowP >= (uint64_t)P * arStride;
    const float *lagRe[AR_ORDER_MAX], *lagIm[AR_ORDER_MAX];

    for (unsigned k = 0; k < P && haveAll; k++) {
      const uint64_t want = nowP - (uint64_t)(k + 1) * arStride;
      const unsigned sl = (unsigned)(want % m_ArRingLen);

      if (m_ArRingPeriod[ringBase + sl] != want) {
        haveAll = false;
        break;
      }
      lagRe[k] = m_ArRingRe.data() + (ringBase + sl) * NBINS_MAX;
      lagIm[k] = m_ArRingIm.data() + (ringBase + sl) * NBINS_MAX;
    }
    // ---- clean-seed gate for the FIT (2026-08-06) --------------------
    // The analogue of the Burg bridge's gate one level up: a grid point
    // whose tuple overlaps the predictor's OWN output teaches the normal
    // equations the predictor's error.  At 5 % concealment that is
    // noise; at the 21 % this lab can reach it is most of the data, and
    // the library falls 4.45 dB behind a numpy prototype that refits
    // from a rectangular window and therefore forgets contamination
    // completely after its span.
    //
    // The tuple reaches back P*arStride periods, and the OLDEST frame in
    // it was analysed over nfft samples, so the whole span is
    // P*arStride + nfft/buf periods.
    if (haveAll && m_GateFit.load(std::memory_order_relaxed)) {
      const uint64_t span
        = (uint64_t)P * arStride + (nfft + m_SamplesPerBuffer - 1)
          / m_SamplesPerBuffer;
      const uint64_t last
        = m_LastConcealedPeriod.load(std::memory_order_acquire);

      s_GateTests.fetch_add(1, std::memory_order_relaxed);
      if (last && nowP >= last && nowP - last <= span) {
        s_GateSkips.fetch_add(1, std::memory_order_relaxed);
        haveAll = false;
      }
    }
    if (haveAll) {
      // A SKIPPED grid point must NOT reset the accumulators.  Each tuple
      // is equally spaced within itself, so a missing term is simply a
      // missing term - and the shadow is not recomputed during a miss
      // streak, which at 600/min drops roughly every ninth grid point.
      // Resetting there kept the fit permanently under-determined
      // (-14.4 dB on two sinusoids where the prototype reaches -95.8).
      const double lam = ArLambda(P);
      const size_t pStride = (size_t)AR_ORDER_MAX * AR_ORDER_MAX * 2;
      const size_t qStride = (size_t)AR_ORDER_MAX * 2;
      // Paar aussen, BIN innen - die innere Schleife laeuft damit
      // zusammenhaengend ueber alle Bins (P planar, Eingangswerte im
      // Spektrenring ohnehin planar) und ist vektorisierbar.  Nur das
      // obere Dreieck k <= l wird gerechnet; der Rest ist die konjugierte
      // Spiegelung und wird erst beim Auffalten in GatherBin erzeugt.
      const size_t pBase = (size_t)ch * AR_PAIRS * NBINS_MAX;
      const size_t qBase = (size_t)ch * AR_ORDER_MAX * NBINS_MAX;

      for (unsigned k = 0; k < P; k++) {
        // q[k] = sum X0 * conj(Xk)
        double *qr = m_ArQRe.data() + qBase + (size_t)k * NBINS_MAX;
        double *qi = m_ArQIm.data() + qBase + (size_t)k * NBINS_MAX;
        const float *kr = lagRe[k], *ki = lagIm[k];

        for (unsigned bin = 0; bin < nbins; bin++) {
          const double x0r = pFreq[bin][0], x0i = pFreq[bin][1];
          const double xkr = kr[bin], xki = ki[bin];

          qr[bin] = lam * qr[bin] + x0r * xkr + x0i * xki;
          qi[bin] = lam * qi[bin] + x0i * xkr - x0r * xki;
        }
        for (unsigned l = k; l < P; l++) {
          // P[k][l] = sum Xk * conj(Xl)
          double *pr = m_ArPRe.data() + pBase + (size_t)PairOf(k, l) * NBINS_MAX;
          double *pi = m_ArPIm.data() + pBase + (size_t)PairOf(k, l) * NBINS_MAX;
          const float *lr = lagRe[l], *li = lagIm[l];

          for (unsigned bin = 0; bin < nbins; bin++) {
            const double xkr = kr[bin], xki = ki[bin];
            const double xlr = lr[bin], xli = li[bin];

            pr[bin] = lam * pr[bin] + xkr * xlr + xki * xli;
            pi[bin] = lam * pi[bin] + xki * xlr - xkr * xli;
          }
        }
      }
      m_ArFitPeriod[ch] = nowP;
      if (m_ArFitCount[ch] < 250)
        m_ArFitCount[ch]++;
      // Hybrid: dieses Rasterupdate zaehlt fuer jedes junge Band als
      // EIN echtes Tupel der langen Kette.
      if (hybOn && m_HybAny[ch])
        for (unsigned b = 0; b < ROUTE_BANDS; b++) {
          const size_t o = (size_t)ch * ROUTE_BANDS + b;

          if (m_HybAge[o] != HYB_DORMANT && m_HybTupA[o] < 60000)
            m_HybTupA[o]++;
        }
    }
  }

  // Per-bin phase-advance rotator WITHOUT transcendentals: the classic
  // formulation is adv = omega + princarg(angle(F1)-angle(F0) - omega),
  // rot = e^{i*adv} - but e^{i*princarg(x)} == e^{ix}, so all the wraps
  // cancel and rot collapses to normalize(F1 * conj(F0)) exactly.  That
  // replaces 2 atan2f + 1 sincos per bin (the dominant scalar cost of
  // the whole prediction) with one complex multiply and one rsqrt.
  // While normalising, accumulate the cosine similarity of the two
  // magnitude spectra for free: |F1*conj(F0)| == |F1||F0|, so the
  // numerator is the |cross| this loop computes anyway.  s discriminates
  // sustained material (s >= 0.999) from transients (<= 0.99) and drives
  // the adaptive seam below.
  // float accumulators on purpose: the compiler may reassociate and
  // vectorize them, and sqrt(N)*eps ~ 6e-6 relative error is far below
  // the 0.999/0.009 decision band.
  float sumCross = 0.0f, sumE1 = 0.0f, sumE0 = 0.0f;
  float bandE[ROUTE_BANDS] = {0.0f};
  // Welche Baender steigen?  Der Hybrid braucht die MENGE, nicht nur die
  // Zahl - sie ist seine Reset-Maske.
  uint8_t riseB[ROUTE_BANDS] = {0};
  // 1024 -> >>1, 2048 -> >>2, 4096 -> >>3, 8192 -> >>4: 256 Baender
  // ueber Nyquist.  Der 1k-Fall FEHLTE beim Einbau des Modus - 1k lief
  // mit >>4 auf 32 Baendern, Detektor und Router entsprechend vergroebert.
  const unsigned bandShift = nfft == 1024
    ? 1u : (nfft == 2048 ? 2u : (nfft == 4096 ? 3u : 4u));

  for (unsigned bin = 0; bin < nbins; bin++) {
    const float re = pFreq[bin][0], im = pFreq[bin][1];
    const float p_re = prevRe[bin], p_im = prevIm[bin];
    float rre = re * p_re + im * p_im;
    float rim = im * p_re - re * p_im;
    const float mag2 = rre * rre + rim * rim;
    const float e0 = p_re * p_re + p_im * p_im;
    const float e1 = re * re + im * im;
    const unsigned band = bin >> bandShift;

    bandE[band < ROUTE_BANDS ? band : ROUTE_BANDS - 1] += e1;
    // Read before write: this slot becomes the older frame next period.
    prevRe[bin] = re;
    prevIm[bin] = im;
    sumE1 += e1;
    sumE0 += e0;
    if (mag2 > 1e-30f) {
      const float inv = 1.0f / sqrtf(mag2);
      const float cross = mag2 * inv; // == |F1||F0| for this bin

      sumCross += cross;
      rre *= inv;
      rim *= inv;
      // Raise the per-analysis-hop rotator to the synthesis hop by
      // repeated complex squaring (mPow is a power of two by
      // construction).  Exact: the wraps cancel inside the exponential.
      for (unsigned e = mPow; e > 1; e >>= 1) {
        const float sr_ = rre * rre - rim * rim;

        rim = 2.0f * rre * rim;
        rre = sr_;
      }
    } else {
      rre = 1.0f; // silent bin: hold phase
      rim = 0.0f;
    }
    specRe[bin] = re;
    specIm[bin] = im;
    rotRe[bin] = rre;
    rotIm[bin] = rim;
  }

  // ---- frequency inertia ------------------------------------------------
  // Blend this analysis' rotator with the one carried over, so the
  // FREQUENCY estimate moves slowly.  Legitimate because rotators are
  // unit vectors: the blend interpolates a direction and is renormalised.
  // Magnitudes deliberately stay fresh - only the frequency drives the
  // self-reinforcing loop when predicted audio is re-analysed.
  const float damping = m_FreqDamping.load(std::memory_order_relaxed);
  const float inertia = 1.0f - damping;

  if (damping > 0.0f) {
    float *kRe = m_RotKeepRe.data() + (size_t)ch * NBINS_MAX;
    float *kIm = m_RotKeepIm.data() + (size_t)ch * NBINS_MAX;

    if (m_RotKeepValid[ch])
      for (unsigned bin = 0; bin < nbins; bin++) {
        const float re
          = kRe[bin] + inertia * (rotRe[bin] - kRe[bin]);
        const float im
          = kIm[bin] + inertia * (rotIm[bin] - kIm[bin]);
        const float m2 = re * re + im * im;
        const float inv = m2 > 1e-30f ? 1.0f / sqrtf(m2) : 1.0f;

        kRe[bin] = rotRe[bin] = re * inv;
        kIm[bin] = rotIm[bin] = im * inv;
      }
    else
      for (unsigned bin = 0; bin < nbins; bin++) {
        kRe[bin] = rotRe[bin];
        kIm[bin] = rotIm[bin];
      }
    m_RotKeepValid[ch] = 1;
  }

  const double sim
    = (double)sumCross / std::max(sqrt((double)sumE1 * sumE0), 1e-30);

  // ---- seam routing, on a FIXED ~21 ms baseline ------------------------
  // Compare this period's coarse band energies against the snapshot from
  // m_RouteSlots-1 periods ago (1024 samples), NOT against the previous
  // analysis frame.  The analysis hop is one audio period, and scaling a
  // 2.7 ms similarity up to the calibration baseline by hop^2 assumes the
  // difference is deterministic partial drift.  Measured, it is not: for
  // a hop ratio of 8 the measured ratio is 1.2 (synthetic sines), 17.5
  // (held organ tone) and 32 (dense chord), never the assumed 64.  The
  // over-compensation pinned real material at the bridge end - a clean
  // held tone read wBridge 0.82 where the anchor measured 4.9 dB better.
  //
  // Calibrated against which seam actually WINS, at 600/min on frame 4k
  // (fill error, anchor vs bridge): synthetic sines 0.0000 -> anchor by
  // 4.7 dB, held organ tone 0.0001 -> anchor by 9.4, single organ tone
  // 0.0008 -> anchor by 4.9, chord buildup with attacks 0.0012 -> anchor
  // by 3.4, full chord 0.0053 -> bridge by 0.5, dense Vierne 0.0402 ->
  // tie.  So the anchor is right below ~0.003 by a wide margin, and the
  // bridge is at best marginally better above it; the band blends them
  // where the corpus stops being decisive.
  //
  // Net effect, measured by running the SAME chain through the pre-fix
  // and post-fix library (adaptive vs adaptive, frame 4k, fill error at
  // 120 / 600 / 2400 misses per minute):
  //
  //   single organ tone   +3.5  +3.5  +1.1
  //   chord + attacks     +2.7  +2.7  +0.8
  //   full chord           0.0  -0.2   0.0
  //   held tone, sines, dense Vierne   unchanged
  //
  // The held tone gains nothing because the OLD router already anchored
  // it - (1 - sim) is tiny enough there to survive the x64.  What the
  // fix recovers is the middle of the corpus, where hop^2 tipped the
  // decision but the material is still anchor material.  Vierne is the
  // one segment whose decision is identical under both formulas: 0.0402
  // is far above the upper threshold, so it was full bridge and stays
  // full bridge - and there the two branches differ by 0.1 dB anyway.
  double onsetFlux = -1.0, onsetWide = -1.0;
  double dens;
  {
    const size_t base = (size_t)ch * m_RouteSlots;
    // The slot about to be overwritten holds the oldest snapshot, which
    // is exactly m_RouteSlots-1 periods back.
    const unsigned oldSlot = (unsigned)((nowP + 1) % m_RouteSlots);
    const float *pOld = m_RouteBandE.data() + (base + oldSlot) * ROUTE_BANDS;
    const bool usable = m_RouteNfft[base + oldSlot] == nfft
      && m_RoutePeriod[base + oldSlot] + (m_RouteSlots - 1) == nowP;
    double c = 0.0, e0 = 0.0, e1 = 0.0;
    double posF = 0.0, totF = 0.0;
    unsigned nRise = 0;

    if (usable)
      for (unsigned i = 0; i < ROUTE_BANDS; i++) {
        const double u = bandE[i], v = pOld[i];
        c += u * v;
        e1 += u * u;
        e0 += v * v;
        // positiver Spektralfluss fuer den Transienten-Trigger: ein
        // zuschaltendes Register hebt Baender von fast null an - relativ
        // ein Riesensprung, egal wie laut der Rest schon ist.  sim bei
        // Hop 128 sieht das nicht: die Registerzuschaltungen tauchten
        // auf 0.9991 ein, waehrend gehaltenes Material auf 0.9971
        // schwankt.
        if (u > v)
          posF += u - v;
        totF += v;
        // BREITE statt Energie: wie viele Baender steigen um mehr als
        // 6 dB?  Ein Anschlag zuendet viele Baender gleichzeitig, auch
        // solche am Rauschgrund - und zwar unabhaengig davon, wie laut
        // es schon war, also auch bei Einsaetzen MITTEN in den Klang.
        // Periodische AMPLITUDENMODULATION - auf dem Kalibriermaterial
        // eine ~9-Hz-Schwebung verstimmter Choere, mit gezogenem
        // Tremulant dieselbe Signatur - hebt dagegen periodisch
        // DIESELBEN Baender: gemessen p99 Breite 0.113 gegen 0.31-0.87
        // bei echten Einsaetzen, waehrend der ENERGIEfluss bei der
        // Modulation mit 0.466 hoeher lag als bei jedem Einsatz.  Weil
        // Schwebung Grundzustand von Orgelklang ist (nicht nur ein
        // Registrier-Sonderfall), ist diese Trennung Pflicht, nicht
        // Feinschliff.  (Einordnung vom Nutzer, 07.08.)
        if (u > 2.0 * v) {
          nRise++;
          riseB[i] = 1;
        }
      }
    // Warm-up, a mode change, or a gap in the snapshot ring: fall back to
    // the short baseline rather than routing on stale bands.
    dens = (usable && e0 > 1e-30 && e1 > 1e-30)
      ? 1.0 - c / sqrt(e1 * e0)
      : (1.0 - sim) * (double)mPow * mPow;

    const unsigned slot = (unsigned)(nowP % m_RouteSlots);
    float *pNew = m_RouteBandE.data() + (base + slot) * ROUTE_BANDS;

    std::copy(bandE, bandE + ROUTE_BANDS, pNew);
    m_RoutePeriod[base + slot] = nowP;
    m_RouteNfft[base + slot] = nfft;
    onsetFlux = usable ? posF / (totF + 1e-12) : -1.0;
    onsetWide = usable ? (double)nRise / ROUTE_BANDS : -1.0;
  }

  // Kalibrierhilfe (XO_SIM_TRACE=<pfad>): Periode, sim, Fluss.
  {
    static FILE *simTrace = [] {
      const char *e = getenv("XO_SIM_TRACE");

      return e ? fopen(e, "w") : (FILE *)nullptr;
    }();

    if (simTrace && ch == 0) {
      // Spalten 5-7: Residuen-Statistik der beiden Ketten (nur mit
      // hybrid+chooser gefuellt, sonst 0) - fuer die Frage des Nutzers,
      // ob der VORHERSAGEFEHLER als Detektor trennt.
      const float *ra = m_HybResA.data();
      const float *rb = m_HybResB.data();
      double ma = 0.0, xa = 0.0, mb = 0.0;

      for (unsigned b = 0; b < ROUTE_BANDS; b++) {
        ma += ra[b];
        mb += rb[b];
        if (ra[b] > xa)
          xa = ra[b];
      }
      fprintf(
        simTrace, "%llu %.6f %.6f %.6f %.6f %.6f %.6f\n",
        (unsigned long long)nowP, sim, onsetFlux, onsetWide,
        ma / ROUTE_BANDS, xa, mb / ROUTE_BANDS);
    }
  }

  // ---- HYBRID: 512er-Vorstufe ------------------------------------------
  // Jede Periode: 512er-Rahmen, 94-Hz-Baender (1 Bin = 1 Band), Breite
  // gegen 8 Perioden zurueck.  Kreuzt sie die Vor-Schwelle, schalten
  // die steigenden Baender VORLAEUFIG auf die kurze Kette.
  if (
    hybOn && m_HybPre.load(std::memory_order_relaxed)
    && !m_HybChooser.load(std::memory_order_relaxed)
    && !m_HybSimple.load(std::memory_order_relaxed)
    && m_HistFill.load(std::memory_order_relaxed) >= PRE_NFFT) {
    float *pT = m_TimeBuf + (size_t)ch * NFFT_MAX;
    fftwf_complex *pF = m_FreqBuf + (size_t)ch * FREQ_STRIDE;
    const float *pW = HistWindow(ch, PRE_NFFT);

    for (unsigned i = 0; i < PRE_NFFT; i++)
      pT[i] = pW[i] * m_PreWin[i];
    fftwf_execute_dft_r2c(m_PreFwd, pT, pF);

    const unsigned sl = (unsigned)(nowP % PRE_RING);
    float *cur = m_PreBandE.data()
      + ((size_t)ch * PRE_RING + sl) * ROUTE_BANDS;

    for (unsigned b = 0; b < ROUTE_BANDS; b++) {
      const float re = pF[b + 1][0], im = pF[b + 1][1]; // DC ueberspringen

      cur[b] = re * re + im * im;
    }
    m_PrePeriod[(size_t)ch * PRE_RING + sl] = nowP;

    const unsigned s8 = (unsigned)((nowP - 8) % PRE_RING);

    if (
      nowP >= 8 && m_PrePeriod[(size_t)ch * PRE_RING + s8] == nowP - 8
      && (RoundMask() & ((1u << (preVeto() + 1)) - 1u)) == 0) {
      // Vorstufen-Veto LAENGER als beim 2k-Detektor (Vorgabe 5 statt 2
      // Perioden): das 512er-Fenster traegt den Wiedereinstiegs-Spleiss
      // 4 Perioden lang mit hohem Gewicht, und bei dichter Verdeckung
      // zuendete die Vorstufe sonst dauernd auf Residuen (Schwebung
      // -4.5 -> +0.9 dB NMR bei 6000/min mit Veto 2).
      const float *old8 = m_PreBandE.data()
        + ((size_t)ch * PRE_RING + s8) * ROUTE_BANDS;
      uint8_t rise512[ROUTE_BANDS];
      unsigned nR = 0;

      for (unsigned b = 0; b < ROUTE_BANDS; b++) {
        rise512[b] = cur[b] > 2.0f * old8[b];
        nR += rise512[b];
      }

      // Kalibriert 08.08. (sauberer Lauf, 128er-Perioden): Stoermax
      // 0.401 (gehalten), Einsatz-Minimum 0.424 - hauchduenn, deshalb
      // nur VORLAEUFIG plus Rueckzug.
      static const float PRE_ON = [] {
        const char *e = getenv("XO_PRE_ON");

        return e ? (float)atof(e) : 0.41f;
      }();

      static const unsigned PRE_CHK = [] {
        const char *e = getenv("XO_PRE_CHK");

        return e ? (unsigned)atoi(e) : 4u;
      }();
      static const float PRE_OFF = [] {
        const char *e = getenv("XO_PRE_OFF");

        // 0.38, knapp unter ON: bei 26.7 % Verdeckung heben die
        // Residuen die Breite dauerhaft ueber die alte 0.32 - Kandidaten
        // beruhigten sich NIE, hingen 256 ms auf Residuen fest, und
        // echte Einsaetze dahinter verhungerten ('zweiter Einsatz
        // notorisch zu spaet, ohne Grund' - exakt der Stau).
        return e ? (float)atof(e) : 0.38f;
      }();
      const float wideNow = (float)nR / ROUTE_BANDS;

      // Kandidaten-Verfall: 48 Perioden (128 ms) - laenger dauert kein
      // Anschlagskern, und im Residuen-Nebel darf kein Kandidat die
      // Buehne blockieren.
      if (
        m_PreCandP[ch] != ~0ull && nowP - m_PreCandP[ch] > 48)
        m_PreCandP[ch] = ~0ull;

      // LAUTSTAERKE-TOR (Nutzer 08.08.): die Transienten der Stille
      // (Motor, Residuen) sind ECHT, aber nicht massgeblich hoerbar -
      // sie verbrauchen Kandidaten, hinter denen echte Einsaetze
      // anstehen.  Gezuendet wird nur ueber XO_PRE_LOUD (-20 dB) des
      // laufenden Maximums (Zerfall ~2.7 s): nach Stille darf leise
      // zuenden, neben lautem Klang nur Lautes.
      static const float PRE_LOUD = [] {
        const char *e = getenv("XO_PRE_LOUD");

        return e ? (float)atof(e) : 0.01f;
      }();
      double eTot = 0.0;

      for (unsigned b = 0; b < ROUTE_BANDS; b++)
        eTot += cur[b];
      m_PreEmax[ch] = std::max((float)eTot, m_PreEmax[ch] * 0.999f);

      const bool loud = eTot >= (double)PRE_LOUD * m_PreEmax[ch];

      if (wideNow >= PRE_ON && loud) {
        m_PreStable[ch] = 0;
        if (m_PreCandP[ch] == ~0ull) {
          // ZUENDUNG - das eine Ereignis dieses Anschlags.
          m_PreCandP[ch] = nowP;
          HybridOnset(ch, rise512, nbins, bandShift, true);
          m_PreChk[ch] = (uint8_t)PRE_CHK;
          m_PreResMax[ch] = 0.0f;
          std::fill_n(m_PreResMaxB.begin() + (size_t)ch * ROUTE_BANDS,
                      ROUTE_BANDS, 0.0f);
          std::copy_n(old8, ROUTE_BANDS,
                      m_PreBase.begin() + (size_t)ch * ROUTE_BANDS);
          std::copy_n(rise512, ROUTE_BANDS,
                      m_PreRise.begin() + (size_t)ch * ROUTE_BANDS);
        } else
          // Anschlag entfaltet sich: neue steigende Baender treten bei.
          HybridOnset(ch, rise512, nbins, bandShift, true);
      } else if (m_PreCandP[ch] != ~0ull && wideNow < PRE_OFF) {
        if (++m_PreStable[ch] >= 4) {
          m_PreCandP[ch] = ~0ull; // beruhigt - bereit fuer den naechsten
          m_PreStable[ch] = 0;
        }
      } else
        m_PreStable[ch] = 0;
    }

    // Waehrend der Check laeuft: Maximum des mittleren
    // A-Vorhersagefehlers mitfuehren.  Der EMA-Momentanwert AM
    // Checkpunkt verpasste den schwaechsten Einsatz (7,26 s), dessen
    // Residuum erst um die Checkzeit kreuzt.
    if (m_PreChk[ch]) {
      const float *ra = m_HybResA.data() + (size_t)ch * ROUTE_BANDS;
      float *rb = m_PreResMaxB.data() + (size_t)ch * ROUTE_BANDS;
      double m = 0.0;

      for (unsigned b = 0; b < ROUTE_BANDS; b++) {
        m += ra[b];
        if (ra[b] > rb[b])
          rb[b] = ra[b];
      }
      m /= ROUTE_BANDS;
      if ((float)m > m_PreResMax[ch])
        m_PreResMax[ch] = (float)m;
    }

    // PERSISTENZ-CHECK: laeuft unabhaengig vom Veto (er vergleicht nur
    // eingefrorene Vergangenheit mit dem aktuellen Rahmen).
    if (m_PreChk[ch] && --m_PreChk[ch] == 0) {
      const float *base = m_PreBase.data() + (size_t)ch * ROUTE_BANDS;
      const uint8_t *orig = m_PreRise.data() + (size_t)ch * ROUTE_BANDS;
      unsigned nOrig = 0, nStill = 0;

      for (unsigned b = 0; b < ROUTE_BANDS; b++)
        if (orig[b]) {
          nOrig++;
          if (cur[b] > 2.0f * base[b])
            nStill++;
        }

      static const float PRE_KEEP = [] {
        const char *e = getenv("XO_PRE_KEEP");

        return e ? (float)atof(e) : 0.5f;
      }();
      // ZWEITES, UNABHAENGIGES Urteil: der mittlere Vorhersagefehler
      // der LANGEN Kette.  Langsame Schwellungen sind fuer das
      // AR-Modell vorhersagbar (sie BESTEHEN die Band-Persistenz,
      // resA bleibt unten), ein neuer phasenkohaerenter Ton nicht
      // (Nutzer-These, kalibriert: Stoermax 0.0258, Einsatz-Min
      // 0.0279, Latenz 16-35 ms = zur Checkzeit gekreuzt).  Zwei
      // 1.06er-Margen, UND-verknuepft.
      static const float PRE_RES = [] {
        const char *e = getenv("XO_PRE_RES");

        return e ? (float)atof(e) : 0.027f;
      }();
      // PER-BAND-Urteil (Nutzer-Go 08.08.): ein Band bleibt nur, wenn
      // es SELBST persistent gestiegen ist UND sein EIGENES Residuum
      // seit der Zuendung gesprungen ist (unvorhersagbar).  Schwebungs-
      // und Schwellungsbaender sind persistent, aber vorhersagbar -
      // sie fliegen einzeln raus, waehrend die Anschlagsbaender
      // bleiben.  Das Kanalmittel scheiterte, weil an den
      // Zuendmomenten alle Kanal-Kriterien korrelieren.
      static const float PRE_RESB = [] {
        const char *e = getenv("XO_PRE_RESB");

        // 0.10 gewann den Sweep: Schwebung bei jeder Schwelle sauber,
        // Motor fast weg, und bei 6000/min bester Schlimmstfall (7.70)
        // und beste Einsatz-Luecken (+0.90) der ganzen Messreihe.
        return e ? (float)atof(e) : 0.10f;
      }();
      const float *rbMax = m_PreResMaxB.data() + (size_t)ch * ROUTE_BANDS;
      const int handRaw = m_HybHandOrder.load(std::memory_order_relaxed);
      const unsigned handP
        = ((unsigned)std::max(1, handRaw > 0 ? handRaw : 6) + 2) * arStride;
      (void)PRE_RES;
      (void)PRE_KEEP;
      (void)nOrig;
      (void)nStill;

      for (unsigned b = 0; b < ROUTE_BANDS; b++) {
        uint8_t &tn = m_HybTent[(size_t)ch * ROUTE_BANDS + b];

        if (!tn)
          continue;
        tn = 0;

        const bool persistB = cur[b] > 2.0f * base[b];
        const bool residB = rbMax[b] >= PRE_RESB;

        if (persistB && residB)
          continue; // fest
        if (m_HybAge[(size_t)ch * ROUTE_BANDS + b] < handP) {
          m_HybAge[(size_t)ch * ROUTE_BANDS + b] = (uint16_t)handP;
          m_HybRetr[(size_t)ch * ROUTE_BANDS + b] = 1;
        }
      }
    }
  }

  // ---- HYBRID: A-Residuum fuer den Waehler -----------------------------
  // Wie gut haetten die aktuellen A-Koeffizienten diesen Rahmen aus den
  // Ring-Lags vorhergesagt?  Relative Fehler, per Band, EMA ~3 Perioden.
  // Auch fuer den Simpel-Detektor: er braucht das Residuum als
  // Trennkriterium (siehe Zuendstelle weiter unten).  Vorher lief der
  // Rechner nur fuer Waehler und Vorstufe - beide in GO aus.
  if (hybOn
      && (m_HybChooser.load(std::memory_order_relaxed)
          || m_HybPre.load(std::memory_order_relaxed)
          || m_HybSimple.load(std::memory_order_relaxed))
      && arUsable && P >= 1) {
    const size_t ringBase = (size_t)ch * m_ArRingLen;
    bool lagOk = nowP >= (uint64_t)P * arStride;
    const float *lRe[AR_ORDER_MAX], *lIm[AR_ORDER_MAX];

    for (unsigned k = 1; lagOk && k <= P; k++) {
      const uint64_t want = nowP - (uint64_t)k * arStride;
      const unsigned ks = (unsigned)(want % m_ArRingLen);

      if (m_ArRingPeriod[ringBase + ks] != want) {
        lagOk = false;
        break;
      }
      lRe[k - 1] = m_ArRingRe.data() + (ringBase + ks) * NBINS_MAX;
      lIm[k - 1] = m_ArRingIm.data() + (ringBase + ks) * NBINS_MAX;
    }
    if (lagOk) {
      const float *cReAll
        = m_ArCoefRe.data() + (size_t)ch * NBINS_MAX * AR_ORDER_MAX;
      const float *cImAll
        = m_ArCoefIm.data() + (size_t)ch * NBINS_MAX * AR_ORDER_MAX;
      float *res = m_HybResA.data() + (size_t)ch * ROUTE_BANDS;
      double err = 0.0, ene = 0.0;
      unsigned curBand = 0;

      for (unsigned bin = 0; bin <= nbins; bin++) {
        const unsigned b
          = bin < nbins ? std::min(bin >> bandShift, ROUTE_BANDS - 1) : ~0u;

        if (b != curBand) {
          const float r = (float)(err / (ene + 1e-20));

          res[curBand] = 0.7f * res[curBand] + 0.3f * std::min(r, 10.0f);
          err = ene = 0.0;
          curBand = b;
          if (bin >= nbins)
            break;
        }

        const float *cR = cReAll + (size_t)bin * AR_ORDER_MAX;
        const float *cI = cImAll + (size_t)bin * AR_ORDER_MAX;
        float pr = 0.0f, pi = 0.0f;

        for (unsigned k = 0; k < P; k++) {
          pr += cR[k] * lRe[k][bin] - cI[k] * lIm[k][bin];
          pi += cR[k] * lIm[k][bin] + cI[k] * lRe[k][bin];
        }

        const float dr = specRe[bin] - pr, di = specIm[bin] - pi;

        err += (double)dr * dr + (double)di * di;
        ene += (double)specRe[bin] * specRe[bin]
          + (double)specIm[bin] * specIm[bin];
      }
    }
  }

  // ---- Transienten-Patch: Fluss-Trigger --------------------------------
  // Einsatz, wenn der positive Bandfluss ueber die 8-Perioden-Baseline
  // springt; stabilisiert, wenn er STABLE_K Perioden am Stueck wieder
  // unten liegt.  Waechter wie gehabt: saubere letzte 16 Perioden,
  // Entprellung, Reichweitenpruefung in BuildBackPatch.
  // Der Detektor laeuft IMMER (Kosten: eine Handvoll Vergleiche) und
  // meldet jeden stabilisierten Einsatz ins Ereignisprotokoll - das Lab
  // zeigt ihn als Linie, auch wenn nicht gepatcht wird.  Nur der
  // PATCH-Bau haengt am Modus und seinen Waechtern.
  // Simpel-Modus: Erkennung durch die 4k-Augen, mit 4k-Schwellen und
  // 4k-Rise-Menge - unabhaengig vom Verdeckungsrahmen.
  const bool detSimple
    = hybSimple && m_D4Wide[ch] >= 0.0f;
  const double detWide = detSimple ? (double)m_D4Wide[ch] : onsetWide;
  const unsigned detNfft = detSimple ? 4096u : nfft;
  const uint8_t *detRise
    = detSimple ? m_D4Rise.data() + (size_t)ch * ROUTE_BANDS : riseB;

  if (detWide >= 0.0) {
    // RAHMENABHAENGIG: die Breite haengt an der Mittelungslaenge des
    // Analyse-Fensters.  4k (85 ms) kalibriert am 07.08.: gehalten p99
    // 0.217, Schwebung p99 0.113, Einsaetze 0.31-0.87 -> ON 0.25.
    // 2k (43 ms) mittelt halb so lang, alles zappelt hoeher: gehalten
    // p99 0.268/max 0.328, Schwebung p99 0.258, Einsaetze 0.363-0.887
    // -> ON 0.33.  Mit den 4k-Schwellen auf 2k feuerte der Detektor 48x
    // gleichverteilt ueber die Datei.  1k ist NICHT kalibriert
    // (konservativ hoch geraten).
    static const double envOn = [] {
      const char *e = getenv("XO_WIDE_ON");

      return e ? atof(e) : 0.0;
    }();
    const double FLUX_ON = envOn > 0.0
      ? envOn
      : (detNfft <= 1024 ? 0.40 : (detNfft <= 2048 ? 0.33 : 0.25));
    // Beruhigung ZWISCHEN Ruhelage-p95 (0.058) und schwaechstem Einsatz
    // (0.121) - die erste Fassung lag mit ON*0.4 = 0.04 UNTER dem
    // Ruhelage-Median: sechs Perioden am Stueck darunter waren ein
    // Wuerfelspiel, drei von fuenf Einsaetzen wurden nie gepatcht und
    // verwaiste Kandidaten feuerten hunderte Millisekunden spaeter.
    static const double envOff = [] {
      const char *e = getenv("XO_WIDE_OFF");

      return e ? atof(e) : 0.0;
    }();
    const double FLUX_OFF = envOff > 0.0
      ? envOff
      : (detNfft <= 1024 ? 0.22 : (detNfft <= 2048 ? 0.15 : 0.12));
    static constexpr unsigned STABLE_K = 6;
    // Zweite Verteidigungslinie WIEDERHOLRATE: die 9-Hz-Schwebung
    // feuert im 110-ms-Takt, echte Zuschaltungen liegen Sekunden
    // auseinander.  0.5 s Entprellung laesst jeden echten Einsatz durch.
    // Die alten 4096 Samples (85 ms) lagen knapp UNTER der
    // Modulationsperiode - der denkbar schlechteste Wert.
    static const uint64_t DEBOUNCE = [] {
      const char *e = getenv("XO_PATCH_DEBOUNCE");

      return e ? (uint64_t)atoll(e) : 24000ull;
    }();
    // Verfallsdatum: wer sich binnen 48 Perioden (~128 ms) nicht
    // beruhigt, ist keine Zuschaltung, sondern laufende Bewegung -
    // dort ist KEIN Patch die richtige Antwort.
    static constexpr uint64_t CANDIDATE_TTL = 96; // 256 ms Wiederhol-Raum

    // RUHE DAVOR ist der Tremulant-Diskriminator, nicht die Hoehe: der
    // Tremulant erreicht 0.345 Fluss, echte Zuschaltungen nur 0.12-0.27
    // - ueber die Amplitude sind sie NICHT trennbar.  Ueber die
    // Struktur schon: gemessen auf dem D-Dur-Material ist die laengste
    // ruhige Straehne vor echten Einsaetzen ueber 1200 ms, im
    // Tremulant-Abschnitt 35 ms.  QUIET_BEFORE = 96 Perioden (256 ms)
    // trennt mit Faktor 7 Sicherheitsabstand.
    //
    // Die erste Fassung eichte ausgerechnet AUF dem Tremulant-Abschnitt
    // und nannte ihn "gehalten" - 44 der 58 Ereignisse waren dann seine
    // 9-Hz-Modulation im 110-ms-Takt, und die 85-ms-Entprellung lag
    // knapp unter seiner Periode.
    // GEMESSEN UND VERWORFEN: die Ruhe vor dem Sprung trennt NICHT -
    // sie trennt falsch herum.  Vor echten Zuschaltungen liegen 1 bis 9
    // ruhige Perioden (3-24 ms), vor Schwebungs-Spitzen median 11 und
    // bis 13.  Der Waechter haette also die Modulation bevorzugt.
    // Bleibt als Knopf (Vorgabe 0 = aus), fuer die Reproduzierbarkeit.
    static const unsigned QUIET_BEFORE = [] {
      const char *e = getenv("XO_QUIET_BEFORE");

      return e ? (unsigned)atoi(e) : 0u;
    }();

    // SPLEISS-VETO.  Der Detektor sieht nie Luecken - der Ring haelt
    // seit dem Fade-im-Ring nur aufbereitetes Material.  Was ihn
    // taeuscht, ist der WIEDEREINSTIEG: der in der Vorhersage
    // kollabierte Rauschgrund springt beim Spleiss breitbandig zurueck,
    // und das ist formal derselbe Puls wie ein Anschlag.  Gemessen am
    // D-Dur-Material bei 6000 Luecken/min (Breite im Schwebungs-
    // Abschnitt, nach Abstand zum Wiedereinstieg):
    //
    //   0-1 Perioden   p99 0.198   max 0.262   2x ueber ON
    //   2-3 Perioden   p99 0.173   max 0.215   nie ueber ON
    //   >=4 Perioden   p99 0.133   max 0.172   = sauberer Lauf
    //
    // Alle vier Falschzuendungen sassen 1-2 Perioden hinter einem
    // Wiedereinstieg.  Zwei Perioden Sperre genuegen also; das Veto ist
    // reine Rueckschau auf die Vorhersagemaske, also vollstaendig
    // kausal und in GO genauso zu haben.
    static const unsigned SPLICE_VETO = [] {
      const char *e = getenv("XO_SPLICE_VETO");

      return e ? (unsigned)atoi(e) : 2u;
    }();
    const uint32_t mask = RoundMask();
    // Bit 0 = diese Periode, Bit 1 = die vorige: waehrend der Analyse
    // echter Audiodaten ist Bit 0 stets frei, Bit 1 markiert genau die
    // Wiedereinstiegsperiode.
    const uint32_t vetoMask = (SPLICE_VETO >= 31)
      ? ~0u
      : ((1u << (SPLICE_VETO + 1)) - 1u);

    if (hybOn && m_HybPend[ch]) {
      m_HybPend[ch]--;
      if ((mask & vetoMask) == 0) {
        HybridOnset(
          ch, m_HybPendRise.data() + (size_t)ch * ROUTE_BANDS, nbins,
          bandShift, false);
        m_HybPend[ch] = 0;
      }
      if (!m_HybPend[ch])
        std::fill_n(
          m_HybPendRise.begin() + (size_t)ch * ROUTE_BANDS, ROUTE_BANDS,
          (uint8_t)0);
    }
    if (detWide < FLUX_OFF) {
      if (m_QuietRun[ch] < 100000u)
        m_QuietRun[ch]++;
    } else
      m_QuietRun[ch] = 0;
    if (
      m_OnsetPeriod[ch] != ~0ull
      && nowP - m_OnsetPeriod[ch] > CANDIDATE_TTL) {
      m_OnsetPeriod[ch] = ~0ull;
      m_StableCount[ch] = 0;
    }
    if (detWide > FLUX_ON) {
      // ZUENDUNG = Erkennung.  Gemeldet wird JETZT, nicht nach der
      // Beruhigung: alles, was frueher zwischen Zuendung und Meldung
      // lag (Beruhigung, Verfallsdatum, Entprellung), war Geruest fuer
      // den RUECKWAERTS-Patch - der musste warten, bis der Transient
      // vorbei war.  Ein Modellwechsel dagegen muss im Moment des
      // Anschlags greifen, und ein nachtraegliches Verwerfen des
      // Kandidaten waere in GO ohnehin unmoeglich (Nutzer, 07.08.).
      // Das Geruest bleibt nur noch fuer den Patch-Zweig stehen.
      if (
        m_OnsetPeriod[ch] == ~0ull && m_QuietRun[ch] >= QUIET_BEFORE
        && (mask & vetoMask) == 0) {
        m_OnsetPeriod[ch] = nowP;
        if (ch == 0) {
          const uint64_t n = s_PatchLogN.load(std::memory_order_relaxed);
          uint64_t *e = s_PatchLog[n % PATCH_LOG];

          // Zone/Laenge 0 heisst "erkannt, nicht gepatcht"; ein
          // spaeterer Patch meldet denselben Einsatz mit Zone nach.
          e[0] = nowP * m_SamplesPerBuffer;
          e[1] = 0;
          e[2] = 0;
          s_PatchLogN.store(n + 1, std::memory_order_relaxed);
        }
      }
      // HYBRID: Baender wechseln nur mit LEBENDEM Kandidaten - also
      // unter genau den Bedingungen (Veto, Verfallsdatum), unter denen
      // die Maschinerie einen Einsatz von Spleiss-Residuen und
      // laufender Bewegung trennt.  Die erste Fassung hing nur am Veto:
      // jeder Breiten-Ausreisser in lauten Abschnitten schaltete
      // Baender mitten im Plateau auf die kurze Kette (Rest-Luecken
      // -2.55 -> -0.66).  Solange der Puls anhaelt, treten weitere
      // steigende Baender bei (der Anschlag entfaltet sich).
      // Eigene Latte fuer den KETTENWECHSEL: 0.28 trennt die echten
      // Einsaetze (schwaechster 0.297) von Doppelzuendungen und dem
      // Schwebungs-Ausreisser (0.262-0.273).  Beim Patch war eine
      // Falschzuendung ein no-op - hier kostet sie 50-80 ms kurze Kette
      // im Plateau, wo die 8.5 dB verliert (Schwebung -6.7 vs +1.8).
      static const double envHyb = [] {
        const char *e = getenv("XO_HYB_WIDE");

        return e ? atof(e) : 0.0;
      }();
      // Simpel: geschaltet wird AN der Erkennungsschwelle - die
      // separate (hoehere) Schaltlatte gab es damals nicht, und sie
      // kostete die zwei leisen Einsaetze (Spitzen 0.26-0.31 bei 4k).
      const double HYB_WIDE = envHyb > 0.0
        ? envHyb
        : (detSimple ? FLUX_ON : (detNfft <= 2048 ? 0.34 : 0.28));

      // Der Wechselwunsch haengt NICHT am Kandidaten: beim
      // schwaechsten Einsatz (7,30 s @ 6000/min) blockierte das Veto
      // alle starken Perioden - dann entsteht auch nie ein Kandidat,
      // und ein kandidatengebundener Wunsch verhungerte mit.  Die
      // Schaltlatte 0.34 ist ohnehin die schaerfere Huerde.
      // ---- RESIDUEN-TOR (XO_D4_RES, Vorgabe 0 = aus) ----------------
      // Die Breite allein kann Schwebung und Ausklang nicht von einem
      // Anschlag trennen: beide heben Bandenergien an, und in den leisen
      // Baendern, wo der eigene Vorhersagefehler wohnt, ist Faktor zwei
      // billig.  Eine PEGEL-Untergrenze scheitert daran, dass echte
      // Anschlagsbaender im selben Bereich liegen (gemessen: schon
      // 0,001 der mittleren Bandenergie loescht alle sechs Einsaetze).
      //
      // Was trennt, ist die VORHERSAGBARKEIT: eine Schwebung setzt die
      // lange Kette fort, ein Anschlag nicht.  m_HybResA traegt genau
      // das je Band - den relativen Vorhersagefehler der A-Koeffizienten
      // auf diesen Rahmen.  Ein Band zaehlt also nur, wenn es steigt UND
      // sein eigenes Residuum gesprungen ist; die Breite wird aus den so
      // bestaetigten Baendern neu gebildet.  Derselbe Gedanke wie beim
      // Per-Band-Urteil der 512er-Vorstufe, dort gemessen mit
      // "Schwebung 0 bestaetigte Baender bei jeder Schwelle".
      const float D4_RES = m_D4ResGate.load(std::memory_order_relaxed);
      const uint8_t *fireRise = detRise;
      double fireWide = detWide;

      if (D4_RES > 0.0f && detSimple) {
        // Abklingrate: 0.85 je Periode haelt das Maximum rund acht
        // Perioden lang oben (0.85^8 = 0.27) - dieselbe Groessenordnung
        // wie das Prueffenster der Vorstufe.
        const float D4_DECAY = m_D4ResDecay.load(std::memory_order_relaxed);
        const float *ra = m_HybResA.data() + (size_t)ch * ROUTE_BANDS;
        float *rm = m_D4ResMax.data() + (size_t)ch * ROUTE_BANDS;
        uint8_t *conf = m_D4Conf.data() + (size_t)ch * ROUTE_BANDS;
        unsigned nConf = 0;

        for (unsigned b = 0; b < ROUTE_BANDS; b++) {
          rm[b] = std::max(ra[b], rm[b] * D4_DECAY);
          conf[b] = detRise[b] && rm[b] >= D4_RES;
          nConf += conf[b];
        }
        fireRise = conf;
        fireWide = (double)nConf / ROUTE_BANDS;
      }
      if (hybOn && fireWide >= HYB_WIDE) {
        if ((mask & vetoMask) == 0) {
          if (m_OnsetPeriod[ch] != ~0ull)
            HybridOnset(ch, fireRise, nbins, bandShift, false);
        } else if (!m_HybSimple.load(std::memory_order_relaxed)) {
          m_HybPend[ch] = 8;
          for (unsigned b = 0; b < ROUTE_BANDS; b++)
            m_HybPendRise[(size_t)ch * ROUTE_BANDS + b] |= riseB[b];
        }
      }
      m_StableCount[ch] = 0;
    } else if (m_OnsetPeriod[ch] != ~0ull && detWide < FLUX_OFF) {
      if (++m_StableCount[ch] >= STABLE_K) {
        const uint64_t onsetSamp = m_OnsetPeriod[ch] * m_SamplesPerBuffer;

        // Blockt der Verdeckungs-Waechter, bleibt der Kandidat AM LEBEN
        // und versucht es naechste Periode erneut (bis die TTL greift).
        // Die erste Fassung verwarf ihn - bei 600 Luecken/min lag fast
        // immer eine Luecke in den letzten 16 Perioden, und 4 von 6
        // korrekt erkannten Einsaetzen verfielen ungenutzt.
        // 8 statt 16 saubere Perioden: der Patch-Bau liest ohnehin ein
        // 32-Perioden-Fenster - die 16 waren immer ein Kompromiss, und
        // bei 600 Luecken/min blockten sie trotz Wiederholens die
        // Haelfte der korrekt erkannten Einsaetze.
        const bool retry = m_PatchMode.load(std::memory_order_relaxed)
          && (mask & 0xFFu) != 0;

        if (
          !retry && m_PatchMode.load(std::memory_order_relaxed)
          && onsetSamp > 256
          && onsetSamp - m_LastPatchSample[ch] >= DEBOUNCE
          && BuildBackPatch(ch, onsetSamp - 256, 768)) {
          m_LastPatchSample[ch] = onsetSamp;

          s_PatchCount.fetch_add(1, std::memory_order_relaxed);
          if (ch == 0) { // sonst zaehlt die Anzeige jeden Einsatz doppelt
            const uint64_t n = s_PatchLogN.load(std::memory_order_relaxed);
            uint64_t *e = s_PatchLog[n % PATCH_LOG];

            e[0] = onsetSamp;
            e[1] = onsetSamp - 256;
            e[2] = 768;
            s_PatchLogN.store(n + 1, std::memory_order_relaxed);
          }
          if (getenv("XO_SIM_TRACE") && ch == 0)
            fprintf(
              stderr, "PATCH %.3f s (Einsatz war %.3f s)\n",
              (double)nowP * m_SamplesPerBuffer / m_SampleRate,
              (double)onsetSamp / m_SampleRate);
        }
        // Wiederholen heisst: Kandidat lebt weiter, eine Periode vor der
        // Beruhigungsschwelle geparkt.  Die vorige Fassung setzte ihn
        // hier bedingungslos zurueck und hob damit genau das auf.
        if (retry) {
          m_StableCount[ch] = STABLE_K - 1;
        } else {
          m_OnsetPeriod[ch] = ~0ull;
          m_StableCount[ch] = 0;
        }
      }
    } else
      m_StableCount[ch] = 0;
  }

  // 0 = pure anchor (sustained), 1 = pure Burg bridge (transient)
  float wBridge = (float)std::min(1.0, std::max(0.0, (dens - 0.003) / 0.017));

  // CLEAN-SEED GATE.  The Burg bridge REPLACES the seam with a free run
  // seeded from the newest ring samples - if any of those came from the
  // predictor itself, the bridge reproduces (and re-amplifies) the
  // prediction instead of bridging towards reality.  Worse, the pollution
  // also drags the similarity down, so the discriminator flips to the
  // bridge exactly when the bridge is at its worst: a second feedback
  // loop through the ROUTING.  Measured on a real organ chord buildup at
  // 27 % concealment: adaptive was pinned at wBridge = 1.00 even on a
  // single tone; gating recovers -4.9 -> -8.8 dB (tone) and -3.0 -> -4.3
  // (full chord) while dense material with clean seeds is unaffected
  // (Vierne: +-0.1 dB at both 5 % and 27 %).
  {
    const unsigned seedPeriods
      = (BURG_WINDOW + m_SamplesPerBuffer - 1) / m_SamplesPerBuffer;
    const uint32_t mask = (seedPeriods >= 32)
      ? ~0u
      : ((1u << seedPeriods) - 1u);

    if (RoundMask() & mask)
      wBridge = 0.0f;
  }

  switch (m_SeamMode.load(std::memory_order_relaxed)) {
  case XORCIST_SEAM_ANCHOR:
    wBridge = 0.0f;
    break;
  case XORCIST_SEAM_BRIDGE:
    wBridge = 1.0f;
    break;
  default:
    break;
  }

  // ---- level trend (GLOBAL, density gated) -----------------------------
  // Continue the LEVEL change between the two frames, not only the phase,
  // so a swell or a decay keeps moving through the gap.
  //
  // Deliberately ONE SCALAR, not a per-bin ratio: |F1_k|/|F0_k| cannot
  // tell "this partial got louder" from "this partial moved into the next
  // bin".  Measured on a glissando, 36 % of all bins exceeded the clamp
  // while the global level did not move at all, and beating on a detuned
  // chord did the same in 11 % - both fed a crescendo that was not there.
  // The RMS ratio is immune, because energy that merely migrates between
  // bins cancels in the sum.  It is also free (both sums already exist
  // for `sim`) and it removes the per-bin divide the earlier version had.
  //
  // Gated by the SAME discriminator as the seam: on dense material the
  // level fluctuation is beating rather than a real envelope, so the
  // exponent fades to 0 there.  Measured effect of the gate: dense
  // orchestral material returns exactly to its trend-off numbers while
  // the swell (-1.0 -> -25.2 dB excess) and the held tone keep the gain.
  // ---- Sollbruchstelle: Analyse | Synthese -----------------------------
  // Alles OBERHALB fuettert Zustand (Spektrum-Cache, Rotator,
  // Routing-Ring, AR-Ring, Normalgleichungen); alles UNTERHALB erzeugt
  // nur den Schatten.  Wenn der Aufrufer weiss, dass keine Luecke nahe
  // ist - das Offline-Lab kennt seine Positionen -, ist der Schatten
  // weggeworfene Arbeit: bei 60 Luecken/min verfallen 6582 von 6634
  // Schatten pro Datei ungehoert.  Absorb() setzt das Flag und spart
  // die Synthese; der Analyse-Zustand bleibt EXAKT der der Engine.
  arClock.Stop(XO_ST_AR_AKKU);
  if (m_AnalyzeOnly)
    return;

  XoClock synClock(XoProfiling());

  // ---- AR(p): solve the recursion for this prediction ------------------
  // The coefficients come out of a ~20-frame memory, so they are solved
  // once per AR_SOLVE_EVERY grid points and cached; only the STATE is
  // refreshed every period, and that is what the extrapolation is
  // sensitive to.
  bool useAr = false;
  const unsigned arSteps = arUsable ? synHop / arHop : 0;

  if (arUsable && wantOrder >= 2) {
    const size_t ringBase = (size_t)ch * m_ArRingLen;

    // Warm-up must scale with the ORDER: after w rank-1 updates the
    // p x p system has rank at most w, and solving it rank-deficient
    // produces garbage coefficients whose roots then get stored as the
    // Durand-Kerner warm start - a self-sustaining attractor of garbage
    // that explained order 12 reading -1.4 dB on a cluster where the
    // prototype reaches -48.  A fixed AR_WARMUP = 4 was enough for order
    // 8 by luck (rank deficit 4), not by design.
    useAr = nowP >= (uint64_t)P * arStride && m_ArNfft[ch] == nfft
      && m_ArFitCount[ch] >= P + 2;
    const bool wantedAr = useAr;

    for (unsigned k = 0; k < P && useAr; k++) {
      const uint64_t want = nowP - (uint64_t)(k + 1) * arStride;

      if (m_ArRingPeriod[ringBase + (unsigned)(want % m_ArRingLen)] != want)
        useAr = false;
    }
    if (wantedAr) {
      s_ArTries.fetch_add(1, std::memory_order_relaxed);
      if (!useAr)
        s_ArDrops.fetch_add(1, std::memory_order_relaxed);
    }
    if (useAr) {
      // State: [0] = X(n), [k] = X(n - k*arHop).  Copied because the
      // recursion overwrites it as it walks the horizon.
      for (unsigned bin = 0; bin < nbins; bin++) {
        stRe[bin] = specRe[bin];
        stIm[bin] = specIm[bin];
      }
      for (unsigned k = 1; k < P; k++) {
        const uint64_t want = nowP - (uint64_t)k * arStride;
        const unsigned sl = (unsigned)(want % m_ArRingLen);
        const float *sRe = m_ArRingRe.data() + (ringBase + sl) * NBINS_MAX;
        const float *sIm = m_ArRingIm.data() + (ringBase + sl) * NBINS_MAX;

        std::copy_n(sRe, nbins, stRe + (size_t)k * NBINS_MAX);
        std::copy_n(sIm, nbins, stIm + (size_t)k * NBINS_MAX);
      }
    }
    if (useAr) {
      // STRIPED solve: a rolling 1/N of the bins every period instead of
      // ALL bins every AR_SOLVE_EVERY-th grid point.
      //
      // MEASURED PRICE of this plus AR_SOLVE_EVERY plus the warm start,
      // against a reference that re-solves every bin at every grid point
      // from a cold seed - 24 cases, orders 2/3/4, four materials, at
      // 5 % and 27 % concealment: worst case 0.31 dB, and it sits at
      // ORDER 2 on a sustained tone (-24.40 -> -24.09), where the
      // prediction is so good that a slightly stale coefficient set is
      // the largest remaining term.  Orders 3 and 4 are within 0.26 dB
      // everywhere and mostly within 0.05; on the full chord and on
      // Vierne the whole effect is under 0.13 dB.  The Durand-Kerner
      // exit (1e-20, 40 iterations) costs exactly 0.000 dB against
      // 1e-30/300 - it converges long before either limit bites.  The point solve was
      // a hidden spike - measured up to 10.8 ms serial at order 4 (six
      // channels), 1.5-2.1 ms on the parallel shadow threads, surfacing
      // in the monitor's new max meter the moment it existed.  The cycle
      // length equals the old solve interval, so each bin's coefficients are
      // exactly as fresh as before; they are merely renewed in turn
      // rather than in one burst.  A change of order re-solves everything
      // at once - stale-order coefficients must not drive the recursion.
      const size_t pStride = (size_t)AR_ORDER_MAX * AR_ORDER_MAX * 2;
      const size_t qStride = (size_t)AR_ORDER_MAX * 2;
      double gP[AR_ORDER_MAX * AR_ORDER_MAX * 2], gQ[AR_ORDER_MAX * 2];
      float *cRe = m_ArCoefRe.data() + (size_t)ch * NBINS_MAX * AR_ORDER_MAX;
      float *cIm = m_ArCoefIm.data() + (size_t)ch * NBINS_MAX * AR_ORDER_MAX;
      float *rRe = m_ArRootRe.data() + (size_t)ch * NBINS_MAX * AR_ORDER_MAX;
      float *rIm = m_ArRootIm.data() + (size_t)ch * NBINS_MAX * AR_ORDER_MAX;
      // Diagnostic knobs for the order-12 hunt: XO_STRIPE_CYCLE=1 solves
      // every bin every period (no coefficient age), XO_NO_WARMSTART=1
      // seeds Durand-Kerner cold each time (no persistent roots).
      static const unsigned envCycle = [] {
        const char *e = getenv("XO_STRIPE_CYCLE");

        return e ? (unsigned)atoi(e) : 0u;
      }();
      // solveEvery: 1 = jeder Bin in jeder Periode neu aufgeloest, wie
      // der Prototyp es bei jeder Luecke tut.  0 = der gestreifte
      // Standard, jeder Bin alle AR_SOLVE_EVERY*arStride = 16 Perioden.
      // Gemessen an den Toneinsaetzen des Nutzers: 1,22 dB von 1,52.
      const int solveEv = m_Replica.load(std::memory_order_relaxed)
        ? 1 : m_SolveEvery.load(std::memory_order_relaxed);
      const unsigned cycle = solveEv > 0
        ? (unsigned)solveEv
        : (envCycle ? envCycle : AR_SOLVE_EVERY * arStride);
      const bool full
        = m_ArSolvedPeriod[ch] == ~0ull || m_ArSolvedOrder[ch] != wantOrder;
      const unsigned stripe = (unsigned)(nowP % cycle);
      const unsigned b0 = full ? 0 : nbins * stripe / cycle;
      const unsigned b1 = full ? nbins : nbins * (stripe + 1) / cycle;

      static const bool envNoWarm = getenv("XO_NO_WARMSTART") != nullptr;
      const bool noWarm
        = envNoWarm || m_Replica.load(std::memory_order_relaxed);

      m_ArSolvedPeriod[ch] = m_ArFitPeriod[ch];
      m_ArSolvedOrder[ch] = wantOrder;
      if (noWarm)
        for (unsigned bin = b0; bin < b1; bin++)
          for (unsigned k = 0; k < AR_ORDER_MAX; k++) {
            rRe[(size_t)bin * AR_ORDER_MAX + k] = 0.0f;
            rIm[(size_t)bin * AR_ORDER_MAX + k] = 0.0f;
          }
      // ---- RECHTECKFENSTER statt exponentiellem Vergessen -----------
      // Der Offline-Prototyp fittet ueber J gleichgewichtete Rahmen und
      // vergisst danach ABRUPT alles; die Akkumulatoren hier vergessen
      // exponentiell und werden nichts je ganz los.  Nominell gleich
      // lang (J=20 gegen 1/(1-lambda)=22), aber an einer Ansprache ist
      // der Unterschied gross: der Prototyp ist den alten Ton nach 20
      // Rahmen los, die Bibliothek schleppt ihn gedaempft weiter und
      // verbraucht Freiheitsgrade darauf, ihn mitzumodellieren.  Das ist
      // die verbliebene Erklaerung dafuer, dass der Prototyp mit
      // nominell vier Freiheitsgraden leistet, wofuer diese Bibliothek
      // acht bis zwoelf braucht.
      //
      // Neu aufgebaut wird aus dem Spektrenring, genau wie der Prototyp
      // es aus seiner Historie tut.  Der Ring haelt AR_ORDER_MAX*stride+1
      // Perioden, also J <= (ringLen-1)/stride - P + 1 Tupel.
      const bool replica = m_Replica.load(std::memory_order_relaxed) != 0;
      const unsigned fitWin = replica
        ? 20u : (unsigned)m_FitWindow.load(std::memory_order_relaxed);

      if (fitWin) {
        const unsigned jMax = (m_ArRingLen - 1) / arStride + 1 - P;
        const unsigned J = std::min(fitWin, jMax);
        // Der Rebuild bleibt je Bin und rechnet in einen DICHTEN
        // Zwischenspeicher - dieselbe Summationsreihenfolge wie vorher,
        // also bitgleiche Werte.  Erst am Ende wird das obere Dreieck in
        // die planare Ablage geschrieben.  Der Pfad ist in GO ohnehin aus
        // (fitWindow 0) und nur im Replica-Modus scharf.
        double wPb[AR_ORDER_MAX * AR_ORDER_MAX * 2], wQb[AR_ORDER_MAX * 2];
        bool ok = J >= P + 2 && nowP >= (uint64_t)(J - 1 + P) * arStride;

        for (unsigned j = 0; ok && j < J + P; j++) {
          const uint64_t want = nowP - (uint64_t)j * arStride;

          if (m_ArRingPeriod[ringBase + (unsigned)(want % m_ArRingLen)] != want)
            ok = false;
        }
        if (ok)
          for (unsigned bin = b0; bin < b1; bin++) {
            // Der Rechteck-Rebuild wuerde den Hybrid-Reset rueckgaengig
            // machen - junge Baender behalten ihren frischen Akkumulator.
            if (hybOn && m_HybReset.load(std::memory_order_relaxed)
                && HybBandYoung(ch, bin >> bandShift))
              continue;

            double *Pb = wPb;
            double *Qb = wQb;

            for (unsigned k = 0; k < P; k++) {
              Qb[2 * k] = Qb[2 * k + 1] = 0.0;
              for (unsigned l = 0; l < P; l++)
                Pb[2 * ((size_t)k * AR_ORDER_MAX + l)]
                  = Pb[2 * ((size_t)k * AR_ORDER_MAX + l) + 1] = 0.0;
            }
            for (unsigned j = 0; j < J; j++) {
              const unsigned sc
                = (unsigned)((nowP - (uint64_t)j * arStride) % m_ArRingLen);
              const double xr = m_ArRingRe[(ringBase + sc) * NBINS_MAX + bin];
              const double xi = m_ArRingIm[(ringBase + sc) * NBINS_MAX + bin];

              for (unsigned k = 0; k < P; k++) {
                const unsigned sk = (unsigned)(
                  (nowP - (uint64_t)(j + 1 + k) * arStride) % m_ArRingLen);
                const double kr = m_ArRingRe[(ringBase + sk) * NBINS_MAX + bin];
                const double ki = m_ArRingIm[(ringBase + sk) * NBINS_MAX + bin];

                Qb[2 * k] += xr * kr + xi * ki;      // x * conj(lag_k)
                Qb[2 * k + 1] += xi * kr - xr * ki;
                for (unsigned l = 0; l < P; l++) {
                  const unsigned sl = (unsigned)(
                    (nowP - (uint64_t)(j + 1 + l) * arStride) % m_ArRingLen);
                  const double lr
                    = m_ArRingRe[(ringBase + sl) * NBINS_MAX + bin];
                  const double li
                    = m_ArRingIm[(ringBase + sl) * NBINS_MAX + bin];

                  Pb[2 * ((size_t)k * AR_ORDER_MAX + l)] += kr * lr + ki * li;
                  Pb[2 * ((size_t)k * AR_ORDER_MAX + l) + 1]
                    += ki * lr - kr * li;
                }
              }
            }
            // Dreieck zurueckschreiben; der Rest ist die Spiegelung.
            {
              const size_t pb
                = (size_t)ch * AR_PAIRS * NBINS_MAX + bin;
              const size_t qb
                = (size_t)ch * AR_ORDER_MAX * NBINS_MAX + bin;

              for (unsigned k = 0; k < P; k++) {
                m_ArQRe[qb + (size_t)k * NBINS_MAX] = Qb[2 * k];
                m_ArQIm[qb + (size_t)k * NBINS_MAX] = Qb[2 * k + 1];
                for (unsigned l = k; l < P; l++) {
                  const size_t o = pb + (size_t)PairOf(k, l) * NBINS_MAX;

                  m_ArPRe[o] = Pb[2 * ((size_t)k * AR_ORDER_MAX + l)];
                  m_ArPIm[o] = Pb[2 * ((size_t)k * AR_ORDER_MAX + l) + 1];
                }
              }
            }
          }
      }
      bool gpuDone = false;

#ifndef XO_NO_OPENCL
      // v1: Geraet nur fuer den Standardpfad - Rechteckfenster (replica)
      // und Schur bleiben CPU.  Scheitert das Geraet, uebernimmt die CPU
      // nahtlos: ihre Akkumulatoren sind weitergelaufen.
      // Detektor-Modus kann der Kernel seit dem 07.08. selbst; nur
      // Rechteckfenster (replica) und Schur bleiben CPU.
      if (
        m_Gpu.load(std::memory_order_relaxed) && !m_GpuBad && !fitWin
        && !s_UseSchur.load(std::memory_order_relaxed))
        gpuDone = GpuSolveRange(ch, b0, b1, P, nbins, full || noWarm);
#endif
      (gpuDone ? s_GpuRanges : s_CpuRanges)
        .fetch_add(1, std::memory_order_relaxed);
      if (!gpuDone)
      for (unsigned bin = b0; bin < b1; bin++) {
        // Junge Hybrid-Baender loest gleich die Leiter bei ihrer
        // Altersordnung - ein voller P-Solve auf dem frischen Akkumulator
        // waere unterbestimmt.
        if (hybOn && m_HybReset.load(std::memory_order_relaxed)
            && HybBandYoung(ch, bin >> bandShift))
          continue;
        GatherBin(ch, bin, P, gP, gQ);
        SolveBin(
          gP, gQ, P,
          cRe + (size_t)bin * AR_ORDER_MAX, cIm + (size_t)bin * AR_ORDER_MAX,
          rRe + (size_t)bin * AR_ORDER_MAX, rIm + (size_t)bin * AR_ORDER_MAX);
      }
      // Leiter-Solve der jungen Baender - JEDE Periode, auch wenn der
      // Streifen sie nicht traf, und NACH der GPU (deren voller P-Solve
      // fuer diese Bins wird hier ueberschrieben).
      if (hybOn && m_HybReset.load(std::memory_order_relaxed))
        HybridSolveLongYoung(ch, P, arStride, bandShift, nbins);

      // Differential dump for the order-12 hunt: one bin's accumulators,
      // solved coefficients and advance state, to be diffed against numpy
      // accumulators built from the identical history.
      static const int dumpBin
        = getenv("XO_DUMP_BIN") ? atoi(getenv("XO_DUMP_BIN")) : -1;

      if (dumpBin >= 0 && ch == 0 && (unsigned)dumpBin >= b0
          && (unsigned)dumpBin < b1) {
        FILE *f = fopen("/tmp/xo_dump.txt", "a");

        if (f) {
          double dP[AR_ORDER_MAX * AR_ORDER_MAX * 2], dQ[AR_ORDER_MAX * 2];

          GatherBin(ch, (unsigned)dumpBin, P, dP, dQ);

          const double *Pb = dP;
          const double *Qb = dQ;

          fprintf(f, "S %llu", (unsigned long long)nowP);
          for (unsigned k = 0; k < P; k++)
            for (unsigned l = 0; l < P; l++)
              fprintf(
                f, " %.17g %.17g", Pb[2 * ((size_t)k * AR_ORDER_MAX + l)],
                Pb[2 * ((size_t)k * AR_ORDER_MAX + l) + 1]);
          for (unsigned k = 0; k < P; k++)
            fprintf(f, " %.17g %.17g", Qb[2 * k], Qb[2 * k + 1]);
          for (unsigned k = 0; k < P; k++)
            fprintf(
              f, " %.9g %.9g", (double)cRe[(size_t)dumpBin * AR_ORDER_MAX + k],
              (double)cIm[(size_t)dumpBin * AR_ORDER_MAX + k]);
          fprintf(f, "\n");
          fclose(f);
        }
      }
    }
  }

  float trend = 1.0f;

  if (
    wBridge < 0.999f && sumE0 > 1e-30f
    && !m_NoTrend.load(std::memory_order_relaxed)) {
    // TREND_MIN/MAX are specified per SYNTHESIS hop; a shorter analysis
    // baseline must clamp proportionally tighter, or anaHop 128 would
    // permit the full 1.10 growth every 2.7 ms instead of every 21 ms.
    const float lo = mPow > 1 ? powf(TREND_MIN, 1.0f / mPow) : TREND_MIN;
    const float hi = mPow > 1 ? powf(TREND_MAX, 1.0f / mPow) : TREND_MAX;

    trend = std::min(hi, std::max(lo, sqrtf(sumE1 / sumE0)));
    // Back to a per-synthesis-hop factor, then gated by density.
    trend = powf(trend, (1.0f - wBridge) * mPow);
    // Order 2 cannot take the trend on its coefficients - scaling `a`
    // alone is not a geometric level change, and scaling both roots would
    // move them off the unit circle the clamp just put them on.  It is
    // applied to the STATE once per synthesis frame instead, which is
    // exactly the same geometric factor.
    if (trend != 1.0f && !useAr)
      for (unsigned bin = 0; bin < nbins; bin++) {
        rotRe[bin] *= trend;
        rotIm[bin] *= trend;
      }
  }

  // Only [0, m_ShadowLen) is ever read back, so neither the clear nor the
  // accumulation below may run past it.  Accumulating each frame over its
  // full overlap with [0, shadowLen + nfft) moved 10240 samples to use
  // 656 - three quarters of the traffic in the most expensive loop, and
  // this loop is squarely memory-bound (measured: with the arithmetic
  // removed it costs exactly as much as a plain copy).
  std::fill(ola, ola + m_ShadowLen, 0.0f);
  // Hindcast des Samples VOR der Grenze (v0Hindcast), samt eigener
  // Fenstergewichtssumme - t = -1 hat einen Frame weniger als t = 0.
  const bool v0Shadow = m_V0Hindcast.load(std::memory_order_relaxed) != 0;
  float shadowM1 = 0.0f, shadowM1W = 0.0f;

  // Exactly the frames whose window overlaps [0, shadowLen) - the former
  // "+2" synthesized one to two full (and unused) iFFT frames per call.
  const unsigned kEnd = (m_ShadowLen + nfft + synHop - 1) / synHop;

  // Koeffizienten einmal je Runde nach [k][bin] umlegen.  Danach liest die
  // Extrapolation nur noch zusammenhaengend.  nbins*P Werte gegen
  // nbins*P*arSteps*(kEnd-1) Multiplikationen dahinter - der Umbau kostet
  // ein Vielfaches weniger, als er einspart.
  if (useAr) {
    const size_t cb = (size_t)ch * NBINS_MAX * AR_ORDER_MAX;
    const float *srcRe = m_ArCoefRe.data() + cb;
    const float *srcIm = m_ArCoefIm.data() + cb;
    float *dstRe = m_ArCoefReP.data() + cb;
    float *dstIm = m_ArCoefImP.data() + cb;

    for (unsigned k = 0; k < P; k++)
      for (unsigned bin = 0; bin < nbins; bin++) {
        dstRe[(size_t)k * NBINS_MAX + bin]
          = srcRe[(size_t)bin * AR_ORDER_MAX + k];
        dstIm[(size_t)k * NBINS_MAX + bin]
          = srcIm[(size_t)bin * AR_ORDER_MAX + k];
      }
  }

  for (unsigned k = 1; k < kEnd; k++) {
    if (useAr) {
      // Advance the order-p recursion arSteps times: it steps by arHop,
      // the synthesis frame by synHop, and arHop divides synHop by
      // construction (nfft/8 into nfft/4, so exactly two steps).
      const size_t cb = (size_t)ch * NBINS_MAX * AR_ORDER_MAX;
      const float *cRe = m_ArCoefReP.data() + cb;
      const float *cIm = m_ArCoefImP.data() + cb;
      float *accRe = m_AdvAccRe.data() + (size_t)ch * NBINS_MAX;
      float *accIm = m_AdvAccIm.data() + (size_t)ch * NBINS_MAX;

      for (unsigned step = 0; step < arSteps; step++)
        switch (P) {
        case 2:
          AdvanceBins<2>(nbins, cRe, cIm, stRe, stIm, accRe, accIm);
          break;
        case 3:
          AdvanceBins<3>(nbins, cRe, cIm, stRe, stIm, accRe, accIm);
          break;
        case 4:
          AdvanceBins<4>(nbins, cRe, cIm, stRe, stIm, accRe, accIm);
          break;
        case 5:
          AdvanceBins<5>(nbins, cRe, cIm, stRe, stIm, accRe, accIm);
          break;
        case 6:
          AdvanceBins<6>(nbins, cRe, cIm, stRe, stIm, accRe, accIm);
          break;
        case 7:
          AdvanceBins<7>(nbins, cRe, cIm, stRe, stIm, accRe, accIm);
          break;
        case 8:
          AdvanceBins<8>(nbins, cRe, cIm, stRe, stIm, accRe, accIm);
          break;
        default:
          // Orders beyond the template cases (the lab builds this library
          // with -DXO_AR_ORDER_MAX=20 to explore them): a RUNTIME trip
          // count, correct for every p, merely not vectorised.  The
          // previous default fell through to AdvanceBins<1> and advanced
          // with ONE coefficient instead of p - that is the bug that made
          // orders 5..8 read -0.4 dB while order 4 read -26.5, and why
          // order 1 must never be the silent fallback.
          AdvanceBinsGeneric(P, nbins, cRe, cIm, stRe, stIm);
          break;
        }
      for (unsigned bin = 0; bin < nbins; bin++) {
        if (trend != 1.0f)
          for (unsigned k = 0; k < P; k++) {
            stRe[(size_t)k * NBINS_MAX + bin] *= trend;
            stIm[(size_t)k * NBINS_MAX + bin] *= trend;
          }
        specRe[bin] = stRe[bin];
        specIm[bin] = stIm[bin];
        pFreq[bin][0] = stRe[bin];
        pFreq[bin][1] = stIm[bin];
      }
    } else {
      // advance every bin by its own frequency: spec *= rot
      for (unsigned bin = 0; bin < nbins; bin++) {
        const float re
          = specRe[bin] * rotRe[bin] - specIm[bin] * rotIm[bin];
        const float im
          = specRe[bin] * rotIm[bin] + specIm[bin] * rotRe[bin];

        specRe[bin] = re;
        specIm[bin] = im;
        pFreq[bin][0] = re;
        pFreq[bin][1] = im;
      }
    }
    // HYBRID: junge Baender aus der langen Kette ausblenden (1 - w);
    // die kurze Kette liefert denselben Anteil mit w, die Summe bleibt
    // punktweise eins.
    static const bool hybNoB = getenv("XO_HYB_NO_B") != nullptr;

    if (hybOn && m_HybAny[ch] && !hybNoB) {
      const float *hw = m_HybW.data() + (size_t)ch * ROUTE_BANDS;

      for (unsigned bin = 0; bin < nbins; bin++) {
        const unsigned b = bin >> bandShift;

        // BINAER, nicht gemischt: bei 0 < w < 1 addierten sich ZWEI
        // verschiedene Vorhersagen desselben Bandes, deren Phasen in der
        // Luecke auseinanderlaufen - die Summe kaemmt (gemessen: Schaden
        // waechst mit der Uebergabezeit, Rest-Luecken -2.55 -> +0.83).
        // Zwischen zwei Luecken gibt es keine Stetigkeitspflicht: jede
        // wird eigenstaendig verdeckt und an der Naht ans echte Signal
        // gebunden.  Die Rampe von w bleibt als ENTPRELLUNG des
        // Umschaltpunkts, gehoert wird nur ganz A oder ganz B.
        if (hw[b < ROUTE_BANDS ? b : ROUTE_BANDS - 1] > 0.5f) {
          pFreq[bin][0] = 0.0f;
          pFreq[bin][1] = 0.0f;
        }
      }
    }
    fftwf_execute_dft_c2r(kit.inv, pFreq, pTime); // destroys pFreq

    // frame k spans [k*synHop - nfft, k*synHop) relative to the boundary
    const int start = (int)(k * synHop) - (int)nfft;
    const unsigned a0 = start < 0 ? 0 : (unsigned)start;
    const unsigned a1 = std::min(m_ShadowLen, (unsigned)(start + (int)nfft));

    if (a0 >= a1)
      continue;

    const float scale = 1.0f / nfft;

    // Hindcast-Anker (XO_V0_SHADOW): dieselben Frames decken das Sample
    // VOR der Grenze mit ab - ein Sample einsammeln statt wegschneiden.
    if (v0Shadow && start <= -1) {
      const unsigned j = (unsigned)(-1 - start);

      shadowM1 += pTime[j] * win[j] * scale;
      shadowM1W += win[j] * win[j];
    }

    for (unsigned t = a0; t < a1; t++)
      ola[t] += pTime[t - start] * win[t - start] * scale;
  }
  // Normalise with the kit's precomputed constant (no per-call weight
  // accumulation, no divide).
  for (unsigned i = 0; i < m_ShadowLen; i++)
    pShadow[i] = ola[i] * kit.olaRecip[i];
  if (v0Shadow && shadowM1W > 1e-9f)
    shadowM1 /= shadowM1W;

  // HYBRID: die kurze Kette traegt ihre (mit w gewichteten) Baender bei.
  // VOR Cap und Naht - beide arbeiten auf der fertigen Summe.
  static const bool hybNoB2 = getenv("XO_HYB_NO_B") != nullptr;

  if (hybOn && m_HybAny[ch] && !hybNoB2)
    HybridSynth(
      ch, nowP, pShadow, ola, v0Shadow ? &shadowM1 : nullptr);

  synClock.Stop(XO_ST_SYNTHESE);

  XoClock capClock(XoProfiling());

  // ---- safety cap ------------------------------------------------------
  // Never louder than the recent material (+20 %).  This SCALES, it does
  // not clip - a limiter would fold harmonics into a signal whose whole
  // purpose is to be inaudible.
  //
  // Runs BEFORE the seam correction: the cap is a gain on the entire
  // shadow, so applying it afterwards would move sample 0 again and undo
  // exactly the continuity the anchor just established.
  //
  // The factor is a knob because it is under suspicion (06.08.2026): the
  // peak is taken over the WHOLE shadow, ~32 ms, while only the played
  // 128 samples are ever heard.  More modes near |z| = 1 means a better
  // chance that the superposition exceeds histPeak SOMEWHERE in that
  // tail, so the cap fires on evidence the listener never gets - and it
  // ratchets: the scaled samples go into the ring, the next histPeak is
  // smaller, the limit tightens.  s_CapHits/s_CapCalls measure how often
  // that actually happens, per order.
  //
  // capSpan ist die ANTWORT auf genau diesen Verdacht und war bis
  // 2026-08-09 eine Attrappe: m_CapSpan wurde in setTuning abgelegt, von
  // getTuning zurueckgelesen - und nie benutzt.  Der Haken "Cap nur
  // Gap+Fade" hat also in Lab UND GO nichts getan; alle Messreihen mit
  // gesetztem Haken haben in Wahrheit den ganzen Schatten gemessen.
  //
  // Gemessen wird jetzt nur ueber die Strecke, die als naechstes wirklich
  // ausgegeben wird (Luecke + Wiedereinstiegsblende).  Divergenz JENSEITS
  // davon ist das Problem des naechsten Puffers - falls es dann noch eine
  // ist.  capSpan 0 heisst weiterhin: ganzer Schatten.
  //
  // Der ermittelte Faktor wirkt bewusst auf den GANZEN Schatten, nicht nur
  // auf die gemessene Strecke: eine Kette aufeinanderfolgender Fehlschlaege
  // laeuft ohne neue Analyse Scheibe fuer Scheibe durch denselben Schatten,
  // und eine nur teilweise skalierte Kurve haette an der Spannengrenze eine
  // Stufe - ein Knacks, den es vorher nicht gab.  Die Spanne steuert also
  // WORAUF GESCHAUT wird, nicht worauf multipliziert wird; der Verdacht
  // oben richtet sich ausschliesslich gegen das Hinschauen.
  {
    const int span = m_CapSpan.load(std::memory_order_relaxed);
    const unsigned nLook = span > 0
      ? std::min((unsigned)span, m_ShadowLen)
      : m_ShadowLen;
    float histPeak = 0.0f, shadowPeak = 0.0f;

    for (unsigned i = len - std::min(len, 2048u); i < len; i++)
      histPeak = std::max(histPeak, std::abs(pHist[i]));
    for (unsigned i = 0; i < nLook; i++)
      shadowPeak = std::max(shadowPeak, std::abs(pShadow[i]));

    const float cap = m_SafetyCap.load(std::memory_order_relaxed);
    const float limit = cap * histPeak;

    s_CapCalls.fetch_add(1, std::memory_order_relaxed);
    if (!m_Replica.load(std::memory_order_relaxed)
        && cap > 0.0f && shadowPeak > limit && shadowPeak > 0.0f) {
      const float g = limit / shadowPeak;

      s_CapHits.fetch_add(1, std::memory_order_relaxed);
      for (unsigned i = 0; i < m_ShadowLen; i++)
        pShadow[i] *= g;
    }
  }

  // ---- adaptive entry seam (anchor <-> bridge) -------------------------
  // The freeze is only approximately continuous at the boundary, and the
  // right repair depends on the material (frame similarity `sim` above):
  //
  // SPARSE spectrum (sustained tone, sim >= 0.999): the freeze itself is
  // nearly exact; a 2 ms Burg crossfade is then the dominant seam error
  // (order-starved free-run = low/mid band thump, independent of gap
  // width).  Repair: keep the freeze and subtract the entry jump as a
  // decaying exponential,
  //
  //     shadow[i] -= (shadow[0] - v0) * exp(-i / tau),   tau ~ 1/3 ms
  //
  // where the anchor v0 is Burg's ONE-step prediction (the residual the
  // fit minimizes - exact where the free-run is not).  A naive linear-
  // extrapolation anchor fails on curved waveforms.
  //
  // DENSE spectrum (sim <= 0.99, held chords included): several partials
  // per bin defeat the per-bin frequency estimate, the freeze leaves the
  // true waveform within a few samples, and a one-point anchor still
  // clicks - there the Burg free-run bridge REPLACES the seam content
  // (see the branch below).  Both corrections blend linearly via wBridge.
  //
  // The fit is refreshed only every BURG_REFIT_PERIODS periods (it was 36%
  // of the entire prediction); the anchor dot product runs every time on
  // the LATEST samples, which is what makes the entry continuous.
  const unsigned bw = std::min(
    {BURG_WINDOW, m_HistFill.load(std::memory_order_relaxed), len});
  const uint64_t fitAge = nowP - m_BurgFitPeriod[ch];
  // Staggered refit slots: channel ch only refits in periods where
  // (period mod interval) == (ch mod interval).  Without the stagger all
  // channels refit in the SAME period, and the measured load profile
  // spiked to +197% of the median there - concentrated cost is exactly
  // what a period-scheduled task must not produce.
  //
  // burgEvery = 1 refits EVERY period, which is what the offline
  // prototype does at every gap.  Under test since 06.08.2026: the
  // anchor subtracts (shadow[0] - v0) * decay^i with tau = 1/3 ms from
  // the shadow, and an exponential that short is BROADBAND - a vertical
  // stripe in the spectrogram, of exactly the amplitude by which v0 is
  // wrong.  A Burg model up to 1024 samples old is still valid on a
  // sustained tone and not at all at an onset, which is where the
  // remaining 3.1 dB (12-18 kHz) and 5.5 dB (18-24 kHz) sit.
  const uint64_t burgEvery = m_Replica.load(std::memory_order_relaxed) ? 1u
    : (uint64_t)std::max(
    1, (int)(m_BurgEvery.load(std::memory_order_relaxed)
               ? m_BurgEvery.load(std::memory_order_relaxed)
               : (int)BURG_REFIT_PERIODS));
  const bool mySlot
    = burgEvery <= 1 || (nowP % burgEvery) == (ch % burgEvery);

  if (
    !s_NoSeam && !v0Shadow && mySlot && fitAge >= burgEvery
    && FitBurg(ch, pHist + len - bw, bw))
    m_BurgFitPeriod[ch] = nowP;

  if (
    !s_NoSeam
    && ((v0Shadow && shadowM1W > 1e-9f)
        || m_BurgFitPeriod[ch] != ~0ull)) {
    const double *a = m_BurgA.data() + ch * BURG_ORDER;
    double v0 = 0.0;

    if (v0Shadow)
      // Hindcast-Anker: Versatz DIREKT am letzten echten Sample gemessen
      // (Schatten[-1] gegen die Wahrheit), statt die Wahrheit an
      // Schatten[0] vorherzusagen.  jump wird unten einmal zusaetzlich
      // mit decay multipliziert - die Exponentiale sitzt ein Sample
      // frueher.
      v0 = (double)pHist[len - 1] + ((double)pShadow[0] - (double)shadowM1);
    else if (s_V0Last)
      v0 = (double)pHist[len - 1]; // Messknopf: letztes echtes Sample
    else
      for (unsigned j = 0; j < BURG_ORDER; j++)
        v0 += a[j] * (double)pHist[len - 1 - j];

    // The jump against the pure freeze - taken BEFORE the bridge branch
    // touches pShadow; both corrections superpose additively on the
    // freeze (out = freeze + w*(bridgeBlend - freeze) - (1-w)*J*decay).
    const double jump = (double)pShadow[0] - v0;

    // ---- dense-spectrum branch: weighted Burg bridge ------------------
    // On a dense complex spectrum (several partials per bin) the frame
    // phase difference no longer identifies a dominant frequency and the
    // freeze leaves the true waveform within a few samples - a one-point
    // anchor then still clicks (measured on the Vierne finale: up to
    // +7 dB above signal, on HELD chords as much as on runs).  There the
    // 2 ms free-run REPLACES the wrong freeze content, and its low-band
    // thump is masked by the very density that made it necessary.
    if (!v0Shadow && wBridge > 0.001f) {
      double *state = burgSt;

      for (unsigned j = 0; j < BURG_ORDER; j++)
        state[j] = pHist[len - 1 - j];
      for (unsigned i = 0; i < m_BridgeLen; i++) {
        double v = 0.0;

        for (unsigned j = 0; j < BURG_ORDER; j++)
          v += a[j] * state[j];
        for (unsigned j = BURG_ORDER - 1; j > 0; j--)
          state[j] = state[j - 1];
        state[0] = v;
        bridge[i] = (float)v;
      }
      for (unsigned i = 0; i < m_BridgeLen; i++)
        pShadow[i]
          += wBridge * (1.0f - m_BridgeWin[i]) * (bridge[i] - pShadow[i]);
    }

    // ---- sparse-spectrum branch: anchor corrector ---------------------
    // Geometric decay: one multiply per sample, no transcendental in the
    // loop.  After m_AnchorLen (= 10 tau) the factor is < 5e-5.
    //
    // tau ist seit 06.08.2026 einstellbar (anchorTau, in Samples): der
    // Anker ist eine BREITBANDIGE Exponentiale - kuerzeres tau heisst
    // mehr Hochtonenergie bei gleicher Sprunghoehe, laengeres verteilt
    // denselben Betrag tiefer und leiser.  Der Nutzer will die Wirkung
    // an Transienten hoeren.
    double k = jump * (1.0 - (double)wBridge);
    const unsigned m = m_Replica.load(std::memory_order_relaxed)
      ? m_ShadowLen : std::min(m_ShadowLen, m_AnchorLen);
    const int tauCfg = m_AnchorTau.load(std::memory_order_relaxed);
    const unsigned tau
      = tauCfg > 0 ? (unsigned)tauCfg : std::max(8u, m_SampleRate / 3000);
    const double decay = exp(-1.0 / (double)tau);
    const unsigned mm = tauCfg > 0
      ? std::min(m_ShadowLen, 10u * tau) : m;

    // Hindcast-Anker: die Exponentiale ist am Sample VOR der Grenze
    // verankert, an Schatten[0] ist sie schon einen Schritt abgeklungen.
    if (v0Shadow)
      k *= decay;

    for (unsigned i = 0; i < mm; i++) {
      pShadow[i] -= (float)k;
      k *= decay;
    }
  }

  // 1:1-Nachbau des Offline-Prototyps: dort laeuft _safety_cap NACH
  // with_anchor_corrector, nicht davor.  Greift er, zieht er pShadow[0]
  // wieder von v0 weg - genau den Sprung, den der Anker beseitigt hat.
  // Die Bibliothek macht es normalerweise umgekehrt (und besser); hier
  // geht es um Gleichheit, nicht um Qualitaet.
  if (m_Replica.load(std::memory_order_relaxed)) {
    float histPeak = 0.0f, shadowPeak = 0.0f;

    for (unsigned i = len - std::min(len, 2000u); i < len; i++)
      histPeak = std::max(histPeak, std::abs(pHist[i]));
    for (unsigned i = 0; i < m_ShadowLen; i++)
      shadowPeak = std::max(shadowPeak, std::abs(pShadow[i]));

    const float limit = 1.2f * histPeak;

    if (shadowPeak > limit && shadowPeak > 0.0f) {
      const float g = limit / shadowPeak;

      for (unsigned i = 0; i < m_ShadowLen; i++)
        pShadow[i] *= g;
    }
  }
  capClock.Stop(XO_ST_CAP_NAHT);
}

// Baut den Rueckwaerts-Patch fuer [zoneStart, zoneStart+len) aus dem
// Zustand der LETZTEN Analyse dieses Kanals: Rahmen k entsteht durch
// k-fache konjugierte Rotation des juengsten Spektrums (Schrittweite =
// Synthese-Hop), synthetisiert per OLA mit denselben Fenstern wie die
// Analyse - jede spaetere Fensterung ueber der Zone sieht damit per
// COLA in sich konsistente, phasenkontinuierliche Rahmen.
bool Xorcist::BuildBackPatch(unsigned ch, uint64_t zoneStart, unsigned len) {
  const int mode = m_Mode.load(std::memory_order_relaxed);

  if (!ValidMode(mode))
    return false;

  const PvKit &kit = m_Pv[KitIndex(mode)];
  const unsigned nfft = kit.nfft;
  const unsigned synHop = nfft / 4;
  const unsigned nbins = nfft / 2 + 1;

  if (len == 0 || len > PATCH_MAX)
    return false;
  if (m_SpecCachePeriod[ch] == ~0ull || m_SpecCacheNfft[ch] != nfft)
    return false; // noch keine gueltige Analyse

  const uint64_t refEnd = m_SpecCachePeriod[ch] * m_SamplesPerBuffer;
  const uint64_t zoneEnd = zoneStart + len;

  if (zoneEnd > refEnd || refEnd - zoneStart > HIST - nfft)
    return false;

  const float *specRe = m_SpecRe.data() + (size_t)ch * NBINS_MAX;
  const float *specIm = m_SpecIm.data() + (size_t)ch * NBINS_MAX;
  const float *rotRe = m_RotRe.data() + (size_t)ch * NBINS_MAX;
  const float *rotIm = m_RotIm.data() + (size_t)ch * NBINS_MAX;
  float *const pTime = m_TimeBuf + (size_t)ch * NFFT_MAX;
  fftwf_complex *const pFreq = m_FreqBuf + (size_t)ch * FREQ_STRIDE;
  const float *win = kit.win.data();
  // Laufende Rueckwaerts-Rotation des Spektrums
  std::vector<float> xr(nbins), xi(nbins);

  for (unsigned b = 0; b < nbins; b++) {
    xr[b] = specRe[b];
    xi[b] = specIm[b];
  }

  std::vector<double> acc(len, 0.0), wsum(len, 1e-12);
  const float scale = 1.0f / (float)nfft;

  // k = 0 ist der ECHTE aktuelle Rahmen, unrotiert - er MUSS beitragen:
  // ohne ihn deckt nur der Fensterauslaeufer von k = 1 die Zone, wsum
  // faellt am rechten Rand gegen null, und die Division verstaerkt jede
  // Rahmen-Unstimmigkeit tausendfach.  Der Sinus-Test konnte das nicht
  // zeigen: fuer ihn ist jeder Rahmen allein exakt, (w^2 x)/(w^2) = x
  // bei beliebig kleinem Gewicht.  Erst dichtes Material, dessen Rahmen
  // sich leicht widersprechen, macht Deckungsluecken zu Spitzen.
  for (unsigned k = 0; k <= 64; k++) {
    const uint64_t fEnd = refEnd - (uint64_t)k * synHop;

    if (fEnd <= zoneStart)
      break; // dieser und alle aelteren Rahmen enden vor der Zone

    const uint64_t fStart = fEnd - nfft;

    if (fStart < zoneEnd) {
      for (unsigned b = 0; b < nbins; b++) {
        pFreq[b][0] = xr[b];
        pFreq[b][1] = xi[b];
      }
      fftwf_execute_dft_c2r(kit.inv, pFreq, pTime);

      const uint64_t lo = fStart > zoneStart ? fStart : zoneStart;
      const uint64_t hi = fEnd < zoneEnd ? fEnd : zoneEnd;

      for (uint64_t a = lo; a < hi; a++) {
        const unsigned j = (unsigned)(a - fStart);
        const float w = win[j];

        acc[a - zoneStart] += (double)pTime[j] * w * scale;
        wsum[a - zoneStart] += (double)w * w;
      }
    }
    // erst NACH der Verwendung zuruedrehen: k=0 bleibt unrotiert
    for (unsigned b = 0; b < nbins; b++) {
      const float re = xr[b] * rotRe[b] + xi[b] * rotIm[b];  // * conj(rot)
      const float im = xi[b] * rotRe[b] - xr[b] * rotIm[b];

      xr[b] = re;
      xi[b] = im;
    }
  }

  // Normieren und mit kurzen Rampen in die echte Historie einbetten:
  // links (Alt-Ton) 128 Samples, rechts (phasenexakt) 64 als Sicherung.
  const float *real = HistWindow(ch, (unsigned)(refEnd - zoneStart));
  float *pb = m_PatchBuf.data() + (size_t)ch * PATCH_MAX;

  for (unsigned i = 0; i < len; i++) {
    float v = (float)(acc[i] / wsum[i]);
    const unsigned lRamp = 128 < len / 4 ? 128u : len / 4;
    const unsigned rRamp = 64 < len / 4 ? 64u : len / 4;

    if (i < lRamp) {
      const float a = 0.5f - 0.5f * cosf((float)M_PI * i / lRamp);

      v = real[i] * (1.0f - a) + v * a;
    } else if (len - 1 - i < rRamp) {
      const float a
        = 0.5f - 0.5f * cosf((float)M_PI * (len - 1 - i) / rRamp);

      v = real[i] * (1.0f - a) + v * a;
    }
    pb[i] = v;
  }
  m_PatchStart[ch] = zoneStart;
  m_PatchEnd[ch] = zoneEnd;
  return true;
}

void Xorcist::Absorb() {
  const int mode = m_Mode.load(std::memory_order_relaxed);

  if (!ValidMode(mode))
    return;

  const PvKit &kit = m_Pv[KitIndex(mode)];

  if (m_HistFill.load(std::memory_order_relaxed) < kit.nfft + kit.nfft / 4)
    return;
  // Kein Slot, keine Publikation, kein Busy-Protokoll: es entsteht kein
  // Schatten.  pShadow zeigt auf Slot 0, wird aber im Analyse-Teil nie
  // beschrieben.
  m_AnalyzeOnly = true;
  for (unsigned ch = 0; ch < m_NChannels; ch++)
    ComputeChannelShadow(
      ch, m_Shadow[0].data() + (size_t)ch * m_ShadowLen, kit);
  m_AnalyzeOnly = false;
}

int Xorcist::Predict() {
  const int mode = m_Mode.load(std::memory_order_relaxed);

  if (!ValidMode(mode))
    return 0; // off: forced misses play hard silence, nothing to predict

  const PvKit &kit = m_Pv[KitIndex(mode)];

  if (m_HistFill.load(std::memory_order_relaxed) < kit.nfft + kit.nfft / 4)
    return 0;

  // Recompute whenever new history exists.  Slices > 0 are a DEGRADED
  // fallback only: once real audio has played past the anchor, a later
  // slice ignores that reality and starts with unbridged drift (user's
  // catch - an every-2nd-period economy here made that the common case).
  // Their legitimate uses are consecutive misses (no real audio in
  // between - the self-fed ring continues ONE coherent extrapolation)
  // and a compute that was lost to an abandoned period.
  const uint64_t histNow = m_HistPeriod.load(std::memory_order_acquire);
  const uint64_t pub = m_ShadowPub.load(std::memory_order_acquire);

  if (pub != SHADOW_INVALID) {
    const uint64_t anchor = pub >> 1;

    if (histNow == anchor)
      return 0; // already anchored on the current history

    // CONSECUTIVE MISS: the last ring append was our own concealment, so
    // re-analysing now would fit a spectrum to PREDICTED material and
    // start a fresh extrapolation - a seam every single period, heard as
    // fluttering.  Keep the running one instead and let the delivery walk
    // to the next slice: one continuous prediction across the whole
    // outage, exactly the case the offline lab measured as inaudible at
    // 512-sample gaps.  It also costs nothing at the very moment the
    // machine is short of time.
    // The same argument reaches FURTHER than the streak itself.  The
    // analysis window is nfft samples - 32 periods at 4k - so after a
    // miss ends, predicted material stays INSIDE that window for another
    // 31 periods, and every re-analysis in between measures phase
    // advances partly on the predictor's own output.  Measured with an
    // oracle that feeds the ring truth while still PLAYING the
    // prediction, that feedback costs 3.6 dB at 2.5 % concealment and
    // 10.2 dB at 23.7 % - more than everything else in this file put
    // together, and it is also the whole of the high-order collapse:
    // with a clean ring order 12 BEATS order 4 (-13.34 vs -12.53), with
    // a dirty one it loses.
    //
    // staleWindow extends the keep-the-extrapolation rule by that many
    // periods past the last concealment.  0 = the streak only, i.e. the
    // behaviour this knob was measured against.  The horizon guard below
    // still applies, so the extrapolation can never outrun its shadow.
    const uint64_t stale = (uint64_t)m_StaleWindow.load(
      std::memory_order_relaxed);
    const uint64_t lastConc
      = m_LastConcealedPeriod.load(std::memory_order_acquire);

    // replica: NIE behalten, immer neu rechnen - der Offline-Prototyp
    // baut fuer JEDE Luecke eine frische Fortsetzung.  Verkettet las die
    // Bibliothek dagegen ab der zweiten Luecke einer Kette nur eine
    // spaetere SCHEIBE des alten Schattens, und die kennt das inzwischen
    // gespielte Material nicht.  Bei 6000/min sind 1768 Luecken in 914
    // Ketten, also wird etwa jede zweite Luecke so bedient.
    if (
      !m_Replica.load(std::memory_order_relaxed)
      && (m_ConcealedThrough.load(std::memory_order_acquire) == histNow
          || (stale && lastConc && histNow >= lastConc
              && histNow - lastConc <= stale))
      && histNow - anchor + 1 < m_HorizonPeriods)
      return 2; // deliberately kept - not a skip
  }

  if (m_ComputeBusy.exchange(true, std::memory_order_acquire)) {
    m_SkipCount.fetch_add(1, std::memory_order_relaxed);
    // Fuer die Ursachen-Zuordnung der spaeten Scheiben: WANN (in
    // Periodenzeit) die letzte Oeffnung am Besetzt-Waechter scheiterte.
    m_LastSkipPeriod.store(histNow, std::memory_order_relaxed);
    return 3; // besetzt - fuer den Wirt unterscheidbar von "nichts zu tun"
  }

  m_PendingSlot = pub == SHADOW_INVALID ? 0u : (unsigned)(~pub & 1ull);
  m_PendingPeriod = histNow;
  // Ab hier liest die Runde aus IHREM Kopf, nicht aus dem laufenden.  Das
  // Etikett m_PendingPeriod und die Daten, die die Kanaele sehen, gehoeren
  // damit garantiert zusammen - egal wer waehrenddessen pusht.
  m_RoundHead = m_HistWrite.load(std::memory_order_acquire);
  m_RoundMask = m_RecentPredMask.load(std::memory_order_relaxed);
  m_RoundFrozen.store(!s_NoFreeze, std::memory_order_release);
  return 1;
}

// Channels are fully independent - own FFT workspace, own accumulators,
// own recursion, nothing shared but the tuning atomics - so a host with
// spare workers can predict them concurrently.  Measured serially the
// cost scales linearly with the channel count, which at eight channels is
// 20 % of the period budget for order 1 alone and 70 % for order 4; that
// number is a SINGLE core's, and it is the reason this split exists.
void Xorcist::PredictChannel(unsigned ch) {
  const int mode = m_Mode.load(std::memory_order_relaxed);

  if (!ValidMode(mode) || ch >= m_NChannels)
    return;
  ComputeChannelShadow(
    ch, m_Shadow[m_PendingSlot].data() + ch * m_ShadowLen,
    m_Pv[KitIndex(mode)]);
}

// Release the compute slot WITHOUT publishing.  Needed when a period's
// channel tasks did not all run: the slot would otherwise stay taken and
// every later prediction would skip.  The half-filled shadow half is
// simply never published, so the delivery keeps using the previous one -
// which is the documented "shadow skipped for a period under load" case.
void Xorcist::PredictAbort() {
  m_RoundFrozen.store(false, std::memory_order_release);
  m_ComputeBusy.store(false, std::memory_order_release);
}

void Xorcist::PredictEnd() {
  // Hier stand bis 10.08.2026 eine Verwerfung: liefen waehrend der Runde
  // Pushes durch, war die Analyse verdorben (die Kanaele lasen relativ zum
  // LAUFENDEN Schreibkopf, das Fenster rutschte den spaeteren unter den
  // Fuessen weg), und die einzige Rettung war, die ganze Runde
  // wegzuwerfen.  Der Preis dafuer war hoch: der Anker blieb stehen, und
  // die naechste Verdeckung spielte eine Scheibe > 0 - ausgerechnet dort,
  // wo die Nahtreparatur nicht sitzt, denn die sitzt nur auf Scheibe 0.
  // Im laufenden GO liess sich beides zaehlen, und beide Zaehler liefen.
  //
  // Die Ursache ist jetzt weg statt behandelt: die Runde friert ihren
  // Schreibkopf in PredictBegin ein (siehe m_RoundHead), ein
  // gleichzeitiger Push kann das Fenster also gar nicht mehr verschieben.
  // Was die Kanaele gelesen haben, gehoert zum Etikett m_PendingPeriod -
  // die Runde ist gueltig und wird veroeffentlicht, auch wenn der Ring
  // inzwischen weitergelaufen ist.  Conceal rechnet die zwischenzeitlichen
  // Perioden ohnehin sauber heraus (slice = histNow - anchor).
  //
  // Gezaehlt wird der Vorfall weiter: er misst, wie oft der Rueckruf in
  // eine laufende Runde hineinschreibt, und bleibt damit das Mass fuer die
  // Nebenlaeufigkeit - nur kostet er jetzt keinen Schatten mehr.
  if (m_HistPeriod.load(std::memory_order_acquire) != m_PendingPeriod) {
    s_StaleDiscards.fetch_add(1, std::memory_order_relaxed);
    if (s_NoFreeze) { // alter Zustand: Runde wegwerfen
      m_RoundFrozen.store(false, std::memory_order_release);
      m_ComputeBusy.store(false, std::memory_order_release);
      return;
    }
  }
  m_ShadowPub.store(
    (m_PendingPeriod << 1) | m_PendingSlot, std::memory_order_release);
  m_RoundFrozen.store(false, std::memory_order_release);
  m_ComputeBusy.store(false, std::memory_order_release);
}

float Xorcist::DegradeGain(uint64_t missedSamples) const {
  if (missedSamples <= m_DegradeHold)
    return 1.0f;

  const uint64_t k = missedSamples - m_DegradeHold;

  if (k >= m_DegradeFade)
    return 0.0f;
  return 0.5f + 0.5f * cosf((float)M_PI * k / m_DegradeFade);
}

int Xorcist::Conceal(float *pOutBuf) {
  const int mode = m_Mode.load(std::memory_order_relaxed);

  m_ConcealCount.fetch_add(1, std::memory_order_relaxed);

  if (mode == XORCIST_OFF) {
    // Presentation baseline: this is what an unconcealed miss sounds like -
    // one period of hard silence, hard edges on both sides, no pipeline.
    std::memset(pOutBuf, 0, sizeof(float) * m_NChannels * m_SamplesPerBuffer);
    m_ReentryPending = false;
    m_MissedSamples = 0;
    return 0;
  }

  // Slice selection by tag: the missed period is m_HistPeriod + 1 (its own
  // mix never snapshotted), the shadow is anchored at `anchor`, so the
  // samples for THIS period start (histNow - anchor) periods into the
  // prediction.  A shadow computed one period ago is simply used one slice
  // further in - no phase error, by construction.
  const uint64_t pub = m_ShadowPub.load(std::memory_order_acquire);
  const uint64_t histNow = m_HistPeriod.load(std::memory_order_acquire);
  int shadowIdx = -1;
  int played = 0;
  unsigned sliceOffset = 0;

  if (pub != SHADOW_INVALID) {
    const uint64_t anchor = pub >> 1;
    const uint64_t slice = histNow - anchor;

    if (slice < m_HorizonPeriods) {
      shadowIdx = (int)(pub & 1ull);
      sliceOffset = (unsigned)slice * m_SamplesPerBuffer;
      // Scheibe > 0 ist LEGITIM in einer Fehlschlagkette (die
      // selbstgefuetterte Reihe ist EINE zusammenhaengende Fortschreibung).
      // Zaehlenswert ist nur der andere Fall: echtes Audio lief
      // dazwischen, der Anker ist trotzdem zurueckgeblieben.
      if (
        slice > 0
        && m_LastConcealedPeriod.load(std::memory_order_acquire) != histNow) {
        s_LateSlices.fetch_add(1, std::memory_order_relaxed);
        const uint64_t lastSkip
          = m_LastSkipPeriod.load(std::memory_order_relaxed);

        if (lastSkip != NO_SKIP_YET && histNow - lastSkip <= 2)
          s_LateAfterSkip.fetch_add(1, std::memory_order_relaxed);
        for (uint64_t old = s_LateSliceMax.load(std::memory_order_relaxed);
             slice > old
             && !s_LateSliceMax.compare_exchange_weak(
               old, slice, std::memory_order_relaxed);)
          ;
      }
    }
    // else: prediction too stale for its horizon - fall through to silence
  }

  // The degradation envelope moves by elapsed samples, independent of the
  // period size.  Within one period it is close to linear (2.7 ms of a
  // 150 ms cosine), so two evaluations and a ramp suffice.
  const float g0 = DegradeGain(m_MissedSamples);
  const float g1 = DegradeGain(m_MissedSamples + m_SamplesPerBuffer);

  m_MissedSamples += m_SamplesPerBuffer;

  if (shadowIdx < 0 || g0 <= 0.0f) {
    // No prediction available yet, or fully faded out (self-fed
    // predictions have gone stale): silence, re-enter with a fade-in.
    std::memset(pOutBuf, 0, sizeof(float) * m_NChannels * m_SamplesPerBuffer);
    m_ReentrySilent = true;
  } else {
    played = 1;
    const float *pShadow = m_Shadow[shadowIdx].data();
    float *pOut = pOutBuf;
    const float gStep = (g1 - g0) / m_SamplesPerBuffer;

    for (unsigned ch = 0; ch < m_NChannels; ch++) {
      const float *pSh = pShadow + ch * m_ShadowLen + sliceOffset;
      float g = g0;

      for (unsigned i = 0; i < m_SamplesPerBuffer; i++) {
        pOut[i * m_NChannels + ch] = pSh[i] * g;
        g += gStep;
      }
    }
    if (g1 > 0.0f) {
      // Re-entry tail = the slice AFTER the one just played (fits because
      // sliceOffset <= (horizon-1)*buf and shadowLen = horizon*buf + fade),
      // scaled to the level the envelope has reached - the returning real
      // signal then cross-fades against exactly what was last audible.
      for (unsigned ch = 0; ch < m_NChannels; ch++) {
        const float *pSh
          = pShadow + ch * m_ShadowLen + sliceOffset + m_SamplesPerBuffer;
        float *pTail = m_PendingTail.data() + ch * m_ReentryFade;

        for (unsigned i = 0; i < m_ReentryFade; i++)
          pTail[i] = pSh[i] * g1;
      }
      m_ReentrySilent = false;
    } else
      m_ReentrySilent = true; // faded out - re-enter with a fade-in
  }
  m_ReentryPending = true;

  // Self-feed: the abandoned mix will never snapshot, so append what was
  // ACTUALLY played to the ring - the next prediction then starts from a
  // time-aligned history.  Marking the position lets Predict() recognise
  // the streak and KEEP its running extrapolation rather than re-analyse
  // its own output.
  // Die SELBSTFUETTERUNG darf nie geblendet werden: sie schreibt die
  // gerade gespielte Vorhersage, keinen Wiedereinstieg.  Ohne dieses
  // Zuruecksetzen mischte die zweite Luecke einer Kette ihre eigene
  // Ausgabe mit dem Schattenschwanz der ersten - bei 914 Ketten die
  // Haelfte aller Verdeckungen.
  m_RingBlendPending = false;
  Push(pOutBuf);
  m_RecentPredMask.store(
    m_RecentPredMask.load(std::memory_order_relaxed) | 1u,
    std::memory_order_relaxed);
  m_ConcealedThrough.store(
    m_HistPeriod.load(std::memory_order_acquire), std::memory_order_release);
  m_LastConcealedPeriod.store(m_HistPeriod.load(std::memory_order_acquire),
                              std::memory_order_release);
  m_RingBlendPending = true;   // erst der NAECHSTE echte Push blendet
  return played;
}

void Xorcist::Pass(float *pOutBuf, const float *pReal) {
  if (pOutBuf != pReal)
    std::memcpy(pOutBuf, pReal,
                sizeof(float) * m_NChannels * m_SamplesPerBuffer);

  if (m_ReentryPending) {
    float *pOut = pOutBuf;
    const unsigned n = std::min(ReentryLen(), m_SamplesPerBuffer);

    for (unsigned i = 0; i < n; i++) {
      const float fo = ReentryCurve(i, n); // 1 -> 0

      for (unsigned ch = 0; ch < m_NChannels; ch++) {
        float v = pOut[i * m_NChannels + ch] * (1.0f - fo);

        if (!m_ReentrySilent)
          v += m_PendingTail[ch * m_ReentryFade + i] * fo;
        pOut[i * m_NChannels + ch] = v;
      }
    }
    m_ReentryPending = false;
    m_ReentrySilent = false;
    m_MissedSamples = 0;
  }
}

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

extern "C" {

Xorcist *Xorcist_new(
  unsigned channels,
  unsigned framesPerPeriod,
  unsigned sampleRate,
  const char *wisdomPath) {
  if (!channels || !framesPerPeriod || !sampleRate)
    return nullptr;
  try {
    return new Xorcist(channels, framesPerPeriod, sampleRate, wisdomPath);
  } catch (...) {
    return nullptr;
  }
}

void Xorcist_free(Xorcist *x) {
#ifndef XO_NO_OPENCL
  if (x)
    x->GpuRelease();
#endif
  delete x;
}

/* Which device the solve runs on, or NULL when the CPU does. */
const char *Xorcist_gpuDevice(void) {
#ifndef XO_NO_OPENCL
  XoCl &cl = XoCl::get();

  if (cl.ok)
    return cl.device;
#endif
  return 0;
}

void Xorcist_solveTapInfo(
  unsigned *pFrames, unsigned *pNbins, unsigned *pOrder, uint64_t *pTotal) {
  std::lock_guard<std::mutex> g(Xorcist::s_TapMx);
  const uint64_t n = Xorcist::s_TapN;

  if (pFrames)
    *pFrames = (unsigned)(n < Xorcist::s_TapFrames ? n : Xorcist::s_TapFrames);
  if (pNbins)
    *pNbins = Xorcist::s_TapNbins;
  if (pOrder)
    *pOrder = Xorcist::s_TapOrder;
  // Insgesamt geschrieben, nicht nur noch abrufbar - nur damit laesst
  // sich waehrend eines Laufs sagen, WIE VIELE Bilder seit dem letzten
  // Blick dazugekommen sind.
  if (pTotal)
    *pTotal = n;
}

unsigned Xorcist_solveTapRead(
  float *pZ, float *pPath, float *pRoots, uint64_t *pPos,
  unsigned maxFrames) {
  std::lock_guard<std::mutex> g(Xorcist::s_TapMx);
  const uint64_t n = Xorcist::s_TapN;
  const unsigned have
    = (unsigned)(n < Xorcist::s_TapFrames ? n : Xorcist::s_TapFrames);
  const unsigned take = maxFrames < have ? maxFrames : have;
  const unsigned nb = Xorcist::s_TapNbins;
  const size_t rw = (size_t)nb * Xorcist::s_TapOrder * 2;   // Pole je Bild

  if (!take || !nb)
    return 0;

  // Die JUENGSTEN take Bilder, aelteste zuerst: Startschlitz ist der
  // Schreibzeiger minus take.
  const uint64_t first = n - take;

  for (unsigned i = 0; i < take; i++) {
    const size_t slot = (size_t)((first + i) % Xorcist::s_TapFrames);

    if (pZ)
      memcpy(
        pZ + (size_t)i * nb, &Xorcist::s_TapZ[slot * nb],
        (size_t)nb * sizeof(float));
    if (pPath)
      memcpy(
        pPath + (size_t)i * nb, &Xorcist::s_TapPath[slot * nb],
        (size_t)nb * sizeof(float));
    if (pRoots && rw)
      memcpy(
        pRoots + (size_t)i * rw, &Xorcist::s_TapRoots[slot * rw],
        rw * sizeof(float));
    if (pPos)
      pPos[i] = Xorcist::s_TapPos[slot];
  }
  return take;
}

void Xorcist_solveTapClear(void) {
  std::lock_guard<std::mutex> g(Xorcist::s_TapMx);

  Xorcist::s_TapN = 0;
}

void Xorcist_analysisGeometry(
  const Xorcist *x, unsigned *pNfft, unsigned *pAnaHop, unsigned *pArHop) {
  const int mode = x->m_Mode.load(std::memory_order_relaxed);

  if (mode == XORCIST_OFF) {
    if (pNfft)
      *pNfft = 0;
    if (pAnaHop)
      *pAnaHop = 0;
    if (pArHop)
      *pArHop = 0;
    return;
  }

  // Dieselben Ausdruecke wie im Rechenpfad - bewusst hier wiederholt und
  // nicht in Python nachgebaut, wo sie beim naechsten Umbau veralten
  // wuerden.
  const Xorcist::PvKit &kit = x->m_Pv[Xorcist::KitIndex(mode)];
  const unsigned nfft = kit.nfft;
  const unsigned synHop = nfft / 4;
  const unsigned ovr
    = (unsigned)x->m_AnaHopOverride.load(std::memory_order_relaxed);

  if (pNfft)
    *pNfft = nfft;
  if (pAnaHop)
    *pAnaHop
      = (ovr && synHop % ovr == 0 && ovr <= synHop) ? ovr : kit.anaHop;
  if (pArHop)
    *pArHop = std::max(nfft / 8, x->m_SamplesPerBuffer);
}

void Xorcist_setTuning(Xorcist *x, const XorcistTuning *t) {
  x->m_SeamMode.store(t->seam, std::memory_order_relaxed);
  x->m_NoTrend.store(t->noTrend, std::memory_order_relaxed);
  x->m_AnaHopOverride.store(t->anaHop, std::memory_order_relaxed);
  x->m_FreqDamping.store(
    t->freqDamping < 0.0f ? 0.0f : (t->freqDamping > 1.0f ? 1.0f : t->freqDamping),
    std::memory_order_relaxed);
  Xorcist::s_UseSchur.store(t->schur ? 1 : 0, std::memory_order_relaxed);
  if (t->safetyCap >= 0.0f)   // negative keeps the built-in default
    x->m_SafetyCap.store(t->safetyCap, std::memory_order_relaxed);
  x->m_GateFit.store(t->gateFit ? 1 : 0, std::memory_order_relaxed);
  x->m_StaleWindow.store(t->staleWindow, std::memory_order_relaxed);
  x->m_Backfill.store(t->backfill ? 1 : 0, std::memory_order_relaxed);
  x->m_FitWindow.store(t->fitWindow, std::memory_order_relaxed);
  x->m_SolveEvery.store(t->solveEvery, std::memory_order_relaxed);
  x->m_BurgEvery.store(t->burgEvery, std::memory_order_relaxed);
  x->m_Replica.store(t->replica ? 1 : 0, std::memory_order_relaxed);
  x->m_FadeInRing.store(t->noFadeInRing ? 0 : 1, std::memory_order_relaxed);
  x->m_CapSpan.store(t->capSpan, std::memory_order_relaxed);
  x->SetReentryFade(t->reentryFade);
  x->m_AnchorTau.store(t->anchorTau, std::memory_order_relaxed);
  x->m_Gpu.store(t->gpu ? 1 : 0, std::memory_order_relaxed);
  x->m_PatchMode.store(t->patchTransients ? 1 : 0, std::memory_order_relaxed);
  x->m_Hybrid.store(t->hybrid ? 1 : 0, std::memory_order_relaxed);
  x->m_HybShortCap.store(t->hybShortCap, std::memory_order_relaxed);
  x->m_HybHandOrder.store(t->hybHandOrder, std::memory_order_relaxed);
  x->m_HybReset.store(t->hybReset ? 1 : 0, std::memory_order_relaxed);
  x->m_HybAllBands.store(t->hybAllBands ? 1 : 0, std::memory_order_relaxed);
  x->m_HybChooser.store(t->hybChooser ? 1 : 0, std::memory_order_relaxed);
  x->m_HybPre.store(t->hybPre ? 1 : 0, std::memory_order_relaxed);
  x->m_HybSimple.store(t->hybSimple ? 1 : 0, std::memory_order_relaxed);
  x->m_HybWarmOrder.store(t->hybWarmOrder, std::memory_order_relaxed);
  if (t->detResGate >= 0.0f)
    x->m_D4ResGate.store(t->detResGate, std::memory_order_relaxed);
  if (t->detResDecay > 0.0f)
    x->m_D4ResDecay.store(t->detResDecay, std::memory_order_relaxed);
  x->m_V0Hindcast.store(t->v0Hindcast ? 1 : 0, std::memory_order_relaxed);
  x->m_SolveTap.store(t->solveTap ? 1 : 0, std::memory_order_relaxed);
  // 0 = aus; JEDES rho > 0 scharf.  Die erste Fassung verwarf rho <= 1
  // stillschweigend als "sinnloses Budget" - der Nutzer hat die
  // Attrappe sofort gefunden: rho = 0.5 MUSS die Luecke hoerbar
  // ausbluten lassen (jede Mode halbiert sich pro Hop), sonst beisst
  // der Knopf nicht.  Extremwerte sind Beisstests, keine Fehler.
  Xorcist::s_DetectRho.store(
    t->detectRho > 0.0f ? t->detectRho : 0.0f, std::memory_order_relaxed);
  Xorcist::s_RootClamp.store(
    t->rootClamp > 0.0f ? t->rootClamp : 0.0f,
    std::memory_order_relaxed);
  x->m_Order.store(
    t->order < 1 ? 1 : (t->order > (int)Xorcist::AR_ORDER_MAX
                          ? (int)Xorcist::AR_ORDER_MAX
                          : t->order),
    std::memory_order_relaxed);
}

void Xorcist_replaceLast(Xorcist *x, const float *interleaved) {
  const uint64_t now = x->m_HistPeriod.load(std::memory_order_acquire);

  if (x->m_ConcealedThrough.load(std::memory_order_acquire) != now)
    return; // the last append was real audio already

  // Rewind one period and append again; the mirrored ring keeps both
  // copies consistent because Push() writes at w and w + HIST.
  x->m_HistWrite.store(
    (x->m_HistWrite.load(std::memory_order_relaxed) + Xorcist::HIST
     - x->m_SamplesPerBuffer)
      % Xorcist::HIST,
    std::memory_order_release);
  x->m_HistPeriod.store(now - 1, std::memory_order_release);
  x->m_ConcealedThrough.store(~0ull, std::memory_order_release);
  x->m_RecentPredMask.store(
    x->m_RecentPredMask.load(std::memory_order_relaxed) >> 1,
    std::memory_order_relaxed);
  x->m_RingBlendPending = false; // Wahrheit da, nichts zu blenden
  x->Push(interleaved); // shifts back in with bit 0 = real

  // The samples are corrected - now say so.  Until 2026-08-06 this
  // function fixed the history and told nobody: any spectrum already
  // computed over a window covering this period still described the
  // PREDICTION, and the cache would hand it out again on the next call.
  // Straggler truth is an advantage GO has and an offline prototype
  // never can, and it was being thrown away one frame short of the
  // finish line.
  //
  // Invalidating is enough - the seed backfill above recomputes on
  // demand, and it now reads a history that holds the truth.
  for (unsigned ch = 0; ch < x->m_NChannels; ch++) {
    x->m_SpecCachePeriod[ch] = ~0ull;

    const size_t ringBase = (size_t)ch * x->m_ArRingLen;
    const unsigned sl = (unsigned)(now % x->m_ArRingLen);

    if (x->m_ArRingPeriod[ringBase + sl] == now)
      x->m_ArRingPeriod[ringBase + sl] = ~0ull;

    const unsigned slB = (unsigned)(now % Xorcist::HYB_RING);

    if (x->m_HybRingPeriod[(size_t)ch * Xorcist::HYB_RING + slB] == now)
      x->m_HybRingPeriod[(size_t)ch * Xorcist::HYB_RING + slB] = ~0ull;
  }
}

void Xorcist_getTuning(const Xorcist *x, XorcistTuning *t) {
  t->seam = x->m_SeamMode.load(std::memory_order_relaxed);
  t->noTrend = x->m_NoTrend.load(std::memory_order_relaxed);
  t->anaHop = x->m_AnaHopOverride.load(std::memory_order_relaxed);
  t->freqDamping = x->m_FreqDamping.load(std::memory_order_relaxed);
  t->order = x->m_Order.load(std::memory_order_relaxed);
  t->schur = Xorcist::s_UseSchur.load(std::memory_order_relaxed);
  t->safetyCap = x->m_SafetyCap.load(std::memory_order_relaxed);
  t->gateFit = x->m_GateFit.load(std::memory_order_relaxed);
  t->staleWindow = x->m_StaleWindow.load(std::memory_order_relaxed);
  t->backfill = x->m_Backfill.load(std::memory_order_relaxed);
  t->fitWindow = x->m_FitWindow.load(std::memory_order_relaxed);
  t->solveEvery = x->m_SolveEvery.load(std::memory_order_relaxed);
  t->burgEvery = x->m_BurgEvery.load(std::memory_order_relaxed);
  t->replica = x->m_Replica.load(std::memory_order_relaxed);
  t->noFadeInRing = x->m_FadeInRing.load(std::memory_order_relaxed) ? 0 : 1;
  t->capSpan = x->m_CapSpan.load(std::memory_order_relaxed);
  t->reentryFade = (int)x->ReentryLen();
  t->anchorTau = x->m_AnchorTau.load(std::memory_order_relaxed);
  t->rootClamp = Xorcist::s_RootClamp.load(std::memory_order_relaxed);
  t->gpu = x->m_Gpu.load(std::memory_order_relaxed);
  t->detectRho = Xorcist::s_DetectRho.load(std::memory_order_relaxed);
  t->patchTransients = x->m_PatchMode.load(std::memory_order_relaxed);
  t->hybrid = x->m_Hybrid.load(std::memory_order_relaxed);
  t->hybShortCap = x->m_HybShortCap.load(std::memory_order_relaxed);
  t->hybHandOrder = x->m_HybHandOrder.load(std::memory_order_relaxed);
  t->hybReset = x->m_HybReset.load(std::memory_order_relaxed);
  t->hybAllBands = x->m_HybAllBands.load(std::memory_order_relaxed);
  t->hybChooser = x->m_HybChooser.load(std::memory_order_relaxed);
  t->hybPre = x->m_HybPre.load(std::memory_order_relaxed);
  t->hybSimple = x->m_HybSimple.load(std::memory_order_relaxed);
  t->hybWarmOrder = x->m_HybWarmOrder.load(std::memory_order_relaxed);
  t->detResGate = x->m_D4ResGate.load(std::memory_order_relaxed);
  t->detResDecay = x->m_D4ResDecay.load(std::memory_order_relaxed);
  t->v0Hindcast = x->m_V0Hindcast.load(std::memory_order_relaxed);
  t->solveTap = x->m_SolveTap.load(std::memory_order_relaxed);
}

void Xorcist_setMode(Xorcist *x, int mode) {
  x->m_Mode.store(mode, std::memory_order_relaxed);
}

int Xorcist_getMode(const Xorcist *x) {
  return x->m_Mode.load(std::memory_order_relaxed);
}

void Xorcist_push(Xorcist *x, const float *interleaved) {
  x->Push(interleaved);
}

void Xorcist_pushMany(
  Xorcist *x, const float *interleaved, unsigned periods) {
  const unsigned stride = x->m_NChannels * x->m_SamplesPerBuffer;

  for (unsigned p = 0; p < periods; p++)
    x->Push(interleaved + (size_t)p * stride);
}

void Xorcist_absorb(Xorcist *x) { x->Absorb(); }

int Xorcist_patchForce(
  Xorcist *x, unsigned ch, uint64_t zoneStart, unsigned len) {
  return ch < x->m_NChannels && x->BuildBackPatch(ch, zoneStart, len) ? 1 : 0;
}

unsigned Xorcist_patchRead(
  Xorcist *x, unsigned ch, uint64_t *pStart, float *out, unsigned maxLen) {
  if (ch >= x->m_NChannels || x->m_PatchEnd[ch] <= x->m_PatchStart[ch])
    return 0;

  const unsigned n = (unsigned)(x->m_PatchEnd[ch] - x->m_PatchStart[ch]);
  const unsigned c = n < maxLen ? n : maxLen;

  if (pStart)
    *pStart = x->m_PatchStart[ch];
  if (out)
    std::memcpy(
      out, x->m_PatchBuf.data() + (size_t)ch * Xorcist::PATCH_MAX,
      sizeof(float) * c);
  return c;
}

void Xorcist_patchClear(Xorcist *x) {
  for (unsigned ch = 0; ch < x->m_NChannels; ch++)
    x->m_PatchStart[ch] = x->m_PatchEnd[ch] = 0;
}

int Xorcist_predict(Xorcist *x) {
  // Convenience: the whole prediction on the calling thread.
  const int r = x->Predict();

  if (r == 1) {
    for (unsigned ch = 0; ch < x->m_NChannels; ch++)
      x->PredictChannel(ch);
    x->PredictEnd();
  }
  return r;
}

int Xorcist_predictBegin(Xorcist *x) { return x->Predict(); }
void Xorcist_predictChannel(Xorcist *x, unsigned ch) {
  x->PredictChannel(ch);
}
void Xorcist_predictEnd(Xorcist *x) { x->PredictEnd(); }
void Xorcist_predictAbort(Xorcist *x) { x->PredictAbort(); }
unsigned Xorcist_channels(const Xorcist *x) { return x->m_NChannels; }

int Xorcist_conceal(Xorcist *x, float *out) { return x->Conceal(out); }

void Xorcist_pass(Xorcist *x, float *out, const float *real) {
  x->Pass(out, real);
}

unsigned Xorcist_peek(const Xorcist *x, float *out, unsigned frames) {
  const uint64_t pub = x->m_ShadowPub.load(std::memory_order_acquire);

  if (pub == Xorcist::SHADOW_INVALID)
    return 0;

  const float *pShadow = x->m_Shadow[pub & 1ull].data();
  const unsigned n = frames < x->m_ShadowLen ? frames : x->m_ShadowLen;

  for (unsigned ch = 0; ch < x->m_NChannels; ch++)
    for (unsigned i = 0; i < n; i++)
      out[i * x->m_NChannels + ch] = pShadow[ch * x->m_ShadowLen + i];
  return n;
}

unsigned Xorcist_horizonFrames(const Xorcist *x) { return x->m_ShadowLen; }

uint64_t Xorcist_concealedCount(const Xorcist *x) {
  return x->m_ConcealCount.load(std::memory_order_relaxed);
}

uint64_t Xorcist_skippedCount(const Xorcist *x) {
  return x->m_SkipCount.load(std::memory_order_relaxed);
}

// Safety-cap diagnostics: how many shadow builds were scaled down, out
// of how many built.  Reading resets, so a caller can bracket one run.
void Xorcist_capStats(uint64_t *pHits, uint64_t *pCalls) {
  if (pHits)
    *pHits = Xorcist::s_CapHits.exchange(0, std::memory_order_relaxed);
  if (pCalls)
    *pCalls = Xorcist::s_CapCalls.exchange(0, std::memory_order_relaxed);
}

uint64_t Xorcist_patchCount(void) {
  return Xorcist::s_PatchCount.exchange(0, std::memory_order_relaxed);
}

unsigned Xorcist_patchLog(uint64_t *pOut, unsigned maxEvents) {
  const uint64_t n = Xorcist::s_PatchLogN.load(std::memory_order_relaxed);
  const unsigned have
    = (unsigned)(n < Xorcist::PATCH_LOG ? n : Xorcist::PATCH_LOG);
  const unsigned c = have < maxEvents ? have : maxEvents;
  const uint64_t first = n - c;

  for (unsigned i = 0; i < c; i++) {
    const uint64_t *e = Xorcist::s_PatchLog[(first + i) % Xorcist::PATCH_LOG];

    pOut[3 * i] = e[0];
    pOut[3 * i + 1] = e[1];
    pOut[3 * i + 2] = e[2];
  }
  return c;
}

unsigned Xorcist_hybLog(uint64_t *pOut, unsigned maxEvents) {
  const uint64_t n = Xorcist::s_HybLogN.load(std::memory_order_relaxed);
  const unsigned have
    = (unsigned)(n < Xorcist::HYB_LOG ? n : Xorcist::HYB_LOG);
  const unsigned c = have < maxEvents ? have : maxEvents;
  const uint64_t first = n - c;

  for (unsigned i = 0; i < c; i++) {
    const uint64_t *e = Xorcist::s_HybLog[(first + i) % Xorcist::HYB_LOG];

    pOut[3 * i] = e[0];
    pOut[3 * i + 1] = e[1];
    pOut[3 * i + 2] = e[2];
  }
  return c;
}

void Xorcist_hybLogClear(void) {
  Xorcist::s_HybLogN.store(0, std::memory_order_relaxed);
}

void Xorcist_patchLogClear(void) {
  Xorcist::s_PatchLogN.store(0, std::memory_order_relaxed);
  Xorcist::s_PatchCount.store(0, std::memory_order_relaxed);
}

void Xorcist_rhoStats(uint64_t *pFails, uint64_t *pTests) {
  if (pFails)
    *pFails = Xorcist::s_RhoFails.exchange(0, std::memory_order_relaxed);
  if (pTests)
    *pTests = Xorcist::s_RhoTests.exchange(0, std::memory_order_relaxed);
}

void Xorcist_hybStats(
  uint64_t *pActive, uint64_t *pInert, uint64_t *pBandPeriods) {
  if (pActive)
    *pActive = Xorcist::s_HybOnPeriods.exchange(0, std::memory_order_relaxed);
  if (pInert)
    *pInert = Xorcist::s_HybOffPeriods.exchange(0, std::memory_order_relaxed);
  if (pBandPeriods)
    *pBandPeriods
      = Xorcist::s_HybBandPeriods.exchange(0, std::memory_order_relaxed);
}

// Welche Diagnoseschalter der Umgebung greifen GERADE?  Ein Wirt soll das
// ANZEIGEN koennen, statt dass jemand nach einem A/B raten muss, ob seine
// Variable ueberhaupt angekommen ist - in diesem Projekt war stille
// Wirkungslosigkeit mehrfach die Fehlerursache.  Gemeldet wird nur, was
// vom Standard abweicht; leerer String heisst "alles auf Vorgabe".
const char *Xorcist_diagOverrides(void) {
  static const std::string s = [] {
    static const char *const names[] = {
      "XO_NO_FREEZE", "XO_D4_FLOOR", "XO_D4_DEN", "XO_PROFILE",
      "XO_NO_AVX2", "XO_STRIPE_CYCLE", "XO_NO_WARMSTART", "XO_SCHUR_RHO",
      "XO_SAFETY_CAP", "XO_WIDE_ON", "XO_WIDE_OFF", "XO_HYB_WIDE",
      "XO_SPLICE_VETO", "XO_HYB_REJOIN", "XO_HYB_ALLBANDS", "XO_HYB_NO_B",
      "XO_COUNT_RHO", "XO_QUIET_BEFORE", "XO_PRE_TTL", "XO_DUMP_BIN",
      "XO_V0_LAST", "XO_NO_SEAM", "XO_V0_SHADOW", "XO_V0_ORDER"};
    std::string out;

    for (const char *n : names) {
      const char *v = getenv(n);

      if (v && *v) {
        if (!out.empty())
          out += ' ';
        out += n;
        out += '=';
        out += v;
      }
    }
    return out;
  }();

  return s.c_str();
}

void Xorcist_setProfiling(int on) {
  s_Profile.store(on ? 1 : 0, std::memory_order_relaxed);
}

int Xorcist_profiling(void) {
  return s_Profile.load(std::memory_order_relaxed);
}

unsigned Xorcist_stageCount(void) { return (unsigned)XO_ST_COUNT; }

const char *Xorcist_stageName(unsigned i) {
  return i < XO_ST_COUNT ? s_StageNames[i] : "";
}

void Xorcist_stageStats(uint64_t *pNs, uint64_t *pCalls, unsigned maxStages) {
  const unsigned n = maxStages < XO_ST_COUNT ? maxStages : XO_ST_COUNT;

  for (unsigned i = 0; i < n; i++) {
    const uint64_t ns = s_StageNs[i].exchange(0, std::memory_order_relaxed);
    const uint64_t c = s_StageCalls[i].exchange(0, std::memory_order_relaxed);

    if (pNs)
      pNs[i] = ns;
    if (pCalls)
      pCalls[i] = c;
  }
}

unsigned Xorcist_detectorPerChannel(unsigned ch, uint64_t *pOnsets) {
  if (ch >= Xorcist::DET_CH_MAX) {
    if (pOnsets)
      *pOnsets = 0;
    return 0;
  }
  if (pOnsets)
    *pOnsets = Xorcist::s_OnsetPerCh[ch].load(std::memory_order_relaxed);
  return Xorcist::s_OccPerCh[ch].load(std::memory_order_relaxed);
}

void Xorcist_onsetStats(uint64_t *pOnsets, uint64_t *pTentative) {
  if (pOnsets)
    *pOnsets = Xorcist::s_HybOnsets.exchange(0, std::memory_order_relaxed);
  if (pTentative)
    *pTentative
      = Xorcist::s_HybOnsetsTent.exchange(0, std::memory_order_relaxed);
}

unsigned Xorcist_onsetLog(uint64_t *pOut, unsigned maxEvents) {
  const uint32_t seq = Xorcist::s_OnsetSeq.load(std::memory_order_relaxed);
  const unsigned have
    = seq < Xorcist::ONSET_LOG ? (unsigned)seq : Xorcist::ONSET_LOG;
  const unsigned n = maxEvents < have ? maxEvents : have;

  // Neueste zuletzt - ein Zeichner laeuft die Liste vorwaerts durch.
  for (unsigned i = 0; i < n; i++) {
    const uint32_t slot = (seq - n + i) % Xorcist::ONSET_LOG;

    pOut[i] = Xorcist::s_OnsetLog[slot].load(std::memory_order_relaxed);
  }
  return n;
}

void Xorcist_raceStats(uint64_t *pStaleDiscards, uint64_t *pLateSlices) {
  if (pStaleDiscards)
    *pStaleDiscards
      = Xorcist::s_StaleDiscards.exchange(0, std::memory_order_relaxed);
  if (pLateSlices)
    *pLateSlices = Xorcist::s_LateSlices.exchange(0, std::memory_order_relaxed);
}

void Xorcist_lateStats(uint64_t *pAfterSkip, uint64_t *pMaxSlice) {
  if (pAfterSkip)
    *pAfterSkip
      = Xorcist::s_LateAfterSkip.exchange(0, std::memory_order_relaxed);
  if (pMaxSlice)
    *pMaxSlice
      = Xorcist::s_LateSliceMax.exchange(0, std::memory_order_relaxed);
}

void Xorcist_gpuStats(uint64_t *pOnDevice, uint64_t *pOnCpu) {
  if (pOnDevice)
    *pOnDevice = Xorcist::s_GpuRanges.exchange(0, std::memory_order_relaxed);
  if (pOnCpu)
    *pOnCpu = Xorcist::s_CpuRanges.exchange(0, std::memory_order_relaxed);
}

uint64_t Xorcist_backfillCount(void) {
  return Xorcist::s_Backfills.exchange(0, std::memory_order_relaxed);
}

void Xorcist_arDropStats(uint64_t *pDrops, uint64_t *pTries) {
  if (pDrops)
    *pDrops = Xorcist::s_ArDrops.exchange(0, std::memory_order_relaxed);
  if (pTries)
    *pTries = Xorcist::s_ArTries.exchange(0, std::memory_order_relaxed);
}

void Xorcist_gateStats(uint64_t *pSkips, uint64_t *pTests) {
  if (pSkips)
    *pSkips = Xorcist::s_GateSkips.exchange(0, std::memory_order_relaxed);
  if (pTests)
    *pTests = Xorcist::s_GateTests.exchange(0, std::memory_order_relaxed);
}

float Xorcist_safetyCap(const Xorcist *x) {
  return x->m_SafetyCap.load(std::memory_order_relaxed);
}

const char *Xorcist_wisdomFileName(void) { return Xorcist::WISDOM_FILE_NAME; }

const char *Xorcist_version(void) { return XORCIST_VERSION; }

} // extern "C"
