# Metal decode investigation, 2026-09-30

Follow-up: [reconstruction, unpacking, fusion, and rendering experiments](DECODE_EXPERIMENTS.md),
2026-10-03. The results below describe the first investigation.

On this Apple M3, the first experiments produce small decode improvements rather
than a substantial reduction at 240 fps. Band batching is the strongest result
so far: roughly 10 microseconds less GPU time under sustained load. Private
output textures show no convincing benefit. All experiments remain optional in
the CLI's private backend; the shared library's default decode shaders are unchanged.
The subsequent encoder scratch-allocation correctness fix also applies to the
shared library.

## Measurement

The main fixture is a deterministic 3440x1440, 8-bit YUV420 frame, precision 1
(FP32 arithmetic, FP16 storage), with a 1,000,000-byte encode budget. Its actual
packetized size is 999,932 bytes. Encoding occurs once in setup; the encoder is
destroyed before decode measurement. Every iteration resets and parses the
cached packets, uploads their payload, records a fresh decode, submits it, and
waits for completion. Network transport and presentation are outside this test.

Each comparison uses the same cached packets for both variants, alternating in
ABBA order. It measures 2,400 frames per variant after 240 warmup frames per
variant and four initial decoder priming frames. The 240 fps comparisons schedule
4,800 measured frames over 20 seconds. Only one command buffer is in flight.
All three final output planes are compared byte-for-byte after measurement.
Disk I/O and output readback are excluded. Both sustained and paced runs inherited
user-interactive worker QoS (class 33), on macOS 27.0.1.

The IDWT comparison logs also record identical compressed-packet FNV-1a64
`a2d764ffb14bd061` in their unpaced and paced runs. Their output checksums are
`1203a486675cf6b8`, `8c7617e30f8950af`, and `de56db3ea1714a85` for Y, Cb, and Cr.

## Phase attribution

Separate diagnostic runs with GPU pass counters measured:

| Mean phase, ms | Paced at 240 fps | Unpaced |
| --- | ---: | ---: |
| Complete decode GPU command | 1.817 | 0.637 |
| Dequant GPU pass | 0.707 | 0.183 |
| IDWT GPU pass | 1.109 | 0.454 |
| Packet parsing CPU | 0.150 | 0.072 |
| Upload CPU, included in command encoding | 0.076 | 0.016 |
| Complete command encoding CPU | 0.223 | 0.028 |
| Commit to GPU start | 0.554 | 0.109 |
| GPU end to CPU return | 0.216 | 0.136 |
| Completed decode wall time | 2.960 | 0.983 |

These are 480-sample profiles with counters enabled, not candidate acceptance
runs. Counter collection can perturb execution. Commit CPU overlaps the
commit-to-GPU interval; upload is part of command encoding. Counter resolution
follows the completion timestamp but can delay the next scheduled iteration.

IDWT accounts for about 61% of GPU time in the paced profile and 71% unpaced.
The larger cross-cadence difference also affects CPU phases. Activity/power state
and scheduling are plausible causes, but these measurements alone do not establish
which one explains it. The unpaced result is not a 240 fps latency result.

## Matching Metal traces

A pair of process-targeted Metal System Traces used the original decoder and
identical cached packet checksum, with the same precision, dimensions, budget,
and inherited interactive QoS. The paced trace measured 1,200 frames; the
unpaced trace measured 2,400. Analysis excludes the first 244 chronological
decode command buffers in each (four priming and 240 warmup frames), groups GPU
subintervals by command buffer, and weights states by actual decode GPU-active
time. All measured decode GPU intervals have recorded state coverage.

| Traced measurement | 240 fps | Unpaced |
| --- | ---: | ---: |
| Mean GPU command span | 1.718 ms | 0.655 ms |
| Decode GPU-active time at Minimum | 99.424% | 3.947% |
| Decode GPU-active time at Medium | 0.576% | 9.844% |
| Decode GPU-active time at Maximum | 0% | 86.209% |

This establishes a large recorded performance-state difference accompanying
the timing gap. It supports prioritizing power-state behavior alongside paced
submission and wakeups. It does not provide MHz or prove that state selection
alone causes the entire gap. Apple describes how GPU utilization, thermals,
system settings, and induced device conditions influence GPU state in
[its Metal performance profiling session](https://developer.apple.com/videos/play/wwdc2021/10157/).

Both trace exports report `IsInduced=Yes` with a narrative about active device
conditions, while desired-state is 0 and recording metadata requests Default.
The origin and precise semantics of that combination remain unresolved; it
does not establish an intentional override or ordinary automatic scaling.
The GPU rows merge dequant and IDWT, so these traces cannot separate their pass
costs. Instruments overhead also makes these diagnostic captures unsuitable
as candidate acceptance runs.
After the captures, `pmset` reported AC power and `lowpowermode 0`; the Mac was
already plugged in with Low Power Mode disabled at that check.

## A/B results

Approximate within-run 95% intervals below use the mean candidate-minus-baseline
difference in each four-frame ABBA block, with a Bartlett/Newey-West serial
correlation correction over 60 blocks. There are 1,200 paired blocks per run.
Negative deltas favor the candidate. No scheduling outliers were removed.
Intervals describe these runs, not different content, Macs, or operating conditions.

| Experiment | Cadence | Baseline GPU, ms | Candidate GPU, ms | GPU delta, microseconds (approx. 95% interval) |
| --- | --- | ---: | ---: | ---: |
| Private output | 240 fps | 1.592 | 1.585 | -6.45 [-21.70, +8.80] |
| Private output | Unpaced | 0.642 | 0.638 | -4.20 [-8.49, +0.08] |
| Batched dequant | 240 fps | 1.641 | 1.625 | -15.83 [-30.79, -0.87] |
| Batched dequant | Unpaced | 0.642 | 0.631 | -10.44 [-13.72, -7.16] |
| Batched dequant, repeat | 240 fps | 1.703 | 1.689 | -14.15 [-32.11, +3.81] |
| Reduced IDWT barriers | Unpaced | 0.636 | 0.632 | -4.59 [-7.44, -1.75] |
| Reduced IDWT barriers | 240 fps | 1.707 | 1.694 | -12.73 [-28.07, +2.61] |

Private output has no established material benefit. The wavelet pyramid already
uses private textures, so this experiment changes only the final output planes.

Band batching reduces 4:2:0 dequant dispatches from 42 to 13, while preserving
the shader's coefficient decoding and missing-block zero fill. Its unpaced GPU
benefit is 1.63%; completed wall time improves by 13.66 microseconds (1.39%).
CPU command encoding improves by about 8% unpaced. Paced GPU means move in the
same direction, but completed latency and deadline reliability remain uncertain.
The paced repeat recorded 160 versus 141 misses out of 2,400 per variant; that
single difference does not establish a cadence improvement.

The IDWT variant removes one redundant apron-helper threadgroup barrier. Its
coordinate proof is in `experimental_idwt.msl.h`, with fingerprints of the reviewed
shader sources and a fixed 64-thread host invariant. All other threadgroup joins
and all four inter-level texture barriers remain. The unpaced improvement is
0.72% GPU and 0.46% wall time. In the paced run, candidate wall p95/p99 were
slightly worse and deadline misses were 55 versus 61. There is no demonstrated
paced latency or reliability benefit.

Early paced comparisons had large host stalls: one 297 ms wall sample spent only
0.855 ms on the GPU and 266 ms between GPU completion and CPU return. Another
211 ms sample spent 1.65 ms on the GPU, with 87 ms before GPU start and 122 ms
after GPU completion. These stalls distort raw wall averages and should not be
attributed to texture storage or shader execution.

## Validation and next work

Both shader experiments matched all three output planes for 3440x1440 420 and
3840x2160 420 at precision 1. Each also passed small 64x48 420, odd 65x49 444,
and flat 64x64 420 fixtures at precisions 0, 1, and 2, with repeated variant
switching. These are full-frame synthetic/fixture checks; changing streams,
partial packets, and real client playback remain outside validation.

At the time of this investigation, synthetic 1920x1080, 3440x1440, and 4K 444
fixtures with a 1 MB budget failed full-frame readiness before timing. This was
subsequently fixed: encoder intermediate coefficient scratch storage must cover
the padded coefficient blocks before rate control, rather than depend on the
final compressed byte budget. For 3440x1440 444, a diagnostic measured 10,151,561
coefficient bytes against the old 9,953,272-byte allocation; the corrected bound
allocates 29,926,656 bytes. Large 444 validation and measurements are now recorded
in the follow-up report.

The next priorities are:

1. Investigate the recorded performance-state selection, paced submission, and
   completion wakeups using the matching traces as a starting point.
   Worker QoS was already interactive; simply requesting it again is not a fix.
   A bounded asynchronous submission experiment could test the cost of waiting
   after every frame, but must measure completion latency and avoid queue growth.
2. Profile IDWT occupancy, register pressure, texture sampling, and threadgroup
   synchronization before a larger rewrite. The final two levels contain about
   87% of its threadgroups at this resolution, making them the useful initial
   target. Fusing levels must preserve neighborhood dependencies and rounding.
3. Defer a direct parser-to-Metal-buffer payload sink until the above work is
   understood. Measured upload CPU cost is only 0.016-0.076 ms here. A sink must
   preserve packet validation, duplicate handling, padded shader reads, partial
   frame updates, and bounded in-flight storage; a vector-backed no-copy buffer
   would not provide the required lifetime guarantees.

## Reproduction and artifacts

See `README.md` for the CLI comparison and profiling commands. Local raw CSVs,
logs, `decode-ab-analysis.json`, `decode-ab-analysis.md`, and the reproducible
`analyze_decode_ab.py` are under `cmake-build-metal/`. That build directory is
ignored by Git. The analysis script reports the paired deltas and sensitivity
to 20-, 60-, and 120-block correlation windows. These local results are evidence
for optional prototypes, not acceptance of a production decoder change.
Matching traces are `decode-metal.trace` and `decode-metal-unpaced.trace`, with
exported XMLs and `decode-metal-analysis.md` / `.json` beside them. Earlier
unprofiled CSVs wrote zero for unavailable optional counter fields; current
CLI runs use NaN. Those fields are not used in the unprofiled A/B analysis.
