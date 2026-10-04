# Metal reconstruction and render experiments, 2026-10-03

The useful decode-only change is balanced reconstruction loading plus band
batching. Coefficient-unpacking rewrites did not show a benefit. The larger
packet-to-RGB improvement comes from recording decode and rendering together,
and, for 4:4:4 at the default precision, fusing final reconstruction with RGB
conversion. Two-level wavelet fusion was rejected because it changes rounding
and adds redundant arithmetic.

All new experiments are opt-in in the CLI's static backend under
`PYROWAVE_METAL_BENCH_HOOKS`. The installed shared library's public API and
default decode shaders are unchanged. This investigation uses the existing
Metal command queue/buffer API; it does not depend on Metal 4.

## Measurement boundaries

`Packet to YUV GPU end` starts immediately before decoder reset and feeding the
cached compressed packets. It ends at the command buffer's `GPUEndTime`.
Parsing, upload, CPU command encoding, queue delay, and GPU decode are included;
the final CPU wakeup after GPU completion is excluded. `Packet to RGB GPU end`
extends that interval through offscreen RGB conversion. These timestamps use
the host clock corresponding to Metal GPU timestamps.

`Decode GPU command` measures the complete GPU command span. With chained
rendering, the GPU row covers decode and RGB conversion. With a CPU wait between
them, it sums their separate command durations and excludes the intervening
CPU/submission gap. Packet-to-output is therefore the primary render latency
comparison. Completed wall time additionally includes the final CPU wait return.

The renderer writes a shared RGBA8Unorm texture using full-range BT.709 and
nearest chroma lookup. It is a fixed reference workload, not a metadata-aware
production renderer. No drawable acquisition, display presentation, scanout,
network arrival gaps, transport wrapper, or changing video content is measured.
GPU-ready YUV can be handed to a renderer with an ordered GPU dependency; the
CPU does not have to wait for it before encoding that renderer's commands.

## Controlled comparisons

Apple M3, macOS 27.0.1, Xcode 27.0 / macOS SDK 27.0, AC power, Low Power Mode
disabled. The main fixture is deterministic mixed synthetic 3440x1440 8-bit
YUV420 or YUV444, precision 1 (FP32 arithmetic / FP16 wavelet storage), and a
1,000,000-byte encode budget. Packet hashes are `a2d764ffb14bd061` for 420 and
`0f43ad24f926aea4` for 444. Encoding happens once in setup and the encoder is
destroyed before timing.

Each main run measures 2,400 frames **per variant**, with 240 warmup frames per
variant and four initial priming decodes. Variant 0 is the baseline and 1 the
candidate, alternating ABBA on the same cached packets. Only one frame is in
flight. Worker QoS is interactive; diagnostic counters are disabled. Readback,
pipeline compilation, disk I/O, and warmup are excluded. Scheduling outliers
remain in the results.

Approximate 95% intervals use candidate-minus-baseline differences in each of
1,200 four-frame blocks, with a Bartlett/Newey-West correction over 60 blocks.
The analysis also supplies 20- and 120-block sensitivity. Negative deltas favor
the candidate. These intervals describe within-run mean effects, not a guarantee
for other content, hardware, power states, or tail latency.

## Reconstruction and coefficient decoding

The native reconstruction loader assigns the same 100 four-band gather tiles
to 64 threads in two rounds, reducing the busiest lanes from three loads to two.
Interior groups reuse one texture coordinate across bands; boundary groups
retain the canonical mirror loader. Lifting arithmetic, intermediate storage
precision, output format, and barriers are preserved. Source fingerprints and
a fixed 64-thread host invariant guard the transformed source.

Packed coefficient decoding transposes bit-plane bytes with 32-bit integer
operations. The hybrid uses simpler gathers for one through four planes and
the packed fallback for longer payloads. Neither changes signs, prefix offsets,
sparse zero fill, or texture writes. CPU parity checks exercised over 190,000
packed cases and 320,000 hybrid cases, including unaligned payloads and partial
chunks. GPU comparisons below passed exact output checks, but did not establish
a speedup, so neither is included in the useful `native-batched` combination.

| Unpaced experiment | Chroma | Baseline GPU, ms | Candidate GPU, ms | Delta, microseconds (95% interval) |
| --- | --- | ---: | ---: | ---: |
| Balanced reconstruction | 420 | 0.626 | 0.609 | -17.72 [-18.53, -16.91] |
| Balanced reconstruction | 444 | 1.262 | 1.235 | -27.01 [-29.10, -24.91] |
| Packed unpacking | 420 | 0.643 | 0.644 | +0.51 [-0.06, +1.07] |
| Packed unpacking | 444 | 1.324 | 1.325 | +0.94 [-5.67, +7.55] |
| Hybrid unpacking | 420 | 0.752 | 0.754 | +2.56 [-12.39, +17.50] |
| Hybrid unpacking | 444 | 1.342 | 1.347 | +4.79 [-7.66, +17.23] |

Baseline timings differ between separate runs. Compare each row within its own
run; percentages from separate experiments should not be added.

The useful `native-batched` combination retains canonical coefficient unpacking:

| Cadence | Chroma | Baseline GPU, ms | Candidate GPU, ms | Baseline packet-to-YUV, ms | Candidate, ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Unpaced | 420 | 0.737 | 0.709 | 0.932 | 0.911 |
| Unpaced | 444 | 1.440 | 1.399 | 1.649 | 1.603 |
| 240 fps | 420 | 1.263 | 1.235 | 1.831 | 1.800 |
| 240 fps | 444 | 1.802 | 1.708 | 2.396 | 2.283 |

The paired unpaced GPU reductions are 3.86% and 2.84%; paced reductions are
2.21% and 5.17% for 420/444 respectively. Paced packet-to-YUV deltas are
-30.85 microseconds [-59.55, -2.15] for 420 and -112.58 [-132.00, -93.15] for
444. Paced final CPU-wait deadline misses are 71 to 63 for 420 and 159 to 114
for 444, out of 2,400 samples per variant. Those counts describe these runs;
they do not establish a general reliability improvement.

## Wavelet-level fusion: rejected

Three prototypes reconstruct the last two levels in one workgroup: a 32x32
coarse patch, a compact 24x24 patch, and an eight-pixel phase-aligned 32x32 patch.
They remove intermediate LL texture traffic, but reconstruct overlapping halos.
The first short large-frame trial increased GPU time from 1.246 to 1.539 ms.
The compact patch still performs about 25% extra coarse lifting work.

Moving the coarse tile changes which eight-pixel main helper or four-pixel
apron helper produces a sample. Half storage and fast arithmetic make those
mathematically equivalent paths round differently. Phase alignment reduced
precision-1 differences from 1,442 pixels to 61 across all planes at 3440x1440
444. It still differed at 16,782 pixels for precision 0 and two for precision 2;
all differences were one LSB. Strict comparisons fail and produce no accepted
performance CSV. These prototypes are excluded from the useful combinations.

A future exact fusion would need canonical coarse tile boundaries, arithmetic
paths, and storage checkpoints, obtaining halos from neighboring canonical
tiles. Pruning redundant work must preserve that floating-point schedule.

## Decode and rendering

`render` compares CPU-wait-separated decode/render commands against recording
both in one command buffer. It mainly saves CPU wakeup and a second submission;
GPU arithmetic is effectively unchanged in the unpaced runs.

`render-fused` goes further: the last wavelet level writes RGB directly. For
444, it reconstructs Y/Cb/Cr in a workgroup, preserves the R8Unorm checkpoint
through explicit byte conversion, and applies the same RGB math as the reference
fragment shader. The 420 prototype reconstructs luma and reads the already
finished chroma planes. All final RGBA channels must match the reference.

| Unpaced comparison | Chroma | Baseline packet-to-RGB, ms | Candidate, ms | Delta, microseconds (95% interval) |
| --- | --- | ---: | ---: | ---: |
| CPU wait to chained render | 420 | 1.391 | 1.167 | -224.31 [-225.79, -222.82] |
| CPU wait to chained render | 444 | 2.669 | 2.143 | -525.77 [-653.39, -398.15] |
| Chained fragment to fused RGB | 420 | 1.411 | 1.319 | -91.74 [-114.01, -69.47] |
| Chained fragment to fused RGB | 444 | 2.254 | 2.055 | -198.79 [-252.12, -145.46] |

Fused RGB reduced unpaced GPU time by 4.86% for 420 and 9.01% for 444. Paced
420 fusion did not establish a gain, so the combined render path keeps fragment
conversion for 420. The 444 fused path improved paced GPU time by 8.70%, and
packet-to-RGB mean by 0.209 ms (95% interval -0.313 to -0.104 ms).

Large-frame precision checks caught an R8 checkpoint issue: manual
`rint(clamp(v) * 255)` in 4K 444 precision-2 RGB fusion differed in 270 channels
by up to two LSBs. Precision 2 now uses Metal's native `pack_float_to_unorm4x8`
intrinsic and passes the failing fixture. Applying that float intrinsic to all
precisions introduced half-variant differences, so precisions 0/1 retain their
tested manual checkpoint. The final precision-specific implementation passes
exact RGB checks at 4K for 420/444 and all precisions, and independent flat,
noise, and edge fixtures at all precisions. The precise compiler/texture
conversion cause has not been established. Performance tables use precision 1,
whose generated shader remains identical to the measured source.

The combined `render-optimized` comparison uses the original decoder followed
by a CPU wait and separate fragment render as baseline. The candidate combines
balanced loading, canonical band batching, and a single command buffer, using
RGB fusion for 444 and fragment conversion for 420. Its
results measure the actual combination, rather than adding separate gains:

| Cadence | Chroma | Baseline packet-to-RGB, ms | Candidate, ms | Change | Delta, microseconds (95% interval) |
| --- | --- | ---: | ---: | ---: | ---: |
| Unpaced | 420 | 1.618 | 1.362 | -15.83% | -256.25 [-276.38, -236.11] |
| Unpaced | 444 | 2.480 | 1.900 | -23.38% | -579.70 [-612.46, -546.94] |
| 240 fps | 420 | 2.647 | 2.198 | -16.99% | -449.70 [-474.25, -425.16] |
| 240 fps | 444 | 3.079 | 2.345 | -23.84% | -733.88 [-772.10, -695.66] |

At 240 fps, packet-to-RGB p99 was 4.275 to 3.738 ms for 420 and 4.644 to
3.681 ms for 444. Completed wall p99 was 4.488 to 4.011 ms for 420 and 5.071
to 4.054 ms for 444. Deadline misses, based on scheduled completion after the
final CPU wait, fell from 179 to 84 and from 423 to 103 respectively. Remaining
misses prevent a claim of reliable 240 Hz playback. GPU spans improve from
1.609 to 1.567 ms for 420 and 2.005 to 1.775 ms for 444; the larger ready-latency
gain also includes the removed CPU/submission gap.

## 4K at 120 frames/s

The same precision-1 combinations were measured at 3840x2160, with 1,200 frames
and 120 warmup frames per variant. These comparisons have 600 complete ABBA
blocks and the same 20-second measurement duration per pair. All output checks
passed. A frame budget is 8.333 ms.

| Combination | Chroma | Baseline packet-to-output, ms | Candidate, ms | Change | Deadline misses, baseline / candidate |
| --- | --- | ---: | ---: | ---: | ---: |
| Decode to YUV | 420 | 2.786 | 2.747 | -1.40% | 5 / 5 |
| Decode to YUV | 444 | 5.139 | 5.062 | -1.50% | 98 / 87 |
| Decode to RGB | 420 | 5.655 | 5.090 | -9.98% | 154 / 85 |
| Decode to RGB | 444 | 7.034 | 5.972 | -15.10% | 485 / 163 |

The decode-only GPU and packet-to-YUV intervals include zero at 4K120, so these
runs do not establish a decode-only GPU benefit at that cadence. The combined
packet-to-RGB deltas are -0.565 ms [-0.625, -0.504] for 420 and -1.062 ms
[-1.171, -0.954] for 444. Candidate RGB-ready p99 is 7.850 ms for 420 and
9.106 ms for 444; candidate completed-wall p99 is 8.161 and 9.362 ms. Deadline
misses remain, especially in 444, despite improved means.

## Validation and remaining limits

Balanced reconstruction and band batching pass exact YUV comparisons at 4K
for 420/444 and all three precisions. The useful render combination passes exact
RGB comparisons for those cases with the precision-specific checkpoint above. Additional
258x146 flat, noise, and sharp-edge fixtures passed both combinations for both
chroma formats at 5 KB and 1 MB budgets. Individual variants were also checked
at 64x48 420 and odd 65x49 444 across precisions, and at 3440x1440.

The first investigation's large-444 readiness failure is fixed by the encoder
scratch-allocation bound described in [PERFORMANCE.md](PERFORMANCE.md). It no
longer prevents large-444 decode fixtures. The parser/scratch CPU test passes.

Paced 240 fps runs still show host/submission stalls and missed deadlines. A
4.167 ms frame budget and a mean below that budget do not establish reliable
240 Hz playback. Tail latency, actual presentation cadence, renderer color
metadata, changing streams, partial-frame updates, and bounded asynchronous
buffer ownership still require client acceptance. No shared-library decode
default or live client integration was promoted by this investigation.

## Reproduction and artifacts

Build and compare the decode-only combination:

```sh
./script/build_and_run.sh --build-only
./cmake-build-metal/pyrowave-metal-bench --decode-only \
    --width 3440 --height 1440 --chroma 444 --bytes 1000000 --precision 1 \
    --frames 2400 --warmup 240 --qos interactive --compare native-batched \
    --samples cmake-build-metal/native-batched.csv
```

For the combined packet-to-RGB experiment, replace the comparison with
`--compare render-optimized`. Add `--fps 240` to measure paced scheduling.
Use `--chroma 420` to test 420. The standalone decode combination is
`--native-idwt --batched-dequant`; default decode remains available without flags.

```sh
python3 script/analyze_metal_bench.py cmake-build-metal/decode-v2-ab-*.csv \
    --outputstem cmake-build-metal/decode-v2-all-analysis
```

The tracked analyzer retains all finite samples, reports missing data/cycles,
and uses only complete ABBA cycles for pairing. Intervals are unavailable when
the block count does not exceed the requested lag window. Unpaced deadline
flags are zero because no deadline was scheduled.

Local raw logs/CSVs are `decode-v2-ab-<comparison>-<chroma>-<cadence>.*` under
the ignored `cmake-build-metal/` directory. Correctness logs use
`decode-v2-validation-*`; rejected fusion diagnostic dumps and source notes are
also local. `decode-v2-all-analysis.json` and `.md` contain distributions,
deadline counts, and interval sensitivity for the successful full comparisons.
There are 34 complete full comparisons in the final local analysis. `4k120`
names identify the separate 4K cadence above; the other full runs use 3440x1440.
