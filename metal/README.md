
This is an AI assisted port of Pyrowave to Metal on Apple platforms.

This has no Granite code, using native Metal and converting the shaders to MSL.

This is covered by the standard MIT license available in ../LICENSE.

Upstream note: This port may be removed entirely at some point if some effort is spend on a more direct implementation.
This code duplication and AI use is a pragmatic compromise since a quick and dirty port was needed by clients and upstream did not have a setup to do a "proper" port. Upstream is not able to maintain or support this port beyond the absolute minimum and is provided here for convenience for said clients.

Port notes:
* The API has been simplified, removing Vulkanisms and using Metal objects for GPU decode.
* You can set the environment variable PYROWAVE_PRECISION to 0, 1, or 2, to make the same precision/speed tradeoffs as the main library.
* Using FP32 math and FP16 storage in the shaders ended up being the fastest and most accurate combination on Apple hardware.
* This implementation is roughly twice as fast as the Vulkan implementation running on KosmicKrisp on macOS

## macOS performance CLI

On an Apple Silicon Mac with Xcode command line tools and CMake, build and run
the native Metal benchmark in Release mode from the repository root:

```sh
./script/build_and_run.sh
```

The executable is `cmake-build-metal/pyrowave-metal-bench`. It links its Metal
backend statically and needs no Granite, Vulkan loader, or separate PyroWave
dylib. To build without starting a benchmark:

```sh
./script/build_and_run.sh --build-only
```

Run the executable directly for subsequent measurements, avoiding a rebuild:

```sh
# Synthetic 1080p, 4:2:0, GPU-resident IOSurface input.
./cmake-build-metal/pyrowave-metal-bench --frames 1000 --warmup 30

# 1080p, 4:4:4, 1 MB maximum encoded payload per frame.
./cmake-build-metal/pyrowave-metal-bench --width 1920 --height 1080 \
    --chroma 444 --bytes 1000000 --frames 1000

# Decode only: 4K 4:2:0 at 120 frames/s for 10 seconds, after one second of warmup.
./cmake-build-metal/pyrowave-metal-bench --decode-only --width 3840 --height 2160 \
    --chroma 420 --bytes 1000000 --fps 120 --frames 1200 --warmup 120

# Decode only: 3440x1440 at 240 frames/s for 10 seconds.
./cmake-build-metal/pyrowave-metal-bench --decode-only \
    --width 3440 --height 1440 --chroma 420 --bytes 1000000 \
    --fps 240 --frames 2400 --warmup 240

# Repeatedly encode/decode the first frame of an 8-bit 420 or 444 Y4M file.
./cmake-build-metal/pyrowave-metal-bench --input test.y4m --frames 1000

# Include CPU texture upload in encode wall time, and compare full FP32.
./cmake-build-metal/pyrowave-metal-bench --input-mode cpu --precision 2

./cmake-build-metal/pyrowave-metal-bench --help
```

Precision 1 is the default (FP32 arithmetic with FP16 wavelet storage); 0 and 2
select FP16 throughout and FP32 throughout. `--precision` overrides
`PYROWAVE_PRECISION`.

The benchmark excludes pipeline creation, input preparation, output readback,
and warmup frames. Setup primes four roundtrips to allocate all decoder upload
slots before measurement, including when `--warmup 0` is requested. It reports
mean and percentile times for the encode and
decode Metal command buffers, CPU packetization, and completed wall-clock
operations. GPU-input encode wall time includes IOSurface wrapping, submission,
and the wait for completion; CPU-input encode wall time also includes texture
upload. Decode wall time includes packet parsing, upload, submission, and the
wait. Roundtrip wall time includes all three stages. GPU times include all work
in each codec command buffer, including encoder scratch-buffer clears.
Equivalent FPS is the reciprocal of each stage's mean time, rather than video
playback throughput.
`--fps` schedules frames at a fixed cadence and reports deadline misses against
the next scheduled frame, processing times exceeding the frame budget, and the
maximum scheduling lateness. Pacing waits are excluded from stage timings. This
tests codec scheduling at the requested frame rate; it does not drive a display.

`--decode-only` encodes and packetizes one frame during setup, caches its packets,
and destroys the encoder before priming and measurement. Each measured frame
replays those packets through the decoder, including parser reset, packet
parsing, GPU upload, compute submission, and completion. Encoding and
packetization contribute no work to the decode timing loop. Four decoder upload
slots are primed before the configured warmup, and output validation still runs
after measurement. This mode tests repeated resident compressed input rather
than transport or changing video content.

The final decoded image is read back for per-plane PSNR and a checksum outside
the timing loop. Synthetic input and a repeated first Y4M frame are controlled
microbenchmarks; they do not measure video playback, network latency, or display
presentation. Keep input, dimensions, chroma, byte budget, precision, and GPU
load the same when comparing runs. The `--bytes` budget applies to the codec
payload; packet headers add overhead to the reported packetized byte count.

The earlier `Packetized frame is not complete for decoding` failure at large
4:4:4 dimensions was fixed. Encoder coefficient scratch storage was undersized:
the final compressed byte budget does not bound the intermediate coefficients
before rate control. Storage now covers the padded coefficient blocks, and
3440x1440 and 3840x2160 4:4:4 fixtures decode successfully with a 1 MB budget.

For a CMake-only build, configure the Metal directory directly:

```sh
cmake -S metal -B cmake-build-metal -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 -DPYROWAVE_METAL_CLI=ON
cmake --build cmake-build-metal --parallel
```

The CLI is enabled by default on macOS. Set `PYROWAVE_METAL_CLI=OFF` for a
library-only build. Its private timing hooks and experimental shaders are
compiled into a separate static backend and do not change the installed shared
library API or its default decode behavior.

### Decode investigation

The CLI can separate parser, command encoding, commit-to-GPU-start, GPU execution,
and GPU-end-to-CPU-return costs. Optional GPU pass counters further separate
dequantization from inverse wavelet transformation:

```sh
./cmake-build-metal/pyrowave-metal-bench --decode-only \
    --width 3440 --height 1440 --chroma 420 --bytes 1000000 --precision 1 \
    --fps 240 --frames 480 --warmup 240 --profile \
    --samples cmake-build-metal/decode-profile.csv
```

Counters are diagnostic: sampling and resolving them can change execution and
scheduling. Counter resolution follows the completion timestamp, so it is
excluded from the current decode wall time but can affect the next frame's
start. Use unprofiled runs for performance comparisons. Commit CPU overlaps the
commit-to-GPU interval and should not be added to it.

For comparisons, `--compare output`, `--compare dequant`, or `--compare idwt`
alternates variants in ABBA order on the same cached packets. `--frames` and
`--warmup` apply **per variant**. At 240 fps, 2400 frames per variant requires
20 seconds of measurement. CSV variant 0 is the baseline and 1 is the candidate.
Readback follows measurement and all three decoded planes must be byte-identical.
RGB fusion comparisons validate the final RGBA output instead, because the
candidate writes RGB directly and does not produce final YUV planes.

```sh
./cmake-build-metal/pyrowave-metal-bench --decode-only \
    --width 3440 --height 1440 --chroma 420 --bytes 1000000 --precision 1 \
    --fps 240 --frames 2400 --warmup 240 --compare dequant \
    --samples cmake-build-metal/decode-dequant-ab.csv
```

`output` compares shared versus private output textures (or the reverse when
`--output-storage private` is specified). `dequant` compares the original shader
against band batching, reducing 42 dispatches to 13 for 4:2:0. `idwt` compares
the original inverse transform against removal of one redundant threadgroup
barrier in its apron helper; inter-level dependencies retain their barriers.
These shader variants are experimental and remain disabled by default.
`--batched-dequant` and `--reduced-idwt-barriers` enable individual variants for
standalone or profiled runs; comparison mode disallows these override flags.

The current reconstruction, unpacking, and offscreen rendering experiments are:

| Comparison | Candidate |
| --- | --- |
| `native-idwt` | Balanced interior wavelet-band loads; original edge loader and lifting arithmetic |
| `native-dequant` | Packed integer bit-plane transposition |
| `hybrid-dequant` | Short-plane unpacking plus packed fallback |
| `native-batched` | Balanced reconstruction plus original coefficient unpacking with band batching |
| `combined` | Packed unpacking plus balanced reconstruction and band batching |
| `optimized` | Hybrid unpacking plus balanced reconstruction and band batching |
| `render` | Decode and fragment RGB conversion in one command buffer; baseline waits on the CPU between them |
| `render-fused` | Final reconstruction writes RGB directly; baseline chains decode and fragment conversion |
| `render-optimized` | Balanced reconstruction, band batching, and chained rendering; additionally uses final RGB fusion for 4:4:4 |

`--native-idwt`, `--native-dequant`, and `--hybrid-dequant` select individual
standalone experiments. The unpacking rewrites have not demonstrated a speedup.
`--fused-idwt`, `--compare fused-idwt`, and `--compare compact-idwt` are rejected
two-level reconstruction prototypes: they change rounding and strict A/B checks
can fail. They remain available for diagnosis and are excluded from the faster
combinations above.
The RGB fusion checkpoint uses precision-specific UNORM conversion and passes
4K checks across all three precisions. `render-optimized` retains fragment
conversion for 4:2:0, whose paced fusion result did not establish a gain.

Test the useful decode combination without pacing:

```sh
./cmake-build-metal/pyrowave-metal-bench --decode-only \
    --width 3440 --height 1440 --chroma 444 --bytes 1000000 --precision 1 \
    --frames 2400 --warmup 240 --qos interactive --compare native-batched \
    --samples cmake-build-metal/native-batched.csv
```

Measure packet submission through offscreen RGB completion at 240 frames/s:

```sh
./cmake-build-metal/pyrowave-metal-bench --decode-only \
    --width 3440 --height 1440 --chroma 444 --bytes 1000000 --precision 1 \
    --fps 240 --frames 2400 --warmup 240 --qos interactive \
    --compare render-optimized --samples cmake-build-metal/render-optimized.csv

python3 script/analyze_metal_bench.py \
    cmake-build-metal/native-batched.csv cmake-build-metal/render-optimized.csv \
    --outputstem cmake-build-metal/variants-analysis
```

`Packet to YUV GPU end` starts immediately before decoder reset and cached packet
parsing, and ends at the decode command buffer's GPU completion timestamp.
It includes parsing, upload, command encoding, queue delay, and GPU work, and
excludes the final CPU wakeup. With rendering, `Packet to RGB GPU end` ends after
RGB conversion. `Decode wall + wait` / `Packet to RGB + wait` includes the final
wait return as well. CSV field `packet_to_gpu_ms` records the GPU-ready interval.

`--render wait` waits for decode before submitting a separate render command;
`--render chain` puts both in one command buffer. The offscreen reference renderer
uses full-range BT.709, nearest chroma, and RGBA8 output. These are fixed benchmark
assumptions, not stream color metadata handling. Render comparisons never acquire
a drawable or present to a display. Their GPU row measures the complete command
buffer, or the sum of both GPU commands for `wait`; that sum excludes the CPU gap.

The analysis script retains scheduling outliers, reports percentiles and paired
ABBA deltas, and writes JSON and Markdown. Unpaced CSV deadline flags are zero
because there is no scheduled deadline; they do not establish reliable playback.

Worker QoS is inherited unless `--qos default` or `--qos interactive` is supplied.
The CLI prints the actual class. Removing `--fps` tests sustained unpaced decode;
that is a useful diagnostic, but it does not establish 240 fps scheduling behavior.
See [the first measured investigation](PERFORMANCE.md) and
[the reconstruction and rendering experiments](DECODE_EXPERIMENTS.md) for results.
