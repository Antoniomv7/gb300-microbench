# Measurement protocol

## Population and environment

One NVIDIA B300 SXM6 AC exposes 148 SMs, ECC enabled, dynamic clocks and a
1100 W power limit. `VERSIONS.env` fixes the container and dependencies.
Every acquisition records the source commit, GPU UUID and a common
GPU/software/power-limit fingerprint. Final analysis requires complete,
finished acquisitions and rejects pilot data and mixed identities.

| Block | Population |
|---|---|
| I | 2 transfer methods x 3 stage counts x 3 in-flight sizes x 30 launches per campaign |
| II | 2 UMMA units x 3 N values x 4 depths x 30 launches per campaign |
| III | 2 UMMA units x isolated/device scale x 30 launches per campaign |
| IV | 5 shapes x NP1, P1, P2 256x128, P2 256x256 and cuBLASLt per campaign |
| V | 3 shapes x 3 formats x 2 implementations x 3 repetitions |
| Sustained power | 2 methods x 3 windows, each with 10 s warm-up and at least 30 s measured |
| GEMM profile | Both P2 tiles and cuBLASLt x 3 shapes = 9 captures |

I-IV have three campaigns. I-III reduce 30 launches to a campaign median,
then report mean, sample SD and CV across campaigns. IV uses a mean time
over 20 timed launches per candidate and campaign. V has three repetitions.
Whiskers consistently show the observed minimum and maximum, as descriptive
dispersion. Individual profiles have no estimated sampling interval.

## Validation and work

Preparation, compilation and numerical validation occur outside timing.
I checks its deterministic vector pattern in a separate kernel. II-III use
exactly representable small integers and read back all TMEM accumulators.
IV shares BF16 operands and an IEEE-FP32 reference across candidates.
V checks identical operand bytes, including NVFP4 block scales, across
libraries, reconstructs the represented operands and validates every result.
A negative control checks the validator; cuBLASLt plans must pass algorithm
checks and remain equivalent across operand sets.

I uses a working set larger than twice L2, 32 passes, 128-thread CTAs and
one resident CTA per SM. Its 256 B-wide unicast TMA boxes have no swizzle;
their volumes are 2-32 KiB. At 8 stages and 16 KiB in flight, each is 2 KiB.
The metric is useful logical bytes divided by whole-kernel time. Separate
DRAM captures check transferred volume. This is the tested copy geometry.

## Timing and effective clock

II-III bracket a `%clock64` read with `%globaltimer` reads. Midpoints
estimate its global time; the sum of start/end bracket widths bounds timestamp
uncertainty and must be below 1% of the loop interval. Raw records retain
cycles, timestamps and SM IDs. For elapsed cycles C and nanoseconds T:

```text
effective_sm_clock_mhz = 1000 C / T
loop_flops_per_cycle_sm = normalized work / C
loop_tflops_per_sm = normalized work / (1000 T)
```

The isolated two-SM unit is observed from the issuing SM and normalized by
two resources. III saves every CTA's interval, including partner SMs; work
is apportioned equally for per-resource normalization. A residency handshake
and timed SM IDs require distinct simultaneous SMs. The aggregate loop
interval is `max(end_ns) - min(start_ns)` across CTAs. The primary III rate
uses whole-kernel CUDA events. Products of aggregated clock/work-cycle
statistics are diagnostic, rather than exact decompositions of event time.
External clock telemetry supplies operating context.

The nominal BF16 bridge is `8192 FLOP/cycle/SM x 148 SM x 1.855799 GHz
= 2.25 PFLOP/s`. This frequency is implied by the nominal peak; effective
loop clocks are measured independently from paired cycles and timestamps.

IV-V use the pinned CUTLASS graph-replay helper as their common primary
policy and an ordinary-host/event diagnostic on the same allocated workspace
and explicit non-default stream. Each policy has 5 warm-up and 20 timed
launches. Allocation, compilation and graph capture precede timed replay.
Unsupported capture is an error. A scoped NVFP4 adapter directs the example's
stream helper to the current stream and restores it after the call.

Timing order alternates in campaign/repetition 2. III configuration order
also reverses in campaign 2; sustained-power order is AB, BA, AB.
NCU captures a separate ordinary validated launch after warm-up. Its duration,
graph time and host/event time have different replay and profiling conditions;
their ratios are diagnostics, not calibrated measurements of Python overhead.

## Sustained operating power

Validated full-device kernels repeat throughout each window. Output copies,
validation and profiling occur outside it. The driver saves exact host
boundaries, monotonic duration, launch count, total work and summed device
event time. Device-event duty fraction must be at least 90% and at most 1.01.

The sampler identifies GPU-board fields, preferring `power.draw.average`
with an explicit `power.draw` fallback, and also retains `power.draw.instant`
when available. Polling at 50 ms does not increase the sensor bandwidth.
Traces must cover both boundaries. Analysis integrates linear interpolation
clipped to the window, rejects non-finite samples and gaps above 0.5 s, and
recomputes power from the archived samples.

The relative difference between first- and second-half means must be at most
5% for power and 2% for device-reported SM clock. Every final window passes.
Power describes sustained operation over at least 30 s after 10 s warm-up.

## Geometry and traffic

For BF16 and exact tile multiples, a joint output tile gives the prediction
`Q_TMA = 2 M N K (1/tile_m + 1/tile_n)` bytes. Widening 256x128 to 256x256
predicts two-thirds of the TMA input supply. A known-volume TMA transfer
calibrates the L2-to-SM counter before profiling; missing calibration fails
acquisition. DRAM traffic is separate because locality may change.
Hot-cache profiles use application replay without L2 flushing and dynamic
clocks. Tile changes also affect resources and scheduling.

## Reproduction

Run `make check-static`, build the pinned image and acquire a passing pilot.
The pilot checks timer/control paths, all formats, graph capture, telemetry
and counter calibration. Its samples are excluded. `make final-study PILOT=...`
requires that pilot's exact source/environment, writes new run directories
and computes the final summaries. The archived dataset in `data/` can be
analyzed without a GPU using the README commands.
