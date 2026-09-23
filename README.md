# GB300 Blackwell Microbenchmarks

Five focused experiments on an NVIDIA B300 SXM6 AC: LDGSTS versus TMA, isolated 1-SM versus 2-SM
BF16 UMMA, whole-device UMMA scaling, three CuTe DSL GEMM variants versus cuBLASLt, and matched
BF16/FP8/NVFP4 GEMMs.

## Results

The four main experiments — `memory_paths`, `umma_throughput`, `umma_device_scaling` and
`gemm_comparison` — come from three complete final campaigns, `final-unified-1`, `final-unified-2`
and `final-unified-3`. Each campaign runs all four experiments back to back in one container on the
GPU selected with `BLACKWELL_GPU_INDEX=6` (`GPU-619f7fdc-…`, driver 610.43.02), records SM-clock
telemetry concurrently with the UMMA scaling measurements, and closes with eight Nsight Compute
captures. Every campaign produced 540 memory samples, 720 isolated UMMA samples, 120 device-scaling
samples and 20 GEMM rows. `make analyze` reduces each campaign by the median within it, then reports
the mean, standard deviation and coefficient of variation across the three campaigns.

The fifth experiment, `precision_comparison`, is not part of those campaigns. It runs separately
through `make precision`, which times its own three repetitions of every format and shape in a
single invocation and writes its CSV and figure directly.

All five CSV summaries and five SVG figures live in `results/`. All statistics are descriptive.

### LDGSTS versus TMA

![LDGSTS and TMA effective transfer rates](results/memory_paths.svg)

| Stages | Bytes in flight per SM | LDGSTS GB/s | TMA GB/s | TMA / LDGSTS |
|---:|---:|---:|---:|---:|
| 2 | 16 KiB | 3105.3 | 3017.7 | 0.972 |
| 2 | 32 KiB | 5146.1 | 5029.2 | 0.977 |
| 2 | 64 KiB | 6949.5 | 6958.5 | 1.001 |
| 4 | 16 KiB | 3217.5 | 2396.4 | 0.745 |
| 4 | 32 KiB | 5111.6 | 4666.8 | 0.913 |
| 4 | 64 KiB | 7023.3 | 6962.6 | 0.991 |
| 8 | 16 KiB | 2002.2 | 1202.6 | 0.601 |
| 8 | 32 KiB | 3674.0 | 2403.6 | 0.654 |
| 8 | 64 KiB | 6698.1 | 4802.2 | 0.717 |

LDGSTS led in eight of the nine configurations; TMA led only at two stages and 64 KiB in flight,
by 0.13%. Both peaked at four stages and 64 KiB: 7023.3 GB/s (7.023 TB/s) for LDGSTS and
6962.6 GB/s (6.963 TB/s) for TMA. The gap widens as stages increase and bytes in flight shrink,
reaching 0.601× at eight stages and 16 KiB. Coefficients of variation stayed at or below 0.057%.

Effective bandwidth is logical useful bytes divided by kernel time; it is not a direct HBM counter.
For the six configurations profiled with Nsight Compute, DRAM bytes read per useful byte ranged
from 1.000008 to 1.000036, so the transferred volume matches the logical working set.

### Isolated BF16 UMMA

![Isolated 1-SM and 2-SM UMMA throughput](results/umma_throughput.svg)

| N | Depth | 1-SM FLOP/cycle/SM | 2-SM FLOP/cycle/SM | 2-SM / 1-SM total |
|---:|---:|---:|---:|---:|
| 64 | 4 | 2350.7 | 1108.3 | 0.943 |
| 64 | 16 | 3714.8 | 2845.3 | 1.532 |
| 64 | 64 | 4933.0 | 4665.4 | 1.892 |
| 64 | 256 | 5318.9 | 5571.9 | 2.095 |
| 128 | 4 | 3477.4 | 2094.8 | 1.205 |
| 128 | 16 | 6641.4 | 4728.3 | 1.424 |
| 128 | 64 | 7695.9 | 6919.7 | 1.798 |
| 128 | 256 | 8062.1 | 7832.0 | 1.943 |
| 256 | 4 | 5599.3 | 3363.2 | 1.201 |
| 256 | 16 | 7007.8 | 5995.8 | 1.711 |
| 256 | 64 | 7838.9 | 7496.4 | 1.913 |
| 256 | 256 | 8100.8 | 8028.3 | 1.982 |

The strongest per-SM configuration was 1-SM UMMA at `N=256`, depth `256`: 8100.8 FLOP/cycle/SM,
or 16.373 TFLOP/s/SM after applying the SM clock measured by Nsight Compute for that
configuration. The strongest aggregate configuration was 2-SM UMMA at the same point,
16056.6 FLOP/cycle across its two SMs.

A two-CTA UMMA only pays off once the pipeline is deep. The 2-SM/1-SM aggregate ratio rises from
0.943 at `N=64`, depth `4` — where the two-CTA launch is slower than a single SM — to 2.095 at
`N=64`, depth `256`, and reaches 1.982 at the strongest point. These rows are timed with the
per-SM `%clock64` counter and are reported in FLOP/cycle; the three campaign medians were identical
for every configuration, giving a coefficient of variation of 0.000%.

### Whole-device BF16 UMMA

![Isolated and whole-device UMMA scaling](results/umma_device_scaling.svg)

| Configuration | Active SMs | Work units | Kernel time (ms) | Total TFLOP/s | TFLOP/s per SM | Mean SM clock (MHz) | Min–max (MHz) | Mean power (W) | Mean temp (°C) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-SM isolated | 1 | 1 | 17.321 | 15.498 | 15.498 | 2032.0 | 2032–2032 | 237.4 | 43.9 |
| 2-SM isolated | 2 | 1 | 17.481 | 30.712 | 15.356 | 2032.0 | 2032–2032 | 217.4 | 43.7 |
| 1-SM whole device | 148 | 148 | 18.882 | 2104.024 | 14.216 | 1922.2 | 1882–1942 | 582.3 | 60.5 |
| 2-SM whole device | 148 | 74 clusters | 18.713 | 2123.020 | 14.345 | 1932.7 | 1897–1935 | 1085.7 | 63.0 |

| Scaling path | Clock ratio device / isolated | Raw ratio | Frequency-normalized ratio |
|---|---:|---:|---:|
| 1-SM → 148 SMs | 0.946 | 0.917 | 0.970 |
| 2-SM → 74 clusters (148 SMs) | 0.951 | 0.934 | 0.982 |

The raw ratio is the measured whole-device throughput divided by the number of work units times the
separately timed isolated work unit. The frequency-normalized ratio divides that by the measured
SM-clock ratio, separating spatial scaling from the clock the device actually ran at. Both are
empirical ratios against an independent baseline, not bounded efficiencies, and the measured
TFLOP/s columns are never frequency-corrected.

Of the 8.3% shortfall against ideal linear scaling on the 1-SM path, the clock accounts for a
factor of 0.946 and a factor of 0.970 remains; on the 2-SM path the 6.6% shortfall splits into
0.951 and 0.982. Expressed per SM and per cycle, throughput falls from 7627 to 7396 FLOP/cycle/SM
on the 1-SM path and from 7557 to 7422 on the 2-SM path. The two whole-device launches land within
0.9% of each other, at 93.5% and 94.4% of the 2250 TFLOP/s dense BF16 vendor reference described
below.

All four configurations are timed with CUDA events. Each runs its warm-up and then its 30 timed
repetitions as one contiguous block, the timed part spanning 0.52 s for the isolated configurations
and 0.56 s for the whole-device ones, while `nvidia-smi` samples the SM clock, power and
temperature every 50 ms throughout. A sample is attributed to a configuration when its
timestamp falls inside one of that configuration's own timed launches, which yielded 10 to 11
samples per configuration per campaign and 31 to 33 across the three; the median offset between the
sampler's timestamps and the benchmark's own clock was 0.48 to 0.56 ms. Clocks were not locked. The
isolated configurations held 2032 MHz with no observed variation, while the whole-device
configurations ran lower and warmer; the min–max columns span all three campaigns, and the 1-SM
whole-device block varied more than the 2-SM block.

Power is indicative only. The two whole-device configurations execute the same instruction stream
at nearly the same rate, yet report 582.3 W and 1085.7 W: the reading rises monotonically across
the first whole-device block in all three campaigns and is already settled during the second, so
the telemetry filter is slow relative to a 0.56 s block. The isolated instruction-throughput
measurements in the previous section use `%clock64` and are not part of this comparison.

### CuTe DSL versus cuBLASLt

![CuTe DSL and cuBLASLt GEMM comparison](results/gemm_comparison.svg)

| Shape `(M,N,K,L)` | Best CuTe DSL variant | Best CuTe DSL TFLOP/s | cuBLASLt TFLOP/s | Ratio |
|---|---|---:|---:|---:|
| `4096x4096x4096x1` | `persistent_2cta` | 1680.3 | 1756.0 | 95.69% |
| `8192x8192x8192x1` | `persistent_2cta` | 1448.9 | 2107.8 | 68.74% |
| `16384x512x4096x1` | `persistent_2cta` | 814.2 | 1438.1 | 56.62% |
| `32768x512x4096x1` | `persistent_2cta` | 756.6 | 1507.7 | 50.18% |
| `512x16384x4096x1` | `persistent_2cta` | 1268.7 | 1501.3 | 84.51% |

`persistent_2cta` was the strongest CuTe DSL variant for every shape, and the ordering
`nonpersistent_1cta` < `persistent_1cta` < `persistent_2cta` held in all five. The gap against
cuBLASLt is smallest on the square 4096 shape and largest on the tall-and-thin shapes, where
cuBLASLt was 1.8–2.0× faster. cuBLASLt peaked at 2107.8 TFLOP/s on `8192x8192x8192`, 93.7% of
the 2250 TFLOP/s dense BF16 vendor reference. Coefficients of variation stayed at or below 1.03%.
These are hot-cache measurements without kernel-level GEMM profiling; all candidates share operands
and pass the same untimed IEEE-FP32 reference.

### BF16 versus FP8 versus NVFP4

![Matched BF16, FP8 and NVFP4 GEMM throughput](results/precision_comparison.svg)

`make precision` compares the pinned official persistent CuTe DSL kernels on three shapes:
`4096x4096x4096`, `8192x8192x8192`, and `32768x512x4096`. All formats use FP32 accumulation and
output, a `256x128` MMA tile, a `2x1` CTA cluster, and TMA stores. NVFP4 uses `Float4E2M1FN` inputs
with one `Float8E4M3FN` scale per 16 values. The [pinned NVFP4
kernel](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/sm103_dense_blockscaled_gemm_persistent.py#L190-L198)
fixes its accumulator to `Float32` and infers two-CTA instructions from `mma_tiler_mn[0] == 256`,
matching the explicit BF16 and FP8 settings.

| Shape `(M,N,K,L)` | BF16 TFLOP/s | FP8 TFLOP/s | NVFP4 TFLOP/s | FP8/BF16 | NVFP4/BF16 |
|---|---:|---:|---:|---:|---:|
| `4096x4096x4096x1` | 1685.1 | 2943.4 | 4044.9 | 1.75× | 2.40× |
| `8192x8192x8192x1` | 1457.3 | 3127.3 | 5562.0 | 2.15× | 3.82× |
| `32768x512x4096x1` | 758.3 | 1724.0 | 3003.7 | 2.27× | 3.96× |

| Shape `(M,N,K,L)` | BF16 vs. 2250 | FP8 vs. 4500 | NVFP4 vs. 13500 |
|---|---:|---:|---:|
| `4096x4096x4096x1` | 74.89% | 65.41% | 29.96% |
| `8192x8192x8192x1` | 64.77% | 69.49% | 41.20% |
| `32768x512x4096x1` | 33.70% | 38.31% | 22.25% |

NVFP4 reaches 5562.0 TFLOP/s (5.562 PFLOP/s) and improves throughput by 2.40–3.96× against the
matched BF16 kernel; FP8 improves it by 1.75–2.27×. Against the cuBLASLt BF16 results measured in
the campaigns above, the NVFP4 ratios are 2.30×, 2.64× and 1.99×. The BF16 rows agree with the
campaigns' `persistent_2cta` results to within 0.6% on the three shapes the two experiments share,
which is consistent across two independent acquisitions. The nine measured configurations pass
numerical validation, and their coefficients of variation range from 0.027% to 0.219%.

Each configuration uses five warm-up iterations and 20 timed iterations per repetition, and three
repetitions per format and shape. Each format verifies its own correctly represented operands
before timing; the format-specific operands are not a cross-format model-accuracy comparison.

The dense per-GPU vendor references — 2250 TFLOP/s for BF16, 4500 TFLOP/s for FP8, and
13500 TFLOP/s for NVFP4 — follow NVIDIA's [official HGX B300
specifications](https://www.nvidia.com/en-us/data-center/hgx/). For the eight-GPU system, the
published sparse BF16 and FP8 values convert to dense single-GPU peaks as `36 / 2 / 8 = 2.25
PFLOP/s` and `72 / 2 / 8 = 4.50 PFLOP/s`; the explicitly published dense NVFP4 value gives
`108 / 8 = 13.50 PFLOP/s`. These are comparison references, not measured hardware ceilings.

## Build and run

The pinned CUDA image, CUTLASS commit and Python package versions are in `VERSIONS.env`.

```bash
make image
make build
export BLACKWELL_GPU_INDEX=6
```

Index 6 identifies the B300 used for this acquisition; select the index of an available B300 on
your system with `nvidia-smi -L`. `scripts/run_gpu.sh` resolves the index to a GPU UUID, refuses a
GPU that already has compute processes, and exposes only that GPU to the container.

Run the three complete final campaigns. Each one runs all four main experiments, records the UMMA
scaling clock telemetry, and finishes with the Nsight Compute captures:

```bash
make campaign CAMPAIGN_KIND=final CAMPAIGN_ID=final-unified-1
make campaign CAMPAIGN_KIND=final CAMPAIGN_ID=final-unified-2
make campaign CAMPAIGN_KIND=final CAMPAIGN_ID=final-unified-3
```

Produce the four CSV summaries and four SVG figures in `results/`:

```bash
make analyze FINAL_CAMPAIGNS="final-unified-1 final-unified-2 final-unified-3"
```

Run the precision comparison, which is independent of the campaigns and writes the fifth CSV and
figure itself:

```bash
make precision
```

`make smoke` is an optional short pilot of the four main experiments; it is not one of the three
final campaigns and its output is not used by `make analyze`. `make sass` optionally generates the
five CUDA disassemblies in `build/sass/`.

## Measurement boundaries

- Numerical correctness is mandatory before timing.
- Each final campaign contains 540 memory samples, 720 isolated UMMA samples, 120 device-scaling
  samples and 20 GEMM rows.
- Whole-device UMMA requires simultaneous residency and observed coverage of every planned SM.
- UMMA scaling is timed with CUDA events and records the SM clock with `nvidia-smi` during the same
  timed campaigns; power and temperature are recorded alongside and are indicative only.
- Nsight Compute contributes eight captures per campaign: the DRAM cross-check on six memory
  configurations and the SM frequency behind the isolated UMMA TFLOP/s estimate.
- The UMMA baseline uses BF16 inputs and FP32 accumulation; GEMM candidates share operands and an
  untimed IEEE-FP32 reference.
- Precision formats share shape, layouts, accumulation/output types, tile, cluster, and store path;
  scaled NVFP4 operands are format-specific.
- Three campaigns, and three repetitions for the precision comparison, support descriptive
  statistics, not significance testing or architectural peak claims.
- Independent TMEM/DSMEM latency, dual-die topology, and per-launch DVFS measurements are outside
  the scope of the closed experimental phase.

BSD 3-Clause; see `LICENSE`.
