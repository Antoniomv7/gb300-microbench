# GB300 Blackwell Microbenchmarks

Five focused experiments on an NVIDIA B300 SXM6 AC: LDGSTS versus TMA, isolated 1-SM versus 2-SM BF16 UMMA, whole-device UMMA scaling, three CuTe DSL GEMM variants versus cuBLASLt, and matched BF16/FP8/NVFP4 GEMMs.

## Results

Experimental acquisition is complete. The five experiments are represented by five CSV summaries and five SVG figures directly in `results/`. The first four experiments summarize three final campaigns using the median within each campaign; the separate precision comparison uses three independently timed repetitions. All statistics are descriptive.

### LDGSTS versus TMA

![LDGSTS and TMA effective transfer rates](results/memory_paths.svg)

LDGSTS led in eight of nine configurations. The highest means were 7.026 TB/s for LDGSTS and 6.965 TB/s for TMA. Effective bandwidth is logical useful bytes divided by kernel time; it is not a direct HBM counter.

### Isolated BF16 UMMA

![Isolated 1-SM and 2-SM UMMA throughput](results/umma_throughput.svg)

The strongest per-SM configuration was 1-SM UMMA at `N=256`, depth `256`: 16.352 TFLOP/s/SM after applying the measured SM clock. The corresponding 2-SM/1-SM ratio was 1.983.

### Whole-device BF16 UMMA

![Isolated and whole-device UMMA scaling](results/umma_device_scaling.svg)

| Method | Scale | Active SMs | Kernel time (ms) | Total TFLOP/s | Mean SM clock (MHz) | Mean power (W) |
|---|---|---:|---:|---:|---:|---:|
| `umma_1sm` | isolated | 1 | 17.285 | 15.530 | 2032.0 | 235.0 |
| `umma_2sm` | isolated | 2 | 17.434 | 30.794 | 2032.0 | 224.4 |
| `umma_1sm` | device | 148 | 19.063 | 2084.110 | 1894.5 | 594.5 |
| `umma_2sm` | device | 148 | 18.982 | 2092.957 | 1911.4 | 1078.7 |

| Scaling path | Clock ratio device/isolated | Raw efficiency | Frequency-normalized efficiency |
|---|---:|---:|---:|
| 1-SM → 148 SMs | 0.932 | 0.907 | 0.973 |
| 2-SM → 74 clusters (148 SMs) | 0.941 | 0.918 | 0.976 |

The 2-SM launch uses 74 simultaneously resident two-CTA clusters. Raw efficiency is the measured
whole-device throughput divided by the number of work units times the separately timed isolated
work unit; the frequency-normalized column divides that ratio by the measured SM-clock ratio, so it
separates spatial scaling from the DVFS state. Both are empirical ratios against an independent
baseline, not bounded efficiencies. All four configurations are timed with CUDA events, each as one
contiguous campaign, while `nvidia-smi` samples the SM clock every 50 ms; each mean clock comes
from the samples that fall inside that configuration's own timed launches. Clocks were not locked.
The mean power column follows a slower telemetry filter than a 0.57 s campaign, so it is indicative
only: the two whole-device campaigns run the same instruction stream at the same rate, and the
lower figure belongs to the first of them. The isolated instruction-throughput measurements above
use `%clock64` and are not part of this comparison.

### CuTe DSL versus cuBLASLt

![CuTe DSL and cuBLASLt GEMM comparison](results/gemm_comparison.svg)

| Shape `(M,N,K,L)` | Best CuTe DSL TFLOP/s | cuBLASLt TFLOP/s | Ratio |
|---|---:|---:|---:|
| `4096x4096x4096x1` | 1658.0 | 1749.0 | 94.80% |
| `8192x8192x8192x1` | 1441.8 | 2106.8 | 68.44% |
| `16384x512x4096x1` | 813.5 | 1432.0 | 56.80% |
| `32768x512x4096x1` | 758.7 | 1507.9 | 50.31% |
| `512x16384x4096x1` | 1272.6 | 1498.4 | 84.93% |

`persistent_2cta` was the strongest CuTe DSL variant for every shape. These are hot-cache measurements without kernel-level GEMM profiling.

### BF16 versus FP8 versus NVFP4

![Matched BF16, FP8 and NVFP4 GEMM throughput](results/precision_comparison.svg)

`make precision` compares the pinned official persistent CuTe DSL kernels on three shapes: `4096x4096x4096`, `8192x8192x8192`, and `32768x512x4096`. All formats use FP32 accumulation and output, a `256x128` MMA tile, a `2x1` CTA cluster, and TMA stores. NVFP4 uses `Float4E2M1FN` inputs with one `Float8E4M3FN` scale per 16 values. The [pinned NVFP4 kernel](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/sm103_dense_blockscaled_gemm_persistent.py#L190-L198) fixes its accumulator to `Float32` and infers two-CTA instructions from `mma_tiler_mn[0] == 256`, matching the explicit BF16 and FP8 settings.

| Shape `(M,N,K,L)` | BF16 TFLOP/s | FP8 TFLOP/s | NVFP4 TFLOP/s | FP8/BF16 | NVFP4/BF16 | NVFP4 vendor peak |
|---|---:|---:|---:|---:|---:|---:|
| `4096x4096x4096x1` | 1660.1 | 2926.2 | 4040.7 | 1.76× | 2.43× | 29.93% |
| `8192x8192x8192x1` | 1444.8 | 3106.6 | 5585.1 | 2.15× | 3.87× | 41.37% |
| `32768x512x4096x1` | 757.9 | 1727.6 | 3033.3 | 2.28× | 4.00× | 22.47% |

NVFP4 reaches 5.585 PFLOP/s and improves throughput by 2.43–4.00× against the matched BF16 kernel; FP8 improves it by 1.76–2.28×. Against the independently measured cuBLASLt BF16 results above, the NVFP4 ratios are 2.31×, 2.65×, and 2.01×, respectively. The nine measured configurations pass numerical validation, and their coefficients of variation range from 0.061% to 0.452%.

Each configuration uses five warm-up iterations and 20 timed iterations per repetition. Each format verifies its own correctly represented operands before timing; the format-specific operands are not a cross-format model-accuracy comparison.

The dense per-GPU vendor references—2250 TFLOP/s for BF16, 4500 TFLOP/s for FP8, and 13500 TFLOP/s for NVFP4—follow NVIDIA's [official HGX B300 specifications](https://www.nvidia.com/en-us/data-center/hgx/). For the eight-GPU system, the published sparse BF16 and FP8 values convert to dense single-GPU peaks as `36 / 2 / 8 = 2.25 PFLOP/s` and `72 / 2 / 8 = 4.50 PFLOP/s`; the explicitly published dense NVFP4 value gives `108 / 8 = 13.50 PFLOP/s`. These are comparison references, not measured hardware ceilings. The complete measurements and figure are published in `results/precision_comparison.csv` and `results/precision_comparison.svg`.

## Build and run

The pinned CUDA image, CUTLASS commit and Python package versions are in `VERSIONS.env`.

```bash
make image
make build
export BLACKWELL_GPU_INDEX=7
make smoke
```

Run one short pilot and three complete final campaigns. Final campaigns include the essential Nsight Compute counters:

```bash
make campaign CAMPAIGN_KIND=pilot CAMPAIGN_ID=pilot
make campaign CAMPAIGN_KIND=final CAMPAIGN_ID=final-1
make campaign CAMPAIGN_KIND=final CAMPAIGN_ID=final-2
make campaign CAMPAIGN_KIND=final CAMPAIGN_ID=final-3
```

Generate or replace four CSV summaries and four SVG figures directly in `results/`:

```bash
make analyze FINAL_CAMPAIGNS="final-1 final-2 final-3"
```

Repeat one experiment on its own — here the whole-device UMMA scaling comparison, which needs no
Nsight Compute pass — and regenerate only its CSV and SVG:

```bash
for i in 1 2 3; do
  make campaign CAMPAIGN_KIND=final CAMPAIGN_ID="umma-scaling-$i" \
    CAMPAIGN_EXPERIMENTS=umma_device_scaling CAMPAIGN_NCU=0
done
make analyze ANALYSIS_ONLY=umma_device_scaling \
  FINAL_CAMPAIGNS="umma-scaling-1 umma-scaling-2 umma-scaling-3"
```

Run the independent low-precision extension without repeating the four completed campaigns:

```bash
make precision
```

`make sass` optionally generates the five CUDA disassemblies in `build/sass/`.

## Measurement boundaries

- Numerical correctness is mandatory before timing.
- Each final campaign contains 540 memory samples, 720 isolated UMMA samples, 120 device-scaling samples and 20 GEMM rows.
- Whole-device UMMA requires simultaneous residency and observed coverage of every planned SM.
- The UMMA baseline uses BF16 inputs and FP32 accumulation; GEMM candidates share operands and an untimed IEEE-FP32 reference.
- Nsight Compute provides the DRAM cross-check and the SM frequency behind the isolated TFLOP/s estimate.
- UMMA scaling records the SM clock with `nvidia-smi` during the same CUDA-event-timed campaigns.
- Precision formats share shape, layouts, accumulation/output types, tile, cluster, and store path; scaled NVFP4 operands are format-specific.
- Three campaigns support descriptive statistics, not significance testing or architectural peak claims.
- Independent TMEM/DSMEM latency, dual-die topology, and per-launch DVFS measurements are outside the scope of the closed experimental phase.

BSD 3-Clause; see `LICENSE`.
