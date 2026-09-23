# GB300 Blackwell Microbenchmarks

Five experiments on one NVIDIA B300 SXM6 AC, an independent Nsight Compute profile of the BF16 GEMM
gap, and the checks that reject incomplete or invalid acquisitions.

## Experiments

| | Experiment | Source | Measurement | Acquired by |
|---|---|---|---|---|
| I | LDGSTS versus TMA | `memory_paths/` | Effective transfer rate: logical useful bytes / CUDA-event kernel time, 3 stage counts × 3 bytes-in-flight sizes | `make campaign` |
| II | Isolated BF16 UMMA | `umma_throughput/umma_{1sm,2sm}.cu` | FLOP/cycle/SM from the per-SM `%clock64` counter, 3 N × 4 pipeline depths | `make campaign` |
| III | Whole-device BF16 UMMA | `umma_throughput/umma_device_scaling.cu` | CUDA-event TFLOP/s of isolated and all-SM launches, with `nvidia-smi` SM-clock telemetry during the timed launches | `make campaign` |
| IV | CuTe DSL versus cuBLASLt | `gemm_comparison/` | CUDA-event TFLOP/s of three BF16 CuTe DSL variants and the first supported cuBLASLt heuristic, 5 shapes | `make campaign` |
| V | BF16 versus FP8 versus NVFP4 | `precision_comparison/` | `cute.testing.benchmark` TFLOP/s of the pinned persistent CuTe DSL kernels and a within-format cuBLASLt baseline, 3 shapes | `make precision-extended` |
| — | GEMM profile | `scripts/profile_gemm.py` | Nsight Compute DRAM and L2→SM bytes, duration and SM clock of one `persistent_2cta` and one cuBLASLt launch, 3 shapes | `make gemm-profile` |

`analysis/analyze.py` reduces Experiments I–IV: the median within each campaign, then the mean,
sample standard deviation and coefficient of variation across three campaigns. Experiment V repeats
each format and shape three times in one run. All statistics are descriptive.

## Acquisition

The pinned CUDA image, CUTLASS commit and Python packages are in `VERSIONS.env`. Acquire from a
committed tree: every run records the commit and the SHA-256 of its sources, and the analyzer and
checker reject runs made with modified tracked files.

```bash
make image
make build
export BLACKWELL_GPU_INDEX=<index>   # nvidia-smi -L
```

`scripts/run_gpu.sh` resolves the index to a GPU UUID, refuses a GPU that already runs compute
processes, and exposes only that GPU to the container.

```bash
# Experiments I–IV: three complete final campaigns on the same GPU
make campaign CAMPAIGN_KIND=final CAMPAIGN_ID=<id1> CAMPAIGN_NCU=1
make campaign CAMPAIGN_KIND=final CAMPAIGN_ID=<id2> CAMPAIGN_NCU=1
make campaign CAMPAIGN_KIND=final CAMPAIGN_ID=<id3> CAMPAIGN_NCU=1
make analyze FINAL_CAMPAIGNS="<id1> <id2> <id3>" ANALYSIS_OUT=<directory>

# Experiment V: nine-row CuTe DSL summary and 18-row CuTe DSL/cuBLASLt comparison
make precision-extended PRECISION_ID=<id>

# GEMM profile against the new analysis, then the diagnostic checks
make gemm-profile PROFILE_CACHE=hot PROFILE_ID=<id> GEMM_SUMMARY=<directory>/gemm_comparison.csv
make check-diagnostics PROFILE_ID=<id> PRECISION_ID=<id>
```

Campaign, precision and profile IDs name new directories under `runs/`; an existing directory is
never reused. `ANALYSIS_OUT` and `GEMM_SUMMARY` are paths inside the repository, because the profile
runs in a container that mounts only the repository; `runs/<analysis-id>` keeps a new analysis out
of `results/`. `make smoke` runs a short pilot campaign of Experiments I–IV. Pilots are never
accepted by `make analyze`.

`results/` is replaced only after the new raw data have passed `make analyze` and
`make check-diagnostics` and have been reviewed.

## Validation

An acquisition is rejected, rather than summarized, when any of these fails:

- **Campaign.** Every benchmark validates its numerical result before its timing is kept, and a
  dataset is written only when all its rows validated: 540 memory, 720 isolated UMMA, 120
  device-scaling and 20 GEMM rows. Whole-device UMMA requires simultaneous residency on every
  planned SM. The clock telemetry must span the whole timed scaling block, agree with the host
  clock to within 1 s, and hold at least 3 samples inside each configuration's timed launches.
  GEMM candidates share operands and must match an untimed IEEE-FP32 reference.
- **Analysis.** Exactly three distinct final campaigns, each with its complete row counts, telemetry
  and eight Nsight Compute captures; one GPU UUID; one source commit with identical source hashes;
  no modified tracked files. `analysis.json` records the campaigns and the SHA-256 of every output.
- **Experiment V.** Each CuTe DSL output, including every timed repetition's, and each cuBLASLt
  output before and after timing must match an IEEE-FP32 product of the exactly represented,
  dequantized operands. cuBLASLt consumes the same bytes as CuTe DSL (data and NVFP4 scales), uses
  FP32 compute and output confirmed by `cublasLtMatmulAlgoCheck`, and keeps one algorithm across
  operand sets. A negative control confirms that the tolerance check rejects a mismatch. The
  protocol is fixed at three repetitions of 5 warm-up and 20 timed launches.
- **GEMM profile.** `GEMM_SUMMARY` must be the `gemm_comparison.csv` of a `make analyze` output
  whose `analysis.json` names the profiled GPU and records that file's SHA-256; all six
  shape/variant rows must exist. Each capture validates before and after the profiled launch, holds
  exactly one kernel inside the NVTX range, and shares operands with the other implementation of
  its shape. The L2→SM metric is used only after a TMA stream reproduces its useful bytes to within
  0.1%.

`make check-diagnostics` repeats these checks from the saved files of a profile and a precision run,
independently of the scripts' own verdicts.

## Timed results and Nsight Compute diagnostics

Throughput comes only from timed launches without a profiler attached: CUDA events for Experiments
I, III, IV and V, and `%clock64` for Experiment II. Clocks are not locked, and the caches are hot
after the warm-up launches.

Nsight Compute always runs in separate processes after the timing. In each campaign it captures
DRAM bytes for six memory configurations, reported as DRAM bytes read per useful byte beside the
Experiment I rates, and the SM clock of both peak-depth UMMA configurations, used only to express
the strongest Experiment II configuration in TFLOP/s/SM. The GEMM profile's durations and byte
counts are diagnostics beside the CUDA-event reference it records, never replacements for it:

- `PROFILE_CACHE=hot` uses application replay with `--cache-control none`: every replay pass reruns
  the operand setup, validation and the campaigns' two warm-up launches before the one NVTX-selected
  launch.
- `PROFILE_CACHE=cold` uses kernel replay with `--cache-control all`, flushing the caches before the
  profiled launch.

Each profile directory holds one cache state, recorded in `index.json` and in every row of
`gemm_profile.csv`; compare DRAM counters only between captures made with the same state.

## Published results

`results/` holds the previous acquisition, made before runs recorded their source commit:

- `memory_paths`, `umma_throughput`, `umma_device_scaling` and `gemm_comparison` (CSV and SVG):
  final campaigns `final-unified-1` to `-3` on `GPU-619f7fdc-5f98-8c37-fe89-0465d6130baf`,
  published in commit `ade92ad`.
- `precision_comparison` (CSV and SVG): the former CuTe DSL-only `make precision` run, last
  updated in commit `ade92ad`.
- `precision_cutedsl_vs_cublaslt` (CSV and SVG) and `gemm_profile.csv`: separate runs on the same
  GPU, published in commit `5b5a95c`. `gemm_profile.csv` is the cold-cache profile; its CUDA-event
  columns come from the `gemm_comparison.csv` above.

These files are kept as historical evidence and are not results of the new campaigns. Their raw
runs predate the provenance fields that `make analyze` and `make check-diagnostics` now require.

BSD 3-Clause; see `LICENSE`.
