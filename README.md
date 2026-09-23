# NVIDIA B300 GEMM Microbenchmarks

Five experiments on one NVIDIA B300 SXM6 AC: memory transfers, Tensor Core throughput, device
scaling, GEMM implementations and numerical formats. CUDA/PTX microbenchmarks characterize the
hardware mechanisms; pinned CUTLASS CuTe DSL examples and cuBLASLt provide the GEMM comparisons.

## Experiments

| # | Purpose | Command | Main CSV in the run directory |
|---|---|---|---|
| I | LDGSTS versus TMA effective transfer rate | `make exp1-memory` | `raw/memory_paths.csv` |
| II | Isolated 1-SM versus 2-SM BF16 UMMA throughput | `make exp2-umma` | `raw/umma_throughput.csv` |
| III | Whole-device BF16 UMMA scaling | `make exp3-scaling` | `raw/umma_device_scaling.csv` |
| IV | BF16 CuTe DSL variants versus cuBLASLt | `make exp4-gemm` | `raw/gemm_comparison.csv` |
| V | BF16, FP8 and NVFP4 with CuTe DSL and cuBLASLt | `make precision` | `precision_comparison.csv`, `precision_cutedsl_vs_cublaslt.csv` |

Experiments I and II also collect six DRAM and two SM-clock Nsight Compute captures, respectively.
Experiment III samples the SM clock, power and temperature every 50 ms during timing. Experiment V
records operands, validation, repetitions and selected cuBLASLt algorithms in `raw/`. A separate
GEMM profile collects DRAM and L2 traffic for two BF16 implementations on three shapes.

## Run the complete study

The host needs an NVIDIA driver, Docker with GPU support, Git, Make and Python 3. The CUDA image
digest, CUTLASS commit and Python package versions are pinned in `VERSIONS.env`.

Build the container once, or after changing the pinned environment:

```bash
make image
```

Select an idle GPU and run the study:

```bash
export BLACKWELL_GPU_INDEX=6
make final-study
```

`scripts/run_gpu.sh` resolves the index to a GPU UUID, checks for existing compute processes and
exposes that GPU as device 0 inside the container. The study requires committed sources and creates
a new `runs/study-<UTC>/` directory. It compiles the binaries once, then:

1. runs Experiments I–IV three times into `campaign-1/`, `campaign-2/` and `campaign-3/`;
2. checks and aggregates those campaigns into `analysis/`;
3. runs Experiment V, including its three repetitions per format and shape, into `precision/`;
4. profiles the six BF16 GEMM cases with hot caches into `gemm-profile-hot/`;
5. checks the complete study and creates `runs/study-<UTC>.tar.gz`.

The archive contains the raw data, summaries, SVG figures, metadata, Nsight Compute reports and
per-step logs. The final message prints its path, SHA-256, GPU UUID and source commit. Execution
stops at the first failure and preserves the run directory for diagnosis. Existing directories
are never overwritten.

## Run experiments individually

After selecting the GPU, execute any command from the experiment table. For example:

```bash
make exp3-scaling
make precision
```

Each command compiles as needed, uses the study's measurement parameters, creates its own
timestamped directory under `runs/` and checks the saved data. Individual runs are useful for
inspection; aggregation requires three complete I–IV campaigns.

For a standalone campaign, analysis or GEMM profile:

```bash
make campaign CAMPAIGN_ID=campaign-a
# Repeat with campaign-b and campaign-c before analysis.
make analyze FINAL_CAMPAIGNS="campaign-a campaign-b campaign-c" ANALYSIS_OUT=runs/analysis-abc
make gemm-profile PROFILE_CACHE=hot GEMM_SUMMARY=runs/analysis-abc/gemm_comparison.csv
```

`make help` lists the targets. `RUNS` sets the output parent; `CAMPAIGN_ID`, `PRECISION_ID` and
`PROFILE_ID` optionally name new run directories. An extracted study can be checked with:

```bash
python3 scripts/check_diagnostics.py --study /path/to/study-YYYYMMDDTHHMMSSZ
```

## Measurement and validation

- **Timing.** CUDA events measure Experiments I, III, IV and V; `%clock64` measures Experiment II.
  Validation and warm-up precede timing. Clocks are unlocked and repeated launches reuse their
  operands. Profiling runs after timing and its durations are reported as diagnostics.
- **Experiments I–IV.** Every configuration validates its output before timing is retained. Complete
  campaigns contain 540 memory, 720 isolated UMMA, 120 scaling and 20 GEMM rows. Device scaling
  verifies simultaneous residency on every planned SM; telemetry spans the timed block and has at
  least three samples per configuration. GEMM candidates share operands and an IEEE-FP32 reference.
- **Experiment V.** Both implementations use the same operand bytes, including NVFP4 block scales,
  with FP32 accumulation and output. Every timed repetition's output is checked against an
  IEEE-FP32 product of the dequantized operands. cuBLASLt keeps one algorithm across operand sets
  and passes `cublasLtMatmulAlgoCheck`. Each shape and format has three repetitions of five warm-up
  and twenty timed launches. A negative control verifies the numerical tolerance check.
- **Kernel identification.** After Experiment V's timing, one CPU/CUDA PyTorch profiler capture
  attempts to name each cuBLASLt kernel. Empty traces are recorded as `UNAVAILABLE`, with an
  unknown launch count; algorithm identity and numerical validation remain required.
- **Aggregation.** Experiments I–IV use the median within each campaign, followed by the mean,
  sample standard deviation and coefficient of variation across three distinct campaigns. The
  campaigns must share one GPU, source commit and source-file hashes. `analysis.json` identifies
  the inputs and hashes every output. Statistics are descriptive.

## GEMM traffic profiling

`make final-study` uses **hot-cache application replay**: every replay pass repeats operand setup,
validation and two warm-up launches, then profiles one NVTX-selected launch with
`--cache-control none`. `PROFILE_CACHE=cold` is also available for an isolated diagnostic using
kernel replay and `--cache-control all`.

Each capture validates before and after the selected launch and keeps its `.ncu-rep`, CSV export
and worker metadata. The two implementations share operands. The L2-to-SM TMA byte counter is
included only if a known TMA stream reproduces its useful bytes within 0.1%.

The CUDA-event reference comes from the supplied analysis directory and must match its manifest
and GPU. The cache state is recorded in every summary row. `compulsory_read_bytes` denotes the
combined A/B operand size: hot-cache DRAM reads can be smaller, including zero. The L2/DRAM ratio
is left empty in the CSV (`null` in JSON) when no DRAM reads were recorded.

## Repository layout and results

| Path | Contents |
|---|---|
| `memory_paths/`, `umma_throughput/` | CUDA/PTX microbenchmarks and shared launch/validation code |
| `gemm_comparison/`, `precision_comparison/` | GEMM drivers and their cuBLASLt bridges |
| `scripts/` | Acquisition, profiling, telemetry, provenance and saved-data checks |
| `analysis/` | Campaign aggregation and figures |
| `results/` | Published CSV summaries and SVG figures |
| `build/`, `runs/` | Generated binaries and evidence; excluded from Git |

`results/` currently contains the earlier acquisition: three `final-unified` campaigns, a
CuTe DSL precision run, and separate cuBLASLt precision and cold-cache GEMM diagnostics. These
predate the current source-provenance checks. The earlier
[GEMM and precision diagnostic archive](https://github.com/Antoniomv7/gb300-microbench/blob/ef5300c77509a0079f93dc1560eebfdf741cc3ee/evidence-gemm-precision.tar.gz)
is preserved in the Git history. New studies write to `runs/`; update `results/` after auditing
the new acquisition and retain its complete evidence archive alongside the thesis.

BSD 3-Clause; see `LICENSE`.
