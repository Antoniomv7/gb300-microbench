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

`results/` contains the summaries and six SVG figures from the complete study
`study-20260923T173150Z` on GPU
`GPU-619f7fdc-5f98-8c37-fe89-0465d6130baf`, acquired from commit
`6868f001b103c4cfd4a6d2019de7f310a2a18a04`. The three campaigns supply
the four Experiment I–IV summaries; Experiment V supplies both the CuTe DSL
precision summary and the 18-row within-format CuTe DSL/cuBLASLt comparison.
`gemm_profile.csv` contains six hot-cache diagnostic captures, including DRAM
and calibrated L2-to-TMA read bytes. The complete archive, including raw runs,
profiler reports, validation records and the analysis manifest, accompanies
the thesis as `supplementary/study-20260923T173150Z.tar.gz` (SHA-256
`71533e13b16c2e85a784c6f6abea3ef95cf4193af17794dd59155ad1777e4101`).
Extract the archive before using `scripts/check_diagnostics.py --study` on
its study directory. Later studies write to `runs/` without overwriting
these published results.

## Published results and findings

The figures below use the published CSVs from the study above. Throughput values are means of the
three final campaigns for Experiments I–IV and of three timed repetitions for Experiment V. The
Nsight Compute traffic captures are separate diagnostics, not timing measurements. Results describe
this GPU, workload and configuration grid; they are not architectural peak specifications.

### I. HBM-to-shared-memory paths

![Effective transfer rate for LDGSTS and TMA](results/memory_paths.svg)

LDGSTS has the higher effective rate in **8 of 9** matched stage/in-flight-byte configurations.
The highest measured means are **7,024 GB/s** for LDGSTS and **6,964 GB/s** for TMA (both at four
stages and 64 KiB in flight). At two stages and 64 KiB, TMA is marginally higher: 6,954 versus
6,943 GB/s. Increasing the bytes in flight helps both paths in this grid, while eight stages can
reduce throughput substantially. The plotted rate is useful bytes divided by kernel time; it is
not a direct DRAM-bandwidth counter or a prediction of GEMM speed.

Source: [`results/memory_paths.csv`](results/memory_paths.csv).

### II–III. UMMA instruction throughput and device scaling

![Isolated BF16 UMMA throughput](results/umma_throughput.svg)

At `N=256`, depth 256, the isolated 1-SM kernel reaches **8,101 FLOP/cycle/SM** (modeled
**16.372 TFLOP/s/SM** using the measured clock); the 2-SM kernel reaches **8,028
FLOP/cycle/SM**, or **1.982×** the total throughput of the 1-SM kernel. The per-cycle figures
come from validated operation counts and `%clock64` cycles.

![BF16 UMMA scaling to 148 SMs](results/umma_device_scaling.svg)

| Execution | Active SMs | Mean throughput | Scaling efficiency | Clock-normalized efficiency |
|---|---:|---:|---:|---:|
| 1-SM work units | 148 | 2,103.7 TFLOP/s | 91.7% | 96.9% |
| 2-SM work units | 148 | 2,119.7 TFLOP/s | 93.3% | 98.1% |

The 2-SM work-unit configuration delivers about **0.8%** more device throughput in this test.
The gap between raw and clock-normalized efficiency shows why a fixed-clock extrapolation from
an isolated SM overstates the scaling loss. The sampled clocks and power in the CSV describe the
timed runs; power samples are indicative telemetry, not a calibrated energy-efficiency comparison.

Sources: [`results/umma_throughput.csv`](results/umma_throughput.csv) and
[`results/umma_device_scaling.csv`](results/umma_device_scaling.csv).

### IV. BF16 GEMM implementation and shape

![BF16 CuTe DSL variants versus cuBLASLt](results/gemm_comparison.svg)

| GEMM shape (M × N × K) | Persistent 2-CTA CuTe DSL | cuBLASLt | CuTe DSL / cuBLASLt |
|---|---:|---:|---:|
| 4096 × 4096 × 4096 | 1,679.7 TFLOP/s | 1,754.6 TFLOP/s | 95.7% |
| 8192 × 8192 × 8192 | 1,450.5 TFLOP/s | 2,111.2 TFLOP/s | 68.7% |
| 16384 × 512 × 4096 | 812.7 TFLOP/s | 1,435.6 TFLOP/s | 56.6% |
| 32768 × 512 × 4096 | 756.9 TFLOP/s | 1,509.8 TFLOP/s | 50.1% |
| 512 × 16384 × 4096 | 1,269.8 TFLOP/s | 1,498.9 TFLOP/s | 84.7% |

Persistent 2-CTA is the fastest of the three tested CuTe DSL BF16 variants on all five shapes,
but its proximity to cuBLASLt varies strongly with matrix shape. This comparison tests specific
implementations and the selected cuBLASLt algorithms, not a general ceiling for CuTe DSL.

Source: [`results/gemm_comparison.csv`](results/gemm_comparison.csv).

### V. BF16, FP8 and NVFP4 GEMM

![CuTe DSL throughput by input precision](results/precision_comparison.svg)

| GEMM shape (M × N × K) | BF16 CuTe DSL | FP8 CuTe DSL | NVFP4 CuTe DSL |
|---|---:|---:|---:|
| 4096 × 4096 × 4096 | 1,685.0 | 2,938.0 (1.74×) | 4,032.9 (2.39×) |
| 8192 × 8192 × 8192 | 1,459.0 | 3,122.9 (2.14×) | 5,568.8 (3.82×) |
| 32768 × 512 × 4096 | 757.6 | 1,721.1 (2.27×) | 2,992.8 (3.95×) |

Throughputs are in TFLOP/s; parentheses show speedup over BF16 for the same CuTe DSL shape.
Lower precision increases measured throughput, but the gain depends on shape and is below the
ratio of the nominal dense arithmetic peaks for several cases. NVFP4 uses block scales; its
operand representation and numerical error differ from BF16 and FP8. All reported outputs pass
validation against the FP32 reference of the corresponding dequantized operands.

![Matched CuTe DSL and cuBLASLt comparisons by precision](results/precision_cutedsl_vs_cublaslt.svg)

In the matched-operand comparison, CuTe DSL reaches **50.0–95.3%** of cuBLASLt throughput in
BF16, **64.4–92.2%** in FP8 and **73.2–84.2%** in NVFP4 across these three shapes. The two
implementations consume the same operand bytes for each shape and format, including NVFP4 scales;
every timed repetition is checked for numerical correctness.

Sources: [`results/precision_comparison.csv`](results/precision_comparison.csv) and
[`results/precision_cutedsl_vs_cublaslt.csv`](results/precision_cutedsl_vs_cublaslt.csv).

### Hot-cache BF16 GEMM traffic diagnostic

In the six separately profiled launches, the persistent 2-CTA CuTe DSL variant records more DRAM
read bytes than cuBLASLt at 8192 × 8192 × 8192 (**3.35 versus 1.16 GB**) and at
32768 × 512 × 4096 (**1.08 versus 0.28 GB**). The calibrated L2-to-SM TMA counter also records
more bytes for CuTe DSL at those shapes. These observations are consistent with the larger
measured performance gaps, but do not establish which operand was reread or prove that traffic
alone caused the gap. Hot-cache DRAM reads may be smaller than the combined A/B operand size.

Source: [`results/gemm_profile.csv`](results/gemm_profile.csv); capture and cache-state details
are given in [GEMM traffic profiling](#gemm-traffic-profiling).

BSD 3-Clause; see `LICENSE`.
