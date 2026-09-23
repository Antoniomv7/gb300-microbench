# NVIDIA B300 GEMM Microbenchmarks

Five experiments on one NVIDIA B300 follow GEMM operands from memory to the Tensor Cores and on
to complete GEMMs. They measure how LDGSTS and TMA move data into shared memory (I), how fast
one-SM and two-SM UMMA instructions run (II), how UMMA throughput scales to the whole device
(III), how CuTe DSL BF16 GEMMs compare with cuBLASLt (IV), and how BF16, FP8 and NVFP4 inputs
change GEMM throughput (V). A Nsight Compute profile of DRAM and L2 traffic accompanies
Experiment IV.

Experiments I–III are hand-written CUDA/PTX (`cp.async`, `cp.async.bulk.tensor`, `tcgen05.mma`
with Tensor Memory). Experiments IV and V run pinned CUTLASS CuTe DSL example kernels and
cuBLASLt. `results/` holds the published summaries and figures of this thesis study.

## Platform

| | |
|---|---|
| GPU | NVIDIA B300 SXM6 AC (GB110), compute capability 10.3 (`sm_103a`), 148 SMs, 126.5 MiB L2, ECC enabled |
| Driver and clocks | 610.43.02; 1,100 W power limit; 2,032 MHz maximum SM clock; clocks not locked |
| Container | `nvidia/cuda:13.1.0-devel-ubuntu24.04`, pinned by digest in `VERSIONS.env` |
| GEMM software | CUTLASS `e05f953` (CuTe DSL 4.6.1), cuBLASLt 13.2.0, PyTorch 2.10.0+cu130 |
| Profiler | Nsight Compute 2025.4.0 |

## Experiments

| # | Question | Source | Command |
|---|---|---|---|
| I | Effective global-to-shared transfer rate of LDGSTS versus TMA | `memory_paths/` | `make exp1-memory` |
| II | Isolated BF16 UMMA throughput: one SM versus a two-SM pair | `umma_throughput/umma_1sm.cu`, `umma_2sm.cu` | `make exp2-umma` |
| III | BF16 UMMA throughput from one work unit to all 148 SMs | `umma_throughput/umma_device_scaling.cu` | `make exp3-scaling` |
| IV | BF16 GEMM: three CuTe DSL kernels versus cuBLASLt | `gemm_comparison/` | `make exp4-gemm` |
| V | BF16, FP8 and NVFP4 GEMM: CuTe DSL versus cuBLASLt on identical operands | `precision_comparison/` | `make precision` |
| – | DRAM and L2 traffic of single BF16 GEMM launches (diagnostic) | `scripts/profile_gemm.py` | `make gemm-profile` |

`scripts/run_campaign.py` holds the parameters of Experiments I–IV and runs them.
`analysis/analyze.py` computes every published number from the raw measurements, and
`analysis/figures.py` draws the figures.

### I. Global-to-shared memory paths

`memory_paths/ldgsts.cu` copies 16-byte vectors per thread with `cp.async.cg.shared.global`;
`memory_paths/tma.cu` has one elected thread issue 2-D `cp.async.bulk.tensor` loads that complete
on an mbarrier transaction count. Both share the host code, data pattern and validation kernel in
`memory_paths/memory_common.cuh`.

- One 128-thread CTA on each of the 148 SMs; reserving more than half of the shared memory
  enforces one CTA per SM.
- Grid: 2, 4 or 8 pipeline stages × 16, 32 or 64 KiB in flight per SM.
- A 512 MiB working set, more than twice the L2 size (enforced), is streamed 32 times per launch.
- Metric: effective rate = useful bytes (working set × passes) / kernel time, in GB/s.
- Nsight Compute records DRAM read and write bytes for six configurations.

### II. Isolated UMMA throughput

`umma_1sm.cu` issues `tcgen05.mma.cta_group::1.kind::f16` (M = 128) on one SM; `umma_2sm.cu`
issues the `cta_group::2` form (M = 256) from a two-CTA cluster spanning two SMs. Both accumulate
in Tensor Memory (`tcgen05.alloc`, read back with `tcgen05.ld`); shared code is in
`umma_throughput/umma_common.cuh`.

- Grid: N = 64, 128 or 256 (K = 16) × pipeline depth 4, 16, 64 or 256, the number of dependent
  UMMAs issued between two `tcgen05.commit` mbarrier waits.
- `%clock64` times the elected thread's issue-and-completion loop of 1,000 iterations.
- Metric: FLOP/cycle = 2·M·N·K·UMMAs / cycles, divided by two for the two-SM pair.
- Nsight Compute measures the SM clock of both N = 256, depth-256 kernels; with that clock the
  analysis converts the best per-SM result into a modeled TFLOP/s per SM.

### III. Whole-device UMMA scaling

`umma_device_scaling.cu` runs the N = 256, depth-256 UMMA loop as an isolated work unit (one CTA,
or one two-CTA cluster) and at device scale (148 one-SM CTAs, or 74 two-CTA clusters).

- Before timing, a residency handshake proves that every planned CTA is resident at the same time
  on a distinct SM, one per SM; otherwise the run fails.
- CUDA events time each launch of 1,000 iterations.
- `scripts/gpu_telemetry.py` samples SM clock, power and temperature with `nvidia-smi` every
  50 ms. Samples are attributed to a configuration by host timestamps; each configuration
  needs at least three samples inside its timed launches.
- Metrics: total TFLOP/s; scaling efficiency = device / (work units × isolated); the
  clock-normalized efficiency also divides by the ratio of the mean sampled SM clocks.

### IV. BF16 GEMM implementations

`gemm_comparison/gemm_comparison.py` loads the pinned CuTe DSL examples `dense_gemm.py`
(non-persistent, one CTA, tile 128×128) and `dense_gemm_persistent.py` (persistent, one CTA,
128×128; persistent, two CTAs, 256×128 with a 2×1 cluster). `gemm_comparison/cublaslt_bridge.cu`
provides cuBLASLt with the first supported of up to 32 heuristic algorithms (best-fit search,
64 MiB workspace).

- A (M×K) and B (N×K) are BF16 and K-major; accumulation and output are FP32.
- Shapes: 4096³, 8192³, 16384×512×4096, 32768×512×4096 and 512×16384×4096.
- All candidates share the operands of a shape and one IEEE-FP32 reference.
- Two warm-up launches, then CUDA events time ten launches; TFLOP/s = 2·M·N·K / time.

### V. BF16, FP8 and NVFP4

`precision_comparison/precision_comparison.py` runs the pinned CuTe DSL examples
`dense_gemm_persistent.py` (BF16, FP8 E4M3) and `sm103_dense_blockscaled_gemm_persistent.py`
(NVFP4: E2M1 values with one E4M3 scale per 16 elements). Every kernel uses a 256×128 tile, a 2×1
cluster, a TMA store, FP32 accumulation and FP32 output.

- Shapes: 4096³, 8192³ and 32768×512×4096.
- Per shape and format: three repetitions of 5 warm-up and 20 timed launches, timed by
  `cute.testing.benchmark` with CUDA events and hot caches.
- `cublaslt_precision.py` records the operands that each CuTe DSL repetition creates and encodes
  the same values for cuBLASLt (`cublaslt_precision_bridge.cu`), NVFP4 block scales included in
  cuBLASLt's documented `VEC16_UE4M3` layout. Every operand and scale buffer must be byte-identical
  to the one the CuTe DSL kernel consumed. cuBLASLt uses the first supported heuristic algorithm,
  which must pass `cublasLtMatmulAlgoCheck` with FP32 C and D and must be the same for every
  operand set of a shape and format. cuBLASLt is timed with the same benchmark call.
- After all timing, one PyTorch profiler capture per plan names the cuBLASLt kernel. An empty trace
  is recorded as `UNAVAILABLE`; the algorithm identity and validation remain required.

### GEMM traffic profile

`scripts/profile_gemm.py` profiles the persistent two-CTA CuTe DSL kernel and cuBLASLt on 4096³,
8192³ and 32768×512×4096. Each capture runs its own worker process under `ncu`. The worker repeats
Experiment IV's operand setup, validation and two warm-up launches. It then wraps one launch in the
NVTX range `gb300_gemm_profile`, the only range that `ncu` profiles.

- Hot caches (the study): application replay with `--cache-control none`, so every pass repeats
  setup, validation and warm-up. Cold caches (diagnostic): kernel replay with `--cache-control all`.
  Clocks stay unlocked in both modes.
- Counters: DRAM read and write bytes, duration and SM clock.
- The L2-to-SM TMA read counter `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum` is
  collected only when it reproduces, within 0.1%, the useful bytes of a known TMA stream from
  Experiment I.
- Profiler durations are diagnostics; the CUDA-event columns repeat Experiment IV's timing.

## Running the experiments

The host needs an NVIDIA driver, Docker with GPU support, Git, Make and Python 3.9 or newer. Build
the pinned container once, choose an idle B300 and run any experiment:

```bash
make image
export BLACKWELL_GPU_INDEX=6
make exp1-memory     # or exp2-umma, exp3-scaling, exp4-gemm, precision
```

`scripts/run_gpu.sh` resolves the index to the GPU's UUID and refuses a GPU that already runs
compute processes. It exposes only that GPU to the container, as device 0. Every target compiles
as needed and writes one new directory `runs/<target>-<UTC>/`, never an existing one. A run
keeps its samples and validation records in `raw/` and its Nsight Compute reports and CSV exports
beside them. Its `metadata.json`, written only when the run completes, records the Git commit and
whether tracked files were modified, the GPU and driver, the CUDA, Nsight Compute, CUTLASS and
Python package versions, the parameters and UTC timestamps.

Experiments I–IV report per-launch samples; their published summaries need three campaigns (see
[Statistics](#statistics)). `make precision` also writes its two summaries and figures to
`analysis/` inside its run directory.

### The complete study

```bash
make final-study
```

The target builds once and then runs six commands; the first failure stops it:

| Step | Output under `runs/study-<UTC>/` |
|---|---|
| Experiments I–IV, three independent campaigns | `campaign-1/`, `campaign-2/`, `campaign-3/` |
| Experiment V | `precision/` |
| Hot-cache GEMM traffic profile | `gemm-profile-hot/` |
| `analysis/analyze.py --study` on the CPU | `analysis/`: the thirteen files that `results/` publishes |

Use `make final-study 2>&1 | tee study.log` to keep a log. The underlying targets are
`make campaign`, `make analyze CAMPAIGNS="dir1 dir2 dir3"` and
`make gemm-profile PROFILE_CACHE=hot|cold GEMM_SUMMARY=.../gemm_comparison.csv`.

## Validation and controls

- **Correctness before timing.** Each configuration's output is validated before any timed launch
  counts. Experiment I compares every transferred vector with its generated pattern. Experiments
  II and III use small integer operands, whose FP32 accumulation is exact, and compare every
  accumulator element. Experiment IV compares every candidate with the IEEE-FP32 reference
  (|error| ≤ 0.1 + 10⁻⁵·|reference|).
- **Every Experiment V output.** The outputs of all repetitions, CuTe DSL and cuBLASLt, before and
  after timing, are compared with the IEEE-FP32 product of the dequantized operands. The absolute
  tolerance is 0.1; the relative tolerance is 10⁻³ for BF16 and FP8 and 10⁻² for NVFP4. A negative
  control first confirms that the check rejects a perturbed value. The run fails if any check,
  operand-byte comparison or cuBLASLt algorithm condition fails.
- **Complete data.** A campaign keeps an experiment's samples only if every configuration produced
  its full set: 540 memory, 720 UMMA, 120 scaling and 20 GEMM rows. CUDA calls are checked
  throughout.
- **Warm-up.** Warm-up precedes timing: 2 s (I), 10 launches (II, III), 2 launches (IV) and 5
  launches per repetition (V).
- **Profiling apart from timing.** Nsight Compute runs after the timed launches, or in separate
  processes, and never changes a timing.
- **Isolation.** The GPU must be idle, and existing run directories are never overwritten.

## Statistics

For Experiments I–IV, each campaign first reduces the 30 launches of a configuration to their
median. The three independent campaigns then give a mean, a sample standard deviation and a
coefficient of variation (CV); the CSVs keep the three campaign values. Ratios such as TMA/LDGSTS
or scaling efficiency are computed within each campaign and then averaged. Experiment V reports the
mean, sample standard deviation and CV of its three repetitions. All statistics are descriptive.
The largest CV among the Experiment I–IV summaries is 1.6%.

## Published results

`results/` holds the summaries and figures of study `study-20260923T173150Z`, measured on GPU
`GPU-619f7fdc-5f98-8c37-fe89-0465d6130baf` with the code at tag `tfm-acquisition`. The figures
show means; whiskers show the range of the three campaigns or repetitions. The results describe
this GPU, software and configuration grid, not architectural peak specifications.

### I. HBM-to-shared-memory paths

![Effective transfer rate for LDGSTS and TMA](results/memory_paths.svg)

LDGSTS reaches the higher effective rate in **8 of 9** stage and in-flight-byte configurations.
The highest means are **7,024 GB/s** for LDGSTS and **6,964 GB/s** for TMA, both at four stages
and 64 KiB in flight. At two stages and 64 KiB, TMA is marginally higher: 6,954 versus
6,943 GB/s. More bytes in flight help both paths in this grid, while eight stages reduce
throughput substantially, most for TMA. The DRAM counters read 1.00 bytes per useful byte in all
six captured configurations. Source: [`results/memory_paths.csv`](results/memory_paths.csv).

### II–III. UMMA instruction throughput and device scaling

![Isolated BF16 UMMA throughput](results/umma_throughput.svg)

At N = 256 and depth 256, the isolated one-SM kernel reaches **8,101 FLOP/cycle/SM**, a modeled
**16.372 TFLOP/s/SM** at the measured clock. The two-SM kernel reaches **8,028 FLOP/cycle/SM**,
**1.982×** the total throughput of the one-SM kernel.

![BF16 UMMA scaling to 148 SMs](results/umma_device_scaling.svg)

| Execution | Active SMs | Mean throughput | Scaling efficiency | Clock-normalized efficiency |
|---|---:|---:|---:|---:|
| One-SM work units | 148 | 2,103.7 TFLOP/s | 91.7% | 96.9% |
| Two-SM work units | 148 | 2,119.7 TFLOP/s | 93.3% | 98.1% |

Two-SM work units deliver about **0.8%** more device throughput. The isolated units ran at the
2,032 MHz maximum clock, the whole device at about 1,922–1,931 MHz. The gap between raw and
clock-normalized efficiency is why a fixed-clock extrapolation from one SM overstates the scaling
loss. Sources: [`results/umma_throughput.csv`](results/umma_throughput.csv),
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

The persistent 2-CTA kernel is the fastest of the three CuTe DSL variants on all five shapes, but
its distance to cuBLASLt depends strongly on the shape. Source:
[`results/gemm_comparison.csv`](results/gemm_comparison.csv).

### V. BF16, FP8 and NVFP4 GEMM

![CuTe DSL throughput by input precision](results/precision_comparison.svg)

| GEMM shape (M × N × K) | BF16 CuTe DSL | FP8 CuTe DSL | NVFP4 CuTe DSL |
|---|---:|---:|---:|
| 4096 × 4096 × 4096 | 1,685.0 | 2,938.0 (1.74×) | 4,032.9 (2.39×) |
| 8192 × 8192 × 8192 | 1,459.0 | 3,122.9 (2.14×) | 5,568.8 (3.82×) |
| 32768 × 512 × 4096 | 757.6 | 1,721.1 (2.27×) | 2,992.8 (3.95×) |

Throughputs are in TFLOP/s; parentheses give the speedup over BF16 on the same shape. Lower
precision raises throughput, but the gain depends on the shape and in several cases stays below
the ratio of the nominal dense peaks. NVFP4's operand representation and numerical error differ
from BF16 and FP8.

![Matched CuTe DSL and cuBLASLt comparisons by precision](results/precision_cutedsl_vs_cublaslt.svg)

On identical operands, CuTe DSL reaches **50.0–95.3%** of cuBLASLt's throughput in BF16,
**64.4–92.2%** in FP8 and **73.2–84.2%** in NVFP4. Every timed output of both implementations
passed validation. Sources: [`results/precision_comparison.csv`](results/precision_comparison.csv),
[`results/precision_cutedsl_vs_cublaslt.csv`](results/precision_cutedsl_vs_cublaslt.csv).

### Hot-cache BF16 GEMM traffic

In the six profiled launches, the persistent 2-CTA CuTe DSL kernel reads more DRAM bytes than
cuBLASLt at 8192 × 8192 × 8192 (**3.35 versus 1.16 GB**) and at 32768 × 512 × 4096 (**1.08 versus
0.28 GB**). The calibrated L2-to-SM TMA counter also records more bytes for CuTe DSL at those
shapes. This matches the larger performance gaps but shows neither which operand was reread nor
that traffic alone caused the gap. `compulsory_read_bytes` is the combined A and B size;
hot-cache DRAM reads can be smaller, including zero, in which case the L2/DRAM ratio is left
empty. Source: [`results/gemm_profile.csv`](results/gemm_profile.csv).

## Reproducing the published summaries

The complete acquisition accompanies the thesis as `supplementary/study-20260923T173150Z.tar.gz`
(SHA-256 `71533e13b16c2e85a784c6f6abea3ef95cf4193af17794dd59155ad1777e4101`). It contains the raw
samples, telemetry, Nsight Compute reports and exports, validation records and logs. The analysis
reads the extracted archive directly, with no GPU:

```bash
mkdir -p /tmp/gb300 && tar -xzf supplementary/study-20260923T173150Z.tar.gz -C /tmp/gb300
make regenerate STUDY=/tmp/gb300/study-20260923T173150Z
diff -r results runs/regenerated-<UTC>
```

Eleven of the thirteen files come out byte-identical, including all six figures and
`gemm_profile.csv`, which is rebuilt from the exported counters. In the two Experiment V CSVs,
6 of 738 values differ by 10⁻⁶, one unit in the last printed digit: the archived
`precision/raw/repetitions.csv` stores six decimals, whereas the published means were computed
from full-precision values. New runs store full precision, so their summaries regenerate exactly.

## Limitations

- One GPU in one system: the results describe this B300 SXM6 AC, driver, software stack and
  configuration grid, and are not architectural peaks.
- Clocks are not locked. Throughput includes the operating clock (DVFS), which Experiment III
  samples; its power samples are indicative telemetry, not a calibrated energy measurement.
- The GEMM measurements reuse operands across launches (hot caches), so the L2 can hold part of
  them.
- Experiment I's effective rate is useful bytes over kernel time, not a DRAM-bandwidth counter or
  a GEMM prediction. Experiment II's TFLOP/s per SM is modeled from cycles and the profiled clock.
- Experiments IV and V compare specific CuTe DSL example kernels with cuBLASLt's first supported
  heuristic algorithm, without autotuning; they do not establish a ceiling for either library.
- The vendor peaks in `precision_comparison.csv` are nominal dense values.
- The traffic profile has one launch per case; aggregate counters cannot attribute rereads to an
  operand.
- Three campaigns or three repetitions support descriptive statistics, not inference.

## Repository layout

| Path | Contents |
|---|---|
| `benchmark_common.cuh` | Device check and argument parsing shared by the CUDA benchmarks |
| `memory_paths/` | Experiment I: LDGSTS and TMA kernels |
| `umma_throughput/` | Experiments II and III: UMMA and Tensor Memory kernels |
| `gemm_comparison/` | Experiment IV driver and its cuBLASLt bridge |
| `precision_comparison/` | Experiment V driver, operand-sharing cuBLASLt baseline and its bridge |
| `scripts/` | Campaign runner, GPU selection, clock telemetry, Nsight Compute captures, GEMM profile, run metadata |
| `analysis/` | Statistics and figures |
| `results/` | Published CSV summaries and SVG figures |
| `Dockerfile`, `VERSIONS.env`, `Makefile` | Pinned environment and commands |
| `build/`, `runs/` | Binaries and measurements; not tracked by Git |

## Versions

- `tfm-acquisition` (commit `6868f00`): the exact code that acquired the published study; the
  archive records this commit.
- `tfm-final`: the cleaned repository. The cleanup removed workflow infrastructure only:
  per-file source hashes, manifests, state files and a separate checker of saved runs, replaced
  by Git history and one small `metadata.json` per run. It did not change the kernels,
  parameters, timing, validation, statistics or published results. For new runs, a study writes
  all thirteen summaries to its `analysis/` directory, and Experiment V's raw tables keep full
  float precision.

BSD 3-Clause; see `LICENSE`.
