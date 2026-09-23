# GB300 Blackwell Microbenchmarks

Five experiments on one NVIDIA B300 SXM6 AC, a supplementary Nsight Compute profile of the BF16
GEMM gap, and the checks that reject incomplete or invalid acquisitions.

## Experiments

| # | Purpose | Command | Output in `runs/` |
|---|---|---|---|
| I | LDGSTS versus TMA effective transfer rate | `make exp1-memory` | `exp1-memory-<UTC>/`: `raw/memory_paths.csv`, 6 NCU DRAM captures |
| II | Isolated 1-SM versus 2-SM BF16 UMMA throughput | `make exp2-umma` | `exp2-umma-<UTC>/`: `raw/umma_throughput.csv`, 2 NCU SM-clock captures |
| III | Whole-device BF16 UMMA scaling | `make exp3-scaling` | `exp3-scaling-<UTC>/`: `raw/umma_device_scaling.csv` and its clock telemetry |
| IV | BF16 CuTe DSL variants versus cuBLASLt | `make exp4-gemm` | `exp4-gemm-<UTC>/`: `raw/gemm_comparison.csv` |
| V | BF16, FP8 and NVFP4 with CuTe DSL and cuBLASLt | `make precision` | `precision-<UTC>/`: 9-row `precision_comparison.csv`, 18-row `precision_cutedsl_vs_cublaslt.csv`, raw records |

Every run directory also holds `metadata.json` (or `index.json`), with the GPU identity, source
commit, source-file hashes and software versions. `analysis/analyze.py` reduces Experiments I–IV
from three complete campaigns: the median within each campaign, then the mean, sample standard
deviation and coefficient of variation across the three. Experiment V repeats each format and shape
three times in one run. All statistics are descriptive.

## Running

The pinned CUDA image, CUTLASS commit and Python packages are in `VERSIONS.env`. Build once:

```bash
make image
make build
```

`scripts/run_gpu.sh` resolves `BLACKWELL_GPU_INDEX` to a GPU UUID, refuses a GPU that already runs
compute processes, and exposes only that GPU to the container. One experiment at a time:

```bash
export BLACKWELL_GPU_INDEX=6
make exp1-memory
make exp2-umma
make exp3-scaling
make exp4-gemm
make precision
```

Each target uses the final measurement parameters, writes a new directory, and finishes by checking
its own saved result with `scripts/check_diagnostics.py`. It runs only its own experiment and only
that experiment's profiling:

- `exp1-memory`: the six Nsight Compute DRAM captures of the memory configurations.
- `exp2-umma`: the two Nsight Compute SM-clock captures of the peak-depth UMMA configurations.
- `exp3-scaling`: no Nsight Compute; `nvidia-smi` samples the SM clock, power and temperature every
  50 ms during the timed launches.
- `exp4-gemm`: none.
- `precision`: no Nsight Compute; after all timing, the PyTorch profiler names the kernel behind each
  selected cuBLASLt algorithm.

Individual runs are for inspecting one experiment. `make analyze` accepts only complete campaigns,
so they never enter the three-campaign statistics.

The complete acquisition is one command:

```bash
BLACKWELL_GPU_INDEX=6 make final-study
```

It runs Experiments I–IV three times, Experiment V once and the supplementary hot-cache GEMM
profile. It refuses to start from a tree with modified tracked files, because every run records its
source commit. It creates `runs/study-<UTC>/` and, stopping at the first failure:

1. runs three complete campaigns, `campaign-1` to `campaign-3`, each with Experiments I–IV, the
   UMMA clock telemetry and all eight Nsight Compute captures;
2. checks that the three are complete, share one GPU and one source commit, and hold their expected
   rows, telemetry and captures;
3. analyzes only those three campaigns into `analysis/`;
4. runs Experiment V once through `make precision` into `precision/`;
5. captures the six supplementary GEMM profiles with the hot cache into `gemm-profile-hot/`, using
   `analysis/gemm_comparison.csv` as their CUDA-event reference;
6. checks the whole study, then writes the evidence archive `runs/study-<UTC>.tar.gz`: the three
   campaigns, the analysis, the precision data, the Nsight Compute reports, the per-step logs in
   `logs/` and `study.json`. It prints the archive's path, the GPU UUID and the source commit.

A failed study keeps its directory for diagnosis and is never reused or archived. Each step is an
ordinary command that can be rerun alone: `make campaign`, `make analyze`, `make precision`,
`make gemm-profile` and `python3 scripts/check_diagnostics.py`; `make help` lists their inputs, and
`check_diagnostics.py --study` also audits an extracted archive. Nothing writes to `results/`; new
summaries replace the published ones only by hand, after the study has been audited.

## Validation

A run is rejected, rather than summarized, when any of these fails:

- **Experiments I–IV.** Every benchmark validates its numerical result before its timing is kept,
  and a dataset is written only when all its rows validated: 540 memory, 720 isolated UMMA, 120
  device-scaling and 20 GEMM rows. Whole-device UMMA requires simultaneous residency on every
  planned SM. The clock telemetry must span the whole timed scaling block, agree with the host
  clock to within 1 s, and hold at least 3 samples inside each configuration's timed launches.
  GEMM candidates share operands and must match an untimed IEEE-FP32 reference. Each Nsight Compute
  capture must hold one kernel with every requested counter, and its report and export are kept.
- **Analysis.** Exactly three distinct complete campaigns on one GPU, from one source commit with
  identical source hashes and no modified tracked files. `analysis.json` records the campaigns and
  the SHA-256 of every output.
- **Experiment V.** Each CuTe DSL output, including every timed repetition's, and each cuBLASLt
  output before and after timing must match an IEEE-FP32 product of the exactly represented,
  dequantized operands. cuBLASLt consumes the same bytes as CuTe DSL (data and NVFP4 scales), uses
  FP32 compute and output confirmed by `cublasLtMatmulAlgoCheck`, and keeps one algorithm across
  operand sets. A negative control confirms that the tolerance check rejects a mismatch. The
  protocol is fixed at three repetitions of 5 warm-up and 20 timed launches.
- **GEMM profile.** `GEMM_SUMMARY` must be the `gemm_comparison.csv` of a `make analyze` output
  whose `analysis.json` names the profiled GPU and records that file's SHA-256, with all six
  shape/variant rows; the published `results/` summary is refused. Each capture validates before
  and after the profiled launch, holds exactly one kernel inside the NVTX range, and shares operands
  with the other implementation of its shape. The L2→SM metric is used only after a TMA stream
  reproduces its useful bytes to within 0.1%.

## Timed results and Nsight Compute diagnostics

Throughput comes only from timed launches without a profiler attached: CUDA events for Experiments
I, III, IV and V, and `%clock64` for Experiment II. Clocks are not locked, and the caches are hot
after the warm-up launches.

Nsight Compute always runs in separate processes after the timing. The campaign captures report DRAM
bytes read per useful byte beside the Experiment I rates, and the SM clock that expresses the
strongest Experiment II configuration in TFLOP/s/SM. The GEMM profile's durations and byte counts
are diagnostics beside the CUDA-event reference it records, never replacements for it:

- **Hot cache** (`PROFILE_CACHE=hot`, used by `final-study`): application replay with
  `--cache-control none`. Every replay pass reruns the operand setup, validation and the campaigns'
  two warm-up launches before the one NVTX-selected launch.
- **Cold cache** (`PROFILE_CACHE=cold`): kernel replay with `--cache-control all`, which flushes the
  caches before the profiled launch. The published `results/gemm_profile.csv` is a cold profile.

Each profile directory holds one cache state, recorded in `index.json` and in every row of
`gemm_profile.csv`; compare DRAM counters only between captures made with the same state.

## Published results

`results/` holds the previous acquisition, made before runs recorded their source commit:

- `memory_paths`, `umma_throughput`, `umma_device_scaling` and `gemm_comparison` (CSV and SVG):
  final campaigns `final-unified-1` to `-3` on `GPU-619f7fdc-5f98-8c37-fe89-0465d6130baf`,
  published in commit `ade92ad`.
- `precision_comparison` (CSV and SVG): the former CuTe DSL-only precision run, last updated in
  commit `ade92ad`.
- `precision_cutedsl_vs_cublaslt` (CSV and SVG) and `gemm_profile.csv`: separate runs on the same
  GPU, published in commit `5b5a95c`. `gemm_profile.csv` is the cold-cache profile; its CUDA-event
  columns come from the `gemm_comparison.csv` above.

These files are kept as historical evidence and are not results of a new study. Their raw runs
predate the provenance fields that `make analyze` and `scripts/check_diagnostics.py` now require.

BSD 3-Clause; see `LICENSE`.
