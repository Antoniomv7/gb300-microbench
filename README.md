# NVIDIA B300 GEMM Microbenchmarks

Five experiments connect global-to-shared data movement, BF16 UMMA issue throughput,
whole-device scaling, BF16 GEMM implementations and lower input precisions on one B300.
The research question is which measured instruction and data-supply limits help explain
the shape-dependent gap between pinned CuTe DSL kernels and cuBLASLt.

**Protocol v2 is ready for a B300 pilot; its GPU paths have not yet been validated on hardware.**
The current `results/`, published study log and `evidence/sass/` belong to the September 2026
acquisition. They are preserved. Read the [historical results](docs/historical_results.md)
for their interpretation and the [review protocol](docs/review_protocol.md) for the new study.

## Experiments

| # | Measurement | Configurations per final campaign |
|---|---|---|
| I | LDGSTS versus unicast 2-D TMA effective global-to-shared rate | 2 methods × 3 stage counts × 3 in-flight sizes × 30 launches |
| II | Isolated BF16 UMMA issue/completion loop | 1-SM/2-SM × 3 N values × 4 depths × 30 launches |
| III | Isolated versus device-scale UMMA, with internal cycle and time counters | 4 configurations × 30 launches; every timed CTA records its SM |
| IV | BF16 CuTe DSL versus cuBLASLt | 5 shapes × 5 candidates, including P2 tiles 256×128 and 256×256 |
| V | BF16, FP8 E4M3 and NVFP4 E2M1, on matched operand bytes | 3 shapes × 3 formats × 2 implementations × 3 repetitions |
| III power | Sustained full-device operation | 2 UMMA methods × 3 windows; 10 s warm-up + at least 30 s measured |
| IV profile | Hot-cache DRAM and calibrated TMA traffic, plus profiler duration | Both P2 tiles and cuBLASLt × 3 shapes = 9 launches |

Experiments I–III use CUDA/PTX. Experiments IV–V use pinned NVIDIA CUTLASS examples and
local cuBLASLt bridges. [Provenance and source links](docs/provenance.md) distinguish the
SemiAnalysis methodological reference, the NVIDIA example implementations and local work.

## Measurement policy

- Numerical validation, compilation, allocation and profiling remain outside timed samples.
- II–III read `%clock64` bracketed by `%globaltimer` over the same internal loop. The raw
  readings, SM IDs and timestamp uncertainty are saved. Internal-loop rates and whole-kernel
  CUDA-event rates remain separate. Device-wide telemetry is operating context.
- IV–V use graph replay as the primary timing policy for both libraries. A paired series of
  ordinary host-enqueued launches, with CUDA events and its own warm-up, uses the same operand
  workspace. Every series has 5 warm-up launches and 20 measured launches on a non-default stream.
  Unsupported graph capture fails the pilot; there is no automatic timing fallback.
- Sustained power uses explicitly identified GPU-board power fields and time-weighted means.
  Polling at 50 ms does not imply 50 ms sensor bandwidth. The first and second halves of each
  final window must differ by at most 5% in mean power and 2% in device-reported SM clock.
- Three I–IV campaigns reduce the 30-launch samples to medians before cross-campaign statistics.
  IV uses one primary block time per candidate in each campaign. V and power have three
  repetitions per configuration. Means, sample standard deviations and CVs are descriptive;
  figure whiskers show the observed minimum and maximum throughout.
- Final analysis requires complete configuration grids, protocol v2, one source commit, one
  GPU and one software/power-limit fingerprint. Pilots and historical data are rejected.

The scope of Experiment I is the tested geometry: 256 B-wide TMA boxes without swizzle,
including 2 KiB boxes at 8 stages and 16 KiB in flight. It does not establish a general
LDGSTS-versus-TMA ordering for GEMM tiles.

The nominal bridge between II and III is
`8192 FLOP/cycle/SM × 148 SM × 1.855799 GHz ≈ 2.25 PFLOP/s`.
That frequency is implied by the nominal peak; the new run measures its own effective loop
clocks. It must not be presented as a measured clock or used to correct short power windows.

## Run the pilot on the server

Use a clean checkout of this branch and an idle B300. Docker with the NVIDIA runtime and
Nsight Compute counter access are required. `VERSIONS.env` pins the container digest,
CUTLASS commit, Python dependencies and `sm_103a` target.

```bash
make check-static
make image
make clean
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used --format=csv
# Set the index of the idle B300 you selected from that output:
export BLACKWELL_GPU_INDEX=<index>
set -o pipefail
pilot_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
make pilot STAMP="$pilot_stamp" 2>&1 | tee "pilot-${pilot_stamp}.log"
tar -czf "pilot-${pilot_stamp}.tar.gz" \
  "runs/pilot-${pilot_stamp}" "pilot-${pilot_stamp}.log"
```

Replace `<index>` before executing the export. `make pilot` compiles the CUDA sources and
bridges, checks the new loop counters against a control, exercises all five IV candidates
on 4096³ and all three V formats, probes sustained power, and captures both P2 tiles plus
cuBLASLt with the calibrated TMA counter. The power probe is functional (2 s warm-up + 3 s),
not a final power estimate. Its directory contains `report.json`, raw data and step logs.

## Collect the final study after reviewing the pilot

Keep the exact passing pilot commit, image, GPU and configured power limit. A source change
requires a fresh pilot. Do not lock clocks or alter the power limit between stages.

```bash
set -o pipefail
study_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
make final-study PILOT="runs/pilot-${pilot_stamp}" STAMP="$study_stamp" \
  2>&1 | tee "study-${study_stamp}.log"
tar -czf "study-${study_stamp}.tar.gz" \
  "runs/study-${study_stamp}" "study-${study_stamp}.log"
sha256sum "study-${study_stamp}.tar.gz" > "study-${study_stamp}.tar.gz.sha256"
```

`final-study` verifies the pilot against the current commit and environment, then runs
I–IV three times, V once with three repetitions, sustained power, nine hot-cache GEMM
profiles and CPU analysis. It creates a fresh `runs/study-<UTC>/` with:

| Directory | Contents |
|---|---|
| `campaign-1/`, `campaign-2/`, `campaign-3/` | I–IV raw rows, per-SM loop rows, telemetry and eight NCU captures each |
| `precision/` | V timing pairs, validation, operand-byte comparisons and cuBLASLt plan details |
| `power/` | Six sustained windows, native work/time accounting and full sensor traces |
| `gemm-profile-hot/` | Nine captures, TMA calibration and validation records |
| `analysis/` | Nine CSVs, seven SVGs and analysis metadata |

Nothing automatically replaces `results/`. Stop on a failure and keep its directory/log;
do not select only the successful windows or mix stages from another source commit.
The [review protocol](docs/review_protocol.md) defines acceptance and publication steps.

Individual targets remain available: `make exp1-memory`, `make exp2-umma`,
`make exp3-scaling`, `make exp4-gemm`, `make precision`, `make power`, and
`make gemm-profile PROFILE_CACHE=hot GEMM_SUMMARY=.../gemm_comparison.csv`.

To regenerate a completed v2 study on a CPU:

```bash
make regenerate STUDY=runs/study-<UTC>
```

Historical archives require the analysis revision recorded in their acquisition metadata.
They cannot be processed by mixing the old CSV schema with the new code.

## Limits and repository layout

One GPU, dynamic clocks, a small configuration grid and deterministic small-integer operands
limit generalization. Reused operands give hot-cache GEMM timings. First-supported cuBLASLt
heuristics and selected CuTe examples do not establish a library ceiling. Aggregate TMA/DRAM
counts do not identify operand rereads or prove a performance cause. A larger tile changes
several resource and scheduling properties; its timing alone cannot isolate traffic effects.
The IV and V BF16 baselines are separate acquisitions and can differ despite the common policy.

`memory_paths/`, `umma_throughput/`, `gemm_comparison/` and `precision_comparison/` contain
the experiments; `scripts/` handles acquisition and metadata; `analysis/` computes summaries
and figures; `tests/` contains CPU checks. `build/` and `runs/` are untracked local outputs.
`results/`, `evidence/sass/` and the historical tags remain the record of the earlier study
until the reviewed acquisition is accepted and published as one clearly identified release.

BSD 3-Clause; see [LICENSE](LICENSE).
