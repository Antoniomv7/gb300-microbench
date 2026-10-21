# Source and methodological provenance

The thesis identifies SemiAnalysis's **microbench-blackwell** as a methodological reference
and NVIDIA CUTLASS as the source of the CuTe DSL example implementations. These are distinct
roles. This page makes their scope explicit without inventing an original upstream revision.

| Part | Reference or implementation source | Local responsibilities |
|---|---|---|
| I: LDGSTS/TMA | SemiAnalysis throughput microbenchmarks are the methodological reference: [`ldgsts_tput.cu`](https://github.com/SemiAnalysisAI/microbench-blackwell/blob/9a068d853d5c3676939eb46fe21ff6d6a2a4133b/ldgsts_throughput/ldgsts_tput.cu), [`tma2d_tput.cu`](https://github.com/SemiAnalysisAI/microbench-blackwell/blob/9a068d853d5c3676939eb46fe21ff6d6a2a4133b/tma2d_throughput/tma2d_tput.cu) | B300 experiment grid, matched useful work, deterministic patterns, separate validation and timing kernels, occupancy/working-set checks, campaigns and counter ingestion |
| II: isolated UMMA | SemiAnalysis [`umma_tput.cu`](https://github.com/SemiAnalysisAI/microbench-blackwell/blob/9a068d853d5c3676939eb46fe21ff6d6a2a4133b/umma_throughput/umma_tput.cu), as methodological reference for issue/completion-loop throughput | BF16 configuration grid, one-/two-SM specializations, exact numerical checks using Tensor Memory readback, paired cycle/time observations and analysis |
| III: device scaling and sustained power | Extension of the local II-style kernels and shared helpers | Full-device launch plans, residency handshake, one CTA per SM, per-CTA timestamps/SM IDs, CUDA-event rates, sustained-work driver and separate sensor traces |
| IV: BF16 GEMM | NVIDIA [`dense_gemm.py`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py), [`dense_gemm_persistent.py`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py) | Candidate/tile selection, common operands and reference, cuBLASLt bridge, paired timing adapter, profiling and tile-traffic model |
| V: input precision | NVIDIA persistent dense example above and [`sm103_dense_blockscaled_gemm_persistent.py`](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/sm103_dense_blockscaled_gemm_persistent.py) | Format choices, hooks around example execution, byte-identical operand comparison, precision cuBLASLt bridge and checks, repetition evidence and common timing adapter |

The benchmark sources and acquisition helpers are present in this repository; CUTLASS is
checked out separately in the container at the exact `VERSIONS.env` commit. Its example
files are not edited by this protocol. The timing adapter changes the benchmark policy
around their `run()` calls; their native timing defaults must not be assumed unchanged.
The pinned NVFP4 example explicitly selects `cutlass_torch.default_stream()`. During that
call only, a scoped adapter routes this helper to `current_stream()` so allocations and
launches use the same non-default stream. The original helper is restored on exit.

The SemiAnalysis links above identify revision
`9a068d853d5c3676939eb46fe21ff6d6a2a4133b`, inspected during the tutor-review audit. The
original acquisition did not record the SemiAnalysis revision initially consulted. This
audit revision must not be described as the original derivation commit. The classification
is methodological attribution and does not establish a line-by-line historical derivation
or a claim that no lines were originally adapted. If a precise derivation statement is
required, recover the development history or the author's record before deposition.

The compact publication repository initially imported its CUDA sources from
[`gb300-gemm-anatomy` at 86f2382](https://github.com/Antoniomv7/gb300-gemm-anatomy/tree/86f2382fb92a957035c067ae725e9e25afacab6f),
as recorded in its first publication README (commit `4657402`). That import history is
separate from the upstream scientific/methodological credits above. This review does not
relicense upstream examples or add their source to the local BSD-licensed files.
