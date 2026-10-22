# Source provenance

The measured acquisition source is commit
`5f6902c536a2d35183d2d2cbe0ec2561364f4884`. The raw dataset records that
source identity. `VERSIONS.env` pins CUTLASS at
`e05f953a5b3d38adc240df2ff928e0421c2abba3` and fixes the software environment.

| Part | Reference or base implementation | Local contribution |
|---|---|---|
| I: LDGSTS/TMA | SemiAnalysis [LDGSTS](https://github.com/SemiAnalysisAI/microbench-blackwell/blob/9a068d853d5c3676939eb46fe21ff6d6a2a4133b/ldgsts_throughput/ldgsts_tput.cu) and [TMA](https://github.com/SemiAnalysisAI/microbench-blackwell/blob/9a068d853d5c3676939eb46fe21ff6d6a2a4133b/tma2d_throughput/tma2d_tput.cu) microbenchmarks as methodological references | B300 grid, matched work, separate validation/timing, occupancy and working-set checks, campaigns and counters |
| II: isolated UMMA | SemiAnalysis [UMMA throughput](https://github.com/SemiAnalysisAI/microbench-blackwell/blob/9a068d853d5c3676939eb46fe21ff6d6a2a4133b/umma_throughput/umma_tput.cu) as methodological reference | One-/two-SM specializations, TMEM readback validation, paired cycle/time instrumentation and analysis |
| III: scaling/power | Extension of the local UMMA kernels | Launch plans, residency handshake, timed SM IDs, whole-device rates, sustained-work driver and sensor integration |
| IV: BF16 GEMM | NVIDIA CUTLASS [dense](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py) and [persistent dense](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py) examples | Candidate/tile selection, common operands/reference, cuBLASLt bridge, paired timing, profiling and tile-supply model |
| V: precision | NVIDIA persistent dense example and [SM103 block-scaled example](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/sm103_dense_blockscaled_gemm_persistent.py) | Format selection, identical operand/scale comparison, cuBLASLt bridge, validation and common timing |

CUTLASS example files and GEMM kernels are used at the pinned revision.
Local adapters define timing and stream selection around their `run()` calls;
the NVFP4 stream helper is restored afterwards. The thesis credits NVIDIA
for the base GEMM implementations.

The SemiAnalysis links identify the corresponding reference benchmarks at
revision `9a068d853d5c3676939eb46fe21ff6d6a2a4133b`. The initially consulted
upstream revision was not recorded; these links identify the methodological
reference, without asserting an exact original derivation commit.
