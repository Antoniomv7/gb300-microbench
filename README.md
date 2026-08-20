# GB300 Blackwell Microbenchmarks

Minimal, reproducible microbenchmarks for studying HBM-to-SMEM transfer paths, fifth-generation BF16 UMMA throughput, and CuTe DSL GEMM variants versus cuBLASLt on NVIDIA GB300. The repository can collect one pilot plus three new final campaigns and generate the tables and figures used by the thesis.

This is the compact publication layer of [gb300-gemm-anatomy](https://github.com/Antoniomv7/gb300-gemm-anatomy), frozen at commit [`86f2382`](https://github.com/Antoniomv7/gb300-gemm-anatomy/commit/86f2382fb92a957035c067ae725e9e25afacab6f). Development protocols, agent instructions, phase gates, campaign orchestration, static checkers, and audit scaffolding are intentionally excluded.

The CUDA sources are preserved with identical executable code: every identifier, constant, template, inline-PTX block and command-line option is unchanged from the build that produced the reference results, so the compiled binaries measure exactly what they measured there. Only comments and message text were rewritten, to remove pointers into the development repository. The equivalence was checked by stripping comments from both versions and comparing the remaining token streams, which are identical.

## Experiments

| Directory | Question | Implementations |
|---|---|---|
| `memory_paths/` | How do equivalent LDGSTS and 2D unicast TMA paths behave over the tested pipeline grid? | `ldgsts.cu`, `tma.cu` |
| `umma_throughput/` | How do 1-SM and 2-SM BF16 `tcgen05.mma` kernels scale with N and instruction depth? | `umma_1sm.cu`, `umma_2sm.cu` |
| `gemm_comparison/` | How closely do three official CuTe DSL execution variants approach cuBLASLt on five shapes? | `gemm_comparison.py`, `cublaslt_bridge.cu` |

The CUDA kernels retain their integrated numerical validation. The GEMM comparison validates every candidate against one FP32 reference before timing it.

## Historical reference results

The tables and figures currently in `results/` are the byte-identical accepted P4.3 outputs from the frozen development repository. They are retained as a reference until the clean repository has completed its own experimental population. They are not mixed with the new campaigns and will not be silently reused as their raw input.

The reference summarizes three complete final campaigns on one NVIDIA B300 SXM6 AC. Campaign `20260812T013848Z` was a pilot and is excluded from every statistic.

### HBM-to-SMEM paths

![LDGSTS and TMA effective transfer rates](results/figures/memory_paths.svg)

The reported GB/s is a timing-derived effective transfer rate (`useful_bytes / kernel_time`), not a direct DRAM-bandwidth counter and not GEMM traffic. LDGSTS was higher in eight of nine matched configurations; TMA was essentially equal and slightly higher only at 2 stages and 64 KiB in flight (mean ratio `1.00097`). The highest means were about `7.024 TB/s` for LDGSTS and `6.962 TB/s` for TMA. Every saturation candidate reached the largest tested value, so no plateau was observed inside the 16/32/64 KiB grid.

### BF16 UMMA throughput

![1-SM and 2-SM UMMA throughput](results/figures/umma_throughput.svg)

The metric is derived from validated operation counts and `%clock64` cycles. The selected per-SM ceiling candidate was `umma_1sm`, `N=256`, depth `256`, corresponding to a modeled mean of `16.358 TFLOP/s/SM` after applying the measured SM clock. The 2-SM/1-SM speedup reached `1.983x` at `N=256`, depth `256`; a `2.096x` value at `N=64`, depth `256` is preserved as a surprising diagnostic rather than treated as an architectural claim.

### CuTe DSL versus cuBLASLt

![CuTe DSL and cuBLASLt GEMM comparison](results/figures/gemm_comparison.svg)

`persistent_2cta` was the stable best CuTe DSL variant for all five shapes.

| Shape `(M,N,K,L)` | CuTe DSL TFLOP/s | cuBLASLt TFLOP/s | CuTe DSL / cuBLASLt | Gap |
|---|---:|---:|---:|---:|
| `4096x4096x4096x1` | 1665.4 | 1751.6 | 95.08% | 4.92% |
| `8192x8192x8192x1` | 1443.7 | 2118.7 | 68.14% | 31.86% |
| `16384x512x4096x1` | 811.1 | 1432.7 | 56.61% | 43.39% |
| `32768x512x4096x1` | 756.2 | 1504.4 | 50.27% | 49.73% |
| `512x16384x4096x1` | 1262.5 | 1495.1 | 84.44% | 15.56% |

These are hot-cache measurements. No GEMM kernel was profiled with Nsight Compute, so the gap is not attributed to memory, Tensor Cores, scheduling, or any other single cause.

## Environment

The original campaigns used:

- NVIDIA B300 SXM6 AC, compute capability 10.3
- driver 610.43.02
- CUDA toolkit 13.1.0 (`nvcc` 13.1.80)
- CuTe DSL / CUTLASS v4.6.1 at commit `e05f953a...`
- PyTorch 2.10.0+cu130
- target `sm_103a`

Exact image, package, upstream-source, and architecture pins are in `VERSIONS.env` and `PHASE3_VERSIONS.env`.

## Build

```bash
make image
make build
```

`make build` compiles the four CUDA binaries into `build/`. UMMA uses the explicit `compute_103a` / `sm_103a` virtual-real architecture pair required by the pinned toolchain.

`make clean` removes `build/`. Run it if you change a compilation flag or the architecture, so the next build cannot reuse a stale binary — the campaign manifest hashes the sources, not the executable.

To disassemble the rebuilt binaries:

```bash
make sass
```

Generated disassembly goes to `build/sass/`. The files under `artifacts/sass/` are the preserved, byte-identical disassemblies from the accepted experimental build and are useful for compiler and instruction-level study.

## Run

Choose one idle physical GPU explicitly:

```bash
export BLACKWELL_GPU_INDEX=7
```

The benchmarks record the Git commit they ran at, and both `make smoke` and every campaign require a clean worktree, so **commit the repository before running anything**. The same commit, container image and physical GPU must be used for the whole population.

Then run the short GPU validation:

```bash
make smoke
```

`make smoke` uses reduced repetitions but exercises numerical validation, the complete configuration grids and all five GEMM shapes, and then checks the three raw CSV files against the campaign contract: required columns, the full configuration grid, contiguous sample indices, a clean commit and one consistent GPU identity across all three files. Only the sample *count* is relaxed. It is never included in a final statistic.

Before spending a full pilot on profiling, require one real Nsight Compute
capture to pass through the exact export and parser path:

```bash
make ncu-check
```

The gate profiles one memory-path case. NCU exports its row-oriented raw
page using stable metric identifiers, base units and floating-point values;
the gate fails unless the CSV describes exactly one launch and resolves at
least one requested counter. Its evidence is kept under `runs/ncu-gate-*`
and is diagnostic only.

To re-check a smoke directory later without a GPU, name it explicitly:

```bash
make smoke-verify SMOKE_DIR=runs/smoke-20260901T090000Z
```

Collect one pilot:

```bash
make campaign CAMPAIGN_KIND=pilot
```

Inspect it, then collect three independent finals without changing any source or build parameter:

```bash
make campaign CAMPAIGN_KIND=final
make campaign CAMPAIGN_KIND=final
make campaign CAMPAIGN_KIND=final
```

Every invocation creates `runs/<UTC-ID>/` containing the three raw CSV files, a manifest and `SHA256SUMS`. A complete campaign contains:

- memory: 18 configurations, 512 MiB working set, 32 passes, 2000 ms warm-up, 30 repetitions;
- UMMA: 24 configurations, 1000 outer iterations, 10 warm-up launches, 30 repetitions;
- GEMM: five shapes × four candidates, two warm-up launches and ten measured launches.

The campaign is accepted only if it contains exactly 540 memory samples, 720 UMMA samples and 20 GEMM rows; all numerical checks must pass and every row must identify the expected clean commit and GPU UUID. A failed run remains visibly incomplete and cannot enter the analysis.

`make campaign CAMPAIGN_KIND=final CAMPAIGN_NCU=1` adds an optional Nsight Compute stage after the three benchmarks. It profiles six of the eighteen memory configurations — one bytes-in-flight size per stage count, for both methods — and all twenty-four UMMA configurations, and stores each `ncu --page raw --csv` export under `runs/<UTC-ID>/ncu/`. The stage is advisory: a missing profiler or a refusal to grant counter access is recorded as a reason and the campaign still completes. Those counters give the memory results a DRAM cross-check (`dram_read_ratio`, `hbm_classification`) and give the UMMA results the measured SM clock behind `estimated_tflops_per_sm`. Without them the analysis still runs and reports every counter-derived metric as `not_applicable`.

Use the same choice for all three final campaigns. `estimated_tflops_per_sm` is withheld unless every campaign carries a usable SM clock, so profiling two of three loses the ceiling figure entirely; the analyzer warns on stderr when coverage is uneven rather than failing.

Aggregate the three final IDs explicitly, in execution order:

```bash
make analyze \
  FINAL_CAMPAIGNS="20260901T090000Z 20260901T100000Z 20260901T110000Z" \
  ANALYSIS_OUT=results/new
```

The analyzer refuses pilots, duplicate IDs, incomplete matrices, changed hashes, mixed commits, mixed GPUs and mixed images. It computes each campaign's median first and only then descriptive statistics across the three campaigns; internal repetitions are never pooled as independent campaigns. It writes three summary CSV files, `summary.json`, three SVG figures and a hash-pinned manifest.

For development or diagnostics, the three direct single-sweep targets remain available:

```bash
make memory-run
make umma-run
make gemm-run
```

The launcher exposes only the requested GPU and refuses to run while it has an active compute process.

## Reference provenance

- final campaigns: `20260817T110330Z`, `20260817T111310Z`, `20260817T112011Z`;
- pilot excluded: `20260812T013848Z`;
- execution commit: `b08e45c2636a3ac17c94ad8b1368084914196d7a`;
- analysis-code commit: `2ef1ac52907c407dd43c41661382fc8d5673cce4`;
- accepted manifest SHA-256: `b95d17910f8384187ddc94afacc9081507858de1fb69292f5f3d73bf4cc2d6ac`.

`provenance/acceptance.json` is the detached acceptance attestation. `provenance/provenance.json` records source identities, published-result hashes, the one path-only Python adaptation made for this layout, and hashes of the preserved SASS.

New campaign manifests supersede this reference for the thesis dataset. The reference remains useful for checking whether the clean implementation preserves the earlier qualitative findings, but it is not part of the new three-campaign population.

## Limitations

- Three final campaigns support descriptive statistics, not significance testing.
- Effective memory-path GB/s is derived from logical useful bytes and kernel time.
- The new direct campaign does not claim DRAM bandwidth without a separate profiler capture.
- The new UMMA analysis reports clock-independent FLOP/cycle/SM, not a measured whole-device peak.
- All GEMM measurements are hot-cache and lack GEMM-level profiling.
- The tested grids ended before a clear memory or UMMA plateau was observed.

## License

BSD 3-Clause. See `LICENSE`.
