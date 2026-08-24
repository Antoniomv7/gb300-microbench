# GB300 Blackwell Microbenchmarks

Minimal, reproducible microbenchmarks for studying HBM-to-SMEM transfer paths, fifth-generation BF16 UMMA throughput, and CuTe DSL GEMM variants versus cuBLASLt on NVIDIA GB300. The repository can collect one pilot plus three new final campaigns and generate the tables and figures used by the thesis.

This is the compact publication layer of [gb300-gemm-anatomy](https://github.com/Antoniomv7/gb300-gemm-anatomy), frozen at commit [`86f2382`](https://github.com/Antoniomv7/gb300-gemm-anatomy/commit/86f2382fb92a957035c067ae725e9e25afacab6f). Development protocols, agent instructions, phase gates, campaign orchestration, static checkers, and audit scaffolding are intentionally excluded.

The original memory, UMMA and GEMM CUDA sources are preserved with identical executable code: every identifier, constant, template, inline-PTX block and command-line option is unchanged from the historical development build, so the compiled binaries measure exactly what they measured there. Only comments and message text were rewritten, to remove pointers into the development repository. The equivalence was checked by stripping comments from both versions and comparing the remaining token streams, which are identical. Whole-device UMMA scaling is implemented as a separate supplementary experiment.

## Experiments

| Directory | Question | Implementations |
|---|---|---|
| `memory_paths/` | How do equivalent LDGSTS and 2D unicast TMA paths behave over the tested pipeline grid? | `ldgsts.cu`, `tma.cu` |
| `umma_throughput/` | How do 1-SM and 2-SM BF16 `tcgen05.mma` kernels scale with N and instruction depth? | `umma_1sm.cu`, `umma_2sm.cu` |
| `gemm_comparison/` | How closely do three official CuTe DSL execution variants approach cuBLASLt on five shapes? | `gemm_comparison.py`, `cublaslt_bridge.cu` |
| `umma_throughput/` (supplementary) | How do the same 1-SM and 2-SM UMMA execution modes behave when replicated across the whole usable SM capacity? | `umma_device_scaling.cu`, `device_scaling.py` |

The CUDA kernels retain their integrated numerical validation. The GEMM comparison validates every candidate against one FP32 reference before timing it.

## Final experimental results

The tables and figures in `results/` summarize three final primary campaigns and three final supplementary UMMA campaigns. Both populations ran on the same NVIDIA B300 SXM6 AC with the same driver and container image. Their execution commits differ and are recorded separately below; the supplementary pilot is excluded from every statistic.

### HBM-to-SMEM paths

![LDGSTS and TMA effective transfer rates](results/figures/memory_paths.svg)

The reported GB/s is a timing-derived effective transfer rate (`useful_bytes / kernel_time`), not a direct DRAM-bandwidth counter and not GEMM traffic. LDGSTS was higher in eight of nine matched configurations; TMA was essentially equal and slightly higher only at 2 stages and 64 KiB in flight (mean ratio `1.00060`). The highest means were about `7.024 TB/s` for LDGSTS and `6.962 TB/s` for TMA. Every saturation candidate reached the largest tested value, so no plateau was observed inside the 16/32/64 KiB grid.

### BF16 UMMA throughput

![1-SM and 2-SM UMMA throughput](results/figures/umma_throughput.svg)

The metric is derived from validated operation counts and `%clock64` cycles. The selected per-SM ceiling candidate was `umma_1sm`, `N=256`, depth `256`, corresponding to a modeled mean of `16.354 TFLOP/s/SM` after applying the measured SM clock. The 2-SM/1-SM speedup reached `1.983x` at `N=256`, depth `256`; a `2.096x` value at `N=64`, depth `256` is preserved as a surprising diagnostic rather than treated as an architectural claim.

### Whole-device BF16 UMMA scaling

![Isolated and whole-device 1-SM and 2-SM UMMA throughput](results/figures/umma_device_scaling.svg)

At `N=256`, depth `256`, both device-scale configurations cover all 148 SMs; `umma_2sm` uses 74 simultaneously resident two-CTA clusters.

| Method | Launch scale | Active SMs | Total TFLOP/s | TFLOP/s/SM | Scaling efficiency |
|---|---|---:|---:|---:|---:|
| `umma_1sm` | isolated | 1 | 15.486 | 15.486 | — |
| `umma_2sm` | isolated | 2 | 30.686 | 15.343 | — |
| `umma_1sm` | whole device | 148 | 2279.568 | 15.402 | 99.46% |
| `umma_2sm` | whole device | 148 | 2259.351 | 15.266 | 99.50% |

The whole-device `umma_2sm` total is `0.89%` below `umma_1sm` at equal 148-SM coverage. Each efficiency uses its own same-campaign isolated baseline; these supplementary baselines use CUDA-event timing and the same shared-memory reservation as their whole-device launches, and are not interchangeable with the `%clock64`-based isolated sweep above. The largest coefficient of variation across the measured throughput configurations is `0.62%`.

### CuTe DSL versus cuBLASLt

![CuTe DSL and cuBLASLt GEMM comparison](results/figures/gemm_comparison.svg)

`persistent_2cta` was the stable best CuTe DSL variant for all five shapes.

| Shape `(M,N,K,L)` | CuTe DSL TFLOP/s | cuBLASLt TFLOP/s | CuTe DSL / cuBLASLt | Gap |
|---|---:|---:|---:|---:|
| `4096x4096x4096x1` | 1650.0 | 1749.1 | 94.33% | 5.67% |
| `8192x8192x8192x1` | 1440.8 | 2102.4 | 68.53% | 31.47% |
| `16384x512x4096x1` | 812.2 | 1430.3 | 56.79% | 43.21% |
| `32768x512x4096x1` | 758.2 | 1502.6 | 50.46% | 49.54% |
| `512x16384x4096x1` | 1270.2 | 1488.6 | 85.32% | 14.68% |

These are hot-cache measurements. No GEMM kernel was profiled with Nsight Compute, so the gap is not attributed to memory, Tensor Cores, scheduling, or any other single cause.

## Environment

Both published campaign populations used:

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

The benchmarks record the Git commit they ran at, and both `make smoke` and every campaign require a clean worktree, so **commit the repository before running anything**. The same commit, container image and physical GPU must be used within each campaign population.

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
  FINAL_CAMPAIGNS="20260822T222634Z 20260822T223050Z 20260822T223504Z" \
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

## Supplementary experiment: UMMA launch scale

The frozen `umma_throughput/` experiment varies the **instruction** dimension — `tcgen05.mma.cta_group::1` against `tcgen05.mma.cta_group::2`, swept over N and instruction depth — always on one CTA or one two-CTA cluster. This supplementary experiment varies a second, orthogonal dimension, **launch scale**, and holds the instruction dimension fixed at the already-identified ceiling shape `N=256`, depth `256`.

"All SM" is not a third UMMA instruction. PTX provides exactly two CTA groups, `cta_group::1` and `cta_group::2`; there is no whole-device UMMA opcode. The four configurations are two instruction modes crossed with two launch scales:

| Method | Scale | Launch | Work unit |
|---|---|---|---|
| `umma_1sm` | `isolated` | one CTA | one `128x256x16` UMMA |
| `umma_2sm` | `isolated` | one two-CTA cluster | one joint `256x256x16` UMMA |
| `umma_1sm` | `device_scale` | one CTA per usable SM | one `128x256x16` UMMA per SM |
| `umma_2sm` | `device_scale` | every simultaneously resident two-CTA cluster | one joint `256x256x16` UMMA per cluster |

Both launch scales are measured inside the **same** campaign, so every scaling efficiency is computed against an isolated baseline taken on the same GPU, at the same commit, minutes apart — never against a historical run at another commit.

`umma_device_scaling.cu` reuses the instruction descriptors, K-major shared-memory packing, TMEM allocate/fence/commit/wait/readback/deallocate/relinquish lifecycle, cluster-wide `cta_group::2` synchronization, validation pattern and per-work-unit operation count of the corresponding `N=256`, depth `256` frozen kernels. Only the launch geometry, the per-work-unit output indexing, the residency/SM-id diagnostics and the host timing differ. The frozen `umma_1sm.cu`, `umma_2sm.cu`, `benchmark.py`, the 24-configuration contract, `analysis/analyze.py` and `results/new` are untouched, and the new binary is a separate `make` goal so `make build` and `make sass` behave exactly as before.

### What it is, and what it is not

Device-scale UMMA is still a compute-focused microbenchmark: A and B are already resident in shared memory and stay there for the whole measured loop. It is **not** a full GEMM, **not** an HBM or DRAM-bandwidth benchmark, and **not** an NVIDIA architectural peak claim. The reported TFLOP/s comes from validated operation counts divided by whole-kernel CUDA-event time.

### How coverage is established

The SM count comes from `cudaGetDeviceProperties()` at run time and is recorded in every row; 148 SMs is never hardcoded, because Blackwell Ultra SKUs may expose a different count.

Each kernel is launched with a dynamic shared-memory reservation derived from the device's own `sharedMemPerMultiprocessor`, large enough that two CTAs cannot share an SM. `cudaOccupancyMaxActiveBlocksPerMultiprocessor` must then confirm exactly one resident CTA per SM, or the binary refuses to run — one CTA per SM is also what keeps two 256-column Tensor Memory allocations off the same SM. For the 2-SM arm, `cudaOccupancyMaxActiveClusters` gives the resident-cluster capacity and the grid is `2 x min(floor(SMs / 2), max_active_clusters)`, always an even number of blocks.

A matching grid alone is not treated as proof. Before any measurement, each configuration runs one untimed **residency probe** launch of the same kernel with the same geometry and reservation: every block arrives at a global counter and then spins, bounded by its own deadline, until it observes every other block having arrived. If all blocks report success then no block had exited when the last one arrived, so all of them were simultaneously resident. A timeout is recorded, never hidden, and never becomes a hang. Every launch also records the `%smid` of each block into an array indexed by **block position**, never by `%smid` itself, since SM IDs are not guaranteed to be contiguous.

Each row therefore carries `coverage_status`:

- `full_device_coverage` — residency proven, observed unique SMs equal the planned count, and the planned count equals the hardware SM count;
- `maximum_resident_coverage` — residency proven and fully observed, but fewer SMs than the device has (an odd SM count, or a cluster-occupancy limit); the unused SMs are reported in `unused_sm_count`;
- `incomplete_coverage` — residency or SM observation fell short; the analyzer refuses such a campaign rather than calling it "all SM";
- `isolated_unit` — the two isolated configurations.

### Timing

`%clock64` is a per-SM counter and is never the primary timer here. Every published sample is a whole-kernel CUDA-event interval: buffers and events are created first, the device is synchronized, the interval encloses exactly one kernel launch, and the readback plus host validation happen after it. Warm-up launches are discarded. Per-work-unit `%clock64` values survive only as the `diagnostic_clock64_cycles_min`/`_max` columns.

Each repetition executes all four configurations; even repetitions use the canonical order and odd repetitions the exact reverse, so neither UMMA method is always measured first as the device heats up. Every row records its own `execution_order`.

Correctness is mandatory: the whole output of every block or cluster is validated against a CPU reference before any measurement and again after every measured repetition, outside the timed interval. Any mismatch aborts before a timing row is written.

### Build, smoke, collect, analyze

```bash
make umma-scaling-build          # compiles only build/umma_throughput/umma_device_scaling
make umma-scaling-check          # GPU-free: syntax and CLI checks
```

`make umma-scaling-sass` disassembles the new binary into `build/sass/`.

Then choose one idle physical GPU and run the short validation:

```bash
export BLACKWELL_GPU_INDEX=7
make umma-scaling-smoke
```

The smoke runs the four-configuration `--self-test` and then a short measured run; it is never part of a statistic. As with the primary campaigns, **commit the repository before collecting evidence** — every campaign requires a clean worktree and records the commit in every row.

```bash
make umma-scaling-campaign UMMA_SCALING_KIND=pilot
make umma-scaling-campaign UMMA_SCALING_KIND=final
make umma-scaling-campaign UMMA_SCALING_KIND=final
make umma-scaling-campaign UMMA_SCALING_KIND=final
```

Each invocation creates `runs/umma_device_scaling/<UTC-ID>/` with `raw/umma_device_scaling.csv`, a manifest and `SHA256SUMS`. One campaign is 4 configurations x 30 repetitions = 120 rows, at 1000 outer iterations and 10 warm-up launches, and is accepted only if the four-configuration matrix is exact, every device-scale row has proven coverage, correctness is clean and every row carries the expected clean commit and GPU UUID. The raw schema is separate from `umma_throughput.csv`; nothing is appended to the frozen dataset.

```bash
make umma-scaling-analyze \
  UMMA_SCALING_FINALS="20260824T122229Z 20260824T122239Z 20260824T122247Z" \
  UMMA_SCALING_ANALYSIS_OUT=results/umma_device_scaling/final
```

The analyzer takes exactly three distinct final campaigns and requires one shared commit, GPU, container image, parameter set and launch geometry. It reduces each campaign's 30 repetitions to a median first and only then computes mean, median, sample standard deviation, minimum, maximum and CV across the three campaign-level medians; the 90 repetitions are never pooled as independent campaigns, and three campaigns support descriptive statistics, not significance testing. Scaling efficiency is computed separately per method, each against its own isolated baseline:

- 1-SM: `device_total / (isolated_1sm x active_1sm_blocks)`
- 2-SM: `device_total / (isolated_2sm x active_2sm_clusters)`

It also reports both device-scale totals, the `2sm / 1sm` total and per-active-SM ratios, and each method's gap from its own ideal linear projection. Values are never clamped, and totals are never compared without exposing the active-SM counts behind them: an unequal count sets `equal_active_sm_coverage` to false and raises a warning. Output goes to a separate location, `results/umma_device_scaling/<analysis-id>`, and never overwrites `results/new`; it contains a summary CSV, `summary.json`, one SVG figure with the isolated totals, device-scale totals and scaling efficiency on separate scales, a manifest and `SHA256SUMS`.

The published supplementary table and figure are `results/umma_device_scaling.csv` and `results/figures/umma_device_scaling.svg`; their measured values are summarized above.

### Relating this to the GEMM comparison

The device-scale figure is an instruction-issue ceiling with operands already in shared memory and nothing else competing. `gemm_comparison/` measures complete GEMMs, including operand movement, tiling, scheduling and epilogue. The useful relation is a fraction — how much of the device-scale UMMA ceiling a real GEMM converts — and is only interpretable when both come from the same GPU, driver and container image with unchanged relevant benchmark code. The published populations meet those conditions but ran at the distinct commits recorded below; their results remain separate measurements rather than one number. This comparison bounds headroom without attributing any GEMM gap to memory, Tensor Cores, scheduling, or any other single cause, and it is not a roofline.

## Published provenance

- primary final campaigns: `20260822T222634Z`, `20260822T223050Z`, `20260822T223504Z`;
- primary execution commit: `1f27767ec9af121103b6d489592a9fd9052bb1b4`;
- primary analysis manifest SHA-256: `1fce4eadaf9fc14c548c70128bbcee9979d73a34a7b629a985c9883c0a120a9b`;
- supplementary final campaigns: `20260824T122229Z`, `20260824T122239Z`, `20260824T122247Z`;
- supplementary pilot excluded: `20260824T122221Z`;
- supplementary execution commit: `060c4aeec4598babf3701c0dcfb74ea15f828585`;
- supplementary analysis manifest SHA-256: `244b93cac3f0f7b6e254867b8a281e70a1e65282288d3bc1ba2a284597283b6b`;
- shared physical GPU: `GPU-40e00845-d89c-1393-2c32-a2dca3ee9442`.

`provenance/provenance.json` records both campaign populations, their execution commits and manifest hashes, the shared container image, all eight published-artifact hashes, original source identities and preserved SASS hashes. `provenance/acceptance.json` remains the historical acceptance attestation of the source development project; it does not certify either newly published campaign population.

## Limitations

- Three final campaigns support descriptive statistics, not significance testing.
- Effective memory-path GB/s is derived from logical useful bytes and kernel time.
- The new direct campaign does not claim DRAM bandwidth without a separate profiler capture.
- The isolated UMMA sweep reports clock-independent FLOP/cycle/SM; whole-device TFLOP/s is a measured microbenchmark throughput, not an architectural peak.
- All GEMM measurements are hot-cache and lack GEMM-level profiling.
- The tested grids ended before a clear memory or UMMA plateau was observed.
- Launch scale is not a third UMMA instruction; the supplementary experiment crosses the two PTX CTA groups with two launch scales and makes no whole-device peak claim.

## License

BSD 3-Clause. See `LICENSE`.
