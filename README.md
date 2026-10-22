# NVIDIA B300 GEMM Microbenchmarks

Five experiments characterize how data movement, matrix instructions, device
scaling, tile geometry and input precision determine GEMM performance on one
NVIDIA B300 with 148 streaming multiprocessors (SMs).

## Results

| Experiment | Main result |
|---|---|
| I - Global-to-shared transfers | LDGSTS has the higher mean in 8/9 tested pairs. Peak logical rates are 7.024 TB/s for LDGSTS and 6.961 TB/s for unicast TMA. |
| II - Isolated BF16 UMMA | The best one-SM sequence reaches 8098.1 FLOP/cycle/SM. A two-SM unit delivers 1.984 times the total work/cycle, with slightly lower throughput per SM. |
| III - Whole-device scaling | One-SM and two-SM organizations reach 2.109 and 2.103 PFLOP/s. Effective loop clocks decrease by about 7-8%, while work/cycle changes little. Sustained operating power is about 1.1 kW for both. |
| IV - BF16 GEMM geometry | Widening the cooperative tile from 256x128 to 256x256 improves all five shapes by 1.017-1.594x and reaches 78.29-96.05% of cuBLASLt throughput. |
| V - Input precision | With a fixed 256x128 CuTe tile, FP8 gives 1.75-2.28x BF16 throughput and NVFP4 2.41-4.06x. Peak NVFP4 rates are 5.615 PFLOP/s for CuTe DSL and 7.504 PFLOP/s for cuBLASLt. |

![BF16 GEMM throughput across configurations and shapes](results/gemm_comparison.svg)

NP1 is non-persistent with one thread block (CTA); P1 is persistent with one
CTA. P2a and P2b are persistent two-CTA kernels with 256x128 and 256x256 tiles.

The central result is a tile-supply model. For BF16 and exact tile multiples,
one supply of each input to a joint output tile predicts

```text
TMA read bytes = 2 M N K (1/tile_m + 1/tile_n)
```

The 256x256 tile predicts two-thirds of the 256x128 tile's TMA reads. All three
profiled shapes match this prediction exactly. DRAM locality remains relevant:
in the tall shape, the wider CuTe tile requests less TMA than cuBLASLt yet
reads more DRAM and remains slower. Tile size also changes resource use and
scheduling, so TMA volume alone does not determine execution time.

Full [CSV summaries and figures](results/) include
[scaling and loop clocks](results/umma_device_scaling.svg),
[sustained power](results/umma_power.svg),
[precision](results/precision_comparison.svg) and
[paired timing diagnostics](results/timing_comparison.csv).

## Data and environment

The dataset is **study-20261021T202311Z**, acquired on 21 October 2026.
Its nine CSV and seven SVG analysis products reproduce byte-for-byte.

- [Raw archive](data/study-20261021T202311Z.tar.gz)
- [SHA-256](data/study-20261021T202311Z.tar.gz.sha256)
- [Execution log](data/study-20261021T202311Z.log)
- [Measurement protocol](docs/methodology.md) and [source provenance](docs/provenance.md)

The measured source is commit `5f6902c536a2d35183d2d2cbe0ec2561364f4884`.
The dataset records that identity and the GPU/software fingerprint.
The B300 SXM6 AC has ECC enabled, dynamic clocks and a 1100 W power limit.
`VERSIONS.env` pins the CUDA 13.1 container digest, `sm_103a` target,
CUTLASS source and Python packages. The acquisition used driver 610.43.02,
Nsight Compute 2025.4, PyTorch 2.10.0+cu130 and CuTe DSL 4.6.1.

## Reproduce the analysis without a GPU

Requirements: Git and Python 3.12. From a fresh checkout:

```bash
git clone https://github.com/Antoniomv7/gb300-microbench.git
cd gb300-microbench

(cd data && sha256sum -c study-20261021T202311Z.tar.gz.sha256)
mkdir -p runs
reproduce_dir="$(mktemp -d runs/reproduce.XXXXXX)"
tar -xzf data/study-20261021T202311Z.tar.gz -C "$reproduce_dir"
python3 analysis/analyze.py \
  --study "$reproduce_dir/runs/study-20261021T202311Z" \
  --output "$reproduce_dir/analysis"
```

The 16 CSV/SVG products match `results/`. Analysis metadata records the new
input paths and creation time. Print PDF figures and their plotting source
are also supplied with the manuscript project.

## Acquire a new study on a B300

The host needs an NVIDIA driver, Docker with NVIDIA Container Toolkit,
Make and Python. Nsight Compute profiling must be permitted. Select an idle
B300; the example below uses device index 0.

```bash
make check-static
make image
export BLACKWELL_GPU_INDEX=0

pilot_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
make pilot STAMP="$pilot_stamp"
make final-study PILOT="runs/pilot-${pilot_stamp}"
```

The final target requires a passing pilot from the same committed source
and GPU/software environment. It runs three I-IV campaigns, three precision
repetitions, six sustained-power windows, nine GEMM profiles and the analysis.
Each acquisition creates a new directory under `runs/`. Use `make help` for
individual experiments and the [protocol](docs/methodology.md) for parameters.

## Interpretation and implementation

Numerical checks, compilation, allocation and profiling stay outside primary
timing. IV-V use CUDA Graph replay with paired ordinary-host/event diagnostics
on the same operands and stream. II-III keep internal-loop and whole-kernel
intervals distinct. Power is integrated over sustained 30 s windows after
10 s warm-up. Whiskers show the observed minimum and maximum of three
campaigns or repetitions; SD and CV are retained in the CSVs.

The findings describe one GPU, dynamic clocks, hot-cache reuse, small-integer
operand patterns and a limited configuration grid. Experiment I uses 256 B-wide
TMA boxes without swizzle, 2-32 KiB each; its comparison applies to that copy
geometry. IV and V have separate matched BF16 acquisitions under a common
timing policy. Selected CuTe configurations and first-supported cuBLASLt
heuristics do not establish library-wide performance ceilings.

`memory_paths/`, `umma_throughput/`, `gemm_comparison/` and
`precision_comparison/` implement the five experiments. `scripts/` handles
acquisition, `analysis/` generates summaries and SVGs, and `tests/` checks the
protocol on the CPU. NVIDIA CUTLASS supplies the pinned GEMM examples;
SemiAnalysis microbenchmarks are acknowledged as the methodological reference
for the local mechanisms. See [provenance](docs/provenance.md).

BSD 3-Clause; see [LICENSE](LICENSE).
