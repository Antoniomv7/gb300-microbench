# Historical study: 24 September 2026

This page preserves the published interpretation before protocol v2. Its methods, commands and limitations describe the earlier acquisition; use the [current protocol](review_protocol.md) for new measurements. The current README reports protocol-v2 results separately.

## Published results

`results/` holds the CSV summaries and SVG/PDF figures from the final study
`study-20260924T140749Z`, measured on GPU
`GPU-619f7fdc-5f98-8c37-fe89-0465d6130baf`. The complete console log is tracked as
[`study-20260924T140749Z.log`](../study-20260924T140749Z.log). Figures show means; whiskers show
the range of the three campaigns or repetitions. These measurements characterize this GPU,
software stack and configuration grid; they are not architectural peak specifications.

### Key findings

| Mechanism | Main observation |
|---|---|
| Global-to-shared movement | LDGSTS is faster than TMA in 8/9 tested configurations; both peak near 7.0 TB/s effective rate at 4 stages and 64 KiB in flight. |
| UMMA issue throughput | The best one-SM case reaches 8,101 FLOP/cycle/SM; a two-SM work unit delivers 1.982× the one-SM total throughput per cycle. |
| Device scaling | One-SM and two-SM work units reach 2,104.1 and 2,119.9 TFLOP/s across 148 SMs, retaining 91.7% and 93.2% of their isolated throughput per SM. |
| BF16 GEMM | The persistent 2-CTA CuTe DSL kernel is the fastest CuTe variant for every tested shape, reaching 50.3–95.0% of the corresponding cuBLASLt throughput. |
| Low precision | Relative to the CuTe DSL BF16 kernel, FP8 reaches up to 2.27× and NVFP4 up to 3.97× higher throughput on the tested shapes. |
| Experiment IV traffic profile (hot cache) | A tile-supply model exactly reproduces the TMA read volumes of P2 and, assuming larger joint CTA tiles, cuBLASLt. The larger DRAM traffic of P2 remains a separate locality question. |

### I. HBM-to-shared-memory paths

![Effective transfer rate for LDGSTS and TMA](https://raw.githubusercontent.com/Antoniomv7/gb300-microbench/5a192bdc35e4bf776d63c9e78e76d636cb35d066/results/memory_paths.svg)

LDGSTS reaches the higher effective rate in **8 of 9** stage and in-flight-byte configurations.
The highest means are **7,024.5 GB/s** for LDGSTS and **6,964.2 GB/s** for TMA, both at four stages
and 64 KiB in flight. At two stages and 64 KiB, TMA is marginally higher: **6,954.6 versus
6,951.8 GB/s**. More bytes in flight help both paths in this grid, while eight stages reduce
throughput substantially, most strongly for TMA. The DRAM counters read approximately one byte
per useful byte in all six captured configurations. Source:
[`results/memory_paths.csv`](https://raw.githubusercontent.com/Antoniomv7/gb300-microbench/5a192bdc35e4bf776d63c9e78e76d636cb35d066/results/memory_paths.csv).

### II–III. UMMA instruction throughput and device scaling

![Isolated BF16 UMMA throughput](https://raw.githubusercontent.com/Antoniomv7/gb300-microbench/5a192bdc35e4bf776d63c9e78e76d636cb35d066/results/umma_throughput.svg)

At N = 256 and depth 256, the isolated one-SM kernel reaches **8,101 FLOP/cycle/SM**. The two-SM
kernel reaches **8,028 FLOP/cycle/SM**, or **1.982×** the total throughput per cycle of the
one-SM kernel.

In the first campaign, the one-SM NCU capture records **17.308384 ms** for 268,435,456,000 FLOP:
work divided by duration gives **15.509 TFLOP/s/SM**, consistent with the **15.502 TFLOP/s/SM**
isolated reference measured with CUDA events in Experiment III. Dividing the 33,136,959 cycles
of the internal loop by that NCU duration gives approximately **1,914.5 MHz**, close to NCU's
minimum of **1,917.1 MHz** rather than its device-wide mean of **2,021.6 MHz**. This is a
consistency check across separate acquisitions and timing intervals.

![Measured BF16 UMMA throughput and scaling to 148 SMs](https://raw.githubusercontent.com/Antoniomv7/gb300-microbench/5a192bdc35e4bf776d63c9e78e76d636cb35d066/docs/umma_device_scaling.svg)

| Execution | Active SMs | Mean throughput | Measured scaling efficiency |
|---|---:|---:|---:|
| One-SM work units | 148 | 2,104.1 TFLOP/s | 91.7% |
| Two-SM work units | 148 | 2,119.9 TFLOP/s | 93.2% |

Two-SM work units deliver about **0.8%** more device throughput. `nvidia-smi` reports 2,032 MHz
for the isolated runs and means of about **1,915 MHz** and **1,931 MHz** at device scale.
These external samples do not establish the active SM's effective clock in the isolated runs,
so they cannot support a clock-normalized scaling conclusion. The measured ratios retain the
combined effects of frequency, work per cycle and coordination. Recording `%clock64` and
`%globaltimer` over the same interval in the scaling kernel would help distinguish them.

**Traceability:** `estimated_tflops_per_sm` in `umma_throughput.csv` and
`scaling_efficiency_freq_normalized` in `umma_device_scaling.csv` retain the original
clock-derived calculations. Those fields and the original scaling SVG/PDF in `results/` are
preserved to keep regeneration byte-identical; they are excluded from the conclusions above.
The figure displayed here is regenerated from the measured columns with
the historical commit’s `docs/plot_scaling.py` (Matplotlib required).
Sources: [`results/umma_throughput.csv`](https://raw.githubusercontent.com/Antoniomv7/gb300-microbench/5a192bdc35e4bf776d63c9e78e76d636cb35d066/results/umma_throughput.csv) and
[`results/umma_device_scaling.csv`](https://raw.githubusercontent.com/Antoniomv7/gb300-microbench/5a192bdc35e4bf776d63c9e78e76d636cb35d066/results/umma_device_scaling.csv).

### IV. BF16 GEMM implementation and shape

![BF16 CuTe DSL variants versus cuBLASLt](https://raw.githubusercontent.com/Antoniomv7/gb300-microbench/5a192bdc35e4bf776d63c9e78e76d636cb35d066/results/gemm_comparison.svg)

| GEMM shape (M × N × K) | Persistent 2-CTA CuTe DSL | cuBLASLt | CuTe DSL / cuBLASLt |
|---|---:|---:|---:|
| 4096 × 4096 × 4096 | 1,671.6 TFLOP/s | 1,758.8 TFLOP/s | 95.0% |
| 8192 × 8192 × 8192 | 1,447.3 TFLOP/s | 2,109.7 TFLOP/s | 68.6% |
| 16384 × 512 × 4096 | 814.9 TFLOP/s | 1,434.8 TFLOP/s | 56.8% |
| 32768 × 512 × 4096 | 758.2 TFLOP/s | 1,508.5 TFLOP/s | 50.3% |
| 512 × 16384 × 4096 | 1,270.3 TFLOP/s | 1,498.7 TFLOP/s | 84.8% |

The persistent 2-CTA kernel is the fastest of the three CuTe DSL variants on all five shapes, but
its distance to cuBLASLt depends strongly on matrix shape. The near-square 4096³ case reaches
**95.0%** of cuBLASLt, whereas the 32768 × 512 × 4096 case reaches **50.3%**. Source:
[`results/gemm_comparison.csv`](https://raw.githubusercontent.com/Antoniomv7/gb300-microbench/5a192bdc35e4bf776d63c9e78e76d636cb35d066/results/gemm_comparison.csv).

### V. BF16, FP8 and NVFP4 GEMM

![CuTe DSL throughput by input precision](https://raw.githubusercontent.com/Antoniomv7/gb300-microbench/5a192bdc35e4bf776d63c9e78e76d636cb35d066/results/precision_comparison.svg)

| GEMM shape (M × N × K) | BF16 CuTe DSL | FP8 CuTe DSL | NVFP4 CuTe DSL |
|---|---:|---:|---:|
| 4096 × 4096 × 4096 | 1,681.3 | 2,936.8 (1.75×) | 4,029.0 (2.40×) |
| 8192 × 8192 × 8192 | 1,458.5 | 3,125.0 (2.14×) | 5,572.2 (3.82×) |
| 32768 × 512 × 4096 | 756.7 | 1,718.5 (2.27×) | 3,001.5 (3.97×) |

Throughputs are in TFLOP/s; parentheses give the speedup over BF16 on the same shape. Lower
precision raises throughput substantially, but the gain is shape-dependent and remains below the
ratio of the nominal dense peaks in several cases. NVFP4 also changes the operand representation
through block scaling, so its numerical and data-movement behavior is not directly equivalent to
BF16 or FP8.

![Matched CuTe DSL and cuBLASLt comparisons by precision](https://raw.githubusercontent.com/Antoniomv7/gb300-microbench/5a192bdc35e4bf776d63c9e78e76d636cb35d066/results/precision_cutedsl_vs_cublaslt.svg)

On identical operands, CuTe DSL reaches **49.9–95.1%** of cuBLASLt's throughput in BF16,
**64.3–92.1%** in FP8 and **73.3–84.2%** in NVFP4. Every timed output of both implementations
passed validation. The highest measured cuBLASLt throughput in this comparison is **7,545.6
TFLOP/s** for NVFP4 on 8192³. Sources:
[`results/precision_comparison.csv`](https://raw.githubusercontent.com/Antoniomv7/gb300-microbench/5a192bdc35e4bf776d63c9e78e76d636cb35d066/results/precision_comparison.csv) and
[`results/precision_cutedsl_vs_cublaslt.csv`](https://raw.githubusercontent.com/Antoniomv7/gb300-microbench/5a192bdc35e4bf776d63c9e78e76d636cb35d066/results/precision_cutedsl_vs_cublaslt.csv).

### IV. Memory traffic results (hot cache)

In the six profiled launches, the persistent 2-CTA CuTe DSL kernel reads more DRAM bytes than
cuBLASLt at 8192 × 8192 × 8192 (**3.33 versus 1.16 GB**) and at 32768 × 512 × 4096
(**1.08 versus 0.28 GB**). The calibrated L2-to-SM TMA counter also records more bytes for CuTe
DSL at those shapes. These are the same shapes where the BF16 throughput gap is largest, which
supports a data-movement interpretation, but the aggregate counters do not identify which operand
was reread and do not prove that traffic alone caused the performance difference.
`compulsory_read_bytes` is the combined A and B size; with hot caches, DRAM reads can be smaller
than this quantity. Source: [`results/gemm_profile.csv`](https://raw.githubusercontent.com/Antoniomv7/gb300-microbench/5a192bdc35e4bf776d63c9e78e76d636cb35d066/results/gemm_profile.csv).

For BF16, supplying each joint output tile of size *m*×*n* once along K predicts
**Q = 2MNK(1/m + 1/n) bytes** of TMA reads, when the matrix dimensions are exact tile multiples.
P2 uses joint tiles of 256×128. If the sizes in cuBLASLt's `nvjet` kernel names describe one CTA,
their `2cta` pairs form 256×256 tiles for the square shapes and 512×128 for the tall shape:

| Shape | P2 joint tile | P2 predicted = measured | Inferred cuBLASLt joint tile | cuBLASLt predicted = measured |
|---|---|---:|---|---:|
| 4096³ | 256×128 | 1,610,612,736 B (1.50 GiB) | 256×256 | 1,073,741,824 B (1.00 GiB) |
| 8192³ | 256×128 | 12,884,901,888 B (12.00 GiB) | 256×256 | 8,589,934,592 B (8.00 GiB) |
| 32768×512×4096 | 256×128 | 1,610,612,736 B (1.50 GiB) | 512×128 | 1,342,177,280 B (1.25 GiB) |

All six predictions match the TMA counters exactly. For cuBLASLt, this supports a larger-joint-tile
interpretation of the L2-to-SM gap; the undocumented names and aggregate counters do not uniquely
identify its implementation. Traversal order and cache reuse remain hypotheses for the separate
DRAM excess.

## Reproducing the analysis

The final study was executed as `study-20260924T140749Z`; its tracked
[`study-20260924T140749Z.log`](../study-20260924T140749Z.log) records the complete command sequence
and successful completion of the three campaigns, precision experiment, Experiment IV hot-cache
traffic profile and analysis step. The raw `runs/` directory is intentionally not tracked by Git. The complete final
study archive, together with its SHA-256 checksum, is published as an asset of the `tfm-final`
GitHub Release, so the published summaries can be regenerated without repeating the GPU campaign.

The `tfm-final` tag freezes the acquisition code, but its `results/` directory still contains an
earlier acquisition (for example, P2 on 4096³ is 1,679.7 TFLOP/s there). The Release archive and
the final summaries on `main` are the references for the thesis (1,671.6 TFLOP/s in that case).

To run the same protocol again and regenerate its summaries:

Use the acquisition and analysis revision recorded in the original archive. Current protocol-v2 analysis deliberately rejects legacy inputs.

`analysis/analyze.py --study` produces the seven CSV summaries and six SVG figures. The PDF files
in `results/` are derived exports of those six SVG figures. These original files and the analysis
code are unchanged by the interpretation corrections; `docs/umma_device_scaling.svg` is a
separate documentation figure drawn from the same CSV.

## Limitations

- One GPU in one system: the results describe this B300 SXM6 AC, driver, software stack and
  configuration grid, and are not architectural peaks.
- Clocks are not locked. Experiment III samples device-reported clock and power, without measuring
  the effective clock of each active SM. Power samples are indicative telemetry, not a calibrated
  energy measurement.
- GEMM inputs are small, exactly representable integers. In BF16, 0, ±1 and ±2 have all seven
  stored fraction bits zero. This restricted set of bit patterns may affect switching activity,
  power and effective frequency; the study does not quantify that dependence. Identical operands
  make the library comparisons fair for these inputs, while other distributions may change both
  absolute rates and their ratios.
- The GEMM measurements reuse operands across launches (hot caches), so the L2 can hold part of
  them.
- Experiment I's effective rate is useful bytes over kernel time, not a DRAM-bandwidth counter or
  a GEMM prediction. Experiment II measures FLOP/cycle/SM; work divided by NCU duration provides a
  separate whole-kernel diagnostic.
- Experiments IV and V compare specific CuTe DSL example kernels with cuBLASLt's first supported
  heuristic algorithm, without autotuning; they do not establish a ceiling for either library.
- The vendor peaks in `precision_comparison.csv` are nominal dense values.
- The Experiment IV traffic profile has one launch per case; aggregate counters cannot attribute
  rereads to an operand.
- Three campaigns or three repetitions support descriptive statistics, not inference.
