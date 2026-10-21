# Tutor review: protocol v2 and controlled closure

This protocol was executed in study-20261021T202311Z on 21 October 2026.
The B300 pilot passed on source commit 5f6902c and all final acquisition stages
completed on that exact source/environment. Independent CPU regeneration
reproduced all nine CSVs and seven SVGs. The archive, checksum and execution
log are linked from the repository README. The rules below define the
measurement design and how to reproduce a new population.

## Fixed scope

| Tutor comment | Change | Evidence required before closing |
|---|---|---|
| Power values came from short blocks and an averaging sensor | Separate sustained GPU-board power experiment, 3 windows per method | At least 10 s warm-up and 30 s measured; sensor names, full traces, duration, work count and device-event duty fraction |
| Scaling mixes frequency and work per cycle | Co-interval cycle/time reads in II and every timed III CTA | Positive intervals; timestamp brackets below 1%; distinct active SM IDs; raw arithmetic matches launch summaries |
| Connect 8192 FLOP/cycle/SM to nominal device peak | Explicit dimensional calculation in README | 148 SM × 8192 × 1.855799 GHz ≈ 2.25 PFLOP/s; label the implied frequency as nominal |
| A 256×256 P2 tile would test the tile hypothesis | Add tile variant to IV and traffic profile | All five IV shapes pass; both tiles and cuBLASLt captured on three profile shapes |
| Python launches may enter CUDA-event intervals | Common graph primary and paired host-event diagnostic in IV/V | Same operand workspace and explicit non-default stream; capture supported for both implementations; retain all time pairs |
| Compare IV with NCU duration | Add host/profile and graph/profile duration ratios | Nine validated captures; duration is diagnostic under separate profiling and dynamic clocks |
| SemiAnalysis attribution | Add source links and distinguish implementation/reference roles | See provenance.md; original upstream revision remains unrecorded |
| Repository final tag points at old results | Preserve current evidence during acquisition, publish one canonical reviewed release after acceptance | One code/result/source mapping, raw archive checksum, regenerated tables/figures and matching manuscript |

Experiment I is repeated with its existing geometry. The memory experiment is a controlled
microbenchmark of that geometry, not evidence that LDGSTS is generally preferable to TMA.
At 16 KiB in flight and 8 stages the TMA box is 256 B × 8 rows = 2 KiB, without swizzle.
All stage-box sizes are 2–32 KiB across this grid. A realistic GEMM TMA/swizzle sweep would
be a separate research extension and is outside this closure study.

## Timing intervals and equations

Let a cycle read have surrounding global timestamps `before_ns` and `after_ns`. Its time is
estimated by their midpoint. The sum of the start/end bracket widths is saved as a conservative
timestamp uncertainty bound. The acceptance threshold is 1% of the measured loop interval.
This bound does not quantify all effects of instrumentation on execution.

For one timed SM interval, with elapsed cycles C and elapsed nanoseconds T:

```
effective_sm_clock_mhz = 1000 C / T
loop_flops_per_cycle_sm = normalized work / C
loop_tflops_per_sm = normalized work / (1000 T)
```

Experiment II observes the issuing SM; the 2-SM work is normalized by the two participating
SMs. Experiment III saves each CTA's own cycle/time pair, including the partner SMs. For a
2-SM pair, work is apportioned equally for normalization; this does not claim that each SM
issues an independent set of UMMA instructions. The 2-SM loop includes its cooperative waits.

The aggregate loop-device rate uses `max(end_ns) - min(start_ns)` across all timed CTAs.
The primary III rate uses whole-kernel CUDA events, including setup and readback. Clock,
work/cycle and loop-rate ratios are descriptive diagnostics; a median clock times a mean
work/cycle is not an exact algebraic decomposition of event-based scaling efficiency.
External `nvidia-smi` readings must not be substituted for an isolated active SM clock.

The pilot compares separate instrumented/control launches. A duration ratio outside
0.8–1.2 stops acquisition for investigation. With dynamic clocks this is a coarse functional
check, not a calibrated estimate of timestamp instruction overhead.

IV and V use 5 warm-up and 20 timed launches per timing method. CUTLASS's pinned timing
helper captures repeated launches into a CUDA Graph for the primary result. The host-event
diagnostic calls the same helper without graph capture. The timing adapter creates the
workspace once and supplies that same object to both methods. Allocation, compilation and
graph capture occur before timed replay. Any graph capture failure is an error.
The NVFP4 example's explicit `cutlass_torch.default_stream()` selection is temporarily
routed to `current_stream()` during its `run()` call, then restored. Source files and
the kernel implementation stay pinned; timing and stream selection are local adapters.
Timing order alternates by IV campaign and by V repetition. III configuration order reverses
in campaign 2; sustained-power method order is AB, BA, AB. This modest balancing does not
make three repetitions a statistical inference study.

NCU profiles one ordinary validated launch after warm-up. Compare its duration with the
saved graph and host-event durations; keep both the profile's clock and cache settings.
Their differences cannot be attributed exclusively to Python because profiling, replay,
scheduling and dynamic clock behavior also differ between acquisitions.

## Sustained power acceptance

The native III executable repeats the full-device loop during warm-up and measurement.
Validation, copies and profiling occur outside the window. It records host start/end,
monotonic duration, launch count, active SM count, per-launch iteration count, total work
and summed CUDA-event duration. Throughput here is total work divided by the sustained wall
duration; the device-event fraction must be at least 90% (and no more than 1.01).

The sampler prefers `power.draw.average`, falls back explicitly to `power.draw`, and records
`power.draw.instant` as an optional diagnostic if the driver supplies it. There is no
substitution of a missing power value with zero. A 50 ms polling interval does not change
the power sensor's averaging window. Readings describe GPU-board operating power.

Every trace must bracket both measurement boundaries. Analysis integrates linearly
interpolated samples, clipped to those boundaries, and rejects gaps above 0.5 s or non-finite
values. It recomputes means from the archived trace rather than trusting the summary alone.
The absolute relative difference of the two half-window means must be ≤5% for power and
≤2% for the device-reported SM clock. Every final window must pass; retain failed traces.
If a longer warm-up or different window is needed, amend and commit the protocol, rerun the
pilot, then start a new coherent study. Do not relax thresholds after choosing results.

The pilot's 2 s/3 s power probe only checks acquisition and integration. Its stationarity
is reported but is not a final power result. This procedure supports sustained operating
power comparisons; it is not a calibrated measurement of the energy of a subsecond kernel.

## Tile test and interpretation

IV has NP1 (128×128), P1 (128×128), P2 (256×128), P2 (256×256) and cuBLASLt, on all five
existing shapes. Profiles cover 4096³, 8192³ and 32768×512×4096 for the last three variants.
For BF16, one supply of each joint output tile along K predicts:

```
Q_TMA = 2 M N K (1/tile_m + 1/tile_n) bytes
```

All tested dimensions are exact tile multiples. Moving 256×128 to 256×256 predicts two-thirds
of the TMA read bytes. The calibrated L2-to-SM TMA counter tests this prediction. DRAM traffic
is measured separately because cache locality can change independently. A timing improvement
would strengthen the explanation but cannot by itself isolate the cause: the tile also
changes storage, pipeline and scheduling properties. A neutral or negative timing result
is retained and constrains the hypothesis.

## Accept and publish

1. Run `make check-static`, build the pinned image, clean/recompile and execute `make pilot`.
   Review its report, correctness, raw clocks, instrumentation/control ratios, timing pairs,
   power sensor identities and all three profile captures. Do not include pilot rows in finals.
2. On the exact same committed sources and environment, use `make final-study PILOT=...`.
   The gate checks source commit, protocol and GPU/software/power-limit fingerprint.
3. Require three complete I–IV campaigns (540/720/120/25 raw primary rows each), 54 V timing
   rows with all operand/validation/plan evidence, six accepted sustained-power windows and
   nine validated GEMM captures. Recompute the summaries once from the saved archive.
4. Review outliers, timing diagnostics, measured frequency changes, TMA predictions and new
   conclusions. Keep interpretation conditional on the data. No source changes halfway
   through a population; an amendment gets a new pilot and study identity.
5. Export the accepted SVG figures to PDF for the manuscript. Use minimum/maximum whiskers
   consistently and identify them in captions. Explain separate IV/V BF16 acquisitions and
   common timing policy; do not silently force their cuBLASLt values to coincide.
6. Update the manuscript's numerical tables, plots, conclusions, abstract and limitations from
   the new summaries. Structural and bibliography/format changes can proceed independently.
7. Publish code and accepted summaries in one final commit, with the raw study archive and
   SHA-256 checksum attached to its canonical release. Regenerate SASS from that build if
   included. Cite that release/tag throughout the manuscript. Preserve the old tag targets
   until existing links and release assets have been checked; do not move old tags to hide
   the mismatch. If Zenodo is used, archive this accepted release and cite its DOI.

The completed acquisition and current summaries are published together on this review branch. Historical SASS and tag targets remain explicitly historical; an optional DOI must refer to the accepted archive and its recorded source identity.
