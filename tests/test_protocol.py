"""CPU checks of scientific identities, paired workspaces and acquisition grain."""

import copy
import csv
import json
import math
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from analysis import analyze, figures
from gemm_comparison.gemm_comparison import CANDIDATES, SHAPES
from scripts.benchmark_timing import PairedTimer, example_current_stream
from scripts.power_measure import weighted_mean, window_summary
from scripts import profile_gemm
from precision_comparison import precision_comparison as precision


def fixture():
    data = {key: [] for key in analyze.EXPECTED_ROWS}
    for method in ("ldgsts", "tma"):
        for stages in (2, 4, 8):
            for size in (16, 32, 64):
                for sample in range(30):
                    data["memory_paths"].append(dict(
                        method=method, stages=str(stages), bytes_in_flight_per_sm=str(size*1024),
                        sample_index=str(sample), effective_gbps=6000.0))
    for method, group in (("umma_1sm", 1), ("umma_2sm", 2)):
        for n in (64, 128, 256):
            for depth in (4, 16, 64, 256):
                work = 2*128*group*n*16*1000*depth
                cycles = max(100000, work//(8000*group))
                ns = cycles//2
                for sample in range(30):
                    data["umma_throughput"].append(dict(
                        method=method, n=str(n), depth=str(depth), sample_index=str(sample),
                        cta_group=str(group), iterations="1000", total_flops=work, elapsed_cycles=cycles,
                        loop_elapsed_ns=ns, timer_bracket_ns=16,
                        effective_sm_clock_mhz=1000*cycles/ns, loop_tflops=work/ns/1000,
                        flops_per_cycle=work/cycles, correctness="OK"))
    loops, telemetry = [], []
    for method, group in (("umma_1sm", 1), ("umma_2sm", 2)):
        for scale in ("isolated", "device_scale"):
            planned = group if scale == "isolated" else 8
            wsm, cycles, ns = 268435456000, 33554432, 16777216
            span = ns+(planned-1)*10
            for sample in range(30):
                host = 10*len(data["umma_device_scaling"])+1
                telemetry.append(dict(unix_s=host+0.005, sm_clock_mhz=2000, temperature_c=60))
                data["umma_device_scaling"].append(dict(
                    method=method, scale=scale, sample_index=str(sample), cta_group=str(group), iterations="1000",
                    planned_active_sm_count=str(planned), hardware_sm_count="8",
                    observed_unique_sm_count=str(planned), work_unit_count=str(planned//group),
                    cluster_count=str(planned//group if group == 2 else 0),
                    instrumented="1", kernel_time_ms=ns/1e6+0.01,
                    total_tflops=wsm*planned/(ns/1e6+0.01)/1e9,
                    tflops_per_planned_active_sm=wsm/(ns/1e6+0.01)/1e9,
                    median_effective_sm_clock_mhz=2000,
                    mean_loop_flops_per_cycle_sm=wsm/cycles, loop_device_span_ns=span,
                    loop_total_tflops=wsm*planned/span/1000,
                    host_start_unix_s=host, host_end_unix_s=host+0.02, correctness="OK"))
                for block in range(planned):
                    start = 1000000000+block*10
                    loops.append(dict(
                        method=method, scale=scale, sample_index=str(sample), block=str(block),
                        smid=str(block), elapsed_cycles=cycles, loop_elapsed_ns=ns,
                        timer_bracket_ns=16, start_ns=start, end_ns=start+ns,
                        effective_sm_clock_mhz=2000, flops_per_cycle_sm=wsm/cycles))
    for i, shape in enumerate(SHAPES):
        for j, candidate in enumerate(CANDIDATES):
            graph = 0.1+j*0.01
            tile, cluster = candidate.get("tile", ("", "")), candidate.get("cluster", ("", ""))
            data["gemm_comparison"].append(dict(
                shape_index=str(i), candidate_index=str(j), variant=candidate["variant"],
                method=candidate["method"], shape_id="x".join(map(str, shape)),
                **dict(zip(("m", "n", "k", "l"), map(str, shape))),
                tile_m=tile[0], tile_n=tile[1], cluster_m=cluster[0], cluster_n=cluster[1],
                kernel_time_ms=graph, host_event_kernel_time_ms=graph*1.1,
                host_to_graph_time_ratio=1.1, timing_method="cuda_graph",
                tflops=2*math.prod(shape)/graph/1e9, throughput_ratio_vs_cublaslt=0.14/graph,
                correctness="PASS"))
    return data, loops, telemetry


class PowerTests(unittest.TestCase):
    def trace(self, field="power_average_w"):
        return [{"unix_s": i/4, field: 10+2*i/4} for i in range(5)]

    def test_clips_window_edges(self):
        self.assertAlmostEqual(weighted_mean(self.trace(), 0.125, 0.875, "power_average_w"), 11.0)

    def test_missing_coverage_rejected(self):
        with self.assertRaises(ValueError):
            weighted_mean(self.trace(), -0.1, 0.8, "power_average_w")

    def test_large_sampling_gap_rejected(self):
        with self.assertRaises(ValueError):
            weighted_mean([{"unix_s": 0, "x": 1}, {"unix_s": 1, "x": 1}], 0, 1, "x")

    def test_nonfinite_rejected(self):
        rows = self.trace()
        rows[2]["power_average_w"] = float("nan")
        with self.assertRaises(ValueError):
            weighted_mean(rows, 0.1, 0.9, "power_average_w")

    def test_steady_window_and_optional_instant(self):
        trace = [dict(unix_s=i/4, power_average_w=1000, sm_clock_mhz=1900,
                      temperature_c=60, power_instant_w="") for i in range(9)]
        result = window_summary(trace, 0.1, 1.9)
        self.assertTrue(result["stationary"])
        self.assertEqual(result["mean_instant_power_w"], "")
        self.assertAlmostEqual(result["mean_power_w"], 1000)

    def power_study(self, path):
        raw = path/"raw"
        raw.mkdir()
        rows = []
        for method in ("umma_1sm", "umma_2sm"):
            for repetition in (1, 2, 3):
                trace = [dict(unix_s=i/4, power_average_w=1000, sm_clock_mhz=1900,
                              temperature_c=60, power_instant_w="") for i in range(122)]
                analyze.write_csv(raw/f"r{repetition}_{method}.telemetry.csv", trace)
                row = dict(method=method, repetition=repetition, scale="device_scale",
                           host_start_unix_s=0.1, host_end_unix_s=30.1, elapsed_s=30,
                           gpu_duty_fraction=0.99, device_elapsed_ms=29700,
                           launch_count=1000, iterations=1000, active_sm_count=148,
                           total_flops=2*128*256*16*256*1000*148*1000, correctness="OK",
                           **window_summary(trace, 0.1, 30.1))
                rows.append(row)
        analyze.write_csv(raw/"sustained.csv", rows)
        metadata = dict(parameters=dict(warmup_ms=10000, duration_ms=30000),
                        sensors=[dict(power_average_field="power.draw.average")]*6)
        return metadata, rows

    def test_sustained_analysis_recomputes_archived_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            metadata, rows = self.power_study(path)
            summary, points = analyze.power_results(path, metadata)
            self.assertEqual(len(summary), 2)
            self.assertEqual(summary[0]["mean_power_w"], 1000)
            self.assertIn("<svg", figures.power_figure(points))
            rows[0]["mean_power_w"] = 900
            analyze.write_csv(path/"raw/sustained.csv", rows)
            with self.assertRaises(ValueError):
                analyze.power_results(path, metadata)

    def test_short_power_windows_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            metadata, _ = self.power_study(path)
            metadata["parameters"]["duration_ms"] = 3000
            with self.assertRaises(ValueError):
                analyze.power_results(path, metadata)


class TimingTests(unittest.TestCase):
    def timer(self):
        timer = PairedTimer.__new__(PairedTimer)
        timer.reverse = False
        timer.last = None
        self.calls = []
        def original(callable, **kwargs):
            self.calls.append(kwargs)
            return 10 if kwargs["use_cuda_graphs"] else 15
        timer.original = original
        return timer

    def test_same_workspace_is_used_by_both_methods(self):
        timer, generated, workspace = self.timer(), [], object()
        def factory():
            generated.append(1)
            return workspace
        value = timer.benchmark(lambda: None, stream=7, workspace_generator=factory,
                                warmup_iterations=5, iterations=20)
        self.assertEqual(value, 10)
        self.assertEqual(len(generated), 1)
        self.assertTrue(all(c["kernel_arguments"] is workspace for c in self.calls))
        self.assertEqual(timer.last["host_to_graph_time_ratio"], 1.5)

    def test_reverse_order(self):
        timer = self.timer()
        timer.reverse = True
        timer.benchmark(lambda: None, stream=7, kernel_arguments=object(),
                        warmup_iterations=5, iterations=20)
        self.assertEqual([c["use_cuda_graphs"] for c in self.calls], [False, True])

    def test_default_stream_and_multiple_workspaces_rejected(self):
        timer = self.timer()
        for kwargs in ({"stream": 0}, {"stream": 7, "workspace_count": 2}):
            with self.assertRaises(ValueError):
                timer.benchmark(lambda: None, warmup_iterations=5, iterations=20, **kwargs)

    def test_example_stream_helper_is_restored_on_failure(self):
        original = lambda: 0
        helper = SimpleNamespace(default_stream=original, current_stream=lambda: 7)
        with self.assertRaises(RuntimeError):
            with example_current_stream(SimpleNamespace(cutlass_torch=helper)):
                self.assertEqual(helper.default_stream(), 7)
                raise RuntimeError("example failed")
        self.assertIs(helper.default_stream, original)

    def test_example_requires_non_default_current_stream(self):
        helper = SimpleNamespace(default_stream=lambda: 0, current_stream=lambda: 0)
        with self.assertRaises(ValueError):
            with example_current_stream(SimpleNamespace(cutlass_torch=helper)):
                pass


class AcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.loops, cls.telemetry = fixture()

    def test_complete_configuration_grain(self):
        analyze.validate_campaign(self.data, self.loops)

    def test_duplicate_or_missing_sm_observation_rejected(self):
        with self.assertRaises(ValueError):
            analyze.validate_campaign(self.data, self.loops[:-1])
        rows = copy.deepcopy(self.loops)
        group = next(r for r in rows if r["scale"] == "device_scale" and r["block"] == "1")
        group["smid"] = "0"
        with self.assertRaises(ValueError):
            analyze.validate_campaign(self.data, rows)

    def test_corrupted_clock_or_tile_rejected(self):
        rows = copy.deepcopy(self.loops)
        rows[0]["effective_sm_clock_mhz"] = 9999
        with self.assertRaises(ValueError):
            analyze.validate_campaign(self.data, rows)
        data = copy.deepcopy(self.data)
        data["gemm_comparison"][0]["tile_n"] = 17
        with self.assertRaises(ValueError):
            analyze.validate_campaign(data, self.loops)

    def test_uncertainty_rejected(self):
        data = copy.deepcopy(self.data)
        data["umma_throughput"][0]["timer_bracket_ns"] = 999999
        with self.assertRaises(ValueError):
            analyze.validate_campaign(data, self.loops)

    def test_environment_change_rejected(self):
        base = dict(gpu_uuid="gpu", source_commit="commit", environment_fingerprint="a")
        analyze.same_acquisition([base, base])
        with self.assertRaises(ValueError):
            analyze.same_acquisition([base, {**base, "environment_fingerprint": "b"}])

    def test_legacy_and_pilot_data_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for protocol, pilot in ((1, False), (2, True)):
                (path/"metadata.json").write_text(json.dumps(
                    dict(completed_utc="now", protocol_version=protocol, pilot=pilot)))
                with self.assertRaises(ValueError):
                    analyze.read_metadata(path)

    def test_new_summaries_and_figures(self):
        record = dict(data=self.data, telemetry=self.telemetry, profile={"cases": []})
        records = [record, record, record]
        for function, figure in ((analyze.memory_results, figures.memory_figure),
                                 (analyze.umma_results, figures.umma_figure),
                                 (analyze.scaling_results, figures.scaling_figure),
                                 (analyze.gemm_results, figures.gemm_figure)):
            rows, summary = function(records)
            image = figure(summary)
            self.assertIn("<svg", image)
            self.assertTrue(rows)
            self.assertNotIn("estimated_tflops_per_sm", rows[0])
            self.assertNotIn("scaling_efficiency_freq_normalized", rows[0])
        self.assertEqual(len(analyze.timing_results(records, None)), 25)


class PrecisionTests(unittest.TestCase):
    def records(self):
        repetitions, validation, operands, plans = [], [], [], []
        for i, shape in enumerate(precision.SHAPES):
            for name in precision.FORMATS:
                context = dict(shape_index=i, shape_id="x".join(map(str, shape)), precision=name,
                               **dict(zip(("m", "n", "k", "l"), shape)))
                for operand_set in ("validated", "timed_1", "timed_2", "timed_3"):
                    operands.append(dict(**context, operand_set=operand_set,
                                         represented_exactly=True, **{
                        f"{k}_bytes_identical_to_cutedsl": True for k in ("a", "b", "a_scale", "b_scale")}))
                    stages = [("cutedsl", "after_example_run"), ("cublaslt", "before_timing")]
                    if operand_set != "validated":
                        stages.append(("cublaslt", "after_timing"))
                    for implementation, stage in stages:
                        validation.append(dict(**context, operand_set=operand_set,
                            implementation=implementation, stage=stage, status="PASS", bit_exact=True))
                for method in precision.IMPLEMENTATIONS:
                    for repetition in (1, 2, 3):
                        repetitions.append(dict(**context, implementation=method, repetition=repetition,
                            operand_set=f"timed_{repetition}", validation="PASS", bit_exact=True,
                            timing_method="cuda_graph", kernel_time_us=100.0,
                            tflops=2*math.prod(shape)/100/1e6, host_event_time_us=110.0,
                            host_to_graph_time_ratio=1.1))
                plans.append(dict(**context, same_algorithm_for_every_set=True,
                    identification_algorithm_matches=True, kernel_names=["test_kernel"],
                    algorithm=dict(check_status=0, algorithm_id=1, tile_id=1,
                                   stages_id=1, cluster_shape_id=1)))
        return repetitions, validation, operands, plans

    def test_complete_paired_precision_comparison(self):
        r, v, o, p = self.records()
        precision.check_records(r, v, o, p)
        self.assertEqual(len(analyze.precision_results(r, v)), 9)
        self.assertEqual(len(analyze.precision_comparison_results(r, v, p)), 18)
        self.assertEqual(len(analyze.timing_results([], (r, v, p, {}))), 18)

    def test_changed_workspace_or_timing_ratio_rejected(self):
        r, v, o, p = self.records()
        o[0]["a_bytes_identical_to_cutedsl"] = False
        with self.assertRaises(ValueError):
            precision.check_records(r, v, o, p)
        o[0]["a_bytes_identical_to_cutedsl"] = True
        r[0]["host_to_graph_time_ratio"] = 2
        with self.assertRaises(ValueError):
            precision.check_records(r, v, o, p)


class ProfileTests(unittest.TestCase):
    def test_both_p2_tiles_and_duration_comparison(self):
        data, _, _ = fixture()
        summary, _ = analyze.gemm_results([dict(data=data)]*3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            analyze.write_csv(path/"gemm.csv", summary)
            metrics = (*profile_gemm.DRAM_METRICS, *profile_gemm.TIMING_METRICS,
                       profile_gemm.L2_READ_METRIC)
            for case, _, _ in profile_gemm.CASES:
                with (path/f"{case}.csv").open("w", newline="") as output:
                    writer = csv.writer(output)
                    writer.writerow(["Kernel Name", "NVTX Push/Pop_Range", *metrics])
                    writer.writerow(["", "", "byte", "byte", "ns", "cycle/second", "byte"])
                    writer.writerow(["test_kernel", profile_gemm.NVTX_RANGE,
                                     100000, 100000, 100000, 1900000000, 100000])
                (path/f"{case}.worker.json").write_text(json.dumps(dict(validation=dict(
                    first_launch="PASS", after_profiled_launch="PASS"))))
            meta = dict(cache_state="hot", captures=[c for c, _, _ in profile_gemm.CASES],
                        l2_read_metric=dict(collected=True))
            rows = analyze.gemm_profile_results(path, "hot", path/"gemm.csv", meta)
            self.assertEqual(len(rows), 9)
            self.assertEqual(rows[0]["graph_to_profile_time_ratio"], 1.2)
            self.assertAlmostEqual(rows[1]["predicted_tma_read_bytes"]/
                                   rows[0]["predicted_tma_read_bytes"], 2/3)
            with self.assertRaises(ValueError):
                analyze.gemm_profile_results(path, "hot", path/"gemm.csv", {**meta, "captures": []})


if __name__ == "__main__":
    unittest.main()
