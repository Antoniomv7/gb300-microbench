#!/usr/bin/env python3
"""Focused checks for the GEMM profile and the extended precision comparison.

Run with `make test` (inside the pinned image) or `python3 -m unittest discover -s tests`.
The NVFP4 encoding tests need PyTorch and are skipped without it.
"""

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import check_diagnostics  # noqa: E402
import profile_gemm  # noqa: E402
from precision_comparison import precision_comparison as precision  # noqa: E402

TORCH = importlib.util.find_spec("torch") is not None


def ncu_export(kernels, metrics):
    """A raw-page Nsight Compute CSV: header, units, then one row per kernel."""
    header = ["ID", "Process ID", "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:"
              "Msg_Type:Msg", "Kernel Name", "Block Size", "Grid Size", *metrics]
    rows = [header, [""] * 6 + ["byte"] * len(metrics)]
    for index, (name, values) in enumerate(kernels):
        rows.append([str(index), "7", f'7  "<default domain>:{profile_gemm.NVTX_RANGE}:none"', name,
                     "(192, 1, 1)", "(2, 1, 74)", *(f"{values[metric]:.2f}" for metric in metrics)])
    with tempfile.TemporaryFile("w+", newline="") as buffer:
        csv.writer(buffer, quoting=csv.QUOTE_ALL).writerows(rows)
        buffer.seek(0)
        return buffer.read()


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_profile(directory):
    metrics = [*profile_gemm.DRAM_METRICS, *profile_gemm.TIMING_METRICS, profile_gemm.L2_READ_METRIC]
    captures, summary = [], []
    pairs = [(shape, variant) for shape in profile_gemm.SHAPES for variant in profile_gemm.VARIANTS]
    for index, (shape, variant) in enumerate(pairs):
        case = f"{index:02d}_{variant}"
        files = {kind: f"{case}.{kind}" for kind in ("ncu-rep", "csv", "log", "worker.json")}
        values = {metric: 1000.0 + position for position, metric in enumerate(metrics)}
        for kind, file in files.items():
            (directory / file).write_text(
                ncu_export([(f"kernel_{variant}", values)], metrics) if kind == "csv" else "")
        captures.append({"case": case, "shape_id": profile_gemm.shape_id(shape),
                         "variant": variant, "files": files, "status": "PASS",
                         "validation": "PASS", "kernel": {"name": f"kernel_{variant}"},
                         "metrics": values, "worker": {"validation": {
                             "first_launch": "PASS", "after_profiled_launch": "PASS"}}})
        summary.append({"shape_id": profile_gemm.shape_id(shape), "variant": variant,
                        "status": "PASS", "validation": "PASS", "kernel_name": f"kernel_{variant}"})
    index = {"state": "COMPLETE", "captures": captures, "protocol": {"metrics": metrics},
             "l2_read_metric": {"collected": True, "l2_read_to_useful": 1.0}}
    (directory / "index.json").write_text(json.dumps(index))
    write_rows(directory / "gemm_profile.csv", summary)
    return index


def write_precision(directory):
    shapes = ["x".join(map(str, shape)) for shape in precision.SHAPES]
    (directory / "metadata.json").write_text(json.dumps(
        {"state": "COMPLETE", "protocol": {"shapes": shapes, "formats": list(precision.FORMATS)}}))
    comparison, repetitions, validation, operands, plans, cute = [], [], [], [], [], []
    sets = check_diagnostics.OPERAND_SETS
    for shape in shapes:
        for name in precision.FORMATS:
            for implementation, value in (("cutedsl", 200.0), ("cublaslt", 250.0)):
                samples = [f"{value + index:.6f}" for index in range(3)]
                comparison.append({
                    "shape_id": shape, "precision": name, "implementation": implementation,
                    "comparison": "matched", "input_dtype": "x",
                    "scale_dtype": "E4M3" if name == "nvfp4" else "",
                    "scale_block_elements": "16" if name == "nvfp4" else "",
                    "accumulator_dtype": "x", "alpha": "1", "beta": "0", "kernel": "k",
                    "output_dtype": check_diagnostics.FP32_OUTPUTS[implementation],
                    **{f"repetition_{index}_tflops": sample
                       for index, sample in enumerate(samples, 1)},
                    "mean_tflops": f"{value + 1:.6f}", "stdev_tflops": "1.000000",
                    "cv_percent": "0.5", "mean_kernel_time_us": "10.0",
                    "throughput_ratio_vs_cublaslt": f"{(value + 1) / 251:.6f}",
                    "validation": "PASS"})
                repetitions += [{"shape_id": shape, "precision": name,
                                 "implementation": implementation, "repetition": index,
                                 "kernel_time_us": "10.0", "tflops": sample, "validation": "PASS"}
                                for index, sample in enumerate(samples, 1)]
                if implementation == "cutedsl":
                    cute.append({"shape_id": shape, "precision": name, "correctness": "PASS",
                                 **{f"repetition_{index}_tflops": sample
                                    for index, sample in enumerate(samples, 1)}})
            for operand_set in sets:
                operands.append({"shape_id": shape, "precision": name,
                                 "operand_set": operand_set, "represented_exactly": "True"})
                stages = [("cutedsl", "after_example_run"), ("cublaslt", "before_timing")]
                stages += [("cublaslt", "after_timing")] if operand_set != "validated" else []
                validation += [{"shape_id": shape, "precision": name, "operand_set": operand_set,
                                "implementation": implementation, "stage": stage,
                                "status": "PASS"} for implementation, stage in stages]
            plans.append({"same_algorithm_for_every_set": True,
                          "identification_algorithm_matches": True, "kernels_per_launch": 1,
                          "algorithm": {"check_status": 0}})
    write_rows(directory / "precision_cutedsl_vs_cublaslt.csv", comparison)
    write_rows(directory / "raw/repetitions.csv", repetitions)
    write_rows(directory / "raw/validation.csv", validation)
    write_rows(directory / "raw/operands.csv", operands)
    write_rows(directory / "precision_comparison.csv", cute)
    (directory / "raw/cublaslt_plans.json").write_text(json.dumps(plans))
    for figure in ("precision_cutedsl_vs_cublaslt.svg", "precision_comparison.svg"):
        (directory / figure).write_text("<svg/>")


def edit_rows(path, change):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    write_rows(path, change(rows))


class NcuExportTest(unittest.TestCase):
    metrics = list(profile_gemm.DRAM_METRICS)

    def test_one_row_per_profiled_kernel(self):
        values = {metric: 64.0 for metric in self.metrics}
        kernels, units = profile_gemm.parse_kernels(
            ncu_export([("first", values), ("second", values)], self.metrics), self.metrics)
        self.assertEqual([kernel["name"] for kernel in kernels], ["first", "second"])
        self.assertEqual(kernels[0]["metrics"]["dram__bytes_read.sum"], 64.0)
        self.assertIn(profile_gemm.NVTX_RANGE, kernels[0]["nvtx_ranges"])
        self.assertEqual(units["dram__bytes_write.sum"], "byte")

    def test_missing_metric_is_rejected(self):
        text = ncu_export([("only", {self.metrics[0]: 1.0})], self.metrics[:1])
        with self.assertRaises(ValueError):
            profile_gemm.parse_kernels(text, self.metrics)


class GemmProfileCheckTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.index = write_profile(self.directory)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        (self.directory / "index.json").write_text(json.dumps(self.index))

    def test_complete_profile_passes(self):
        self.assertEqual(check_diagnostics.check_gemm_profile(self.directory), [])

    def test_missing_capture_is_reported(self):
        self.index["captures"].pop()
        self.save()
        problems = check_diagnostics.check_gemm_profile(self.directory)
        self.assertTrue(any("5 captures" in problem for problem in problems), problems)

    def test_failed_validation_is_reported(self):
        self.index["captures"][2]["worker"]["validation"]["after_profiled_launch"] = "FAIL"
        self.save()
        problems = check_diagnostics.check_gemm_profile(self.directory)
        self.assertTrue(any("validation did not pass" in problem for problem in problems))

    def test_second_kernel_in_the_range_is_reported(self):
        capture = self.index["captures"][0]
        text = ncu_export([("a", capture["metrics"]), ("b", capture["metrics"])],
                          self.index["protocol"]["metrics"])
        (self.directory / capture["files"]["csv"]).write_text(text)
        problems = check_diagnostics.check_gemm_profile(self.directory)
        self.assertTrue(any("2 kernels" in problem for problem in problems), problems)


class PrecisionCheckTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        write_precision(self.directory)

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_comparison_passes(self):
        self.assertEqual(check_diagnostics.check_precision(self.directory), [])

    def test_missing_comparison_row_is_reported(self):
        edit_rows(self.directory / "precision_cutedsl_vs_cublaslt.csv", lambda rows: rows[:-1])
        problems = check_diagnostics.check_precision(self.directory)
        self.assertTrue(any("has 17 rows" in problem for problem in problems), problems)

    def test_empty_field_is_reported(self):
        def blank(rows):
            rows[4]["kernel"] = ""
            return rows
        edit_rows(self.directory / "precision_cutedsl_vs_cublaslt.csv", blank)
        problems = check_diagnostics.check_precision(self.directory)
        self.assertTrue(any("empty fields ['kernel']" in problem for problem in problems))

    def test_failed_validation_record_is_reported(self):
        def fail(rows):
            rows[7]["status"] = "FAIL"
            return rows
        edit_rows(self.directory / "raw/validation.csv", fail)
        problems = check_diagnostics.check_precision(self.directory)
        self.assertTrue(any("1 validation records failed" in problem for problem in problems))

    def test_missing_repetition_is_reported(self):
        edit_rows(self.directory / "raw/repetitions.csv", lambda rows: rows[1:])
        problems = check_diagnostics.check_precision(self.directory)
        self.assertTrue(any("has 53 rows" in problem for problem in problems), problems)


@unittest.skipUnless(TORCH, "PyTorch is required for the NVFP4 encoding checks")
class BlockScaledEncodingTest(unittest.TestCase):
    def setUp(self):
        import torch

        from precision_comparison import cublaslt_precision

        self.torch, self.module = torch, cublaslt_precision

    def test_scale_offsets_are_a_permutation(self):
        offsets = self.module.scale_offsets(self.torch, 256, 8, "cpu").flatten()
        self.assertTrue(self.torch.equal(offsets.sort().values, self.torch.arange(256 * 8)))

    def test_scale_offsets_follow_the_documented_tile(self):
        offsets = self.module.scale_offsets(self.torch, 256, 8, "cpu")
        # Rows 0, 1, 32 and 96 in the first tile, the next K tile, then the next row tile.
        self.assertEqual([int(offsets[row, block]) for row, block in
                          ((0, 0), (1, 0), (32, 0), (96, 0), (0, 3), (0, 4), (128, 0))],
                         [0, 16, 4, 12, 3, 512, 1024])

    def test_scales_round_trip(self):
        torch = self.torch
        scales = torch.tensor([1.0, 2.0, 0.5, 448.0])[torch.randint(0, 4, (256, 8))]
        packed = self.module.scale_pack(torch, scales)
        self.assertTrue(torch.equal(self.module.scale_unpack(torch, packed, 256, 8), scales))

    def test_fp4_round_trip_and_nibble_order(self):
        torch = self.torch
        magnitudes = torch.tensor(self.module.E2M1_MAGNITUDES)
        values = torch.cat((magnitudes, -magnitudes[1:], torch.tensor([1.0]))).reshape(2, 8)
        packed = self.module.fp4_pack(torch, values)
        self.assertTrue(torch.equal(self.module.fp4_unpack(torch, packed), values))
        # Element 2i sits in the low nibble: 1.0 is code 0x2 and 2.0 is code 0x4.
        pair = self.module.fp4_pack(torch, torch.tensor([[1.0, 2.0]]))
        self.assertEqual(int(pair[0, 0]), 0x42)

    def test_unrepresentable_values_are_rejected(self):
        torch = self.torch
        with self.assertRaises(ValueError):
            self.module.fp4_pack(torch, torch.tensor([[2.5, 1.0]]))
        with self.assertRaises(ValueError):
            self.module.e4m3_encode(torch, torch.tensor([1.0625]))

    def test_compare_counts_mismatches(self):
        torch = self.torch
        expected = torch.full((4, 4), 100.0)
        output = expected.clone()
        output[1, 2] += 1
        record = self.module.compare(torch, "fp8", output, expected)
        self.assertEqual((record["status"], record["mismatches"]), ("FAIL", 1))
        self.assertEqual(self.module.compare(torch, "fp8", expected, expected)["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
