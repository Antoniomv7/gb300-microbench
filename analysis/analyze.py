#!/usr/bin/env python3
"""Compute every published summary and figure from the raw measurements.

Experiments I–III: the median of 30 launches per configuration within each campaign.
Experiment IV: the mean time of ten launches per candidate within each campaign.
The three campaigns give the mean, sample standard deviation and coefficient of variation.
Experiment V: the same statistics over the three repetitions of each shape and format.
GEMM traffic profile: Nsight Compute counters relative to the compulsory operand bytes.

--study regenerates the thirteen files of results/ from a make final-study directory, such as
the extracted study-20260923T173150Z archive. The analysis needs no GPU.
"""

import argparse
import csv
import datetime as dt
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import figures  # noqa: E402
from precision_comparison import precision_comparison as precision  # noqa: E402
from precision_comparison.cublaslt_precision import CUTE_KERNELS, arithmetic  # noqa: E402
from scripts import ncu_capture, profile_gemm  # noqa: E402
from scripts.run_campaign import (MEMORY_METHODS, MINIMUM_SAMPLES_PER_CONFIGURATION,  # noqa: E402
                                  UMMA_METHODS)

# Final parameters: 18 memory and 24 UMMA configurations, 4 scaling configurations,
# 30 repetitions each, and 5 GEMM shapes with 4 candidates.
EXPECTED_ROWS = {"memory_paths": 540, "umma_throughput": 720, "umma_device_scaling": 120,
                 "gemm_comparison": 20}
# Six memory and two UMMA captures; the scaling and GEMM experiments have none.
NCU_CASES = sorted(case["case"] for case in ncu_capture.PLAN)
# Dense peak throughput NVIDIA lists for each input format, for the percent-of-peak column.
VENDOR_DENSE_TFLOPS = {"bf16": 2250.0, "fp8": 4500.0, "nvfp4": 13500.0}
TIME_SCALE_NS = {"ns": 1.0, "nsecond": 1.0, "us": 1e3, "usecond": 1e3, "ms": 1e6, "msecond": 1e6}
CLOCK_SCALE_HZ = {"hz": 1.0, "cycle/second": 1.0, "cycle/nsecond": 1e9}


def stats(values):
    if len(values) != 3 or any(not math.isfinite(value) for value in values):
        raise ValueError("three finite campaign values are required")
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values)
    return {"mean": mean, "median": statistics.median(values), "stdev_sample": deviation,
            "cv_percent": abs(100 * deviation / mean) if mean else 0,
            "minimum": min(values), "maximum": max(values)}


def compact_stats(values, unit):
    summary = stats(values)
    return {f"campaign_{index}_{unit}": value for index, value in enumerate(values, 1)} | {
        f"mean_{unit}": summary["mean"], f"stdev_{unit}": summary["stdev_sample"],
        "cv_percent": summary["cv_percent"]}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.6f}" if isinstance(value, float) else value
                             for key, value in row.items()})


def read_metadata(directory, profile=False):
    """Require a completed acquisition from clean sources; accept the published archive too."""
    path = directory / "metadata.json"
    if profile and not path.exists():
        path = directory / "index.json"  # The published profile's completion record.
    record = json.loads(path.read_text(encoding="utf-8"))
    complete = (record["state"] == "COMPLETE" if "state" in record
                else bool(record.get("completed_utc")))
    if not complete:
        raise ValueError(f"{directory}: the acquisition did not complete")
    if "environment" in record:
        environment = record["environment"]
        repository = environment.get("repository", {})
        commit = repository.get("commit")
        dirty = any(not line.startswith("??")
                    for line in repository.get("status", ["missing source status"]))
    else:
        environment, commit, dirty = record, record.get("git_commit"), record.get("git_dirty")
    gpu = environment.get("gpu", {}).get("uuid")
    if not commit or not gpu or dirty is not False:
        raise ValueError(f"{directory}: missing GPU/source identity or modified tracked sources")
    return {**record, "source_commit": commit, "gpu_uuid": gpu}


def same_acquisition(records):
    """Every input, including precision, profiling and a supplied timing summary, must agree."""
    identities = {(record.get("gpu_uuid"), record.get("source_commit")) for record in records}
    if len(identities) != 1 or any(not gpu or not commit for gpu, commit in identities):
        raise ValueError("the inputs must share one GPU and one source commit")
    gpu, commit = next(iter(identities))
    return {"gpu_uuid": gpu, "source_commit": commit}


# Experiments I–IV ------------------------------------------------------------------------------

def load_campaign(path):
    """One campaign's validated samples, clock telemetry and Nsight Compute counters."""
    metadata = read_metadata(path)
    data = {}
    for experiment, expected in EXPECTED_ROWS.items():
        rows = read_csv(path / "raw" / f"{experiment}.csv")
        if len(rows) != expected or any(row["correctness"] not in ("OK", "PASS") for row in rows):
            raise ValueError(f"{path}: {experiment} needs {expected} validated rows, "
                             f"found {len(rows)}")
        data[experiment] = rows
    profile = json.loads((path / "ncu/index.json").read_text(encoding="utf-8"))
    if sorted(case["case"] for case in profile["cases"]) != NCU_CASES:
        raise ValueError(f"{path}: expected the Nsight Compute captures {NCU_CASES}")
    return {"path": path, "metadata": metadata, "data": data, "profile": profile,
            "telemetry": read_csv(path / "raw/umma_device_scaling_telemetry.csv")}


def load_campaigns(paths):
    """Three distinct complete campaigns, measured on one GPU from one source commit."""
    campaigns = [load_campaign(Path(path).resolve()) for path in paths]
    if len(campaigns) != 3 or len({campaign["path"] for campaign in campaigns}) != 3:
        raise ValueError("three distinct campaigns are required")
    same_acquisition([campaign["metadata"] for campaign in campaigns])
    return campaigns


def grouped_medians(rows, fields, value):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(float(row[value]))
    return {key: statistics.median(samples) for key, samples in groups.items()}


def profile_case(record, section, method, **parameters):
    return next((case for case in record["profile"].get("cases", [])
                 if case["section"] == section and case["method"] == method and
                 all(case.get(key) == value for key, value in parameters.items())), None)


def memory_results(records):
    # Reduce repetitions within each campaign before comparing independent campaigns.
    campaigns = [grouped_medians(record["data"]["memory_paths"],
                                 ("method", "stages", "bytes_in_flight_per_sm"),
                                 "effective_gbps") for record in records]
    rows, points = [], []
    for stages in (2, 4, 8):
        for size in (16, 32, 64):
            ratios = [campaign[("tma", str(stages), str(size * 1024))] /
                      campaign[("ldgsts", str(stages), str(size * 1024))]
                      for campaign in campaigns]
            for method in MEMORY_METHODS:
                key = (method, str(stages), str(size * 1024))
                values = [campaign[key] for campaign in campaigns]
                dram = []
                # DRAM counters exist only for the selected NCU configurations.
                for record in records:
                    case = profile_case(record, "memory_paths", method,
                                        stages=stages, bytes_in_flight_kib=size)
                    if case and case.get("useful_bytes"):
                        dram.append(case["metrics"]["dram__bytes_read.sum"] / case["useful_bytes"])
                rows.append({"method": method, "stages": stages, "bytes_in_flight_kib": size,
                             **compact_stats(values, "gbps"),
                             "tma_to_ldgsts_ratio": stats(ratios)["mean"],
                             "dram_read_ratio": stats(dram)["mean"] if len(dram) == 3 else ""})
                points.append({"method": method, "stages": stages,
                               "bytes_in_flight_kib": size, **stats(values)})
    maxima = {method: max((point for point in points if point["method"] == method),
                          key=lambda point: point["mean"]) for method in MEMORY_METHODS}
    return rows, {"configurations": points, "maximum_by_method": maxima}


def umma_results(records):
    campaigns = [grouped_medians(record["data"]["umma_throughput"],
                                 ("method", "n", "depth"), "flops_per_cycle")
                 for record in records]
    rows, points = [], []
    for n in (64, 128, 256):
        for depth in (4, 16, 64, 256):
            ratios = [campaign[("umma_2sm", str(n), str(depth))] /
                      campaign[("umma_1sm", str(n), str(depth))] for campaign in campaigns]
            for method in UMMA_METHODS:
                group = 1 if method == "umma_1sm" else 2
                totals = [campaign[(method, str(n), str(depth))] for campaign in campaigns]
                # A two-CTA UMMA spans two SMs, so compare throughput per SM.
                per_sm = [value / group for value in totals]
                rows.append({"method": method, "n": n, "depth": depth, "cta_group": group,
                             **compact_stats(per_sm, "flops_cycle_sm"),
                             "mean_flops_per_cycle": stats(totals)["mean"],
                             "two_sm_to_one_sm_ratio": stats(ratios)["mean"],
                             "estimated_tflops_per_sm": ""})
                points.append({"method": method, "n": n, "depth": depth,
                               "cta_group": group, **stats(per_sm), "total": stats(totals)})

    best = max(points, key=lambda point: point["mean"])
    estimates = []
    # Convert FLOP/cycle to TFLOP/s with the clock measured by Nsight Compute.
    metric = "sm__cycles_elapsed.avg.per_second"
    for record, campaign in zip(records, campaigns):
        case = profile_case(record, "umma_throughput", best["method"],
                            n=best["n"], depth=best["depth"])
        if case:
            factor = CLOCK_SCALE_HZ.get(case["units"].get(metric, "").lower())
            if factor:
                cycles = campaign[(best["method"], str(best["n"]), str(best["depth"]))]
                estimates.append(cycles / best["cta_group"] * case["metrics"][metric] * factor / 1e12)
    ceiling = stats(estimates) if len(estimates) == 3 else None
    if ceiling:
        next(row for row in rows if row["method"] == best["method"] and
             row["n"] == best["n"] and row["depth"] == best["depth"])[
                 "estimated_tflops_per_sm"] = ceiling["mean"]
    return rows, {"configurations": points, "best_per_sm_configuration": best,
                  "estimated_tflops_per_sm": ceiling}


def clock_campaigns(record):
    """Aggregate the clock samples recorded inside each configuration's timed launches."""
    samples = [{"moment": float(row["unix_s"]), "sm_clock_mhz": float(row["sm_clock_mhz"]),
                "power_w": float(row["power_w"]),
                "temperature_c": float(row["temperature_c"])} for row in record["telemetry"]]
    windows = defaultdict(list)
    for row in record["data"]["umma_device_scaling"]:
        windows[(row["method"], row["scale"])].append(
            (float(row["host_start_unix_s"]), float(row["host_end_unix_s"])))
    summary = {}
    for key, spans in windows.items():
        inside = [sample for sample in samples
                  if any(start <= sample["moment"] <= end for start, end in spans)]
        if len(inside) < MINIMUM_SAMPLES_PER_CONFIGURATION:
            raise ValueError(f"{'/'.join(key)}: fewer than "
                             f"{MINIMUM_SAMPLES_PER_CONFIGURATION} concurrent clock samples")
        summary[key] = {"mean_sm_clock_mhz": statistics.fmean(s["sm_clock_mhz"] for s in inside),
                        "min_sm_clock_mhz": min(s["sm_clock_mhz"] for s in inside),
                        "max_sm_clock_mhz": max(s["sm_clock_mhz"] for s in inside),
                        "mean_power_w": statistics.fmean(s["power_w"] for s in inside),
                        "mean_temperature_c": statistics.fmean(s["temperature_c"] for s in inside),
                        "clock_sample_count": len(inside)}
    return summary


def scaling_results(records):
    campaigns = []
    for record in records:
        values = {"clocks": clock_campaigns(record)}
        for field in ("total_tflops", "kernel_time_ms", "tflops_per_planned_active_sm"):
            values[field] = grouped_medians(record["data"]["umma_device_scaling"],
                                            ("method", "scale"), field)
        campaigns.append(values)

    rows, points, geometry = [], [], {}
    for method in UMMA_METHODS:
        for scale in ("isolated", "device_scale"):
            key = (method, scale)
            sample = next(row for row in records[0]["data"]["umma_device_scaling"]
                          if (row["method"], row["scale"]) == key)
            geometry[key] = sample
            throughputs = [campaign["total_tflops"][key] for campaign in campaigns]
            times = [campaign["kernel_time_ms"][key] for campaign in campaigns]
            per_sm = [campaign["tflops_per_planned_active_sm"][key] for campaign in campaigns]
            clocks = [campaign["clocks"][key] for campaign in campaigns]
            clock = stats([entry["mean_sm_clock_mhz"] for entry in clocks])
            rows.append({"method": method, "scale": scale,
                         "active_sms": int(sample["planned_active_sm_count"]),
                         "work_units": int(sample["work_unit_count"]),
                         "cluster_count": int(sample["cluster_count"]),
                         **compact_stats(throughputs, "tflops"),
                         "mean_kernel_time_ms": stats(times)["mean"],
                         "mean_tflops_per_sm": stats(per_sm)["mean"],
                         "mean_sm_clock_mhz": clock["mean"],
                         "stdev_sm_clock_mhz": clock["stdev_sample"],
                         "min_sm_clock_mhz": min(entry["min_sm_clock_mhz"] for entry in clocks),
                         "max_sm_clock_mhz": max(entry["max_sm_clock_mhz"] for entry in clocks),
                         "mean_power_w": stats([entry["mean_power_w"] for entry in clocks])["mean"],
                         "mean_temperature_c":
                             stats([entry["mean_temperature_c"] for entry in clocks])["mean"],
                         "clock_sample_count": sum(entry["clock_sample_count"] for entry in clocks),
                         "per_sm_ratio_vs_isolated": "", "sm_clock_ratio_vs_isolated": "",
                         "scaling_efficiency_raw": "",
                         "scaling_efficiency_freq_normalized": ""})
            points.append({"method": method, "scale": scale,
                           "active_sms": int(sample["planned_active_sm_count"]),
                           "work_units": int(sample["work_unit_count"]),
                           "clock": clock, **stats(throughputs)})

    scaling_ratios, efficiency = {}, {}
    for method in UMMA_METHODS:
        # Independent empirical baselines define a ratio, not a bounded efficiency.
        units = int(geometry[(method, "device_scale")]["work_unit_count"])
        raw, normalized, clock_ratios = [], [], []
        for campaign in campaigns:
            isolated = campaign["total_tflops"][(method, "isolated")]
            device = campaign["total_tflops"][(method, "device_scale")]
            clock_ratio = (campaign["clocks"][(method, "device_scale")]["mean_sm_clock_mhz"] /
                           campaign["clocks"][(method, "isolated")]["mean_sm_clock_mhz"])
            raw.append(device / (isolated * units))
            # Dividing by the clock ratio separates spatial scaling from the DVFS state.
            normalized.append(device / (isolated * units * clock_ratio))
            clock_ratios.append(clock_ratio)
        scaling_ratios[method] = stats(raw)
        efficiency[method] = {"units": units, "raw": stats(raw),
                              "frequency_normalized": stats(normalized),
                              "sm_clock_ratio": stats(clock_ratios)}
        row = next(row for row in rows
                   if row["method"] == method and row["scale"] == "device_scale")
        row["per_sm_ratio_vs_isolated"] = scaling_ratios[method]["mean"]
        row["sm_clock_ratio_vs_isolated"] = efficiency[method]["sm_clock_ratio"]["mean"]
        row["scaling_efficiency_raw"] = efficiency[method]["raw"]["mean"]
        row["scaling_efficiency_freq_normalized"] = \
            efficiency[method]["frequency_normalized"]["mean"]
    ratios = [campaign["total_tflops"][("umma_2sm", "device_scale")] /
              campaign["total_tflops"][("umma_1sm", "device_scale")]
              for campaign in campaigns]
    return rows, {"configurations": points, "per_sm_ratio_vs_isolated": scaling_ratios,
                  "scaling_efficiency": efficiency,
                  "device_total_ratio_2sm_over_1sm": stats(ratios),
                  "hardware_sm_count": int(next(iter(geometry.values()))["hardware_sm_count"])}


def gemm_results(records):
    campaigns = [{(int(row["shape_index"]), int(row["candidate_index"])): row
                  for row in record["data"]["gemm_comparison"]} for record in records]
    rows, points, shapes = [], [], []
    for key in sorted(campaigns[0]):
        entries = [campaign[key] for campaign in campaigns]
        first = entries[0]
        values = [float(entry["tflops"]) for entry in entries]
        durations = [float(entry["kernel_time_ms"]) for entry in entries]
        ratios = [float(entry["throughput_ratio_vs_cublaslt"]) for entry in entries]
        rows.append({field: first[field] for field in
                     ("shape_index", "shape_id", "m", "n", "k", "l", "variant", "method")} |
                    compact_stats(values, "tflops") |
                    {"mean_kernel_time_ms": stats(durations)["mean"],
                     "ratio_vs_cublaslt": stats(ratios)["mean"]})
        points.append({"shape_index": int(first["shape_index"]), "shape_id": first["shape_id"],
                       "variant": first["variant"], "method": first["method"], **stats(values)})
    for shape_index in sorted({point["shape_index"] for point in points}):
        best = max((point for point in points if point["shape_index"] == shape_index
                    and point["method"] == "cutedsl"), key=lambda point: point["mean"])
        shapes.append({"shape_index": shape_index, "shape_id": best["shape_id"],
                       "best_cutedsl_variant": best["variant"]})
    return rows, {"configurations": points, "shapes": shapes}


# Experiment V ----------------------------------------------------------------------------------

def load_precision(path):
    """The per-repetition timings, validation records and cuBLASLt plans of a precision run."""
    metadata = read_metadata(path)
    repetitions = [{**row, "shape_index": int(row["shape_index"]),
                    "repetition": int(row["repetition"]),
                    "kernel_time_us": float(row["kernel_time_us"]), "tflops": float(row["tflops"])}
                   for row in read_csv(path / "raw/repetitions.csv")]
    validation = [{**row, "shape_index": int(row["shape_index"]),
                   "bit_exact": row["bit_exact"] == "True"}
                  for row in read_csv(path / "raw/validation.csv")]
    plans = json.loads((path / "raw/cublaslt_plans.json").read_text(encoding="utf-8"))
    operands = [{key: value == "True" if key == "represented_exactly" or
                 key.endswith("_bytes_identical_to_cutedsl") else value
                 for key, value in row.items()} for row in read_csv(path / "raw/operands.csv")]
    precision.check_records(repetitions, validation, operands, plans)
    return repetitions, validation, plans, metadata


def configuration(record):
    return record["shape_index"], record["precision"], record["implementation"]


def repetition_summary(repetitions, validation, key):
    """The three timed repetitions of one implementation, shape and format, and their checks."""
    samples = sorted((record for record in repetitions if configuration(record) == key),
                     key=lambda record: record["repetition"])
    checks = [record for record in validation if configuration(record) == key]
    return {"samples": samples, "throughput": stats([record["tflops"] for record in samples]),
            "time": stats([record["kernel_time_us"] for record in samples]),
            "validation": "PASS" if len(samples) == precision.REPETITIONS and checks and all(
                check["status"] == "PASS" for check in checks) else "FAIL",
            "bit_exact": all(check["bit_exact"] for check in checks)}


def precision_results(repetitions, validation):
    """CuTe DSL throughput by shape and input format (precision_comparison.csv)."""
    rows = []
    for shape_index in sorted({record["shape_index"] for record in repetitions}):
        for name in precision.FORMATS:
            summary = repetition_summary(repetitions, validation, (shape_index, name, "cutedsl"))
            first, types = summary["samples"][0], arithmetic("cutedsl", name)
            peak = VENDOR_DENSE_TFLOPS[name]
            rows.append({
                "shape_index": shape_index, "shape_id": first["shape_id"],
                **{key: first[key] for key in ("m", "n", "k", "l")}, "precision": name,
                **{key: types[key] for key in ("input_dtype", "scale_dtype", "scale_block_elements",
                                               "accumulator_dtype", "output_dtype")},
                "mma_tile": "x".join(map(str, precision.TILE)),
                "cluster_shape": "x".join(map(str, precision.CLUSTER)), "tma_store": "yes",
                **{f"repetition_{record['repetition']}_tflops": record["tflops"]
                   for record in summary["samples"]},
                "mean_tflops": summary["throughput"]["mean"],
                "stdev_tflops": summary["throughput"]["stdev_sample"],
                "cv_percent": summary["throughput"]["cv_percent"],
                "mean_kernel_time_us": summary["time"]["mean"],
                "speedup_vs_bf16": "", "vendor_dense_peak_tflops": peak,
                "percent_vendor_dense_peak": 100 * summary["throughput"]["mean"] / peak,
                "correctness": summary["validation"]})
        baseline = next(row for row in rows
                        if row["shape_index"] == shape_index and row["precision"] == "bf16")
        for row in rows:
            if row["shape_index"] == shape_index:
                row["speedup_vs_bf16"] = row["mean_tflops"] / baseline["mean_tflops"]
    return rows


def precision_comparison_results(repetitions, validation, plans):
    """CuTe DSL and cuBLASLt on the same operands (precision_cutedsl_vs_cublaslt.csv)."""
    plans = {(plan["shape_index"], plan["precision"]): plan for plan in plans}
    keys = sorted({(record["shape_index"], record["precision"]) for record in repetitions},
                  key=lambda key: (key[0], precision.FORMATS.index(key[1])))
    rows = []
    for shape_index, name in keys:
        summaries = {implementation: repetition_summary(repetitions, validation,
                                                        (shape_index, name, implementation))
                     for implementation in precision.IMPLEMENTATIONS}
        plan = plans.get((shape_index, name), {})
        # Matched: both validated against the same reference, FP32 output confirmed by
        # cublasLtMatmulAlgoCheck, and one algorithm across every operand set.
        matched = (all(summary["validation"] == "PASS" for summary in summaries.values()) and
                   plan.get("algorithm", {}).get("check_status") == 0 and
                   plan.get("same_algorithm_for_every_set", False) and
                   plan.get("identification_algorithm_matches", False))
        cublaslt_mean = summaries["cublaslt"]["throughput"]["mean"]
        for implementation, summary in summaries.items():
            first = summary["samples"][0]
            if implementation == "cutedsl":
                tile, cluster = precision.TILE, precision.CLUSTER
                kernel = (f"{CUTE_KERNELS[name]}; MMA tile {tile[0]}x{tile[1]}; "
                          f"cluster {cluster[0]}x{cluster[1]}; TMA store")
            else:
                algorithm = plan["algorithm"]
                kernel_name = (" + ".join(plan.get("kernel_names", []))
                               or "name unavailable in profiler")
                kernel = (f"{kernel_name} (algorithm "
                          f"{algorithm['algorithm_id']}; tile {algorithm['tile_id']}; stages "
                          f"{algorithm['stages_id']}; cluster {algorithm['cluster_shape_id']})")
            rows.append({
                **{key: first[key] for key in ("shape_index", "shape_id", "m", "n", "k", "l",
                                               "precision")},
                "implementation": implementation,
                "comparison": "matched" if matched else "not_matched",
                **arithmetic(implementation, name), "kernel": kernel,
                **{f"repetition_{record['repetition']}_tflops": record["tflops"]
                   for record in summary["samples"]},
                "mean_tflops": summary["throughput"]["mean"],
                "stdev_tflops": summary["throughput"]["stdev_sample"],
                "cv_percent": summary["throughput"]["cv_percent"],
                "mean_kernel_time_us": summary["time"]["mean"],
                "throughput_ratio_vs_cublaslt": summary["throughput"]["mean"] / cublaslt_mean,
                "validation": summary["validation"], "bit_exact_outputs": summary["bit_exact"]})
    return rows


# GEMM traffic profile --------------------------------------------------------------------------

def gemm_profile_results(directory, cache_state, gemm_summary, metadata):
    """DRAM and L2 traffic of the six profiled launches beside their CUDA-event timing."""
    if cache_state != metadata.get("cache_state"):
        raise ValueError(f"{directory}: the requested cache state differs from the acquisition")
    reference = {(row["shape_id"], row["variant"]): row for row in read_csv(gemm_summary)}
    rows = []
    for case, shape, variant in profile_gemm.CASES:
        text = (directory / f"{case}.csv").read_text(encoding="utf-8")
        # The L2 read metric is exported only when its calibration passed.
        l2_metric = profile_gemm.L2_READ_METRIC if profile_gemm.L2_READ_METRIC in \
            text.partition("\n")[0] else None
        metrics = (*profile_gemm.DRAM_METRICS, *profile_gemm.TIMING_METRICS,
                   *((l2_metric,) if l2_metric else ()))
        kernels, units = ncu_capture.parse_kernels(text, metrics)
        if len(kernels) != 1 or not kernels[0]["name"] or \
                profile_gemm.NVTX_RANGE not in kernels[0]["nvtx_ranges"]:
            raise ValueError(f"{case}: expected one named kernel inside the NVTX range")
        values = kernels[0]["metrics"]
        if any(not math.isfinite(value) or value < 0 for value in values.values()) or \
                any(values[metric] <= 0 for metric in profile_gemm.TIMING_METRICS):
            raise ValueError(f"{case}: invalid counter, duration or SM clock")
        validation = json.loads((directory / f"{case}.worker.json").read_text(
            encoding="utf-8"))["validation"]
        if validation["first_launch"] != "PASS" or validation["after_profiled_launch"] != "PASS":
            raise ValueError(f"{case}: the profiled launch failed numerical validation")
        m, n, k, batch = shape
        compulsory = 2 * m * k * batch + 2 * n * k * batch  # BF16 A and B
        output_bytes = 4 * m * n * batch  # FP32 D
        read, write = values["dram__bytes_read.sum"], values["dram__bytes_write.sum"]
        duration_ns = values["gpu__time_duration.sum"] * TIME_SCALE_NS[
            units["gpu__time_duration.sum"].lower()]
        clock_hz = values["sm__cycles_elapsed.avg.per_second"] * CLOCK_SCALE_HZ[
            units["sm__cycles_elapsed.avg.per_second"].lower()]
        l2_bytes = values[l2_metric] if l2_metric else None
        timing = reference[(profile_gemm.shape_id(shape), variant)]
        rows.append({
            "shape_id": profile_gemm.shape_id(shape), "m": m, "n": n, "k": k, "l": batch,
            "variant": variant, "method": "cutedsl" if variant == "persistent_2cta" else "cublaslt",
            # Acquisition completion and the exported capture were both checked above.
            "cache_state": cache_state, "status": "PASS", "kernel_name": kernels[0]["name"],
            "validation": "PASS",
            "dram_read_bytes": read, "dram_write_bytes": write,
            "compulsory_read_bytes": compulsory, "dram_read_to_compulsory": read / compulsory,
            "dram_read_excess_bytes": read - compulsory, "output_bytes": output_bytes,
            "dram_write_to_output": write / output_bytes,
            "l2_tma_read_bytes": "" if l2_bytes is None else l2_bytes,
            # A hot-cache launch can have no DRAM reads; its L2/DRAM ratio is then undefined.
            "l2_tma_read_to_dram_read": "" if l2_bytes is None or not read else l2_bytes / read,
            "profiled_duration_us": duration_ns / 1e3,
            "profiled_sm_clock_mhz": clock_hz / 1e6,
            # Performance data remain the CUDA-event results of Experiment IV.
            "cuda_event_mean_kernel_time_us": 1e3 * float(timing["mean_kernel_time_ms"]),
            "cuda_event_mean_tflops": float(timing["mean_tflops"])})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for the summaries; an existing one is never reused")
    parser.add_argument("--study", type=Path,
                        help="a make final-study directory: regenerate all thirteen results")
    parser.add_argument("--campaigns", type=Path, nargs=3, metavar="CAMPAIGN",
                        help="three complete Experiment I–IV campaigns")
    parser.add_argument("--precision", type=Path, help="an Experiment V run")
    parser.add_argument("--gemm-profile", type=Path, help="a GEMM traffic profile")
    parser.add_argument("--cache-state", choices=("hot", "cold"), help="the profile's cache state")
    parser.add_argument("--gemm-summary", type=Path,
                        help="the gemm_comparison.csv whose CUDA-event timing --gemm-profile "
                             "reports; defaults to the one computed from --campaigns")
    args = parser.parse_args()
    if args.study:
        # The layout of make final-study, whose GEMM profile always uses hot caches.
        args.campaigns = [args.study / f"campaign-{index}" for index in (1, 2, 3)]
        args.precision = args.study / "precision"
        args.gemm_profile, args.cache_state = args.study / "gemm-profile-hot", "hot"
    if not (args.campaigns or args.precision or args.gemm_profile):
        parser.error("name --study, --campaigns, --precision or --gemm-profile")
    if args.gemm_profile and not (args.cache_state and (args.campaigns or args.gemm_summary)):
        parser.error("--gemm-profile needs --cache-state and --campaigns or --gemm-summary")

    campaigns = load_campaigns(args.campaigns) if args.campaigns else []
    precision_data = load_precision(args.precision) if args.precision else None
    profile_metadata = read_metadata(args.gemm_profile, profile=True) if args.gemm_profile else None
    records = [campaign["metadata"] for campaign in campaigns]
    if precision_data:
        records.append(precision_data[3])
    if profile_metadata:
        records.append(profile_metadata)
    if args.gemm_profile and args.gemm_summary:
        manifest = args.gemm_summary.parent / "metadata.json"
        if not manifest.exists():
            manifest = args.gemm_summary.parent / "analysis.json"  # Published archive.
        records.append(json.loads(manifest.read_text(encoding="utf-8")))
    identity = same_acquisition(records)

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    if args.campaigns:
        for name, results, figure in (
                ("memory_paths", memory_results, figures.memory_figure),
                ("umma_throughput", umma_results, figures.umma_figure),
                ("umma_device_scaling", scaling_results, figures.scaling_figure),
                ("gemm_comparison", gemm_results, figures.gemm_figure)):
            rows, summary = results(campaigns)
            write_csv(output / f"{name}.csv", rows)
            (output / f"{name}.svg").write_text(figure(summary), encoding="utf-8")
    if args.precision:
        repetitions, validation, plans, _ = precision_data
        rows = precision_results(repetitions, validation)
        write_csv(output / "precision_comparison.csv", rows)
        (output / "precision_comparison.svg").write_text(figures.precision_figure(rows),
                                                         encoding="utf-8")
        rows = precision_comparison_results(repetitions, validation, plans)
        write_csv(output / "precision_cutedsl_vs_cublaslt.csv", rows)
        (output / "precision_cutedsl_vs_cublaslt.svg").write_text(
            figures.precision_comparison_figure(rows, precision.WARMUP_ITERATIONS,
                                                precision.ITERATIONS), encoding="utf-8")
    if args.gemm_profile:
        rows = gemm_profile_results(args.gemm_profile, args.cache_state,
                                    args.gemm_summary or output / "gemm_comparison.csv",
                                    profile_metadata)
        # Unlike the other summaries, this table keeps Python's full float representation.
        with (output / "gemm_profile.csv").open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    record = {**identity, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "campaigns": args.campaigns, "precision": args.precision,
              "gemm_profile": args.gemm_profile, "cache_state": args.cache_state,
              "gemm_summary": args.gemm_summary}
    (output / "metadata.json").write_text(json.dumps(record, indent=2, default=str) + "\n",
                                          encoding="utf-8")
    print(f"analysis: complete {output}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"analysis: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
