#!/usr/bin/env python3
"""Compute every published summary and figure from the raw measurements.

Experiments I–III: the median of 30 launches per configuration within each campaign.
Experiment IV: twenty graph-replayed launches per candidate within each campaign.
The three campaigns give the mean, sample standard deviation and coefficient of variation.
Experiment V: the same statistics over the three repetitions of each shape and format.
GEMM traffic profile: Nsight Compute counters relative to the compulsory operand bytes.

--study produces nine CSV summaries and seven SVG figures from a protocol-v2
make final-study directory. Historical acquisitions require their original analysis commit.
The analysis needs no GPU.
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
from scripts import ncu_capture, profile_gemm, power_measure  # noqa: E402
from scripts.run_campaign import (MEMORY_METHODS, MINIMUM_SAMPLES_PER_CONFIGURATION,  # noqa: E402
                                  UMMA_METHODS)

# Final parameters: 18 memory and 24 UMMA configurations, 4 scaling configurations,
# 30 repetitions each, and 5 GEMM shapes with 5 candidates.
EXPECTED_ROWS = {"memory_paths": 540, "umma_throughput": 720, "umma_device_scaling": 120,
                 "gemm_comparison": 25}
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
    if record.get("protocol_version") != 2 or record.get("pilot", False):
        raise ValueError(f"{directory}: requires final protocol-v2 data, not a pilot or legacy study")
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
    fingerprints = {record.get("environment_fingerprint") for record in records}
    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError("the GPU/software/power-limit environment changed between acquisitions")
    return {"gpu_uuid": gpu, "source_commit": commit, "protocol_version": 2,
            "environment_fingerprint": next(iter(fingerprints))}


# Experiments I–IV ------------------------------------------------------------------------------

def positive(row, field):
    value = float(row[field])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {field}: {row[field]}")
    return value


def validate_campaign(data, loop_rows):
    """Check configuration grain, sample identities and matched timer arithmetic."""
    from gemm_comparison.gemm_comparison import CANDIDATES, SHAPES

    def require(rows, fields, expected):
        keys = [tuple(str(row[field]) for field in fields) for row in rows]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError(f"missing, duplicate or unexpected configuration: {fields}")

    require(data["memory_paths"], ("method", "stages", "bytes_in_flight_per_sm", "sample_index"),
            {(m, str(s), str(b*1024), str(r)) for m in MEMORY_METHODS for s in (2, 4, 8)
             for b in (16, 32, 64) for r in range(30)})
    require(data["umma_throughput"], ("method", "n", "depth", "sample_index"),
            {(m, str(n), str(d), str(r)) for m in UMMA_METHODS for n in (64, 128, 256)
             for d in (4, 16, 64, 256) for r in range(30)})
    require(data["umma_device_scaling"], ("method", "scale", "sample_index"),
            {(m, s, str(r)) for m in UMMA_METHODS for s in ("isolated", "device_scale")
             for r in range(30)})
    require(data["gemm_comparison"], ("shape_index", "candidate_index"),
            {(str(i), str(j)) for i in range(len(SHAPES)) for j in range(len(CANDIDATES))})
    for row in data["umma_throughput"]:
        cycles, ns = positive(row, "elapsed_cycles"), positive(row, "loop_elapsed_ns")
        bracket = float(row["timer_bracket_ns"])
        if not 0 <= bracket < 0.01*ns or not math.isclose(
                positive(row, "effective_sm_clock_mhz"), 1000*cycles/ns, rel_tol=1e-6):
            raise ValueError("isolated timer mismatch or excessive uncertainty")
        if not math.isclose(positive(row, "loop_tflops"), positive(row, "total_flops")/ns/1000,
                            rel_tol=1e-6):
            raise ValueError("isolated work/time mismatch")
        group = 1 if row["method"] == "umma_1sm" else 2
        work = 2*128*group*int(row["n"])*16*int(row["depth"])*int(row["iterations"])
        if int(row["cta_group"]) != group or int(row["total_flops"]) != work or not math.isclose(
                positive(row, "flops_per_cycle"), work/cycles, rel_tol=1e-6):
            raise ValueError("isolated work/cycle mismatch")
    expected_loops = set()
    summary_lookup = {}
    for row in data["umma_device_scaling"]:
        key = row["method"], row["scale"], row["sample_index"]
        summary_lookup[key] = row
        planned = int(row["planned_active_sm_count"])
        expected = int(row["cta_group"]) if row["scale"] == "isolated" else int(row["hardware_sm_count"])
        if planned != expected or int(row["observed_unique_sm_count"]) != planned or row["instrumented"] != "1":
            raise ValueError("scaling residency or instrumentation mismatch")
        for field in ("kernel_time_ms", "total_tflops", "median_effective_sm_clock_mhz",
                      "mean_loop_flops_per_cycle_sm", "loop_device_span_ns", "loop_total_tflops"):
            positive(row, field)
        expected_loops.update((*key, str(block)) for block in range(planned))
    require(loop_rows, ("method", "scale", "sample_index", "block"), expected_loops)
    groups = defaultdict(list)
    for row in loop_rows:
        cycles, ns = positive(row, "elapsed_cycles"), positive(row, "loop_elapsed_ns")
        bracket = float(row["timer_bracket_ns"])
        if not 0 <= bracket < 0.01*ns or int(row["end_ns"])-int(row["start_ns"]) != int(ns):
            raise ValueError("invalid SM timer interval")
        if not math.isclose(positive(row, "effective_sm_clock_mhz"), 1000*cycles/ns, rel_tol=1e-7):
            raise ValueError("SM clock arithmetic mismatch")
        groups[(row["method"], row["scale"], row["sample_index"])].append(row)
    for key, rows in groups.items():
        summary = summary_lookup[key]
        if len({row["smid"] for row in rows}) != len(rows):
            raise ValueError("timed CTAs did not use distinct SMs")
        median = statistics.median(float(r["effective_sm_clock_mhz"]) for r in rows)
        span = max(int(r["end_ns"]) for r in rows)-min(int(r["start_ns"]) for r in rows)
        if int(summary["loop_device_span_ns"]) != span or not math.isclose(
                float(summary["median_effective_sm_clock_mhz"]), median, rel_tol=1e-7):
            raise ValueError("per-SM observations differ from launch summary")
        work_sm = 2*128*256*16*256*int(summary["iterations"])
        for row in rows:
            if not math.isclose(positive(row, "flops_per_cycle_sm"),
                                work_sm/float(row["elapsed_cycles"]), rel_tol=1e-7):
                raise ValueError("per-SM work/cycle mismatch")
        work = work_sm*len(rows)
        checks = {"mean_loop_flops_per_cycle_sm": statistics.fmean(
                    float(r["flops_per_cycle_sm"]) for r in rows),
                  "loop_total_tflops": work/span/1000,
                  "total_tflops": work/float(summary["kernel_time_ms"])/1e9}
        if any(not math.isclose(positive(summary, field), value, rel_tol=1e-7)
               for field, value in checks.items()):
            raise ValueError("scaling work/time/cycle arithmetic mismatch")
    for row in data["gemm_comparison"]:
        i, j = int(row["shape_index"]), int(row["candidate_index"])
        candidate = CANDIDATES[j]
        if row["variant"] != candidate["variant"] or row["method"] != candidate["method"] or \
                tuple(int(row[k]) for k in ("m", "n", "k", "l")) != SHAPES[i] or \
                row["timing_method"] != "cuda_graph":
            raise ValueError("GEMM candidate or timing policy mismatch")
        if candidate["method"] == "cutedsl" and tuple(int(row[k]) for k in ("tile_m", "tile_n")) != candidate["tile"]:
            raise ValueError("GEMM tile differs from its candidate")
        if candidate["method"] == "cutedsl" and tuple(int(row[k]) for k in ("cluster_m", "cluster_n")) != candidate["cluster"]:
            raise ValueError("GEMM cluster differs from its candidate")
        graph, host = positive(row, "kernel_time_ms"), positive(row, "host_event_kernel_time_ms")
        if not math.isclose(positive(row, "host_to_graph_time_ratio"), host/graph, rel_tol=1e-7):
            raise ValueError("paired host/graph time mismatch")


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
    validate_campaign(data, read_csv(path / "raw/umma_device_scaling_loops.csv"))
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
    fields = ("method", "n", "depth")
    columns = ("flops_per_cycle", "effective_sm_clock_mhz", "loop_tflops")
    campaigns = [{column: grouped_medians(r["data"]["umma_throughput"], fields, column)
                  for column in columns} for r in records]
    rows, points = [], []
    for n in (64, 128, 256):
        for depth in (4, 16, 64, 256):
            ratios = [c["flops_per_cycle"][("umma_2sm", str(n), str(depth))] /
                      c["flops_per_cycle"][("umma_1sm", str(n), str(depth))] for c in campaigns]
            for method in UMMA_METHODS:
                group = 1 if method == "umma_1sm" else 2
                key = method, str(n), str(depth)
                totals = [c["flops_per_cycle"][key] for c in campaigns]
                per_sm = [v/group for v in totals]
                rows.append({"method": method, "n": n, "depth": depth, "cta_group": group,
                             **compact_stats(per_sm, "flops_cycle_sm"),
                             "mean_flops_per_cycle": stats(totals)["mean"],
                             "two_sm_to_one_sm_ratio": stats(ratios)["mean"],
                             "mean_issuing_sm_clock_mhz": stats([
                                 c["effective_sm_clock_mhz"][key] for c in campaigns])["mean"],
                             "mean_loop_tflops_per_sm": stats([
                                 c["loop_tflops"][key]/group for c in campaigns])["mean"]})
                points.append({"method": method, "n": n, "depth": depth,
                               "cta_group": group, **stats(per_sm), "total": stats(totals)})
    return rows, {"configurations": points,
                  "best_per_sm_configuration": max(points, key=lambda p: p["mean"])}


def clock_campaigns(record):
    """Aggregate the clock samples recorded inside each configuration's timed launches."""
    samples = [{"moment": float(row["unix_s"]), "sm_clock_mhz": float(row["sm_clock_mhz"]),
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
                        "mean_temperature_c": statistics.fmean(s["temperature_c"] for s in inside),
                        "clock_sample_count": len(inside)}
    return summary


def scaling_results(records):
    fields = ("total_tflops", "kernel_time_ms", "tflops_per_planned_active_sm",
              "median_effective_sm_clock_mhz", "mean_loop_flops_per_cycle_sm", "loop_total_tflops")
    campaigns = [{**{f: grouped_medians(r["data"]["umma_device_scaling"],
                                       ("method", "scale"), f) for f in fields},
                  "clocks": clock_campaigns(r)} for r in records]
    rows, points, geometry = [], [], {}
    for method in UMMA_METHODS:
        for scale in ("isolated", "device_scale"):
            key = method, scale
            sample = next(r for r in records[0]["data"]["umma_device_scaling"]
                          if (r["method"], r["scale"]) == key)
            geometry[key] = sample
            throughput = [c["total_tflops"][key] for c in campaigns]
            clock = stats([c["median_effective_sm_clock_mhz"][key] for c in campaigns])
            context = [c["clocks"][key] for c in campaigns]
            rows.append({"method": method, "scale": scale,
                         "active_sms": int(sample["planned_active_sm_count"]),
                         "work_units": int(sample["work_unit_count"]),
                         "cluster_count": int(sample["cluster_count"]),
                         **compact_stats(throughput, "tflops"),
                         "mean_kernel_time_ms": stats([c["kernel_time_ms"][key] for c in campaigns])["mean"],
                         "mean_tflops_per_sm": stats([c["tflops_per_planned_active_sm"][key]
                                                      for c in campaigns])["mean"],
                         "mean_effective_sm_clock_mhz": clock["mean"],
                         "mean_loop_flops_per_cycle_sm": stats([c["mean_loop_flops_per_cycle_sm"][key]
                                                               for c in campaigns])["mean"],
                         "mean_loop_total_tflops": stats([c["loop_total_tflops"][key]
                                                         for c in campaigns])["mean"],
                         "smi_mean_sm_clock_mhz": statistics.fmean(c["mean_sm_clock_mhz"] for c in context),
                         "mean_temperature_c": statistics.fmean(c["mean_temperature_c"] for c in context),
                         "clock_sample_count": sum(c["clock_sample_count"] for c in context),
                         "per_sm_ratio_vs_isolated": "", "loop_clock_ratio_vs_isolated": "",
                         "loop_work_per_cycle_ratio_vs_isolated": "", "loop_scaling_ratio": ""})
            points.append({"method": method, "scale": scale,
                           "active_sms": int(sample["planned_active_sm_count"]),
                           "work_units": int(sample["work_unit_count"]),
                           "clock": clock, **stats(throughput)})
    efficiency = {}
    for method in UMMA_METHODS:
        isolated, device = (method, "isolated"), (method, "device_scale")
        units = int(geometry[device]["work_unit_count"])
        ratios = [c["total_tflops"][device]/(units*c["total_tflops"][isolated]) for c in campaigns]
        efficiency[method] = {"raw": stats(ratios)}
        row = next(r for r in rows if (r["method"], r["scale"]) == device)
        row["per_sm_ratio_vs_isolated"] = stats(ratios)["mean"]
        for column, field, denominator in (
                ("loop_clock_ratio_vs_isolated", "median_effective_sm_clock_mhz", 1),
                ("loop_work_per_cycle_ratio_vs_isolated", "mean_loop_flops_per_cycle_sm", 1),
                ("loop_scaling_ratio", "loop_total_tflops", units)):
            row[column] = stats([c[field][device]/(denominator*c[field][isolated])
                                 for c in campaigns])["mean"]
    total_ratios = [c["total_tflops"][("umma_2sm", "device_scale")]/
                    c["total_tflops"][("umma_1sm", "device_scale")] for c in campaigns]
    return rows, {"configurations": points, "scaling_efficiency": efficiency,
                  "device_total_ratio_2sm_over_1sm": stats(total_ratios),
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
                     ("shape_index", "shape_id", "m", "n", "k", "l", "variant", "method",
                      "tile_m", "tile_n", "cluster_m", "cluster_n", "timing_method")} |
                    compact_stats(values, "tflops") |
                    {"mean_kernel_time_ms": stats(durations)["mean"],
                     "ratio_vs_cublaslt": stats(ratios)["mean"],
                     "mean_host_event_kernel_time_ms": statistics.fmean(
                         float(entry["host_event_kernel_time_ms"]) for entry in entries),
                     "mean_host_to_graph_time_ratio": statistics.fmean(
                         float(entry["host_to_graph_time_ratio"]) for entry in entries)})
        points.append({"shape_index": int(first["shape_index"]), "shape_id": first["shape_id"],
                       "variant": first["variant"], "method": first["method"], **stats(values)})
    for shape_index in sorted({point["shape_index"] for point in points}):
        best = max((point for point in points if point["shape_index"] == shape_index
                    and point["method"] == "cutedsl"), key=lambda point: point["mean"])
        shapes.append({"shape_index": shape_index, "shape_id": best["shape_id"],
                       "best_cutedsl_variant": best["variant"]})
    return rows, {"configurations": points, "shapes": shapes}


# Experiment V ----------------------------------------------------------------------------------

def timing_results(campaigns, precision_data):
    groups = defaultdict(list)
    for campaign in campaigns:
        for row in campaign["data"]["gemm_comparison"]:
            groups[("IV", row["shape_id"], row["variant"], "bf16")].append((
                float(row["kernel_time_ms"])*1000, float(row["host_event_kernel_time_ms"])*1000))
    if precision_data:
        for row in precision_data[0]:
            groups[("V", row["shape_id"], row["implementation"], row["precision"])].append((
                float(row["kernel_time_us"]), float(row["host_event_time_us"])))
    return [{"experiment": key[0], "shape_id": key[1], "implementation": key[2], "precision": key[3],
             "timing_method": "cuda_graph", "mean_graph_time_us": stats([x[0] for x in values])["mean"],
             "mean_host_event_time_us": stats([x[1] for x in values])["mean"],
             **compact_stats([host/graph for graph, host in values], "host_to_graph_ratio")}
            for key, values in sorted(groups.items())]


def power_results(directory, metadata):
    rows = read_csv(directory / "raw/sustained.csv")
    keys = [(r["method"], int(r["repetition"])) for r in rows]
    expected = {(m, r) for m in UMMA_METHODS for r in (1, 2, 3)}
    parameters = metadata["parameters"]
    if len(keys) != len(expected) or set(keys) != expected or \
            parameters["warmup_ms"] < power_measure.WARMUP_MS or \
            parameters["duration_ms"] < power_measure.DURATION_MS:
        raise ValueError("incomplete or short sustained-power protocol")
    sensors = metadata.get("sensors", [])
    if len(sensors) != 6 or any(s.get("power_average_field") not in
                              ("power.draw.average", "power.draw") for s in sensors):
        raise ValueError("missing or ambiguous sustained-power sensor identity")
    for row in rows:
        start, end = float(row["host_start_unix_s"]), float(row["host_end_unix_s"])
        if end-start < parameters["duration_ms"]/1000-0.001 or row["correctness"] != "OK" or \
                row["scale"] != "device_scale" or positive(row, "gpu_duty_fraction") < 0.90:
            raise ValueError("invalid sustained-power window")
        elapsed, duty = positive(row, "elapsed_s"), positive(row, "gpu_duty_fraction")
        work = 2*128*256*16*256*int(row["iterations"])*int(row["active_sm_count"])*int(row["launch_count"])
        if not math.isclose(end-start, elapsed, abs_tol=0.01) or not math.isclose(
                positive(row, "device_elapsed_ms")/(1000*elapsed), duty, rel_tol=1e-7) or \
                not math.isclose(positive(row, "total_flops"), work, rel_tol=1e-12) or duty > 1.01:
            raise ValueError("sustained work/time accounting mismatch")
        trace = read_csv(directory / f"raw/r{row['repetition']}_{row['method']}.telemetry.csv")
        recomputed = power_measure.window_summary(trace, start, end)
        if not recomputed["stationary"] or row["stationary"] != "True":
            raise ValueError("sustained power/clock window exceeds the declared drift limits")
        for field, value in recomputed.items():
            if isinstance(value, float) and not math.isclose(float(row[field]), value, rel_tol=1e-8):
                raise ValueError("power summary differs from archived telemetry")
    summary_rows, configurations = [], []
    for method in UMMA_METHODS:
        samples = sorted((r for r in rows if r["method"] == method), key=lambda r: int(r["repetition"]))
        powers = [positive(r, "mean_power_w") for r in samples]
        throughput = [positive(r, "total_flops")/positive(r, "elapsed_s")/1e12 for r in samples]
        clocks = [positive(r, "mean_sm_clock_mhz") for r in samples]
        summary_rows.append({"method": method, **compact_stats(powers, "power_w"),
                             "mean_sustained_tflops": stats(throughput)["mean"],
                             "mean_sm_clock_mhz": stats(clocks)["mean"],
                             "mean_temperature_c": statistics.fmean(float(r["mean_temperature_c"]) for r in samples),
                             "minimum_gpu_duty_fraction": min(float(r["gpu_duty_fraction"]) for r in samples),
                             "warmup_ms": parameters["warmup_ms"], "duration_ms": parameters["duration_ms"]})
        configurations.append({"method": method, "power": stats(powers),
                               "throughput": stats(throughput), "clock": stats(clocks)})
    return summary_rows, {"configurations": configurations}

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
                "timing_method": "cuda_graph",
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
                "timing_method": "cuda_graph",
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
    """DRAM and L2 traffic of nine launches beside graph and host-event timing."""
    if cache_state != metadata.get("cache_state"):
        raise ValueError(f"{directory}: the requested cache state differs from the acquisition")
    if sorted(metadata.get("captures", [])) != sorted(case for case, _, _ in profile_gemm.CASES) or \
            not metadata.get("l2_read_metric", {}).get("collected"):
        raise ValueError("missing GEMM profile cases or calibrated TMA counter")
    reference = {(row["shape_id"], row["variant"]): row for row in read_csv(gemm_summary)}
    rows = []
    for case, shape, variant in profile_gemm.CASES:
        text = (directory / f"{case}.csv").read_text(encoding="utf-8")
        # The L2 read metric is exported only when its calibration passed.
        l2_metric = profile_gemm.L2_READ_METRIC if profile_gemm.L2_READ_METRIC in \
            text.partition("\n")[0] else None
        if not l2_metric:
            raise ValueError(f"{case}: the calibrated TMA counter is missing")
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
            "variant": variant, "method": "cutedsl" if variant.startswith("persistent_") else "cublaslt",
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
            "graph_mean_kernel_time_us": 1e3 * float(timing["mean_kernel_time_ms"]),
            "graph_mean_tflops": float(timing["mean_tflops"]),
            "timing_method": timing["timing_method"],
            "host_event_mean_kernel_time_us": 1e3*float(timing["mean_host_event_kernel_time_ms"]),
            # Separate acquisitions under dynamic clocks: descriptive ratios,
            # not a calibrated subtraction of host overhead or profiler effects.
            "host_to_profile_time_ratio": 1e6*float(timing["mean_host_event_kernel_time_ms"])/duration_ns,
            "graph_to_profile_time_ratio": 1e6*float(timing["mean_kernel_time_ms"])/duration_ns,
            "predicted_tma_read_bytes": (2*m*n*k*batch*(1/int(timing["tile_m"])+1/int(timing["tile_n"]))
                                         if timing["method"] == "cutedsl" else "")})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for the summaries; an existing one is never reused")
    parser.add_argument("--study", type=Path,
                        help="a protocol-v2 make final-study directory: nine CSVs and seven SVGs")
    parser.add_argument("--campaigns", type=Path, nargs=3, metavar="CAMPAIGN",
                        help="three complete Experiment I–IV campaigns")
    parser.add_argument("--power", type=Path, help="sustained full-device UMMA power run")
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
        args.power = args.study / "power"
        args.gemm_profile, args.cache_state = args.study / "gemm-profile-hot", "hot"
    if not (args.campaigns or args.precision or args.gemm_profile or args.power):
        parser.error("name --study, --campaigns, --precision, --power or --gemm-profile")
    if args.gemm_profile and not (args.cache_state and (args.campaigns or args.gemm_summary)):
        parser.error("--gemm-profile needs --cache-state and --campaigns or --gemm-summary")

    campaigns = load_campaigns(args.campaigns) if args.campaigns else []
    precision_data = load_precision(args.precision) if args.precision else None
    profile_metadata = read_metadata(args.gemm_profile, profile=True) if args.gemm_profile else None
    power_metadata = read_metadata(args.power) if args.power else None
    records = [campaign["metadata"] for campaign in campaigns]
    if power_metadata:
        records.append(power_metadata)
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
    if args.power:
        rows, summary = power_results(args.power, power_metadata)
        write_csv(output / "umma_power.csv", rows)
        (output / "umma_power.svg").write_text(figures.power_figure(summary), encoding="utf-8")
    if args.campaigns or args.precision:
        write_csv(output / "timing_comparison.csv", timing_results(campaigns, precision_data))
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
              "campaigns": args.campaigns, "precision": args.precision, "power": args.power,
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
