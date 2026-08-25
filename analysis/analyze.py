#!/usr/bin/env python3
"""Summarize three campaigns and generate four thesis figures."""

import argparse
import csv
import html
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ldgsts", "tma")
UMMA_METHODS = ("umma_1sm", "umma_2sm")
EXPERIMENTS = ("memory_paths", "umma_throughput", "umma_device_scaling", "gemm_comparison")
COLORS = {"ldgsts": "#2563eb", "tma": "#d97706",
          "umma_1sm": "#2563eb", "umma_2sm": "#d97706"}
GEMM_COLORS = {"nonpersistent_1cta": "#2563eb", "persistent_1cta": "#7c3aed",
               "persistent_2cta": "#d97706", "heuristic_first_supported": "#15803d"}


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


def read_campaign(path):
    path = path.resolve()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["kind"] != "final":
        raise ValueError(f"{path.name} is not a final campaign")
    datasets = {}
    for experiment in EXPERIMENTS:
        with (path / "raw" / f"{experiment}.csv").open(newline="", encoding="utf-8") as source:
            datasets[experiment] = list(csv.DictReader(source))
    profile_path = path / "ncu/index.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}
    return {"metadata": metadata, "data": datasets, "profile": profile}


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
            for method in METHODS:
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
                          key=lambda point: point["mean"]) for method in METHODS}
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
            unit = case["units"].get(metric, "").lower()
            factor = {"cycle/nsecond": 1e9, "hz": 1.0}.get(unit)
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


def scaling_results(records):
    campaigns = []
    for record in records:
        values = {}
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
            rows.append({"method": method, "scale": scale,
                         "active_sms": int(sample["planned_active_sm_count"]),
                         "work_units": int(sample["work_unit_count"]),
                         "cluster_count": int(sample["cluster_count"]),
                         **compact_stats(throughputs, "tflops"),
                         "mean_kernel_time_ms": stats(times)["mean"],
                         "mean_tflops_per_sm": stats(per_sm)["mean"],
                         "scaling_efficiency_percent": ""})
            points.append({"method": method, "scale": scale,
                           "active_sms": int(sample["planned_active_sm_count"]),
                           "work_units": int(sample["work_unit_count"]), **stats(throughputs)})

    efficiencies = {}
    for method in UMMA_METHODS:
        # The device launches one work unit per CTA or per two-CTA cluster.
        units = int(geometry[(method, "device_scale")]["work_unit_count"])
        values = [100 * campaign["total_tflops"][(method, "device_scale")] /
                  (campaign["total_tflops"][(method, "isolated")] * units)
                  for campaign in campaigns]
        efficiencies[method] = stats(values)
        next(row for row in rows if row["method"] == method and row["scale"] == "device_scale")[
            "scaling_efficiency_percent"] = efficiencies[method]["mean"]
    ratios = [campaign["total_tflops"][("umma_2sm", "device_scale")] /
              campaign["total_tflops"][("umma_1sm", "device_scale")]
              for campaign in campaigns]
    return rows, {"configurations": points, "scaling_efficiency": efficiencies,
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


def svg_text(x, y, value, **attributes):
    values = {"x": f"{x:.1f}", "y": f"{y:.1f}", "font-family": "Arial, sans-serif",
              "font-size": "12", "fill": "#334155"}
    values.update({name.replace("_", "-"): item for name, item in attributes.items()})
    properties = " ".join(f'{name}="{html.escape(str(item))}"'
                          for name, item in values.items())
    return f"<text {properties}>{html.escape(str(value))}</text>"


def svg_start(title, subtitle, width=1260, height=490):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            svg_text(34, 40, title, font_size="22", font_weight="700", fill="#0f172a"),
            svg_text(34, 66, subtitle, font_size="13", fill="#64748b")]


def line_figure(title, subtitle, panels, x_labels, y_label, footnote):
    output = svg_start(title, subtitle)
    panel_width, gap, left, top, bottom = 330, 65, 82, 126, 382
    for index, panel in enumerate(panels):
        x0 = left + index * (panel_width + gap)
        points = [point for series in panel["series"].values() for point in series]
        lower = min(point["minimum"] for point in points) * 0.96
        upper = max(point["maximum"] for point in points) * 1.04
        if upper <= lower:
            upper = lower + 1
        y = lambda value: bottom - (value - lower) * (bottom - top) / (upper - lower)
        x = lambda position: x0 + position * panel_width / max(len(x_labels) - 1, 1)
        output.append(svg_text(x0 + panel_width / 2, 106, panel["title"],
                               text_anchor="middle", font_size="14", font_weight="700"))
        for tick in range(5):
            value = lower + tick * (upper - lower) / 4
            yy = y(value)
            output.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x0 + panel_width:.1f}" '
                          f'y2="{yy:.1f}" stroke="#e2e8f0"/>')
            output.append(svg_text(x0 - 8, yy + 4, f"{value:,.0f}", text_anchor="end",
                                   font_size="10", fill="#64748b"))
        for position, label in enumerate(x_labels):
            output.append(svg_text(x(position), bottom + 23, label,
                                   text_anchor="middle", font_size="11"))
        for method, series in panel["series"].items():
            color = COLORS[method]
            coordinates = " ".join(f"{x(i):.1f},{y(point['mean']):.1f}"
                                   for i, point in enumerate(series))
            output.append(f'<polyline points="{coordinates}" fill="none" '
                          f'stroke="{color}" stroke-width="2.4"/>')
            for position, point in enumerate(series):
                xx = x(position)
                output.append(f'<line x1="{xx:.1f}" y1="{y(point["minimum"]):.1f}" '
                              f'x2="{xx:.1f}" y2="{y(point["maximum"]):.1f}" '
                              f'stroke="{color}" stroke-width="1.5"/>')
                output.append(f'<circle cx="{xx:.1f}" cy="{y(point["mean"]):.1f}" '
                              f'r="4.1" fill="{color}"/>')
    for position, method in enumerate(panels[0]["series"]):
        xx = 875 + position * 175
        output.append(f'<rect x="{xx}" y="80" width="12" height="12" fill="{COLORS[method]}"/>')
        output.append(svg_text(xx + 18, 90, method, font_size="11"))
    output.append(svg_text(17, 256, y_label, text_anchor="middle", font_size="12",
                           transform="rotate(-90 17 256)"))
    output.append(svg_text(34, 457, footnote, font_size="11", fill="#64748b"))
    return "\n".join([*output, "</svg>"]) + "\n"


def memory_figure(summary):
    lookup = {(point["method"], point["stages"], point["bytes_in_flight_kib"]): point
              for point in summary["configurations"]}
    panels = [{"title": f"Stages = {stages}",
               "series": {method: [lookup[(method, stages, size)] for size in (16, 32, 64)]
                          for method in METHODS}} for stages in (2, 4, 8)]
    return line_figure("LDGSTS versus TMA: effective transfer rate",
                       "Mean of three campaign medians; whiskers show the observed range.",
                       panels, ("16 KiB", "32 KiB", "64 KiB"), "Effective GB/s",
                       "Logical useful bytes divided by kernel time; not a direct DRAM bandwidth counter.")


def umma_figure(summary):
    lookup = {(point["method"], point["n"], point["depth"]): point
              for point in summary["configurations"]}
    panels = [{"title": f"N = {n}",
               "series": {method: [lookup[(method, n, depth)] for depth in (4, 16, 64, 256)]
                          for method in UMMA_METHODS}} for n in (64, 128, 256)]
    return line_figure("BF16 UMMA: isolated 1-SM versus 2-SM throughput",
                       "Mean of three campaign medians; timing uses the per-SM %clock64 counter.",
                       panels, ("4", "16", "64", "256"), "FLOP/cycle/SM",
                       "Pipeline depth on the x-axis; 2-SM throughput is normalized by its two active SMs.")


def gemm_figure(summary):
    width, height, left, right, top, bottom = 1260, 510, 82, 36, 135, 402
    output = svg_start("CuTe DSL versus cuBLASLt: BF16 GEMM throughput",
                       "Mean of three campaigns; whiskers show their minimum and maximum.", width, height)
    points = summary["configurations"]
    shapes = sorted({(point["shape_index"], point["shape_id"]) for point in points})
    variants = tuple(GEMM_COLORS)
    maximum = max(point["maximum"] for point in points) * 1.10
    y = lambda value: bottom - value * (bottom - top) / maximum
    for tick in range(6):
        value = maximum * tick / 5
        yy = y(value)
        output.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" '
                      'stroke="#e2e8f0"/>')
        output.append(svg_text(left - 9, yy + 4, f"{value:,.0f}", text_anchor="end",
                               font_size="10", fill="#64748b"))
    labels = {"nonpersistent_1cta": "NP1", "persistent_1cta": "P1",
              "persistent_2cta": "P2", "heuristic_first_supported": "cuBLASLt"}
    for index, variant in enumerate(variants):
        xx = 535 + index * 158
        output.append(f'<rect x="{xx}" y="83" width="12" height="12" fill="{GEMM_COLORS[variant]}"/>')
        output.append(svg_text(xx + 18, 94, labels[variant], font_size="11"))
    lookup = {(point["shape_index"], point["variant"]): point for point in points}
    group_width = (width - left - right) / len(shapes)
    bar_width = (group_width - 48) / len(variants)
    for index, (shape_index, shape_id) in enumerate(shapes):
        for position, variant in enumerate(variants):
            point = lookup[(shape_index, variant)]
            xx = left + index * group_width + 20 + position * (bar_width + 2)
            yy = y(point["mean"])
            output.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bar_width:.1f}" '
                          f'height="{bottom-yy:.1f}" fill="{GEMM_COLORS[variant]}"/>')
            center = xx + bar_width / 2
            output.append(f'<line x1="{center:.1f}" y1="{y(point["minimum"]):.1f}" '
                          f'x2="{center:.1f}" y2="{y(point["maximum"]):.1f}" stroke="#0f172a"/>')
        label = "×".join(shape_id.removesuffix("x1").split("x")[:2])
        output.append(svg_text(left + (index + 0.5) * group_width, bottom + 26,
                               label, text_anchor="middle", font_size="11"))
    output.append(svg_text(18, 268, "TFLOP/s", text_anchor="middle",
                           transform="rotate(-90 18 268)"))
    output.append(svg_text(34, 477,
                           "Hot-cache kernel timing; all variants share operands and pass the same FP32 reference.",
                           font_size="11", fill="#64748b"))
    return "\n".join([*output, "</svg>"]) + "\n"


def scaling_panel(output, x0, width, title, bars, unit, reference=None):
    top, bottom = 145, 376
    maximum = max([point["maximum"] for _, point in bars] + ([reference] if reference else [])) * 1.12
    y = lambda value: bottom - value * (bottom - top) / maximum
    output.append(svg_text(x0 + width / 2, 119, title, text_anchor="middle",
                           font_size="14", font_weight="700"))
    for tick in range(5):
        value = maximum * tick / 4
        yy = y(value)
        output.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x0+width:.1f}" '
                      f'y2="{yy:.1f}" stroke="#e2e8f0"/>')
        output.append(svg_text(x0 - 8, yy + 4, f"{value:,.0f}", text_anchor="end", font_size="10"))
    if reference:
        output.append(f'<line x1="{x0:.1f}" y1="{y(reference):.1f}" x2="{x0+width:.1f}" '
                      f'y2="{y(reference):.1f}" stroke="#15803d" stroke-dasharray="5 4"/>')
    bar_width = width / (len(bars) * 2.1)
    for index, (method, point) in enumerate(bars):
        center = x0 + (index + 0.5) * width / len(bars)
        yy = y(point["mean"])
        output.append(f'<rect x="{center-bar_width/2:.1f}" y="{yy:.1f}" width="{bar_width:.1f}" '
                      f'height="{bottom-yy:.1f}" fill="{COLORS[method]}"/>')
        output.append(f'<line x1="{center:.1f}" y1="{y(point["minimum"]):.1f}" '
                      f'x2="{center:.1f}" y2="{y(point["maximum"]):.1f}" stroke="#0f172a"/>')
        output.append(svg_text(center, bottom + 23, method, text_anchor="middle", font_size="11"))
    output.append(svg_text(x0 + width / 2, bottom + 44, unit,
                           text_anchor="middle", font_size="11", fill="#64748b"))


def scaling_figure(summary):
    output = svg_start("BF16 UMMA: isolated work unit versus all usable SMs",
                       "Independent throughput axes avoid mixing isolated and whole-device scales.")
    lookup = {(point["method"], point["scale"]): point for point in summary["configurations"]}
    scaling_panel(output, 86, 302, "Isolated work unit",
                  [(method, lookup[(method, "isolated")]) for method in UMMA_METHODS], "Total TFLOP/s")
    scaling_panel(output, 485, 302, "Whole device",
                  [(method, lookup[(method, "device_scale")]) for method in UMMA_METHODS], "Total TFLOP/s")
    scaling_panel(output, 886, 290, "Scaling efficiency",
                  [(method, summary["scaling_efficiency"][method]) for method in UMMA_METHODS],
                  "Percentage of ideal linear scaling", reference=100.0)
    output.append(svg_text(34, 465,
                           "Whole-kernel CUDA-event timing; coverage requires simultaneous residency.",
                           font_size="11", fill="#64748b"))
    return "\n".join([*output, "</svg>"]) + "\n"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.6f}" if isinstance(value, float) else value
                             for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description="Summarize three GB300 campaigns.")
    parser.add_argument("--campaign", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.campaign) != 3:
        parser.error("exactly three final campaigns are required")
    records = [read_campaign(path) for path in args.campaign]
    if len({record["metadata"]["campaign_id"] for record in records}) != 3:
        raise ValueError("the three campaigns must be distinct")
    if len({record["metadata"]["gpu"]["uuid"] for record in records}) != 1:
        raise ValueError("all campaigns must use the same GPU")

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    processors = {"memory_paths": (memory_results, memory_figure),
                  "umma_throughput": (umma_results, umma_figure),
                  "umma_device_scaling": (scaling_results, scaling_figure),
                  "gemm_comparison": (gemm_results, gemm_figure)}
    for name, (analyze, figure) in processors.items():
        rows, summary = analyze(records)
        write_csv(output / f"{name}.csv", rows)
        (output / f"{name}.svg").write_text(figure(summary), encoding="utf-8")
    print(f"analysis: COMPLETE {output}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"analysis: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
