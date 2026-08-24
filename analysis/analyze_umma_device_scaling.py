#!/usr/bin/env python3
"""Aggregate exactly three supplementary UMMA device-scaling campaigns.

The question this answers is not "which tcgen05.mma instruction is faster" but
"how does each of the two UMMA execution modes behave when it is replicated
across the maximum simultaneously usable SM capacity of the GPU".  Both
launch scales are measured inside the same campaign, so every scaling
efficiency is computed against an isolated baseline collected on the same
GPU, at the same commit, minutes apart.

Each campaign's 30 repetitions are reduced to a median first; only the three
campaign-level medians ever enter a cross-campaign statistic.  The 90
repetitions are never pooled as if they were independent campaigns, and three
campaigns support descriptive statistics only, never significance testing.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
import shutil
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_umma_scaling_campaign as contract  # noqa: E402


SCHEMA = "gb300.umma-scaling-analysis.v1"
CAMPAIGN_COUNT = 3
NA = "not_applicable"
CV_REVIEW_PERCENT = 5.0

MEASURED_NOTE = "median_of_repetitions_then_descriptive_statistics_across_three_campaigns"
DERIVED_NOTE = "derived_from_per_campaign_medians"
DIAGNOSTIC_NOTE = "diagnostic_only_not_aggregated"

CONFIG_COLORS = {
    ("umma_1sm", "isolated"): "#1f6feb",
    ("umma_2sm", "isolated"): "#d97706",
    ("umma_1sm", "device_scale"): "#1f6feb",
    ("umma_2sm", "device_scale"): "#d97706",
}


class AnalysisError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise AnalysisError(f"{path}: expected one JSON object")
    return document


def campaign(path: Path) -> dict:
    path = path.resolve()
    manifest = read_json(path / "manifest.json")
    if manifest.get("schema_version") != contract.SCHEMA:
        raise AnalysisError(f"{path}: unsupported campaign schema")
    if manifest.get("state") != "COMPLETE" or manifest.get("campaign_kind") != "final":
        raise AnalysisError(f"{path}: only COMPLETE final campaigns are accepted")
    campaign_id = manifest.get("campaign_id")
    if not isinstance(campaign_id, str) or not contract.CAMPAIGN_RE.fullmatch(campaign_id):
        raise AnalysisError(f"{path}: invalid campaign ID")
    if path.name != campaign_id:
        raise AnalysisError(f"{path}: directory name must equal campaign ID")
    commit = manifest.get("git_commit")
    gpu = manifest.get("gpu")
    image = manifest.get("container_image")
    parameters = manifest.get("parameters")
    if not isinstance(commit, str) or len(commit) != 40:
        raise AnalysisError(f"{path}: invalid Git commit")
    if not isinstance(gpu, dict) or not isinstance(gpu.get("uuid"), str):
        raise AnalysisError(f"{path}: missing GPU identity")
    if not isinstance(image, dict) or not isinstance(image.get("id"), str):
        raise AnalysisError(f"{path}: missing container image identity")
    if parameters != contract.FROZEN_PARAMETERS:
        raise AnalysisError(f"{path}: benchmark parameters do not match the frozen contract")
    sources = manifest.get("source_sha256")
    if (
        not isinstance(sources, dict)
        or set(sources) != set(contract.SOURCE_FILES)
        or any(not isinstance(value, str) or not contract.SHA256_RE.fullmatch(value)
               for value in sources.values())
    ):
        raise AnalysisError(f"{path}: source inventory or SHA-256 digests are not exact")

    recorded = manifest.get("artifact_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(contract.RAW_FILES):
        raise AnalysisError(f"{path}: raw artifact inventory is not exact")
    for relative in contract.RAW_FILES:
        artifact = path / relative
        if not artifact.is_file() or sha256_file(artifact) != recorded[relative]:
            raise AnalysisError(f"{path}: hash mismatch for {relative}")

    raw_path = path / contract.RAW_FILE
    facts = contract.validate_scaling_csv(
        raw_path, commit=commit, gpu_uuid=gpu["uuid"], campaign_kind="final",
        repetitions=parameters["repetitions"],
    )
    _, rows = contract.read_csv(raw_path)
    return {
        "path": path,
        "manifest": manifest,
        "campaign_id": campaign_id,
        "facts": facts,
        "rows": rows,
        "repetitions": parameters["repetitions"],
    }


def validate_population(records: list[dict]) -> None:
    if len(records) != CAMPAIGN_COUNT:
        raise AnalysisError("exactly three --campaign arguments are required")
    ids = [record["campaign_id"] for record in records]
    if len(set(ids)) != CAMPAIGN_COUNT:
        raise AnalysisError("the three campaign IDs must be distinct")
    for field, getter in (
        ("Git commit", lambda r: r["manifest"]["git_commit"]),
        ("GPU UUID", lambda r: r["manifest"]["gpu"]["uuid"]),
        ("GPU name", lambda r: r["manifest"]["gpu"]["name"]),
        ("driver version", lambda r: r["manifest"]["gpu"]["driver_version"]),
        ("container image", lambda r: r["manifest"]["container_image"]["id"]),
        ("parameter set", lambda r: json.dumps(r["manifest"]["parameters"], sort_keys=True)),
        ("source hashes", lambda r: json.dumps(r["manifest"]["source_sha256"], sort_keys=True)),
        ("hardware SM count", lambda r: r["facts"]["hardware_sm_count"]),
        ("shared-memory reservation",
         lambda r: r["facts"]["shared_memory_reservation_bytes"]),
        ("launch geometry", lambda r: json.dumps(r["facts"]["configurations"], sort_keys=True)),
    ):
        values = {getter(record) for record in records}
        if len(values) != 1:
            raise AnalysisError(f"the three campaigns do not share one {field}")


# ---------------------------------------------------------------------------
# Statistics.  Within-campaign first, then across the three campaign medians.
# ---------------------------------------------------------------------------
def stats(values: list[float]) -> dict[str, float]:
    if len(values) != CAMPAIGN_COUNT or any(not math.isfinite(value) for value in values):
        raise AnalysisError("every cross-campaign statistic requires three finite values")
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values)
    return {
        "mean": mean,
        "median": statistics.median(values),
        "stdev_sample": stdev,
        "cv_percent": abs(100.0 * stdev / mean) if mean else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def within_campaign(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": mean,
        "cv_percent": abs(100.0 * stdev / mean) if mean else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def fmt(value, decimals: int = 6) -> str:
    if value is None:
        return NA
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return NA
    return f"{value:.{decimals}f}"


def metric_row(section: str, keys: dict[str, str], metric: str, unit: str, values: list,
               notes: str, *, evidence: str, aggregate: bool = True,
               decimals: int = 6) -> dict:
    row = {"schema_version": SCHEMA, "section": section, **keys, "metric": metric, "unit": unit,
           "evidence_class": evidence, "campaign_count": CAMPAIGN_COUNT, "notes": notes}
    for index, value in enumerate(values, 1):
        row[f"campaign_{index}_value"] = fmt(value, decimals)
    if aggregate:
        summary = stats([float(value) for value in values])
        row.update({
            "mean": fmt(summary["mean"], decimals),
            "median": fmt(summary["median"], decimals),
            "stdev_sample": fmt(summary["stdev_sample"], decimals),
            "cross_campaign_cv_percent": fmt(summary["cv_percent"], 4),
            "minimum": fmt(summary["minimum"], decimals),
            "maximum": fmt(summary["maximum"], decimals),
            "cross_campaign_cv_review_flag": str(summary["cv_percent"] > CV_REVIEW_PERCENT),
        })
    else:
        for field in ("mean", "median", "stdev_sample", "cross_campaign_cv_percent",
                      "minimum", "maximum", "cross_campaign_cv_review_flag"):
            row[field] = NA
    return row


def campaign_medians(record: dict) -> dict:
    """Reduces one campaign's repetitions to per-configuration medians."""
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {
        key: {"total_tflops": [], "kernel_time_ms": [], "per_planned_sm": [], "per_evidenced_sm": []}
        for key in contract.CONFIGURATIONS
    }
    for row in record["rows"]:
        bucket = grouped[(row["method"], row["scale"])]
        bucket["total_tflops"].append(float(row["total_tflops"]))
        bucket["kernel_time_ms"].append(float(row["kernel_time_ms"]))
        bucket["per_planned_sm"].append(float(row["tflops_per_planned_active_sm"]))
        evidenced = row["tflops_per_evidenced_active_sm"]
        if evidenced != NA:
            bucket["per_evidenced_sm"].append(float(evidenced))
    result = {}
    for key, bucket in grouped.items():
        entry = {name: within_campaign(values) for name, values in bucket.items() if values}
        if "total_tflops" not in entry:
            raise AnalysisError(f"{record['campaign_id']}: {key} has no throughput samples")
        result[key] = entry
    return result


def derived(record: dict, medians: dict) -> dict:
    """Per-campaign derived quantities, from that campaign's own medians."""
    configurations = record["facts"]["configurations"]
    iso_one = medians[("umma_1sm", "isolated")]["total_tflops"]["median"]
    iso_two = medians[("umma_2sm", "isolated")]["total_tflops"]["median"]
    dev_one = medians[("umma_1sm", "device_scale")]["total_tflops"]["median"]
    dev_two = medians[("umma_2sm", "device_scale")]["total_tflops"]["median"]
    blocks = configurations["umma_1sm/device_scale"]["work_unit_count"]
    clusters = configurations["umma_2sm/device_scale"]["work_unit_count"]
    planned_one = configurations["umma_1sm/device_scale"]["planned_active_sm_count"]
    planned_two = configurations["umma_2sm/device_scale"]["planned_active_sm_count"]
    ideal_one = iso_one * blocks
    ideal_two = iso_two * clusters
    return {
        "isolated_1sm_tflops": iso_one,
        "isolated_2sm_tflops": iso_two,
        "device_1sm_tflops": dev_one,
        "device_2sm_tflops": dev_two,
        "active_1sm_blocks": blocks,
        "active_2sm_clusters": clusters,
        "planned_active_sm_1sm": planned_one,
        "planned_active_sm_2sm": planned_two,
        "ideal_linear_1sm_tflops": ideal_one,
        "ideal_linear_2sm_tflops": ideal_two,
        "scaling_efficiency_1sm_percent": 100.0 * dev_one / ideal_one,
        "scaling_efficiency_2sm_percent": 100.0 * dev_two / ideal_two,
        "gap_from_ideal_1sm_percent": 100.0 * (1.0 - dev_one / ideal_one),
        "gap_from_ideal_2sm_percent": 100.0 * (1.0 - dev_two / ideal_two),
        "device_total_ratio_2sm_over_1sm": dev_two / dev_one,
        "device_per_active_sm_ratio_2sm_over_1sm":
            (dev_two / planned_two) / (dev_one / planned_one),
    }


def analysis(records: list[dict]) -> tuple[list[dict], dict, list[str]]:
    per_campaign = [campaign_medians(record) for record in records]
    per_derived = [derived(record, medians) for record, medians in zip(records, per_campaign)]
    geometry = records[0]["facts"]
    warnings: list[str] = []

    rows: list[dict] = []
    configurations = []
    for method, scale in contract.CONFIGURATIONS:
        key = (method, scale)
        keys = {"method": method, "scale": scale}
        entries = [medians[key] for medians in per_campaign]
        totals = [entry["total_tflops"]["median"] for entry in entries]
        times = [entry["kernel_time_ms"]["median"] for entry in entries]
        per_planned = [entry["per_planned_sm"]["median"] for entry in entries]
        rows.append(metric_row("configuration", keys, "median_total_tflops", "TFLOP/s", totals,
                               MEASURED_NOTE, evidence="cuda_event_measured"))
        rows.append(metric_row("configuration", keys, "median_kernel_time_ms", "ms", times,
                               MEASURED_NOTE, evidence="cuda_event_measured"))
        rows.append(metric_row("configuration", keys, "median_tflops_per_planned_active_sm",
                               "TFLOP/s/SM", per_planned, MEASURED_NOTE,
                               evidence="cuda_event_measured"))
        if all("per_evidenced_sm" in entry for entry in entries):
            per_evidenced = [entry["per_evidenced_sm"]["median"] for entry in entries]
            rows.append(metric_row("configuration", keys, "median_tflops_per_evidenced_active_sm",
                                   "TFLOP/s/SM", per_evidenced, MEASURED_NOTE,
                                   evidence="cuda_event_measured"))
        else:
            rows.append(metric_row("configuration", keys, "median_tflops_per_evidenced_active_sm",
                                   "TFLOP/s/SM", [NA] * CAMPAIGN_COUNT,
                                   "coverage_not_evidenced", evidence="diagnostic",
                                   aggregate=False))
        for metric, field in (("within_campaign_sample_count", "count"),
                              ("within_campaign_total_tflops_cv_percent", "cv_percent")):
            rows.append(metric_row("configuration", keys, metric, NA,
                                   [entry["total_tflops"][field] for entry in entries],
                                   DIAGNOSTIC_NOTE, evidence="diagnostic", aggregate=False))
        placement = geometry["configurations"][f"{method}/{scale}"]
        for metric in ("grid_blocks", "cluster_size", "work_unit_count",
                       "planned_active_sm_count", "unused_sm_count", "coverage_status",
                       "max_active_clusters"):
            rows.append(metric_row("coverage", keys, metric, NA,
                                   [placement[metric]] * CAMPAIGN_COUNT, DIAGNOSTIC_NOTE,
                                   evidence="diagnostic", aggregate=False))
        configurations.append({
            "method": method, "scale": scale, **placement,
            "total_tflops": stats(totals), "kernel_time_ms": stats(times),
            "tflops_per_planned_active_sm": stats(per_planned),
        })

    scaling_keys = {"method": NA, "scale": NA}
    scaling = {}
    for metric, unit, decimals, evidence in (
        ("ideal_linear_1sm_tflops", "TFLOP/s", 6, "derived"),
        ("ideal_linear_2sm_tflops", "TFLOP/s", 6, "derived"),
        ("scaling_efficiency_1sm_percent", "percent", 6, "derived"),
        ("scaling_efficiency_2sm_percent", "percent", 6, "derived"),
        ("gap_from_ideal_1sm_percent", "percent", 6, "derived"),
        ("gap_from_ideal_2sm_percent", "percent", 6, "derived"),
        ("device_total_ratio_2sm_over_1sm", "ratio", 6, "derived"),
        ("device_per_active_sm_ratio_2sm_over_1sm", "ratio", 6, "derived"),
    ):
        values = [item[metric] for item in per_derived]
        note = DERIVED_NOTE
        if metric.startswith("scaling_efficiency") and any(
            not 0.0 <= value <= 100.0 for value in values
        ):
            note = "value_outside_0_100_preserved_unclamped"
        rows.append(metric_row("scaling", scaling_keys, metric, unit, values, note,
                               evidence=evidence, decimals=decimals))
        scaling[metric] = stats(values)

    equal = geometry["equal_active_sm_coverage"]
    planned_one = geometry["configurations"]["umma_1sm/device_scale"]["planned_active_sm_count"]
    planned_two = geometry["configurations"]["umma_2sm/device_scale"]["planned_active_sm_count"]
    rows.append(metric_row("scaling", scaling_keys, "equal_active_sm_coverage", NA,
                           [str(equal)] * CAMPAIGN_COUNT,
                           DIAGNOSTIC_NOTE if equal
                           else "unequal_active_sm_counts_totals_are_not_directly_comparable",
                           evidence="diagnostic", aggregate=False))
    if not equal:
        warnings.append(
            f"the two device-scale configurations do not cover the same number of SMs "
            f"({planned_one} vs {planned_two}); their total throughputs are not directly "
            "comparable and device_total_ratio_2sm_over_1sm must be read together with "
            "device_per_active_sm_ratio_2sm_over_1sm"
        )
    unused = geometry["configurations"]["umma_2sm/device_scale"]["unused_sm_count"]
    if unused:
        warnings.append(
            f"the 2-SM device-scale configuration leaves {unused} of "
            f"{geometry['hardware_sm_count']} SMs unused"
        )
    for method, scale in contract.DEVICE_SCALE_CONFIGURATIONS:
        status = geometry["configurations"][f"{method}/{scale}"]["coverage_status"]
        if status != "full_device_coverage":
            warnings.append(f"{method}/{scale} coverage is {status}, not full-device coverage")
    for row in rows:
        if row.get("cross_campaign_cv_review_flag") == "True":
            warnings.append(
                f"{row['section']}/{row.get('method')}/{row.get('scale')}/{row['metric']}: "
                f"cross-campaign CV {row['cross_campaign_cv_percent']}% exceeds "
                f"{CV_REVIEW_PERCENT}%"
            )

    summary = {
        "hardware_sm_count": geometry["hardware_sm_count"],
        "shared_memory_reservation_bytes": geometry["shared_memory_reservation_bytes"],
        "equal_active_sm_coverage": equal,
        "configurations": configurations,
        "scaling": scaling,
        "per_campaign_derived": [
            {"campaign_id": record["campaign_id"], **item}
            for record, item in zip(records, per_derived)
        ],
    }
    return rows, summary, warnings


# ---------------------------------------------------------------------------
# Figure: throughput on two separate scales, plus scaling efficiency.
# ---------------------------------------------------------------------------
def svg_text(x: float, y: float, value: str, **attrs: str) -> str:
    properties = {"x": f"{x:.1f}", "y": f"{y:.1f}", **attrs}
    encoded = " ".join(f'{key.replace("_", "-")}="{html.escape(str(val))}"'
                       for key, val in properties.items())
    return f"<text {encoded}>{html.escape(value)}</text>"


def nice_step(value: float) -> float:
    """Rounds an axis step up to 1, 1.5, 2, 2.5, 3, 4, 5, 6 or 8 times a power of ten."""
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10.0 ** exponent
    for step in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if value <= step * base * (1.0 + 1e-12):
            return step * base
    return 10.0 * base


def bar_panel(out: list[str], *, x0: float, width: float, top: float, bottom: float,
              title: str, unit: str, bars: list[dict], reference: float | None = None) -> None:
    highs = [bar["maximum"] for bar in bars] + ([reference] if reference else [])
    y_max = max(highs) * 1.18
    if y_max <= 0:
        y_max = 1.0
    y_max = 4.0 * nice_step(y_max / 4.0)

    def y(value: float) -> float:
        return bottom - value * (bottom - top) / y_max

    out.append(f'<rect x="{x0:.1f}" y="{top:.1f}" width="{width:.1f}" '
               f'height="{bottom - top:.1f}" fill="none" stroke="#9aa0a6" stroke-width="1"/>')
    out.append(svg_text(x0 + width / 2, top - 28, title, text_anchor="middle",
                        font_family="Arial, sans-serif", font_size="14", font_weight="700",
                        fill="#202124"))
    out.append(svg_text(x0 + width / 2, top - 11, unit, text_anchor="middle",
                        font_family="Arial, sans-serif", font_size="11", fill="#5f6368"))
    for tick in range(5):
        value = tick * y_max / 4
        yy = y(value)
        out.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x0 + width:.1f}" y2="{yy:.1f}" '
                   'stroke="#e5e7eb" stroke-width="1"/>')
        out.append(svg_text(x0 - 7, yy + 4, f"{value:,.6g}", text_anchor="end",
                            font_family="Arial, sans-serif", font_size="10", fill="#5f6368"))
    if reference:
        out.append(f'<line x1="{x0:.1f}" y1="{y(reference):.1f}" x2="{x0 + width:.1f}" '
                   f'y2="{y(reference):.1f}" stroke="#637329" stroke-width="1.5" '
                   'stroke-dasharray="6 4"/>')

    slot = width / len(bars)
    bar_width = min(slot * 0.5, 74.0)
    for index, bar in enumerate(bars):
        center = x0 + slot * (index + 0.5)
        left = center - bar_width / 2
        out.append(f'<rect x="{left:.1f}" y="{y(bar["mean"]):.1f}" width="{bar_width:.1f}" '
                   f'height="{max(bottom - y(bar["mean"]), 0.5):.1f}" fill="{bar["color"]}" '
                   'fill-opacity="0.82" stroke="' + bar["color"] + '" stroke-width="1"/>')
        out.append(f'<line x1="{center:.1f}" y1="{y(bar["minimum"]):.1f}" x2="{center:.1f}" '
                   f'y2="{y(bar["maximum"]):.1f}" stroke="#202124" stroke-width="1.4"/>')
        for edge in ("minimum", "maximum"):
            out.append(f'<line x1="{center - 6:.1f}" y1="{y(bar[edge]):.1f}" '
                       f'x2="{center + 6:.1f}" y2="{y(bar[edge]):.1f}" stroke="#202124" '
                       'stroke-width="1.4"/>')
        out.append(svg_text(center, y(bar["maximum"]) - 8, f"{bar['mean']:,.4g}",
                            text_anchor="middle", font_family="Arial, sans-serif",
                            font_size="11", font_weight="700", fill="#202124"))
        for line_index, line in enumerate(bar["label"]):
            out.append(svg_text(center, bottom + 18 + line_index * 14, line, text_anchor="middle",
                                font_family="Arial, sans-serif", font_size="11", fill="#3c4043"))


def figure_svg(summary: dict) -> str:
    width, height = 1200, 620
    top, bottom = 150, 470
    lookup = {(item["method"], item["scale"]): item for item in summary["configurations"]}

    def throughput_bars(scale: str) -> list[dict]:
        bars = []
        for method in contract.METHODS:
            item = lookup[(method, scale)]
            stat = item["total_tflops"]
            bars.append({
                "mean": stat["mean"], "minimum": stat["minimum"], "maximum": stat["maximum"],
                "color": CONFIG_COLORS[(method, scale)],
                "label": [method,
                          f"{item['work_unit_count']} unit(s)",
                          f"{item['planned_active_sm_count']} SM"],
            })
        return bars

    efficiency_bars = []
    for method in contract.METHODS:
        stat = summary["scaling"][f"scaling_efficiency_{method.split('_')[1]}_percent"]
        efficiency_bars.append({
            "mean": stat["mean"], "minimum": stat["minimum"], "maximum": stat["maximum"],
            "color": CONFIG_COLORS[(method, "device_scale")],
            "label": [method, "device / isolated x N", ""],
        })

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-label="UMMA isolated versus device-scale throughput and scaling efficiency">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(28, 36, "BF16 UMMA: isolated work unit versus device-scale replication",
                 font_family="Arial, sans-serif", font_size="22", font_weight="700",
                 fill="#202124"),
        svg_text(28, 62,
                 "N=256, depth=256, K=16, operands in SMEM. Mean of three per-campaign medians; "
                 "whiskers show min–max across the three campaigns (n=3).",
                 font_family="Arial, sans-serif", font_size="13", fill="#5f6368"),
        svg_text(28, 82,
                 "Isolated and device-scale totals differ by more than two orders of magnitude "
                 "and therefore use separate y-axes; they are never drawn on one scale.",
                 font_family="Arial, sans-serif", font_size="12", fill="#5f6368"),
    ]
    panel_width = 320.0
    gap = 66.0
    left = 96.0
    bar_panel(out, x0=left, width=panel_width, top=top, bottom=bottom,
              title="Isolated work unit", unit="total TFLOP/s (own scale)",
              bars=throughput_bars("isolated"))
    bar_panel(out, x0=left + panel_width + gap, width=panel_width, top=top, bottom=bottom,
              title="Device scale", unit="total TFLOP/s (own scale)",
              bars=throughput_bars("device_scale"))
    bar_panel(out, x0=left + 2 * (panel_width + gap), width=panel_width * 0.72, top=top,
              bottom=bottom, title="Scaling efficiency", unit="percent of each method's own ideal",
              bars=efficiency_bars, reference=100.0)
    out.append(svg_text(left + 2 * (panel_width + gap) + 6, bottom + 62,
                        "dashed line = 100% linear scaling", font_family="Arial, sans-serif",
                        font_size="10", fill="#637329"))
    out.append(svg_text(
        28, height - 26,
        "Compute-focused microbenchmark with operands already in shared memory: not a GEMM, not "
        "an HBM benchmark, and not an NVIDIA architectural peak claim.",
        font_family="Arial, sans-serif", font_size="11", fill="#5f6368"))
    out.append(svg_text(
        28, height - 10,
        "Whole-kernel CUDA-event timing; %clock64 is a per-SM counter and is kept only as a "
        "diagnostic. Three campaigns support descriptive statistics, not significance testing.",
        font_family="Arial, sans-serif", font_size="11", fill="#5f6368"))
    out.append("</svg>")
    return "\n".join(out) + "\n"


SUMMARY_FIELDS = (
    "schema_version", "section", "method", "scale", "metric", "unit", "evidence_class",
    "campaign_count", "campaign_1_value", "campaign_2_value", "campaign_3_value",
    "mean", "median", "stdev_sample", "cross_campaign_cv_percent", "minimum", "maximum",
    "cross_campaign_cv_review_flag", "notes",
)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, NA) for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate exactly three final UMMA device-scaling campaigns."
    )
    parser.add_argument("--campaign", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [campaign(path) for path in args.campaign]
    validate_population(records)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise AnalysisError(f"analysis output already exists: {output}")
    partial.mkdir(parents=True)
    try:
        rows, section, warnings = analysis(records)
        write_csv(partial / "umma_device_scaling.csv", SUMMARY_FIELDS, rows)
        (partial / "figures").mkdir()
        (partial / "figures" / "umma_device_scaling.svg").write_text(
            figure_svg(section), encoding="utf-8")
        summary = {
            "schema_version": SCHEMA,
            "experiment": "umma_device_scaling",
            "question": (
                "How do the cta_group::1 and cta_group::2 BF16 UMMA execution modes behave when "
                "replicated across the maximum simultaneously usable SM capacity of the GPU?"
            ),
            "campaign_ids": [record["campaign_id"] for record in records],
            "campaign_count": CAMPAIGN_COUNT,
            "git_commit": records[0]["manifest"]["git_commit"],
            "gpu": records[0]["manifest"]["gpu"],
            "container_image": records[0]["manifest"]["container_image"],
            "parameters": records[0]["manifest"]["parameters"],
            "statistics": (
                "per-campaign median of 30 repetitions, followed by descriptive statistics "
                "across the three campaign-level medians"
            ),
            "warnings": warnings,
            **section,
            "limitations": [
                "Launch scale is not a third UMMA instruction: PTX provides only cta_group::1 "
                "and cta_group::2.",
                "Three campaigns support descriptive statistics, not significance testing.",
                "Operands are already resident in shared memory, so this is not a GEMM and not "
                "an HBM benchmark.",
                "Throughput is derived from validated operation counts and whole-kernel "
                "CUDA-event time; it is not an NVIDIA architectural peak claim.",
                "Coverage claims rest on a bounded device-side residency handshake plus the "
                "occupancy APIs, at one resident CTA per SM.",
            ],
        }
        (partial / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts = sorted(path for path in partial.rglob("*") if path.is_file())
        hashes = {str(path.relative_to(partial)): sha256_file(path) for path in artifacts}
        (partial / "manifest.json").write_text(
            json.dumps({
                "schema_version": "gb300.umma-scaling-analysis-manifest.v1",
                "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
                    .isoformat().replace("+00:00", "Z"),
                "campaign_ids": [record["campaign_id"] for record in records],
                "campaign_manifest_sha256": {
                    record["campaign_id"]: sha256_file(record["path"] / "manifest.json")
                    for record in records
                },
                "artifact_sha256": hashes,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (partial / "SHA256SUMS").write_text(
            "".join(f"{digest}  {relative}\n" for relative, digest in sorted(hashes.items())),
            encoding="utf-8")
        partial.rename(output)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    for message in warnings:
        print(f"umma-scaling-analysis: WARNING: {message}", file=sys.stderr)
    print(f"umma-scaling-analysis: COMPLETE {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AnalysisError, contract.ScalingCampaignError) as exc:
        print(f"umma-scaling-analysis: ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
