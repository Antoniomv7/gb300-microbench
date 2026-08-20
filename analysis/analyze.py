#!/usr/bin/env python3
"""Aggregate exactly three complete GB300 campaigns into thesis-ready results."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
import random
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncu_capture as profiles  # noqa: E402
import run_campaign as contract  # noqa: E402


SCHEMA = "gb300.analysis.v1"
CAMPAIGN_COUNT = 3
NA = "not_applicable"

# Deterministic percentile bootstrap: the seed is fixed and every draw is
# taken in one frozen configuration order, so the same campaigns always
# produce the same confidence intervals on any machine.
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_LO_PERCENTILE = 0.025
BOOTSTRAP_HI_PERCENTILE = 0.975
MEMORY_BOOTSTRAP_SEED = 20260728
UMMA_BOOTSTRAP_SEED = 20260804

# A configuration whose repetitions inside one campaign vary by more than
# this is marked for review; it is never dropped from any statistic.
CV_REVIEW_PERCENT = 5.0
# Smallest tested value whose median is within this fraction of the group
# maximum, and whose median CI overlaps the maximum's, is the saturation
# candidate.
SATURATION_FRACTION_OF_MAX = 0.95
# DRAM read traffic below this multiple of the logical useful bytes cannot
# support an HBM claim; above the upper bound the read is amplified.
HBM_VALIDATED_MIN_RATIO = 0.90
READ_AMPLIFICATION_MAX_RATIO = 1.10

METHOD_COLORS = {"ldgsts": "#1f6feb", "tma": "#d97706",
                 "umma_1sm": "#1f6feb", "umma_2sm": "#d97706"}
GEMM_COLORS = {
    "nonpersistent_1cta": "#1f6feb",
    "persistent_1cta": "#c69026",
    "persistent_2cta": "#d97706",
    "heuristic_first_supported": "#637329",
}
GEMM_LABELS = {
    "nonpersistent_1cta": "NP1",
    "persistent_1cta": "P1",
    "persistent_2cta": "P2",
    "heuristic_first_supported": "cuBLASLt",
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
    if manifest.get("schema_version") != "gb300.campaign.v1":
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
    if not isinstance(commit, str) or len(commit) != 40:
        raise AnalysisError(f"{path}: invalid Git commit")
    if not isinstance(gpu, dict) or not isinstance(gpu.get("uuid"), str):
        raise AnalysisError(f"{path}: missing GPU identity")
    if not isinstance(image, dict) or not isinstance(image.get("id"), str):
        raise AnalysisError(f"{path}: missing container image identity")

    recorded = manifest.get("artifact_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(contract.RAW_FILES):
        raise AnalysisError(f"{path}: raw artifact inventory is not exact")
    for relative in contract.RAW_FILES:
        artifact = path / relative
        if not artifact.is_file() or sha256_file(artifact) != recorded[relative]:
            raise AnalysisError(f"{path}: hash mismatch for {relative}")

    memory_path = path / "raw/memory_paths.csv"
    umma_path = path / "raw/umma_throughput.csv"
    gemm_path = path / "raw/gemm_comparison.csv"
    contract.validate_memory(memory_path, commit=commit, gpu_uuid=gpu["uuid"])
    contract.validate_umma(umma_path, commit=commit, gpu_uuid=gpu["uuid"])
    contract.validate_gemm(gemm_path, commit=commit, gpu_uuid=gpu["uuid"])
    _, memory_rows = contract.read_csv(memory_path)
    _, umma_rows = contract.read_csv(umma_path)
    _, gemm_rows = contract.read_csv(gemm_path)
    return {
        "path": path,
        "manifest": manifest,
        "campaign_id": campaign_id,
        "memory": memory_rows,
        "umma": umma_rows,
        "gemm": gemm_rows,
        "profiles": load_profiles(path, manifest),
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
        ("source hashes", lambda r: json.dumps(r["manifest"]["source_sha256"], sort_keys=True)),
    ):
        values = {getter(record) for record in records}
        if len(values) != 1:
            raise AnalysisError(f"the three campaigns do not share one {field}")


def group(rows: list[dict[str, str]], key_fields: tuple[str, ...]) -> dict[tuple, list[dict]]:
    result: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in key_fields)].append(row)
    return dict(result)


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


def fmt(value: float | int | str | None, decimals: int = 6) -> str:
    if value is None:
        return NA
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return NA
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Within-campaign statistics.  Repetitions inside one campaign are summarised
# here; they are never pooled across campaigns.  Tukey fences are diagnostic
# counters only: no sample is ever removed from any statistic.
# ---------------------------------------------------------------------------
def percentile_linear(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile over an ascending-sorted sequence."""
    count = len(sorted_values)
    if count == 1:
        return sorted_values[0]
    position = p * (count - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return sorted_values[int(position)]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def iqr_bounds(values: list[float]) -> tuple[float, float, int]:
    """Classic Tukey 1.5*IQR fence and the count of samples outside it."""
    ordered = sorted(values)
    q1 = percentile_linear(ordered, 0.25)
    q3 = percentile_linear(ordered, 0.75)
    spread = q3 - q1
    lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
    return lower, upper, sum(1 for value in values if value < lower or value > upper)


def sample_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def bootstrap_median_ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    """Percentile-bootstrap 95% interval for the median of one campaign's
    repetitions, resampling len(values) items with replacement."""
    count = len(values)
    medians = sorted(
        statistics.median([values[rng.randrange(count)] for _ in range(count)])
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    low = max(int(BOOTSTRAP_LO_PERCENTILE * BOOTSTRAP_RESAMPLES) - 1, 0)
    high = min(int(BOOTSTRAP_HI_PERCENTILE * BOOTSTRAP_RESAMPLES) - 1, BOOTSTRAP_RESAMPLES - 1)
    return medians[low], medians[high]


def bootstrap_ratio_ci(
    left: list[float], right: list[float], rng: random.Random
) -> tuple[float, float]:
    """Percentile-bootstrap 95% interval for median(right)/median(left),
    resampling both inputs independently on every iteration."""
    ratios = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        denominator = statistics.median([left[rng.randrange(len(left))] for _ in range(len(left))])
        numerator = statistics.median([right[rng.randrange(len(right))] for _ in range(len(right))])
        if denominator:
            ratios.append(numerator / denominator)
    if not ratios:
        raise AnalysisError("bootstrap ratio interval: every resample had a zero denominator")
    ratios.sort()
    low = max(int(BOOTSTRAP_LO_PERCENTILE * len(ratios)) - 1, 0)
    high = min(int(BOOTSTRAP_HI_PERCENTILE * len(ratios)) - 1, len(ratios) - 1)
    return ratios[low], ratios[high]


def config_stats(values: list[float]) -> dict:
    """Descriptive statistics for one configuration inside one campaign."""
    if not values or any(not math.isfinite(value) for value in values):
        raise AnalysisError("a configuration repetition is missing or not finite")
    mean = statistics.fmean(values)
    stdev = sample_stdev(values)
    cv_percent = abs(100.0 * stdev / mean) if mean else 0.0
    _, _, flagged = iqr_bounds(values)
    return {
        "count": len(values),
        "median": statistics.median(values),
        "cv_percent": cv_percent,
        "iqr_flagged_count": flagged,
        "stability_review": "REVIEW" if cv_percent > CV_REVIEW_PERCENT else "ok",
    }


def saturation_candidate(medians: dict[int, float], intervals: dict[int, tuple[float, float]]) -> int:
    """Earliest tested axis value whose median reaches SATURATION_FRACTION_OF_MAX
    of the group maximum and whose median interval overlaps the maximum's own.
    The maximum itself is always a valid fallback, so a result always exists."""
    axis = sorted(medians)
    best = max(medians.values())
    best_axis = min(value for value in axis if medians[value] == best)
    best_low, best_high = intervals[best_axis]
    for value in axis:
        if medians[value] < SATURATION_FRACTION_OF_MAX * best:
            continue
        low, high = intervals[value]
        if low <= best_high and best_low <= high:
            return value
    return best_axis


def classify_hbm(dram_read_bytes: float | None, useful_bytes: float | None) -> tuple:
    """Returns (classification, flags, ratio).  The ratio is None whenever the
    classification is inconclusive for a missing or malformed input rather than
    for a real, measured, too-low ratio."""
    if dram_read_bytes is None:
        return "INCONCLUSIVE", ["DRAM_READ_METRIC_UNAVAILABLE"], None
    if not math.isfinite(dram_read_bytes) or dram_read_bytes < 0:
        return "INCONCLUSIVE", ["DRAM_READ_BYTES_MALFORMED"], None
    if useful_bytes is None or not math.isfinite(useful_bytes) or useful_bytes <= 0:
        return "INCONCLUSIVE", ["USEFUL_BYTES_MALFORMED"], None
    ratio = dram_read_bytes / useful_bytes
    if ratio < HBM_VALIDATED_MIN_RATIO:
        return "INCONCLUSIVE", ["RATIO_BELOW_THRESHOLD"], ratio
    if ratio > READ_AMPLIFICATION_MAX_RATIO:
        return "HBM_VALIDATED", ["READ_AMPLIFICATION"], ratio
    return "HBM_VALIDATED", [], ratio


# ---------------------------------------------------------------------------
# Nsight Compute evidence.  The profiler stage is optional: when a campaign
# carries no profile directory, or a planned case was never captured, every
# counter-derived metric is reported as unavailable instead of estimated.
# ---------------------------------------------------------------------------
# Closed unit allowlists for the two counters whose value reaches a result.
# Base units are matched case- and whitespace-insensitively, never rescaled:
# a counter reported in any other unit is treated as unavailable.
SM_CLOCK_METRIC = "sm__cycles_elapsed.avg.per_second"
SM_CLOCK_UNIT_TO_HZ = {"cycle/nsecond": 1e9, "hz": 1.0}
DRAM_READ_METRIC = "dram__bytes_read.sum"
DRAM_READ_UNITS = ("byte",)


NcuParseError = profiles.NcuParseError


def parse_ncu_raw_csv(text: str, candidates: tuple[str, ...]) -> dict:
    """Compatibility wrapper around the capture path's NCU parser."""
    return profiles.parse_ncu_raw_csv(text, candidates)


def load_profiles(path: Path, manifest: dict) -> dict:
    """Reads a campaign's profile directory when the campaign recorded one.
    Every stored file is checked against the hash the campaign recorded, so
    an unrecorded directory can never inject counters into the analysis."""
    empty = {"state": "UNAVAILABLE", "memory_paths": {}, "umma_throughput": {}, "cases": []}
    recorded = manifest.get("ncu")
    if not isinstance(recorded, dict) or recorded.get("state") not in ("COMPLETE", "PARTIAL"):
        return empty
    hashes = recorded.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise AnalysisError(f"{path}: the profile stage recorded no artifact hashes")
    for relative, digest in sorted(hashes.items()):
        artifact = path / relative
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise AnalysisError(f"{path}: hash mismatch for {relative}")

    index = read_json(path / profiles.DIRECTORY / profiles.INDEX_NAME)
    if index.get("schema_version") != profiles.INDEX_SCHEMA:
        raise AnalysisError(f"{path}: unsupported profile index schema")
    result = dict(empty, state=index.get("state", "UNAVAILABLE"), cases=index.get("cases", []))
    for case in result["cases"]:
        if not isinstance(case, dict) or case.get("status") != "captured":
            continue
        section = case.get("section")
        if section not in ("memory_paths", "umma_throughput"):
            raise AnalysisError(f"{path}: profile case of unknown section {section!r}")
        try:
            parsed = parse_ncu_raw_csv(
                (path / profiles.DIRECTORY / case["csv"]).read_text(encoding="utf-8"),
                profiles.MEMORY_METRICS if section == "memory_paths" else profiles.UMMA_METRICS)
            key = ((case["method"], int(case["stages"]), int(case["bytes_in_flight_kib"]))
                   if section == "memory_paths"
                   else (case["method"], int(case["n"]), int(case["depth"])))
        except (OSError, KeyError, TypeError, UnicodeDecodeError, ValueError) as exc:
            raise AnalysisError(f"{path}: profile {case.get('case')!r} is unusable: {exc}") from exc
        parsed["useful_bytes"] = case.get("useful_bytes")
        result[section][key] = parsed
    return result


def sm_clock_hz(parsed: dict | None) -> tuple[float | None, str]:
    """Returns (clock in hertz, status).  The recorded unit alone decides the
    scale to hertz; a magnitude is never used to guess one.  The status is the
    reason the clock is unusable whenever no value can be reported."""
    if parsed is None:
        return None, "not_profiled"
    value = parsed["metrics"].get(SM_CLOCK_METRIC)
    if value is None:
        return None, "metric_unavailable"
    if value <= 0:
        return None, "non_positive"
    unit = parsed["units"].get(SM_CLOCK_METRIC, "").strip().lower()
    scale = SM_CLOCK_UNIT_TO_HZ.get(unit)
    if scale is None:
        return None, f"unknown_unit:{unit or 'empty'}"
    return value * scale, "OK"


def dram_read_bytes(parsed: dict) -> float | None:
    """The DRAM read counter, or None when it is absent or was reported in a
    unit outside the closed allowlist."""
    unit = parsed["units"].get(DRAM_READ_METRIC, "").strip().lower()
    if unit not in DRAM_READ_UNITS:
        return None
    return parsed["metrics"].get(DRAM_READ_METRIC)


# ---------------------------------------------------------------------------
# Result rows.  Every result is one row naming its metric and unit, the three
# campaign values behind it, and the descriptive statistics across those three
# campaigns.  Aggregates are reported only when all three campaigns supplied a
# finite value for that metric.
# ---------------------------------------------------------------------------
AGGREGATE_FIELDS = ("mean", "median", "stdev_sample", "cross_campaign_cv_percent",
                    "minimum", "maximum", "cross_campaign_cv_review_flag")
DIAGNOSTIC_NOTE = "preserved_source_diagnostic"
MEMORY_MEDIAN_NOTE = ("campaign_level_median_of_that_campaign_own_timing_derived_"
                      "effective_transfer_rates")
UMMA_MEDIAN_NOTE = "clock_independent_operation_and_cycle_derived_campaign_level_median"
DERIVED_NOTE = "derived_within_campaign_value_never_pooled"
UNPROFILED_NOTE = "ncu_profile_unavailable"
RATIO_DECIMALS = 9

# How each metric was arrived at, so a reader can tell a raw observation from a
# quantity derived inside one campaign, from one that needed an outside model.
# Anything reported only to describe the inputs is a diagnostic.
SOURCE_DIAGNOSTIC_METRICS = frozenset({
    "ncu_coverage", "profile_sm_clock_status",
    "profile_diagnostic_metrics_resolved_count", "surprising_value_flag",
    "diagnostic_flags",
})
MEASURED_METRICS = frozenset({"kernel_time_ms"})
MODELED_METRICS = frozenset({"estimated_tflops_per_sm"})


def evidence_class_of(metric: str) -> str:
    if metric.startswith("within_campaign_") or metric in SOURCE_DIAGNOSTIC_METRICS:
        return "source_diagnostic"
    if metric in MEASURED_METRICS:
        return "measured_source_observation"
    if metric in MODELED_METRICS:
        return "modeled_estimate"
    return "within_campaign_derived_estimate"


def metric_row(section: str, keys: dict[str, str], metric: str, unit: str,
               values: list, notes: str, *, decimals: int = 6,
               aggregate: bool = True, signed: bool = False) -> dict:
    row = {"schema_version": SCHEMA, "section": section, **keys, "metric": metric,
           "unit": unit, "evidence_class": evidence_class_of(metric),
           "campaign_count": str(CAMPAIGN_COUNT)}
    for index, value in enumerate(values, 1):
        row[f"campaign_{index}_value"] = fmt(value, decimals)
    numbers = [float(value) for value in values
               if isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value)]
    if aggregate and len(numbers) == CAMPAIGN_COUNT:
        summary = stats(numbers)
        review = "REVIEW" if summary["cv_percent"] > CV_REVIEW_PERCENT else "ok"
        row.update({
            "mean": fmt(summary["mean"], decimals),
            "median": fmt(summary["median"], decimals),
            "stdev_sample": fmt(summary["stdev_sample"], decimals),
            "cross_campaign_cv_percent": NA if signed else fmt(summary["cv_percent"], 6),
            "minimum": fmt(summary["minimum"], decimals),
            "maximum": fmt(summary["maximum"], decimals),
            "cross_campaign_cv_review_flag": NA if signed else review,
        })
    else:
        row.update({field: NA for field in AGGREGATE_FIELDS})
    row["notes"] = notes
    return row


def consensus(values: list) -> str:
    return str(values[0]) if len(set(values)) == 1 else "mixed"


def numeric_field(row: dict[str, str], field: str) -> float | None:
    try:
        value = float(row[field])
    except (KeyError, ValueError):
        return None
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------------------
# Memory paths.
# ---------------------------------------------------------------------------
MEMORY_BIF_KIB = tuple(size // 1024 for size in contract.MEMORY_BIF_BYTES)


def memory_campaign_stats(record: dict, rng: random.Random) -> dict:
    """Per-campaign statistics over that campaign's own repetitions, drawn in
    one frozen configuration order so the bootstrap stays reproducible."""
    grouped = group(record["memory"], ("method", "stages", "bytes_in_flight_per_sm"))
    samples, useful = {}, {}
    for (method, stages, bif_bytes), rows in grouped.items():
        key = (method, int(stages), int(bif_bytes) // 1024)
        samples[key] = [float(row["effective_gbps"]) for row in rows]
        useful[key] = numeric_field(rows[0], "useful_bytes")
    result = {}
    for key in sorted(samples, key=lambda item: (item[1], item[2], item[0])):
        entry = config_stats(samples[key])
        entry["interval"] = bootstrap_median_ci(samples[key], rng)
        entry["samples"] = samples[key]
        entry["useful_bytes"] = useful[key]
        result[key] = entry
    return result


def memory_analysis(records: list[dict]) -> tuple[list[dict], dict]:
    generators = [random.Random(MEMORY_BOOTSTRAP_SEED) for _ in records]
    per_campaign = [memory_campaign_stats(record, rng)
                    for record, rng in zip(records, generators)]
    profiled = [record["profiles"]["memory_paths"] for record in records]

    rows, configurations, comparisons = [], [], []
    for stages in contract.MEMORY_STAGES:
        for bif_kib in MEMORY_BIF_KIB:
            for method in contract.MEMORY_METHODS:
                key = (method, stages, bif_kib)
                entries = [campaign[key] for campaign in per_campaign]
                keys = {"method": method, "stages": str(stages),
                        "bytes_in_flight_kib": str(bif_kib)}
                medians = [entry["median"] for entry in entries]
                rows.append(metric_row("configuration", keys, "median_effective_gbps",
                                       "GB/s", medians, MEMORY_MEDIAN_NOTE))
                for metric, field in (("within_campaign_sample_count", "count"),
                                      ("within_campaign_cv_percent", "cv_percent"),
                                      ("within_campaign_stability_review", "stability_review"),
                                      ("within_campaign_iqr_flagged_count", "iqr_flagged_count")):
                    rows.append(metric_row("configuration", keys, metric, NA,
                                           [entry[field] for entry in entries],
                                           DIAGNOSTIC_NOTE, aggregate=False))
                rows.append(metric_row(
                    "configuration", keys, "ncu_coverage", NA,
                    ["ncu_profiled" if key in profile else "not_profiled"
                     for profile in profiled], DIAGNOSTIC_NOTE, aggregate=False))
                summary = stats(medians)
                configurations.append({"method": method, "stages": stages,
                                       "bytes_in_flight_kib": bif_kib, **summary})

    for stages in contract.MEMORY_STAGES:
        for bif_kib in MEMORY_BIF_KIB:
            ldgsts = [campaign[("ldgsts", stages, bif_kib)] for campaign in per_campaign]
            tma = [campaign[("tma", stages, bif_kib)] for campaign in per_campaign]
            ratios = [right["median"] / left["median"] for left, right in zip(ldgsts, tma)]
            intervals = [bootstrap_ratio_ci(left["samples"], right["samples"], rng)
                         for left, right, rng in zip(ldgsts, tma, generators)]
            summary = stats(ratios)
            reading = ("tma_higher" if summary["mean"] > 1
                       else "ldgsts_higher" if summary["mean"] < 1 else "equal")
            rows.append(metric_row(
                "pair_ratio",
                {"method": NA, "stages": str(stages), "bytes_in_flight_kib": str(bif_kib)},
                "tma_to_ldgsts_ratio", "ratio", ratios, f"interpretation={reading}",
                decimals=RATIO_DECIMALS))
            comparisons.append({"stages": stages, "bytes_in_flight_kib": bif_kib,
                                "ratio": summary, "interpretation": reading,
                                "ratio_interval_by_campaign": intervals})

    saturation = []
    for method in contract.MEMORY_METHODS:
        for stages in contract.MEMORY_STAGES:
            candidates = [
                saturation_candidate(
                    {bif: campaign[(method, stages, bif)]["median"] for bif in MEMORY_BIF_KIB},
                    {bif: campaign[(method, stages, bif)]["interval"] for bif in MEMORY_BIF_KIB})
                for campaign in per_campaign
            ]
            rows.append(metric_row(
                "saturation", {"method": method, "stages": str(stages),
                               "bytes_in_flight_kib": NA},
                "earliest_tested_candidate_saturation_bif_kib", "KiB", candidates,
                f"consensus={consensus(candidates)}", aggregate=False))
            saturation.append({"method": method, "stages": stages,
                               "by_campaign": candidates,
                               "consensus": consensus(candidates)})

    validation = []
    for entry in profiles.MEMORY_PLAN:
        key = (entry["method"], entry["stages"], entry["bytes_in_flight_kib"])
        keys = {"method": entry["method"], "stages": str(entry["stages"]),
                "bytes_in_flight_kib": str(entry["bytes_in_flight_kib"])}
        ratios: list = []
        classifications: list = []
        flag_sets: list[str] = []
        for profile, campaign_stats in zip(profiled, per_campaign):
            parsed = profile.get(key)
            if parsed is None:
                ratios.append(None)
                classifications.append(NA)
                flag_sets.append(NA)
                continue
            useful = parsed.get("useful_bytes")
            if useful is None:
                useful = campaign_stats[key]["useful_bytes"]
            label, flags, ratio = classify_hbm(dram_read_bytes(parsed), useful)
            ratios.append(ratio)
            classifications.append(label)
            flag_sets.append("+".join(flags) if flags else "present_and_empty")
        # A captured profile whose DRAM counter is missing still yields a
        # classification and a flag set, so coverage is what decides the note.
        available = any(label != NA for label in classifications)
        rows.append(metric_row(
            "ncu_validation", keys, "dram_read_ratio", "ratio", ratios,
            f"hbm={consensus(classifications)}" if available else UNPROFILED_NOTE,
            decimals=RATIO_DECIMALS))
        rows.append(metric_row(
            "ncu_validation", keys, "hbm_classification", NA, classifications,
            DIAGNOSTIC_NOTE if available else UNPROFILED_NOTE, aggregate=False))
        rows.append(metric_row(
            "ncu_validation", keys, "diagnostic_flags", NA, [NA] * CAMPAIGN_COUNT,
            f"{DIAGNOSTIC_NOTE};diagnostic_field_present;flag_set={','.join(flag_sets)}"
            if available else UNPROFILED_NOTE, aggregate=False))
        validation.append({"method": entry["method"], "stages": entry["stages"],
                           "bytes_in_flight_kib": entry["bytes_in_flight_kib"],
                           "dram_read_ratio_by_campaign": ratios,
                           "hbm_classification": consensus(classifications)})

    ldgsts_higher = sum(1 for item in comparisons if item["interpretation"] == "ldgsts_higher")
    maxima = {
        method: max((item for item in configurations if item["method"] == method),
                    key=lambda item: item["mean"])
        for method in contract.MEMORY_METHODS
    }
    section = {
        "configurations": configurations,
        "comparisons": comparisons,
        "saturation": saturation,
        "ncu_validation": validation,
        "ncu_profiled_count": sum(len(campaign) for campaign in profiled),
        "ldgsts_higher_count": ldgsts_higher,
        "tma_higher_count": len(comparisons) - ldgsts_higher,
        "maximum_by_method": {
            method: {"stages": item["stages"],
                     "bytes_in_flight_kib": item["bytes_in_flight_kib"],
                     "mean_median_effective_gbps": item["mean"]}
            for method, item in maxima.items()
        },
    }
    return rows, section


# ---------------------------------------------------------------------------
# UMMA throughput.
# ---------------------------------------------------------------------------
def umma_campaign_stats(record: dict, rng: random.Random) -> dict:
    grouped = group(record["umma"], ("method", "n", "depth", "cta_group"))
    samples = {}
    for (method, n_value, depth, cta_group), rows in grouped.items():
        key = (method, int(n_value), int(depth))
        total = [float(row["flops_per_cycle"]) for row in rows]
        samples[key] = {"cta_group": int(cta_group), "flops_per_cycle": total,
                        "per_sm": [value / int(cta_group) for value in total]}
    result = {}
    for key in sorted(samples, key=lambda item: (item[1], item[2], item[0])):
        entry = samples[key]
        total = config_stats(entry["flops_per_cycle"])
        per_sm = config_stats(entry["per_sm"])
        result[key] = {
            "cta_group": entry["cta_group"], "total": total, "per_sm": per_sm,
            "samples": entry["flops_per_cycle"],
            "interval": bootstrap_median_ci(entry["flops_per_cycle"], rng),
        }
    return result


def umma_analysis(records: list[dict]) -> tuple[list[dict], dict]:
    generators = [random.Random(UMMA_BOOTSTRAP_SEED) for _ in records]
    per_campaign = [umma_campaign_stats(record, rng)
                    for record, rng in zip(records, generators)]
    profiled = [record["profiles"]["umma_throughput"] for record in records]
    diagnostics = tuple(name for name in profiles.UMMA_METRICS if name != SM_CLOCK_METRIC)

    rows, configurations = [], []
    clocks: dict[tuple, list] = {}
    for n_value in contract.UMMA_N:
        for depth in contract.UMMA_DEPTHS:
            for method in contract.UMMA_METHODS:
                key = (method, n_value, depth)
                entries = [campaign[key] for campaign in per_campaign]
                cta_group = entries[0]["cta_group"]
                keys = {"method": method, "n": str(n_value), "depth": str(depth),
                        "cta_group": str(cta_group)}
                total = [entry["total"]["median"] for entry in entries]
                per_sm = [entry["per_sm"]["median"] for entry in entries]
                rows.append(metric_row("configuration", keys, "median_flops_per_cycle",
                                       "FLOP/cycle", total, UMMA_MEDIAN_NOTE))
                rows.append(metric_row("configuration", keys, "median_flops_per_cycle_per_sm",
                                       "FLOP/cycle/SM", per_sm, UMMA_MEDIAN_NOTE))
                for metric, group_name, field in (
                    ("within_campaign_sample_count", "total", "count"),
                    ("within_campaign_cv_percent", "total", "cv_percent"),
                    ("within_campaign_stability_review", "total", "stability_review"),
                    ("within_campaign_flops_per_cycle_per_sm_cv_percent", "per_sm", "cv_percent"),
                    ("within_campaign_flops_per_cycle_iqr_flagged_count", "total",
                     "iqr_flagged_count"),
                    ("within_campaign_flops_per_cycle_per_sm_iqr_flagged_count", "per_sm",
                     "iqr_flagged_count"),
                ):
                    rows.append(metric_row(
                        "configuration", keys, metric, NA,
                        [entry[group_name][field] for entry in entries],
                        DIAGNOSTIC_NOTE, aggregate=False))
                readings = [sm_clock_hz(profile.get(key)) for profile in profiled]
                clocks[key] = readings
                rows.append(metric_row(
                    "configuration", keys, "profile_sm_clock_status", NA,
                    [NA if status == "not_profiled" else status for _, status in readings],
                    DIAGNOSTIC_NOTE, aggregate=False))
                rows.append(metric_row(
                    "configuration", keys, "profile_diagnostic_metrics_resolved_count", NA,
                    [NA if key not in profile
                     else sum(1 for name in diagnostics if name in profile[key]["metrics"])
                     for profile in profiled],
                    DIAGNOSTIC_NOTE, aggregate=False))
                configurations.append({"method": method, "n": n_value, "depth": depth,
                                       "cta_group": cta_group,
                                       "total": stats(total), "per_sm": stats(per_sm)})

    scaling = []
    for n_value in contract.UMMA_N:
        for depth in contract.UMMA_DEPTHS:
            one = [campaign[("umma_1sm", n_value, depth)] for campaign in per_campaign]
            two = [campaign[("umma_2sm", n_value, depth)] for campaign in per_campaign]
            speedups = [right["total"]["median"] / left["total"]["median"]
                        for left, right in zip(one, two)]
            efficiencies = [100.0 * value / 2.0 for value in speedups]
            intervals = [bootstrap_ratio_ci(left["samples"], right["samples"], rng)
                         for left, right, rng in zip(one, two, generators)]
            keys = {"method": NA, "n": str(n_value), "depth": str(depth), "cta_group": NA}
            rows.append(metric_row("scaling", keys, "speedup_2sm_over_1sm", "ratio",
                                   speedups, DERIVED_NOTE, decimals=RATIO_DECIMALS))
            surprising = [not 0.0 <= value <= 100.0 for value in efficiencies]
            rows.append(metric_row(
                "scaling", keys, "scaling_efficiency_percent", "percent", efficiencies,
                "value_outside_0_100_preserved_unclamped" if any(surprising) else DERIVED_NOTE))
            rows.append(metric_row("scaling", keys, "surprising_value_flag", NA,
                                   [str(flag) for flag in surprising], DIAGNOSTIC_NOTE,
                                   aggregate=False))
            scaling.append({"n": n_value, "depth": depth, "speedup": stats(speedups),
                            "scaling_efficiency_percent": stats(efficiencies),
                            "speedup_interval_by_campaign": intervals})

    saturation = []
    for method in contract.UMMA_METHODS:
        for n_value in contract.UMMA_N:
            candidates = [
                saturation_candidate(
                    {depth: campaign[(method, n_value, depth)]["total"]["median"]
                     for depth in contract.UMMA_DEPTHS},
                    {depth: campaign[(method, n_value, depth)]["interval"]
                     for depth in contract.UMMA_DEPTHS})
                for campaign in per_campaign
            ]
            cta_group = per_campaign[0][(method, n_value, contract.UMMA_DEPTHS[0])]["cta_group"]
            rows.append(metric_row(
                "saturation",
                {"method": method, "n": str(n_value), "depth": NA, "cta_group": str(cta_group)},
                "earliest_tested_candidate_saturation_depth", "depth", candidates,
                f"consensus={consensus(candidates)}", aggregate=False))
            saturation.append({"method": method, "n": n_value, "by_campaign": candidates,
                               "consensus": consensus(candidates)})

    ceiling_rows, ceiling = umma_ceiling(per_campaign, clocks, profiled)
    rows.extend(ceiling_rows)
    best_speedup = max(scaling, key=lambda item: item["speedup"]["mean"])
    section = {
        "configurations": configurations,
        "scaling": scaling,
        "saturation": saturation,
        "per_sm_ceiling_candidate": ceiling,
        "maximum_2sm_speedup": {"n": best_speedup["n"], "depth": best_speedup["depth"],
                                "mean": best_speedup["speedup"]["mean"]},
    }
    return rows, section


def umma_ceiling(per_campaign: list[dict], clocks: dict[tuple, list],
                 profiled: list[dict]) -> tuple[list[dict], dict]:
    """Selects the ceiling configuration in clock-independent FLOP/cycle/SM
    space first, then converts it with that same configuration's own measured
    SM clock.  Without a valid clock for every profiled configuration the
    TFLOP/s figure stays unavailable; it is never modelled.  The note says
    which of the three reasons applies, because a figure that vanishes without
    a stated reason is worse than one that is openly absent."""
    selections = [
        max(sorted(campaign, key=lambda key: (key[1], key[2], key[0])),
            key=lambda key: campaign[key]["per_sm"]["median"])
        for campaign in per_campaign
    ]
    covered = sum(1 for profile in profiled if profile)
    every_clock_valid = all(status == "OK" for readings in clocks.values()
                            for _, status in readings)
    values: list = []
    for index, (campaign, key) in enumerate(zip(per_campaign, selections)):
        hertz, _ = clocks[key][index]
        if not every_clock_valid or hertz is None:
            values.append(None)
            continue
        values.append(campaign[key]["total"]["median"] * hertz
                      / 1e12 / campaign[key]["cta_group"])
    stable = len(set(selections)) == 1
    if not covered:
        note = "ncu_profile_absent_in_every_campaign"
    elif covered < len(profiled):
        note = f"ncu_profile_present_in_{covered}_of_{len(profiled)}_campaigns"
    elif not every_clock_valid:
        note = "sm_clock_unresolvable_in_at_least_one_profiled_configuration"
    elif not stable:
        note = "selection_unstable_across_campaigns"
    else:
        note = "selection_stable_and_all_clocks_valid"
    method, n_value, depth = selections[0]
    row = metric_row("ceiling", {"method": NA, "n": NA, "depth": NA, "cta_group": NA},
                     "estimated_tflops_per_sm", "TFLOP/s/SM", values, note,
                     decimals=RATIO_DECIMALS)
    ceiling = {
        "method": method, "n": n_value, "depth": depth,
        "selection_stable": stable,
        "mean_median_flops_per_cycle_per_sm": stats(
            [campaign[key]["per_sm"]["median"]
             for campaign, key in zip(per_campaign, selections)])["mean"],
        "estimated_tflops_per_sm_by_campaign": values,
        "campaigns_with_profiles": covered,
        "sm_clock_status": note,
    }
    return [row], ceiling


# ---------------------------------------------------------------------------
# GEMM comparison.
# ---------------------------------------------------------------------------
GEMM_METRICS = (
    ("kernel_time_ms", "ms", "kernel_time_ms", "measured_campaign_level_input", False),
    ("tflops", "TFLOP/s", "tflops", "derived_within_campaign_value", False),
    ("throughput_ratio_vs_cublaslt", "ratio", "ratio",
     "derived_within_campaign_ratio_never_recomputed_from_aggregates", False),
    ("gap_to_cublaslt_pct", "percent", "gap",
     "signed_metric_cv_not_computed_negative_gaps_never_clamped", True),
)


def gemm_analysis(records: list[dict]) -> tuple[list[dict], dict]:
    per_campaign = []
    for record in records:
        values = {}
        for row in record["gemm"]:
            values[(int(row["shape_index"]), int(row["candidate_index"]))] = {
                "shape_id": row["shape_id"], "method": row["method"],
                "variant": row["variant"], "m": row["m"], "n": row["n"],
                "k": row["k"], "l": row["l"],
                "kernel_time_ms": float(row["kernel_time_ms"]),
                "tflops": float(row["tflops"]),
                "ratio": float(row["throughput_ratio_vs_cublaslt"]),
                "gap": float(row["gap_to_cublaslt_pct"]),
                "best_cutedsl_variant": row["best_cutedsl_variant"],
            }
        per_campaign.append(values)

    rows, shapes = [], []
    for shape_index in contract.GEMM_SHAPES:
        best = [campaign[(shape_index, 1)]["best_cutedsl_variant"] for campaign in per_campaign]
        agreed = consensus(best)
        candidates = []
        for candidate_index in contract.GEMM_CANDIDATES:
            entries = [campaign[(shape_index, candidate_index)] for campaign in per_campaign]
            shape = entries[0]
            keys = {"shape_index": str(shape_index), "shape_id": shape["shape_id"],
                    "m": shape["m"], "n": shape["n"], "k": shape["k"], "l": shape["l"],
                    "candidate_index": str(candidate_index), "variant": shape["variant"],
                    "method": shape["method"]}
            summaries = {}
            for metric, unit, field, notes, signed in GEMM_METRICS:
                values = [entry[field] for entry in entries]
                rows.append(metric_row("candidate", keys, metric, unit, values, notes,
                                       signed=signed))
                summaries[metric] = stats(values)
            candidates.append({"candidate_index": candidate_index, "variant": shape["variant"],
                               "method": shape["method"], **summaries})
        shape = per_campaign[0][(shape_index, 1)]
        rows.append(metric_row(
            "best_cutedsl",
            {"shape_index": str(shape_index), "shape_id": shape["shape_id"],
             "m": shape["m"], "n": shape["n"], "k": shape["k"], "l": shape["l"],
             "candidate_index": NA, "variant": NA, "method": "cutedsl"},
            "best_cutedsl_variant", "variant", best, f"stable_best={agreed}", aggregate=False))
        shapes.append({"shape_index": shape_index, "shape_id": shape["shape_id"],
                       "best_cutedsl_variant_consensus": agreed, "candidates": candidates})
    return rows, {"shapes": shapes}


def profile_warnings(records: list[dict], umma_summary: dict) -> list[str]:
    """Counter-derived aggregates need the same profile in all three campaigns.
    Campaigns that disagree, or a clock that cannot be resolved, withhold them:
    say so on stderr so a missing headline figure is never a silent surprise."""
    messages = []
    uneven = False
    for section, withheld in (("memory_paths", "dram_read_ratio"),
                              ("umma_throughput", "estimated_tflops_per_sm")):
        present = sum(1 for record in records if record["profiles"][section])
        if 0 < present < len(records):
            uneven = True
            messages.append(f"NCU profiles present in {present} of {len(records)} "
                            f"campaigns; {withheld} withheld")
    ceiling = umma_summary["per_sm_ceiling_candidate"]
    if not uneven and any(value is None
                          for value in ceiling["estimated_tflops_per_sm_by_campaign"]):
        messages.append(f"estimated_tflops_per_sm withheld: {ceiling['sm_clock_status']}")
    return messages


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, NA) for field in fields})


def svg_text(x: float, y: float, value: str, **attrs: str) -> str:
    properties = {"x": f"{x:.1f}", "y": f"{y:.1f}", **attrs}
    encoded = " ".join(f'{key.replace("_", "-")}="{html.escape(str(val))}"'
                       for key, val in properties.items())
    return f"<text {encoded}>{html.escape(value)}</text>"


def line_panels_svg(
    *, title: str, subtitle: str, panels: list[dict], series_order: tuple[str, str],
    x_labels: list[str], y_unit: str, footnote: str,
) -> str:
    width, height = 1200, 600
    left, right, top, bottom, gap = 82, 26, 122, 500, 42
    panel_width = (width - left - right - gap * (len(panels) - 1)) / len(panels)
    lows = [point["minimum"] for panel in panels for series in panel["series"].values()
            for point in series]
    highs = [point["maximum"] for panel in panels for series in panel["series"].values()
             for point in series]
    y_min, y_max = min(lows), max(highs)
    padding = max((y_max - y_min) * 0.08, abs(y_max) * 0.01, 1.0)
    y_min, y_max = y_min - padding, y_max + padding

    def x(panel_index: int, point_index: int) -> float:
        x0 = left + panel_index * (panel_width + gap)
        return x0 + 28 + point_index * (panel_width - 56) / (len(x_labels) - 1)

    def y(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(28, 34, title, font_family="Arial, sans-serif", font_size="22",
                 font_weight="700", fill="#202124"),
        svg_text(28, 60, subtitle, font_family="Arial, sans-serif", font_size="13",
                 fill="#5f6368"),
    ]
    legend_x = 760
    for index, name in enumerate(series_order):
        color = METHOD_COLORS[name]
        lx = legend_x + index * 190
        out.append(f'<line x1="{lx}" y1="88" x2="{lx + 30}" y2="88" '
                   f'stroke="{color}" stroke-width="3"/>')
        marker = "circle" if index == 0 else "rect"
        if marker == "circle":
            out.append(f'<circle cx="{lx + 15}" cy="88" r="4" fill="#fff" '
                       f'stroke="{color}" stroke-width="2"/>')
        else:
            out.append(f'<rect x="{lx + 11}" y="84" width="8" height="8" fill="#fff" '
                       f'stroke="{color}" stroke-width="2"/>')
        out.append(svg_text(lx + 38, 93, name, font_family="Arial, sans-serif",
                            font_size="12", fill="#202124"))

    for tick in range(5):
        value = y_min + tick * (y_max - y_min) / 4
        yy = y(value)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" '
                   'stroke="#e5e7eb" stroke-width="1"/>')
        out.append(svg_text(left - 9, yy + 4, f"{value:,.0f}", text_anchor="end",
                            font_family="Arial, sans-serif", font_size="11", fill="#5f6368"))
    out.append(svg_text(18, (top + bottom) / 2, y_unit, text_anchor="middle",
                        font_family="Arial, sans-serif", font_size="12", fill="#3c4043",
                        transform=f"rotate(-90 18 {(top + bottom) / 2:.1f})"))

    for panel_index, panel in enumerate(panels):
        x0 = left + panel_index * (panel_width + gap)
        x1 = x0 + panel_width
        out.append(f'<rect x="{x0:.1f}" y="{top}" width="{panel_width:.1f}" '
                   f'height="{bottom-top}" fill="none" stroke="#9aa0a6" stroke-width="1"/>')
        out.append(svg_text((x0 + x1) / 2, 108, panel["title"], text_anchor="middle",
                            font_family="Arial, sans-serif", font_size="13",
                            font_weight="700", fill="#202124"))
        for point_index, label in enumerate(x_labels):
            xx = x(panel_index, point_index)
            out.append(svg_text(xx, bottom + 22, label, text_anchor="middle",
                                font_family="Arial, sans-serif", font_size="11",
                                fill="#5f6368"))
        for series_index, name in enumerate(series_order):
            color = METHOD_COLORS[name]
            points = panel["series"][name]
            coords = " ".join(
                f"{x(panel_index, idx):.1f},{y(point['mean']):.1f}"
                for idx, point in enumerate(points)
            )
            out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                       'stroke-width="2.5" stroke-linejoin="round"/>')
            for point_index, point in enumerate(points):
                xx = x(panel_index, point_index)
                out.append(f'<line x1="{xx:.1f}" y1="{y(point["minimum"]):.1f}" '
                           f'x2="{xx:.1f}" y2="{y(point["maximum"]):.1f}" '
                           f'stroke="{color}" stroke-width="1.5"/>')
                out.append(f'<line x1="{xx-4:.1f}" y1="{y(point["minimum"]):.1f}" '
                           f'x2="{xx+4:.1f}" y2="{y(point["minimum"]):.1f}" '
                           f'stroke="{color}" stroke-width="1.5"/>')
                out.append(f'<line x1="{xx-4:.1f}" y1="{y(point["maximum"]):.1f}" '
                           f'x2="{xx+4:.1f}" y2="{y(point["maximum"]):.1f}" '
                           f'stroke="{color}" stroke-width="1.5"/>')
                if series_index == 0:
                    out.append(f'<circle cx="{xx:.1f}" cy="{y(point["mean"]):.1f}" r="4" '
                               f'fill="#fff" stroke="{color}" stroke-width="2"/>')
                else:
                    out.append(f'<rect x="{xx-4:.1f}" y="{y(point["mean"])-4:.1f}" '
                               f'width="8" height="8" fill="#fff" stroke="{color}" '
                               'stroke-width="2"/>')
    out.append(svg_text(28, 572, footnote, font_family="Arial, sans-serif",
                        font_size="11", fill="#5f6368"))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def gemm_svg(summary: dict) -> str:
    width, height = 1200, 600
    left, right, top, bottom = 80, 28, 122, 495
    variants = tuple(GEMM_COLORS)
    shapes = sorted((item["shape_index"], item["shape_id"]) for item in summary["shapes"])
    lookup = {(item["shape_index"], candidate["variant"]): candidate["tflops"]
              for item in summary["shapes"] for candidate in item["candidates"]}
    maximum = max(entry["maximum"] for entry in lookup.values()) * 1.10

    def y(value: float) -> float:
        return bottom - value * (bottom - top) / maximum

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="GEMM throughput comparison">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(28, 34, "GEMM throughput by shape and candidate",
                 font_family="Arial, sans-serif", font_size="22", font_weight="700",
                 fill="#202124"),
        svg_text(28, 60, "Mean TFLOP/s; whiskers show min–max across three final campaigns; hot cache",
                 font_family="Arial, sans-serif", font_size="13", fill="#5f6368"),
    ]
    legend_x = 500
    for index, variant in enumerate(variants):
        lx = legend_x + index * 165
        out.append(f'<rect x="{lx}" y="80" width="14" height="14" '
                   f'fill="{GEMM_COLORS[variant]}" stroke="#3c4043" stroke-width="0.8"/>')
        out.append(svg_text(lx + 20, 92, GEMM_LABELS[variant],
                            font_family="Arial, sans-serif", font_size="11", fill="#202124"))
    for tick in range(6):
        value = tick * maximum / 5
        yy = y(value)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" '
                   'stroke="#e5e7eb" stroke-width="1"/>')
        out.append(svg_text(left - 9, yy + 4, f"{value:,.0f}", text_anchor="end",
                            font_family="Arial, sans-serif", font_size="11", fill="#5f6368"))
    out.append(svg_text(18, (top + bottom) / 2, "TFLOP/s", text_anchor="middle",
                        font_family="Arial, sans-serif", font_size="12", fill="#3c4043",
                        transform=f"rotate(-90 18 {(top + bottom) / 2:.1f})"))
    group_width = (width - left - right) / len(shapes)
    bar_gap, inner_pad = 4, 18
    bar_width = (group_width - 2 * inner_pad - bar_gap * 3) / 4
    short_shape = {
        "4096x4096x4096x1": "4096²",
        "8192x8192x8192x1": "8192²",
        "16384x512x4096x1": "16384×512",
        "32768x512x4096x1": "32768×512",
        "512x16384x4096x1": "512×16384",
    }
    for group_index, (shape_index, shape_id) in enumerate(shapes):
        x0 = left + group_index * group_width + inner_pad
        for variant_index, variant in enumerate(variants):
            entry = lookup[(shape_index, variant)]
            xx = x0 + variant_index * (bar_width + bar_gap)
            yy = y(entry["mean"])
            out.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bar_width:.1f}" '
                       f'height="{bottom-yy:.1f}" fill="{GEMM_COLORS[variant]}" '
                       'stroke="#3c4043" stroke-width="0.8"/>')
            center = xx + bar_width / 2
            out.append(f'<line x1="{center:.1f}" y1="{y(entry["minimum"]):.1f}" '
                       f'x2="{center:.1f}" y2="{y(entry["maximum"]):.1f}" '
                       'stroke="#202124" stroke-width="1.2"/>')
        center = left + (group_index + 0.5) * group_width
        out.append(svg_text(center, bottom + 24, short_shape.get(shape_id, shape_id),
                            text_anchor="middle", font_family="Arial, sans-serif",
                            font_size="11", fill="#3c4043"))
    out.append(svg_text(28, 570,
                        "NP1=nonpersistent_1cta; P1=persistent_1cta; P2=persistent_2cta. "
                        "TFLOP/s is derived from exact operation count and measured kernel time.",
                        font_family="Arial, sans-serif", font_size="11", fill="#5f6368"))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def figures(output: Path, memory: dict, umma: dict, gemm: dict) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir()
    memory_lookup = {(item["method"], item["stages"], item["bytes_in_flight_kib"]): item
                     for item in memory["configurations"]}
    memory_panels = []
    for stages in contract.MEMORY_STAGES:
        panel = {"title": f"stages={stages}", "series": {}}
        for method in contract.MEMORY_METHODS:
            panel["series"][method] = [memory_lookup[(method, stages, bif)]
                                       for bif in MEMORY_BIF_KIB]
        memory_panels.append(panel)
    (figure_dir / "memory_paths.svg").write_text(
        line_panels_svg(
            title="Memory paths: timing-derived effective transfer rate",
            subtitle="Mean of per-campaign medians; whiskers show min\u2013max; n=3; focused y-scale",
            panels=memory_panels, series_order=contract.MEMORY_METHODS,
            x_labels=[f"{bif} KiB" for bif in MEMORY_BIF_KIB], y_unit="GB/s",
            footnote="Logical useful bytes / kernel time; not a direct DRAM-bandwidth counter.",
        ), encoding="utf-8")

    umma_lookup = {(item["method"], item["n"], item["depth"]): item["per_sm"]
                   for item in umma["configurations"]}
    umma_panels = []
    for n_value in contract.UMMA_N:
        panel = {"title": f"N={n_value}", "series": {}}
        for method in contract.UMMA_METHODS:
            panel["series"][method] = [umma_lookup[(method, n_value, depth)]
                                       for depth in contract.UMMA_DEPTHS]
        umma_panels.append(panel)
    (figure_dir / "umma_throughput.svg").write_text(
        line_panels_svg(
            title="UMMA throughput per participating SM",
            subtitle="Mean FLOP/cycle/SM from per-campaign medians; min\u2013max; n=3; focused y-scale",
            panels=umma_panels, series_order=contract.UMMA_METHODS,
            x_labels=[f"d{depth}" for depth in contract.UMMA_DEPTHS],
            y_unit="FLOP/cycle/SM",
            footnote="Derived from validated operation counts and measured cycles; not a whole-device peak.",
        ), encoding="utf-8")
    (figure_dir / "gemm_comparison.svg").write_text(gemm_svg(gemm), encoding="utf-8")


COMMON_FIELDS = (
    "metric", "unit", "evidence_class", "campaign_count", "campaign_1_value",
    "campaign_2_value", "campaign_3_value", "mean", "median", "stdev_sample",
    "cross_campaign_cv_percent", "minimum", "maximum",
    "cross_campaign_cv_review_flag", "notes",
)
MEMORY_FIELDS = ("schema_version", "section", "method", "stages",
                 "bytes_in_flight_kib") + COMMON_FIELDS
UMMA_FIELDS = ("schema_version", "section", "method", "n", "depth",
               "cta_group") + COMMON_FIELDS
GEMM_FIELDS = ("schema_version", "section", "shape_index", "shape_id", "m", "n", "k",
               "l", "candidate_index", "variant", "method") + COMMON_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate exactly three explicit final campaign directories."
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
        memory_rows, memory_summary = memory_analysis(records)
        umma_rows, umma_summary = umma_analysis(records)
        gemm_rows, gemm_summary = gemm_analysis(records)
        warnings = profile_warnings(records, umma_summary)
        write_csv(partial / "memory_paths.csv", MEMORY_FIELDS, memory_rows)
        write_csv(partial / "umma_throughput.csv", UMMA_FIELDS, umma_rows)
        write_csv(partial / "gemm_comparison.csv", GEMM_FIELDS, gemm_rows)
        figures(partial, memory_summary, umma_summary, gemm_summary)
        summary = {
            "schema_version": SCHEMA,
            "campaign_ids": [record["campaign_id"] for record in records],
            "campaign_count": CAMPAIGN_COUNT,
            "git_commit": records[0]["manifest"]["git_commit"],
            "gpu": records[0]["manifest"]["gpu"],
            "container_image": records[0]["manifest"]["container_image"],
            "statistics": "per-campaign median followed by descriptive statistics across three campaigns",
            "nsight_compute": {
                record["campaign_id"]: record["profiles"]["state"] for record in records
            },
            "memory_paths": memory_summary,
            "umma_throughput": umma_summary,
            "gemm_comparison": gemm_summary,
            "limitations": [
                "Three campaigns support descriptive statistics, not significance testing.",
                "Memory GB/s is logical useful bytes divided by kernel time, not a DRAM counter.",
                "UMMA FLOP/cycle/SM is derived from operation counts and cycles.",
                "GEMM uses hot-cache timings and does not identify the cause of performance gaps.",
                "Counter-derived metrics exist only for the configurations Nsight Compute profiled.",
            ],
        }
        (partial / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact_paths = sorted(
            path for path in partial.rglob("*") if path.is_file()
        )
        hashes = {str(path.relative_to(partial)): sha256_file(path) for path in artifact_paths}
        analysis_manifest = {
            "schema_version": "gb300.analysis-manifest.v1",
            "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
                .isoformat().replace("+00:00", "Z"),
            "campaign_ids": [record["campaign_id"] for record in records],
            "campaign_manifest_sha256": {
                record["campaign_id"]: sha256_file(record["path"] / "manifest.json")
                for record in records
            },
            "artifact_sha256": hashes,
        }
        (partial / "manifest.json").write_text(
            json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (partial / "SHA256SUMS").write_text(
            "".join(f"{digest}  {relative}\n" for relative, digest in sorted(hashes.items())),
            encoding="utf-8",
        )
        partial.rename(output)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    for message in warnings:
        print(f"analysis: WARNING: {message}", file=sys.stderr)
    print(f"analysis: COMPLETE {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AnalysisError, contract.CampaignError) as exc:
        print(f"analysis: ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
