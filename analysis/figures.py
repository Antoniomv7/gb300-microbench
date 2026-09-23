"""SVG figures of the published results; analysis/analyze.py computes every plotted value."""

import html
import math

from precision_comparison.precision_comparison import FORMATS, IMPLEMENTATIONS, REPETITIONS
from scripts.run_campaign import MEMORY_METHODS, UMMA_METHODS

COLORS = {"ldgsts": "#2563eb", "tma": "#d97706",
          "umma_1sm": "#2563eb", "umma_2sm": "#d97706"}
GEMM_COLORS = {"nonpersistent_1cta": "#2563eb", "persistent_1cta": "#7c3aed",
               "persistent_2cta": "#d97706", "heuristic_first_supported": "#15803d"}
SCALE_COLORS = {("umma_1sm", "isolated"): "#93c5fd", ("umma_1sm", "device_scale"): "#2563eb",
                ("umma_2sm", "isolated"): "#fcd34d", ("umma_2sm", "device_scale"): "#d97706"}
SCALE_LABELS = {("umma_1sm", "isolated"): "1-SM iso", ("umma_1sm", "device_scale"): "1-SM dev",
                ("umma_2sm", "isolated"): "2-SM iso", ("umma_2sm", "device_scale"): "2-SM dev"}
FORMAT_COLORS = {"bf16": "#2563eb", "fp8": "#7c3aed", "nvfp4": "#d97706"}
FORMAT_LABELS = {"bf16": "BF16", "fp8": "FP8 E4M3", "nvfp4": "NVFP4 E2M1"}
# CuTe DSL keeps the persistent_2cta amber of the GEMM figure; the pair passes the CVD checks.
IMPLEMENTATION_COLORS = {"cutedsl": "#d97706", "cublaslt": "#166534"}
IMPLEMENTATION_LABELS = {"cutedsl": "CuTe DSL persistent 2-CTA",
                         "cublaslt": "cuBLASLt first supported heuristic"}


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
                          for method in MEMORY_METHODS}} for stages in (2, 4, 8)]
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


def scaling_panel(output, x0, width, title, bars, unit, reference=None, decimals=0,
                  baseline=0.0):
    top, bottom = 145, 376
    values = [point["maximum"] for _, _, point in bars] + ([reference] if reference else [])
    maximum = max(values) * 1.12
    span = maximum - baseline
    y = lambda value: bottom - (value - baseline) * (bottom - top) / span
    output.append(svg_text(x0 + width / 2, 119, title, text_anchor="middle",
                           font_size="14", font_weight="700"))
    for tick in range(5):
        value = baseline + span * tick / 4
        yy = y(value)
        output.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x0+width:.1f}" '
                      f'y2="{yy:.1f}" stroke="#e2e8f0"/>')
        output.append(svg_text(x0 - 8, yy + 4, f"{value:,.{decimals}f}",
                               text_anchor="end", font_size="10"))
    if reference:
        output.append(f'<line x1="{x0:.1f}" y1="{y(reference):.1f}" x2="{x0+width:.1f}" '
                      f'y2="{y(reference):.1f}" stroke="#15803d" stroke-dasharray="5 4"/>')
    bar_width = width / (len(bars) * 2.1)
    for index, (label, color, point) in enumerate(bars):
        center = x0 + (index + 0.5) * width / len(bars)
        yy = y(point["mean"])
        output.append(f'<rect x="{center-bar_width/2:.1f}" y="{yy:.1f}" width="{bar_width:.1f}" '
                      f'height="{bottom-yy:.1f}" fill="{color}"/>')
        output.append(f'<line x1="{center:.1f}" y1="{y(point["minimum"]):.1f}" '
                      f'x2="{center:.1f}" y2="{y(point["maximum"]):.1f}" stroke="#0f172a"/>')
        output.append(svg_text(center, bottom + 23, label, text_anchor="middle", font_size="11"))
    output.append(svg_text(x0 + width / 2, bottom + 44, unit,
                           text_anchor="middle", font_size="11", fill="#64748b"))


def scaling_figure(summary):
    output = svg_start("BF16 UMMA: isolated work unit versus all usable SMs",
                       "Independent throughput axes; SM clock sampled during the same timed campaigns.",
                       width=1640, height=500)
    lookup = {(point["method"], point["scale"]): point for point in summary["configurations"]}
    efficiency = summary["scaling_efficiency"]
    scaling_panel(output, 86, 300, "Isolated work unit",
                  [(SCALE_LABELS[(method, "isolated")], SCALE_COLORS[(method, "isolated")],
                    lookup[(method, "isolated")]) for method in UMMA_METHODS], "Total TFLOP/s")
    scaling_panel(output, 482, 300, "Whole device",
                  [(SCALE_LABELS[(method, "device_scale")], SCALE_COLORS[(method, "device_scale")],
                    lookup[(method, "device_scale")]) for method in UMMA_METHODS], "Total TFLOP/s")
    scaling_panel(output, 878, 300, "Mean SM clock in the same campaign",
                  [(SCALE_LABELS[key], SCALE_COLORS[key], lookup[key]["clock"])
                   for key in SCALE_LABELS], "MHz", baseline=1000.0)
    scaling_panel(output, 1274, 300, "Scaling efficiency",
                  [(label, SCALE_COLORS[(method, scale)], efficiency[method][field])
                   for method, label, scale, field in (
                       ("umma_1sm", "1-SM raw", "device_scale", "raw"),
                       ("umma_1sm", "1-SM freq", "isolated", "frequency_normalized"),
                       ("umma_2sm", "2-SM raw", "device_scale", "raw"),
                       ("umma_2sm", "2-SM freq", "isolated", "frequency_normalized"))],
                  "Whole-device / (units x isolated)", reference=1.0, decimals=2)
    output.append(svg_text(34, 470,
                           "Separate CUDA-event timings; raw efficiency also carries the DVFS "
                           "difference that the frequency-normalized bars divide out.",
                           font_size="11", fill="#64748b"))
    return "\n".join([*output, "</svg>"]) + "\n"


def precision_figure(rows):
    width, height, left, right, top, bottom = 1260, 500, 86, 38, 140, 395
    output = svg_start("Low-precision GEMM: BF16 versus FP8 versus NVFP4",
                       "Three repeated hot-cache measurements per matched configuration.",
                       width, height)
    shapes = sorted({(row["shape_index"], row["shape_id"]) for row in rows})
    maximum = max(row[f"repetition_{index}_tflops"]
                  for row in rows for index in range(1, REPETITIONS + 1)) * 1.12
    y = lambda value: bottom - value * (bottom - top) / maximum

    for tick in range(6):
        value = maximum * tick / 5
        yy = y(value)
        output.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" '
                      f'y2="{yy:.1f}" stroke="#e2e8f0"/>')
        output.append(svg_text(left - 9, yy + 4, f"{value:,.0f}",
                               text_anchor="end", font_size="10", fill="#64748b"))

    for index, precision in enumerate(FORMATS):
        xx = 685 + index * 165
        output.append(f'<rect x="{xx}" y="83" width="12" height="12" '
                      f'fill="{FORMAT_COLORS[precision]}"/>')
        output.append(svg_text(xx + 18, 94, FORMAT_LABELS[precision], font_size="11"))

    lookup = {(row["shape_index"], row["precision"]): row for row in rows}
    group_width = (width - left - right) / len(shapes)
    bar_width = (group_width - 70) / len(FORMATS)
    for index, (shape_index, shape_id) in enumerate(shapes):
        for position, precision in enumerate(FORMATS):
            row = lookup[(shape_index, precision)]
            samples = [row[f"repetition_{sample}_tflops"]
                       for sample in range(1, REPETITIONS + 1)]
            xx = left + index * group_width + 30 + position * (bar_width + 3)
            yy = y(row["mean_tflops"])
            output.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bar_width:.1f}" '
                          f'height="{bottom-yy:.1f}" fill="{FORMAT_COLORS[precision]}"/>')
            center = xx + bar_width / 2
            output.append(f'<line x1="{center:.1f}" y1="{y(min(samples)):.1f}" '
                          f'x2="{center:.1f}" y2="{y(max(samples)):.1f}" '
                          'stroke="#0f172a"/>')
        label = "×".join(shape_id.removesuffix("x1").split("x"))
        output.append(svg_text(left + (index + 0.5) * group_width, bottom + 26,
                               label, text_anchor="middle", font_size="11"))

    output.append(svg_text(19, 268, "TFLOP/s", text_anchor="middle",
                           transform="rotate(-90 19 268)"))
    output.append(svg_text(34, 470,
                           "FP32 accumulation/output; tile 256×128; cluster 2×1; "
                           "NVFP4 uses one E4M3 scale per 16 values.",
                           font_size="11", fill="#64748b"))
    return "\n".join([*output, "</svg>"]) + "\n"


def tick_step(span, count=5):
    """Smallest round step (1 to 5 times a power of ten) giving at most count intervals."""
    magnitude = 10 ** math.floor(math.log10(span / count))
    return next(step * magnitude for step in (1, 1.5, 2, 2.5, 3, 4, 5, 10)
                if step * magnitude * count >= span)


def precision_comparison_figure(rows, warmup, iterations):
    width, height, left, right, top, bottom = 1260, 540, 86, 38, 140, 408
    output = svg_start("CuTe DSL versus cuBLASLt by precision",
                       "Same logical operands and FP32 output within each format; mean of three "
                       "repetitions, whiskers show their range.", width, height)
    shapes = sorted({(row["shape_index"], row["shape_id"]) for row in rows})
    lookup = {(row["shape_index"], row["precision"], row["implementation"]): row for row in rows}
    samples = lambda row: [row[f"repetition_{index}_tflops"]
                           for index in range(1, REPETITIONS + 1)]
    highest = max(max(samples(row)) for row in rows) * 1.10
    step = tick_step(highest)
    maximum = step * math.ceil(highest / step)
    y = lambda value: bottom - value * (bottom - top) / maximum

    for tick in range(round(maximum / step) + 1):
        value = step * tick
        yy = y(value)
        output.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" '
                      f'y2="{yy:.1f}" stroke="#e2e8f0"/>')
        output.append(svg_text(left - 9, yy + 4, f"{value:,.0f}",
                               text_anchor="end", font_size="10", fill="#64748b"))
    for index, implementation in enumerate(IMPLEMENTATIONS):
        xx = 640 + index * 300
        output.append(f'<rect x="{xx}" y="83" width="12" height="12" '
                      f'fill="{IMPLEMENTATION_COLORS[implementation]}"/>')
        output.append(svg_text(xx + 18, 94, IMPLEMENTATION_LABELS[implementation],
                               font_size="11"))

    group_width = (width - left - right) / len(shapes)
    bar_width, bar_gap, pair_gap = 24, 2, 64
    pair_width = 2 * bar_width + bar_gap
    span = len(FORMATS) * pair_width + (len(FORMATS) - 1) * pair_gap
    for index, (shape_index, shape_id) in enumerate(shapes):
        start = left + index * group_width + (group_width - span) / 2
        for position, precision in enumerate(FORMATS):
            x0 = start + position * (pair_width + pair_gap)
            pair = [lookup[(shape_index, precision, implementation)]
                    for implementation in IMPLEMENTATIONS]
            for offset, row in enumerate(pair):
                xx = x0 + offset * (bar_width + bar_gap)
                yy = y(row["mean_tflops"])
                values = samples(row)
                title = (f"{IMPLEMENTATION_LABELS[row['implementation']]}, "
                         f"{FORMAT_LABELS[precision]}, "
                         f"{shape_id}: {row['mean_tflops']:,.1f} TFLOP/s "
                         f"(range {min(values):,.1f} to {max(values):,.1f})")
                output.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bar_width}" '
                              f'height="{bottom-yy:.1f}" '
                              f'fill="{IMPLEMENTATION_COLORS[row["implementation"]]}">'
                              f'<title>{html.escape(title)}</title></rect>')
                center = xx + bar_width / 2
                output.append(f'<line x1="{center:.1f}" y1="{y(min(values)):.1f}" '
                              f'x2="{center:.1f}" y2="{y(max(values)):.1f}" stroke="#0f172a"/>')
            highest = max(max(samples(row)) for row in pair)
            output.append(svg_text(x0 + pair_width / 2, y(highest) - 8,
                                   f"{pair[0]['throughput_ratio_vs_cublaslt']:.2f}×",
                                   text_anchor="middle", font_size="11", fill="#334155"))
            output.append(svg_text(x0 + pair_width / 2, bottom + 18, FORMAT_LABELS[precision],
                                   text_anchor="middle", font_size="11"))
        label = "×".join(shape_id.removesuffix("x1").split("x"))
        output.append(svg_text(left + (index + 0.5) * group_width, bottom + 42,
                               label, text_anchor="middle", font_size="12", font_weight="700"))

    output.append(svg_text(19, 274, "TFLOP/s", text_anchor="middle",
                           transform="rotate(-90 19 274)"))
    output.append(svg_text(34, 508,
                           "Ratio above each pair: CuTe DSL / cuBLASLt mean throughput. Each "
                           f"repetition times {iterations} hot-cache launches after {warmup} "
                           "warm-up launches.", font_size="11", fill="#64748b"))
    return "\n".join([*output, "</svg>"]) + "\n"
