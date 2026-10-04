#!/usr/bin/env python3
"""Analyze Metal benchmark CSVs without removing scheduling outliers.

Example:
  python3 script/analyze_metal_bench.py run.csv --outputstem run-analysis

Print Markdown by default, or JSON with --format json. --outputstem writes both
formats. Variant 0 is baseline, 1 is candidate; paired cycles must be ABBA/BAAB.
All finite measurements contribute to distributions. Missing/nonfinite values
are counted explicitly; only complete cycles with two observations per variant
contribute to a metric's paired estimate. Intervals are approximate normal 95%
Bartlett/Newey-West intervals, with lag windows measured in four-frame cycles.
"""

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = (
    "decode_gpu_ms", "decode_wall_ms", "packet_to_gpu_ms", "commands_cpu_ms",
)
WINDOWS = (20, 60, 120)
LABELS = ("baseline", "candidate")


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def integer(value):
    value = number(value)
    return int(value) if value is not None and value.is_integer() else None


def variant(value):
    if value in LABELS:
        return LABELS.index(value)
    value = integer(value)
    return value if value in (0, 1) else None


def mean(values):
    return math.fsum(values) / len(values) if values else None


def percentile(ordered, probability):
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    finite = sorted(value for value in values if value is not None)
    return {
        "observations": len(finite), "missing_or_nonfinite": len(values) - len(finite),
        "mean_ms": mean(finite), "p50_ms": percentile(finite, 0.50),
        "p95_ms": percentile(finite, 0.95), "p99_ms": percentile(finite, 0.99),
    }


def hac_interval(deltas, window):
    """Use cycle IDs to preserve serial lag distances across missing cycles."""
    count = len(deltas)
    if count <= window:
        return {"lag_window_blocks": window, "ci95_ms": None,
                "unavailable_reason": "Paired block count must exceed the lag window."}
    average = mean(list(deltas.values()))
    centered = {cycle: value - average for cycle, value in deltas.items()}
    covariance = math.fsum(value * value for value in centered.values()) / count
    for lag in range(1, window + 1):
        cross = math.fsum(value * centered[cycle - lag]
                          for cycle, value in centered.items() if cycle - lag in centered)
        covariance += 2 * (1 - lag / (window + 1)) * cross / count
    radius = 1.96 * math.sqrt(max(0.0, covariance) / count)
    return {"lag_window_blocks": window, "ci95_ms": [average - radius, average + radius]}


def paired_summary(groups, metric):
    deltas = {}
    baseline_means = []
    candidate_means = []
    for cycle, rows in groups:
        if any(row[metric] is None for row in rows):
            continue
        baseline = mean([row[metric] for row in rows if row["variant"] == 0])
        candidate = mean([row[metric] for row in rows if row["variant"] == 1])
        baseline_means.append(baseline)
        candidate_means.append(candidate)
        deltas[cycle] = candidate - baseline
    baseline = mean(baseline_means)
    effect = mean(list(deltas.values()))
    return {
        "paired_blocks": len(deltas), "blocks_missing_metric": len(groups) - len(deltas),
        "baseline_mean_ms": baseline, "candidate_mean_ms": mean(candidate_means),
        "mean_delta_ms": effect,
        "delta_percent_of_paired_baseline":
            100 * effect / baseline if baseline is not None and baseline != 0 else None,
        "p50_delta_ms": percentile(sorted(deltas.values()), 0.50),
        "ci95": {str(window): hac_interval(deltas, window) for window in WINDOWS},
    }


def analyze(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = []
        for row in reader:
            sample = {metric: number(row.get(metric)) for metric in METRICS}
            sample["variant"] = variant(row.get("variant"))
            sample["index"] = integer(row.get("index"))
            flag = integer(row.get("deadline_missed"))
            sample["deadline_missed"] = flag if flag in (0, 1) else None
            rows.append(sample)

    missing_fields = [field for field in ("variant", "deadline_missed") + METRICS if field not in fields]
    warnings = []
    if not rows:
        warnings.append("No sample rows.")
    if missing_fields:
        warnings.append("Missing fields: " + ", ".join(missing_fields) + ".")
    counts = {label: sum(row["variant"] == value for row in rows)
              for value, label in enumerate(LABELS)}
    unknown = sum(row["variant"] is None for row in rows)
    if unknown:
        warnings.append(str(unknown) + " rows have an unknown/missing variant; retained in all-sample distributions.")

    # Benchmark index aligns four-frame cycles even if the CSV is cropped or
    # has lost a row. When index is unavailable, validate consecutive row groups.
    indices = [row["index"] for row in rows]
    indexed = bool(rows) and all(index is not None and index >= 0 for index in indices)
    indexed = indexed and all(a < b for a, b in zip(indices, indices[1:]))
    if rows and not indexed:
        warnings.append("Index absent/invalid/nonmonotonic; cycles use CSV row order.")
    cycles = {}
    for position, row in enumerate(rows):
        cycle = row["index"] // 4 if indexed else position // 4
        cycles.setdefault(cycle, []).append(row)
    cycle_ids = list(cycles)
    missing_cycles = sum(b - a - 1 for a, b in zip(cycle_ids, cycle_ids[1:])) if indexed else 0
    if missing_cycles:
        warnings.append(str(missing_cycles) + " whole cycles are absent; HAC lag distances preserve those gaps.")

    complete = []
    incomplete = []
    invalid = []
    comparison = counts["baseline"] > 0 and counts["candidate"] > 0
    for cycle, group in cycles.items():
        if len(group) != 4:
            incomplete.append({"cycle": cycle, "rows": len(group)})
        elif comparison:
            order = [row["variant"] for row in group]
            if order in ([0, 1, 1, 0], [1, 0, 0, 1]):
                complete.append((cycle, group))
            else:
                invalid.append({"cycle": cycle, "variant_order": order})
    if incomplete:
        warnings.append(str(len(incomplete)) + " incomplete cycles; their measurements remain in distributions.")
    if invalid:
        warnings.append(str(len(invalid)) + " cycles do not have ABBA/BAAB order; excluded only from pairing.")
    if rows and not comparison:
        warnings.append("Both baseline and candidate are required for paired effects.")

    summaries = {}
    for value, label in enumerate(LABELS):
        selected = [row for row in rows if row["variant"] == value]
        flags = [row["deadline_missed"] for row in selected if row["deadline_missed"] is not None]
        summaries[label] = {
            "samples": len(selected), "deadline_miss_observations": len(flags),
            "deadline_misses": sum(flags) if flags else None,
            "deadline_miss_percent": 100 * sum(flags) / len(flags) if flags else None,
            "metrics": {metric: distribution([row[metric] for row in selected]) for metric in METRICS},
        }
    return {
        "path": str(path), "rows": len(rows), "warnings": warnings,
        "validation": {
            "variant_counts": counts, "unknown_variant_rows": unknown,
            "balanced": bool(rows) and not unknown and counts["baseline"] == counts["candidate"],
            "cycle_basis": "index // 4" if indexed else "CSV row order / 4",
            "complete_abba_blocks": len(complete), "incomplete_cycles": incomplete,
            "invalid_cycles": invalid, "missing_whole_cycles": missing_cycles,
            "missing_fields": missing_fields,
        },
        "all_samples": {metric: distribution([row[metric] for row in rows]) for metric in METRICS},
        "variants": summaries,
        "paired": {metric: paired_summary(complete, metric) for metric in METRICS},
    }


def format_number(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


def interval_text(interval):
    bounds = interval["ci95_ms"]
    return "—" if bounds is None else f"[{1000 * bounds[0]:+.3f}, {1000 * bounds[1]:+.3f}]"


def markdown(report):
    lines = ["# Metal benchmark analysis", "", "All finite samples retained; no outlier removal. "
             "Percentiles use linear interpolation. Paired deltas are candidate minus baseline; "
             "negative values favor candidate. Approximate normal 95% Bartlett/Newey–West intervals "
             "use lag windows of 20/60/120 four-frame cycles."]
    for run in report["runs"]:
        lines.extend(["", "## " + run["path"].replace("|", "\\|"), ""])
        if "error" in run:
            lines.append("Error: " + run["error"])
            continue
        validation = run["validation"]
        counts = validation["variant_counts"]
        lines.append(f"{run['rows']} rows; baseline {counts['baseline']}, candidate {counts['candidate']}; "
                     f"balanced: {str(validation['balanced']).lower()}; "
                     f"complete ABBA/BAAB blocks: {validation['complete_abba_blocks']}.")
        for label in LABELS:
            summary = run["variants"][label]
            misses = summary["deadline_misses"]
            lines.append(f"{label.capitalize()} deadline misses: " +
                         ("unavailable." if misses is None else
                          f"{misses}/{summary['deadline_miss_observations']} "
                          f"({format_number(summary['deadline_miss_percent'], 3)}%)."))
        if validation["unknown_variant_rows"]:
            lines.extend(["", "All-sample distributions, including unknown variants:", "",
                          "| Metric | N | Missing | Mean ms | p50 ms | p95 ms | p99 ms |",
                          "|---|---:|---:|---:|---:|---:|---:|"])
            for metric in METRICS:
                values = run["all_samples"][metric]
                stats = " | ".join(format_number(values[field]) for field in
                                   ("mean_ms", "p50_ms", "p95_ms", "p99_ms"))
                lines.append(f"| {metric} | {values['observations']} | {values['missing_or_nonfinite']} | {stats} |")
        lines.extend(["", "| Variant | Metric | N | Missing | Mean ms | p50 ms | p95 ms | p99 ms |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"])
        for label in LABELS:
            for metric in METRICS:
                values = run["variants"][label]["metrics"][metric]
                stats = " | ".join(format_number(values[field]) for field in
                                   ("mean_ms", "p50_ms", "p95_ms", "p99_ms"))
                lines.append(f"| {label} | {metric} | {values['observations']} | "
                             f"{values['missing_or_nonfinite']} | {stats} |")
        lines.extend(["", "| Paired metric | Blocks | Missing blocks | Mean delta µs | Delta % | "
                      "95% NW20 µs | 95% NW60 µs | 95% NW120 µs |",
                      "|---|---:|---:|---:|---:|---|---|---|"])
        for metric in METRICS:
            values = run["paired"][metric]
            delta = values["mean_delta_ms"]
            intervals = " | ".join(interval_text(values["ci95"][str(window)]) for window in WINDOWS)
            lines.append(f"| {metric} | {values['paired_blocks']} | {values['blocks_missing_metric']} | "
                         f"{format_number(1000 * delta if delta is not None else None, 3)} | "
                         f"{format_number(values['delta_percent_of_paired_baseline'], 3)} | {intervals} |")
        if run["warnings"]:
            lines.append("")
            lines.extend("- " + warning for warning in run["warnings"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_paths", nargs="+", type=Path)
    parser.add_argument("--outputstem", type=Path, help="Write STEM.json and STEM.md (no results merged from earlier runs)")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown", help="Standard output format")
    args = parser.parse_args()
    report = {"units": "milliseconds", "ci": "approximate normal 95% Bartlett/Newey-West",
              "lag_windows_blocks": list(WINDOWS), "samples_per_variant_per_block": 2,
              "outlier_removal": False, "runs": []}
    failed = False
    for path in args.csv_paths:
        try:
            report["runs"].append(analyze(path))
        except (OSError, UnicodeError, csv.Error) as error:
            report["runs"].append({"path": str(path), "error": str(error)})
            failed = True
    json_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    md_text = markdown(report)
    if args.outputstem:
        try:
            Path(str(args.outputstem) + ".json").write_text(json_text, encoding="utf-8")
            Path(str(args.outputstem) + ".md").write_text(md_text, encoding="utf-8")
        except OSError as error:
            parser.exit(1, "Cannot write analysis output: " + str(error) + "\n")
    print(json_text if args.format == "json" else md_text, end="")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
