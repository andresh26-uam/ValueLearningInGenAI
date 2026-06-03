#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


@dataclass
class GroupDefinition:
    label: str
    folders: list[Path]
    metric_sources: dict[str, str] = field(default_factory=dict)


@dataclass
class GroupMetrics:
    definition: GroupDefinition
    frame: pd.DataFrame
    weights_frame: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class MetricDefinition:
    source: str
    label: str


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = value
    for key, replacement in replacements.items():
        escaped = escaped.replace(key, replacement)
    return escaped


def _latex_row(cells: list[str]) -> str:
    return " & ".join(cells) + r" \\"


def _resolve_metrics_folder(path: Path) -> Optional[Path]:
    if path.is_file() and path.name == "test_metrics.csv":
        return path.parent
    if path.is_dir() and (path / "test_metrics.csv").is_file():
        return path
    if path.is_dir():
        matches = [csv_path.parent for csv_path in path.rglob("test_metrics.csv")]
        if len(matches) == 1:
            return matches[0]
    return None


def _load_metrics_frame(folder: Path) -> pd.DataFrame:
    csv_path = folder / "test_metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing test_metrics.csv in {folder}")

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"Empty metrics CSV: {csv_path}")

    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    numeric_frame = numeric_frame.dropna(axis=1, how="all")
    return numeric_frame


def _load_value_system_weights_frame(folder: Path) -> pd.DataFrame:
    csv_path = folder / "value_system_weights.csv"
    if not csv_path.is_file():
        return pd.DataFrame()

    frame = pd.read_csv(csv_path)
    if frame.empty:
        return pd.DataFrame()

    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    numeric_frame = numeric_frame.dropna(axis=1, how="all")
    return numeric_frame


def _load_group_from_folders(group: GroupDefinition) -> GroupMetrics:
    frames = []
    weight_frames = []
    for folder in group.folders:
        resolved_folder = _resolve_metrics_folder(folder)
        if resolved_folder is None:
            raise FileNotFoundError(f"Could not locate test_metrics.csv for folder: {folder}")
        frames.append(_load_metrics_frame(resolved_folder))
        weights_frame = _load_value_system_weights_frame(resolved_folder)
        if not weights_frame.empty:
            weight_frames.append(weights_frame)

    combined = pd.concat(frames, ignore_index=True, sort=True)
    combined = combined.select_dtypes(include=[np.number])
    combined_weights = pd.concat(weight_frames, ignore_index=True, sort=True) if weight_frames else pd.DataFrame()
    combined_weights = combined_weights.select_dtypes(include=[np.number]) if not combined_weights.empty else combined_weights
    return GroupMetrics(group, combined, combined_weights)


def _load_groups_from_json(json_file: Path, results_root: Path) -> list[GroupDefinition]:
    payload = json.loads(json_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        groups_payload = payload.get("groups", payload.get("group", []))
    else:
        groups_payload = payload

    groups: list[GroupDefinition] = []
    for index, item in enumerate(groups_payload):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid group entry at index {index}: expected an object")

        raw_folders = item.get("folders") or item.get("paths") or item.get("folder")
        if raw_folders is None:
            raise ValueError(f"Group entry {index} does not define folders")
        if isinstance(raw_folders, (str, Path)):
            raw_folders = [raw_folders]

        resolved_folders: list[Path] = []
        for raw_folder in raw_folders:
            candidate = Path(raw_folder).expanduser()
            if not candidate.is_absolute():
                candidate = (results_root / candidate).resolve()
            resolved = _resolve_metrics_folder(candidate)
            if resolved is None:
                matching_checkpoints = [
                    folder.parent
                    for folder in results_root.rglob("test_metrics.csv")
                    if raw_folder.casefold() in folder.parent.as_posix().casefold()
                ]
                if len(matching_checkpoints) == 1:
                    resolved = matching_checkpoints[0]
                else:
                    raise FileNotFoundError(
                        f"Could not resolve a metrics folder for '{raw_folder}' in group {index}"
                    )
            resolved_folders.append(resolved)

        metric_sources: dict[str, str] = {}
        raw_metrics = item.get("metrics") or item.get("metric_mapping") or item.get("metric_map")
        if isinstance(raw_metrics, dict):
            for label, source_name in raw_metrics.items():
                metric_sources[str(label)] = str(source_name)
        elif isinstance(raw_metrics, list):
            for metric_index, metric_item in enumerate(raw_metrics):
                if isinstance(metric_item, str):
                    metric_sources[metric_item] = metric_item
                    continue
                if isinstance(metric_item, dict):
                    source_name = metric_item.get("source") or metric_item.get("name") or metric_item.get("metric")
                    if source_name is None:
                        raise ValueError(f"Metric entry {metric_index} in group {index} does not define a source field")
                    label = metric_item.get("label") or metric_item.get("display") or metric_item.get("latex") or source_name
                    metric_sources[str(label)] = str(source_name)
                    continue
                raise ValueError(f"Invalid metric entry at index {metric_index} in group {index}: expected a string or object")

        label = item.get("name") or item.get("label") or resolved_folders[0].name
        groups.append(GroupDefinition(label=label, folders=resolved_folders, metric_sources=metric_sources))

    return groups


def _load_metric_definitions(json_file: Path) -> list[MetricDefinition]:
    payload = json.loads(json_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []

    raw_metrics = payload.get("metrics") or payload.get("metric_mapping") or payload.get("metric_map")
    if raw_metrics is None:
        return []

    metric_definitions: list[MetricDefinition] = []
    if isinstance(raw_metrics, dict):
        for source_name, display_name in raw_metrics.items():
            metric_definitions.append(MetricDefinition(source=str(source_name), label=str(display_name)))
        return metric_definitions

    if isinstance(raw_metrics, list):
        for index, item in enumerate(raw_metrics):
            if isinstance(item, str):
                metric_definitions.append(MetricDefinition(source=item, label=item))
                continue
            if isinstance(item, dict):
                source_name = item.get("source") or item.get("name") or item.get("metric")
                if source_name is None:
                    raise ValueError(f"Metric entry {index} does not define a source field")
                display_name = item.get("label") or item.get("latex") or item.get("display") or source_name
                metric_definitions.append(MetricDefinition(source=str(source_name), label=str(display_name)))
                continue
            raise ValueError(f"Invalid metric entry at index {index}: expected a string or object")
        return metric_definitions

    raise ValueError("'metrics' must be a mapping or a list")


def _collect_group_metrics(groups: list[GroupDefinition]) -> list[GroupMetrics]:
    return [_load_group_from_folders(group) for group in groups]


def _selected_metric_definitions(groups: list[GroupMetrics], metric_definitions: list[MetricDefinition]) -> list[MetricDefinition]:
    if metric_definitions:
        return metric_definitions

    metric_set: set[str] = set()
    for group in groups:
        metric_set.update(group.definition.metric_sources.keys())

    if metric_set:
        return [MetricDefinition(source=metric_name, label=metric_name) for metric_name in sorted(metric_set)]

    for group in groups:
        metric_set.update(column for column in group.frame.columns if column != "__source_folder__")

    return [MetricDefinition(source=metric_name, label=metric_name) for metric_name in sorted(metric_set)]


def _selected_weight_definitions(groups: list[GroupMetrics]) -> list[MetricDefinition]:
    weight_columns: set[str] = set()
    for group in groups:
        weight_columns.update(column for column in group.weights_frame.columns if column != "__source_folder__")

    return [MetricDefinition(source=weight_column, label=weight_column) for weight_column in sorted(weight_columns)]


def _group_column(group: GroupMetrics, source_name: str) -> Optional[pd.Series]:
    if source_name in group.frame.columns:
        return group.frame[source_name]
    if source_name in group.weights_frame.columns:
        return group.weights_frame[source_name]
    return None


def _normalize_group_metrics(group: GroupMetrics, metric_definitions: list[MetricDefinition]) -> GroupMetrics:
    normalized_columns: dict[str, pd.Series] = {}
    for metric in metric_definitions:
        source_name = group.definition.metric_sources.get(metric.label, metric.source)
        column = _group_column(group, source_name)
        if column is not None:
            normalized_columns[metric.label] = column

    normalized_frame = pd.DataFrame(normalized_columns)
    return GroupMetrics(group.definition, normalized_frame, group.weights_frame)


def _format_mean_std(values: pd.Series) -> str:
    clean = values.dropna()
    if clean.empty:
        return "--"
    mean = clean.mean()
    std = clean.std(ddof=1) if len(clean) > 1 else 0.0
    return f"\\makecell[r]{{{mean:.3f} \\\\ $\\pm${std:.4f}}}"


def _format_mean_std_weights(values: pd.Series) -> str:
    clean = values.dropna()
    if clean.empty:
        return "--"
    mean = clean.mean()
    std = clean.std(ddof=1) if len(clean) > 1 else 0.0
    return f"\\makecell[r]{{{mean:.3f} \\\\ $\\pm${std:.3f}}}"


def _format_number(
    value: float | int | np.floating | np.integer | None,
    decimals: int = 4,
) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{decimals}f}"


def _summary_table(groups: list[GroupMetrics], metric_definitions: list[MetricDefinition]) -> str:
    metric_labels = [_latex_escape(metric.label) for metric in metric_definitions]
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        r"\caption{Average and standard deviation of each metric and value system weight per group.}",
        r"\label{tab:group-summary}",
        rf"\begin{{tabular}}{{{'l' + 'r' * len(metric_definitions)}}}",
        r"\toprule",
        _latex_row(["Group"] + metric_labels),
        r"\midrule",
    ]

    if groups:
        for group in groups:
            row = [_latex_escape(group.definition.label)]
            for metric in metric_definitions:
                column = _group_column(group, metric.label)
                if column is not None:
                    row.append(_format_mean_std(column))
                else:
                    row.append("--")
            lines.append(_latex_row(row))
    else:
        lines.append(r"\multicolumn{1}{l}{No data available} \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _weights_table(groups: list[GroupMetrics], weight_definitions: list[MetricDefinition]) -> str:
    weight_labels = [_latex_escape(weight.label) for weight in weight_definitions]
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        r"\caption{Average and standard deviation of value system weights per group.}",
        r"\label{tab:group-weights}",
        rf"\begin{{tabular}}{{{'l' + 'r' * len(weight_definitions)}}}",
        r"\toprule",
        _latex_row(["Group"] + weight_labels),
        r"\midrule",
    ]

    if groups:
        for group in groups:
            row = [_latex_escape(group.definition.label)]
            for weight in weight_definitions:
                column = _group_column(group, weight.label)
                if column is not None:
                    row.append(_format_mean_std_weights(column))
                else:
                    row.append("--")
            lines.append(_latex_row(row))
    else:
        lines.append(r"\multicolumn{1}{l}{No data available} \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _welch_test_adaptive(a: pd.Series, b: pd.Series) -> tuple[float, float, str]:
    a_values = a.dropna().to_numpy(dtype=float)
    b_values = b.dropna().to_numpy(dtype=float)
    if len(a_values) < 2 or len(b_values) < 2:
        return float("nan"), float("nan"), "--"

    mean_a = float(np.mean(a_values))
    mean_b = float(np.mean(b_values))
    if mean_a > mean_b:
        alternative = "A>B"
    elif mean_a < mean_b:
        alternative = "A<B"
    else:
        alternative = "A=B"

    result = ttest_ind(a_values, b_values, equal_var=False, nan_policy="omit")
    statistic = float(result.statistic)
    p_two_sided = float(result.pvalue)

    if pd.isna(statistic) or pd.isna(p_two_sided):
        return statistic, float("nan"), alternative

    # Choose one-tailed p-value based on observed mean ordering.
    if alternative == "A>B":
        if statistic > 0:
            p_one_tailed = p_two_sided / 2.0
        elif statistic < 0:
            p_one_tailed = 1.0 - (p_two_sided / 2.0)
        else:
            p_one_tailed = 0.5
    elif alternative == "A<B":
        if statistic < 0:
            p_one_tailed = p_two_sided / 2.0
        elif statistic > 0:
            p_one_tailed = 1.0 - (p_two_sided / 2.0)
        else:
            p_one_tailed = 0.5
    else:
        p_one_tailed = 1.0

    return statistic, p_one_tailed, alternative


def _significance_marker(p_value: float) -> str:
    if pd.isna(p_value):
        return "--"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _pairwise_table(groups: list[GroupMetrics], metric_definitions: list[MetricDefinition]) -> str:
    lines = [
        r"\begin{longtable}{llllrrl}",
        r"\caption{Pairwise one-tailed Welch t-tests with data-adaptive direction per metric: test $H_1: \mu_{A} > \mu_{B}$ when Group A's sample mean exceeds Group B's, otherwise test $H_1: \mu_{A} < \mu_{B}$.}\label{tab:pairwise-tests}\\",
        r"\toprule",
        r"Group A & Group B & Metric & Test & $t$ & One-tailed $p$-value & Sig. \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Group A & Group B & Metric & Test & $t$ & One-tailed $p$-value & Sig. \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]

    rows: list[str] = []
    for left, right in combinations(groups, 2):
        for metric in metric_definitions:
            if metric.label in left.frame.columns and metric.label in right.frame.columns:
                statistic, p_value, alternative = _welch_test_adaptive(
                    left.frame[metric.label], right.frame[metric.label]
                )
            else:
                statistic, p_value, alternative = float("nan"), float("nan"), "--"

            rows.append(
                _latex_row(
                    [
                        _latex_escape(left.definition.label),
                        _latex_escape(right.definition.label),
                        _latex_escape(metric.label),
                        f"Welch t-test ({alternative})",
                        _format_number(statistic, decimals=2),
                        _format_number(p_value),
                        _significance_marker(p_value),
                    ]
                )
            )

    if rows:
        lines.extend(rows)
    else:
        lines.append(r"\multicolumn{7}{l}{No statistical tests were run.} \\")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def _build_document(
    groups: list[GroupMetrics],
    metric_definitions: list[MetricDefinition],
    weight_definitions: list[MetricDefinition],
) -> str:
    sections = [
        r"\section*{Summary}",
        _summary_table(groups, metric_definitions),
        r"\section*{Value System Weights}",
        _weights_table(groups, weight_definitions),
    ]

    if len(groups) > 1:
        sections.extend([
            r"\section*{Pairwise statistical tests}",
            _pairwise_table(groups, metric_definitions),
        ])
    else:
        sections.append(r"\paragraph{} Only one group was provided, so no statistical tests were run.")

    body = "\n\n".join(sections)
    lines = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{makecell}",
        r"\usepackage{graphicx}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        r"\title{Analyzed Results}",
        r"\date{}",
        r"\maketitle",
        body,
        r"\end{document}",
    ]
    return "\n\n".join(lines)


def _write_output(tex_content: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex_content, encoding="utf-8")
    print(f"LaTeX results written to: {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze test_metrics.csv files and write a LaTeX report.")
    parser.add_argument(
        "--results_root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory that contains the results folders.",
    )
    parser.add_argument(
        "--json_file",
        type=Path,
        required=True,
        help="JSON file that defines the groups and folders to analyze.",
    )
    parser.add_argument(
        "--output_tex",
        type=Path,
        default=None,
        help="Optional explicit output path for the generated .tex file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    json_file = args.json_file.resolve()

    groups = _load_groups_from_json(json_file, results_root)
    if not groups:
        raise SystemExit("No groups were defined in the JSON file.")

    raw_group_metrics = _collect_group_metrics(groups)
    metric_definitions = _selected_metric_definitions(raw_group_metrics, _load_metric_definitions(json_file))
    weight_definitions = _selected_weight_definitions(raw_group_metrics)
    group_metrics = [_normalize_group_metrics(group, metric_definitions) for group in raw_group_metrics]
    document = _build_document(group_metrics, metric_definitions, weight_definitions)

    output_path = args.output_tex
    if output_path is None:
        output_path = Path.cwd() / f"analyzed_results_{json_file.stem}_{datetime.now().strftime('%Y%m%d')}.tex"
    _write_output(document, output_path)


if __name__ == "__main__":
    main()
