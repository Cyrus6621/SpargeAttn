#!/usr/bin/env python3
"""Focused matched-budget fallback analysis for Q-block disagreement.

The K-block disagreement target is always computed from the original Q:
``selection_disagreement_adjusted_r50``.  The only compared Q-block gates are
``1-selfsim(q)`` and ``1-selfsim(q-kbar)``.  Each gate selects exactly the same
number of Q blocks inside every document x layer x Q-head group.

The output intentionally contains only confusion-matrix quantities,
precision, recall, F1, and a decomposition of missed dangerous blocks into a
budget-forced part and an avoidable ranking part.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = (
    SCRIPT_DIR.parents[1]
    / "output"
    / "q_sim"
    / "govreport_test_qwen3_8b_32k"
)
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR.parents[1] / "output" / "q_sim" / "small_recall_analysis"
)
SAMPLE_PATTERN = re.compile(r"^sample-(\d+)\.parquet$")

DANGER_FRACTION = 0.20
FALLBACK_BUDGETS = (0.10, 0.20)
EXPECTED_SAMPLES = 973
EXPECTED_LAYERS = 36
EXPECTED_HEADS = 32
GROUPS_PER_DOCUMENT = EXPECTED_LAYERS * EXPECTED_HEADS

GATES = (
    ("raw_q", "q_selfsim_raw", r"$1-s(q)$"),
    ("q_minus_kbar", "q_selfsim_q_minus_kbar", r"$1-s(q-\bar{k})$"),
)
TARGET_COLUMN = "selection_disagreement_adjusted_r50"

DOCUMENT_METRICS = (
    "actual_fallback_fraction",
    "danger_fraction",
    "tp_fraction",
    "fp_fraction",
    "fn_fraction",
    "tn_fraction",
    "precision",
    "recall",
    "f1",
    "max_possible_recall",
    "budget_forced_miss",
    "ranking_miss",
)


@dataclass(frozen=True)
class SampleInfo:
    dataset_index: int
    parquet_path: Path
    metadata_path: Path
    rows: int
    blocks: int
    original_tokens: int
    used_tokens: int
    truncated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw-q and q-kbar self-sim gates with matched fallback "
            "budgets using TP/FP/FN/TN, precision, recall, and F1."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def atomic_savefig(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.stem}.tmp.{os.getpid()}{path.suffix}"
    )
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, path)


def sample_index(path: Path) -> int:
    match = SAMPLE_PATTERN.match(path.name)
    if match is None:
        raise RuntimeError(f"Unexpected sample filename: {path}")
    return int(match.group(1))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def discover_samples(run_dir: Path) -> tuple[list[SampleInfo], dict[str, Any]]:
    sample_dir = run_dir / "samples"
    paths = sorted(sample_dir.glob("sample-*.parquet"), key=sample_index)
    indices = [sample_index(path) for path in paths]
    expected = list(range(EXPECTED_SAMPLES))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLES} samples, found {len(paths)}; "
            f"missing={missing[:20]}"
        )

    infos: list[SampleInfo] = []
    schemas: set[str] = set()
    total_rows = 0
    for path in paths:
        index = sample_index(path)
        metadata_path = path.with_suffix(".metadata.json")
        if not metadata_path.exists():
            raise RuntimeError(f"Missing metadata: {metadata_path}")
        metadata = read_json(metadata_path)
        parquet_file = pq.ParquetFile(path)
        rows = parquet_file.metadata.num_rows
        schemas.add(str(parquet_file.schema_arrow))
        if int(metadata["dataset_index"]) != index:
            raise RuntimeError(f"Metadata index mismatch: {metadata_path}")
        if int(metadata["rows"]) != rows:
            raise RuntimeError(f"Metadata row mismatch: {metadata_path}")
        if rows % GROUPS_PER_DOCUMENT:
            raise RuntimeError(
                f"{path}: rows={rows} not divisible by "
                f"{GROUPS_PER_DOCUMENT}"
            )
        blocks = rows // GROUPS_PER_DOCUMENT
        infos.append(
            SampleInfo(
                dataset_index=index,
                parquet_path=path,
                metadata_path=metadata_path,
                rows=rows,
                blocks=blocks,
                original_tokens=int(metadata["original_tokens"]),
                used_tokens=int(metadata["used_tokens"]),
                truncated=bool(metadata["truncated"]),
            )
        )
        total_rows += rows

    if len(schemas) != 1:
        raise RuntimeError(f"Sample schemas differ: {len(schemas)} schemas")
    nonempty = [info for info in infos if info.rows > 0]
    empty = [info for info in infos if info.rows == 0]
    validation = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "sample_files": len(infos),
        "nonempty_documents": len(nonempty),
        "short_zero_row_documents": len(empty),
        "short_zero_row_indices": [
            info.dataset_index for info in empty
        ],
        "truncated_documents": sum(info.truncated for info in infos),
        "truncated_indices": [
            info.dataset_index for info in infos if info.truncated
        ],
        "total_rows": total_rows,
        "groups_per_document": GROUPS_PER_DOCUMENT,
        "danger_definition": (
            "top ceil(20% * Q-blocks) by original-q D_adj@50% inside each "
            "document x layer x Q-head"
        ),
        "gate_definition": (
            "top ceil(budget * Q-blocks) by 1-selfsim inside the same group"
        ),
    }
    return nonempty, validation


def top_mask(values: np.ndarray, count: int) -> np.ndarray:
    """Stable top-count mask over the last axis."""

    if not 0 < count <= values.shape[-1]:
        raise ValueError((count, values.shape))
    order = np.argsort(values, axis=-1, kind="stable")
    mask = np.zeros(values.shape, dtype=bool)
    np.put_along_axis(
        mask,
        order[..., -count:],
        np.ones((*values.shape[:-1], count), dtype=bool),
        axis=-1,
    )
    return mask


def safe_divide(
    numerator: np.ndarray, denominator: np.ndarray | float
) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    output = np.full(
        np.broadcast_shapes(numerator.shape, denominator.shape),
        np.nan,
        dtype=np.float64,
    )
    np.divide(
        numerator,
        denominator,
        out=output,
        where=np.broadcast_to(denominator, output.shape) > 0,
    )
    return output


def confusion_metrics(
    truth: np.ndarray,
    selected: np.ndarray,
) -> dict[str, np.ndarray]:
    if truth.shape != selected.shape or truth.ndim != 2:
        raise ValueError((truth.shape, selected.shape))
    blocks = truth.shape[1]
    tp = (truth & selected).sum(axis=1).astype(np.float64)
    fp = ((~truth) & selected).sum(axis=1).astype(np.float64)
    fn = (truth & (~selected)).sum(axis=1).astype(np.float64)
    tn = ((~truth) & (~selected)).sum(axis=1).astype(np.float64)
    positives = tp + fn
    predicted = tp + fp

    precision = safe_divide(tp, predicted)
    recall = safe_divide(tp, positives)
    f1_denominator = precision + recall
    f1 = np.zeros_like(f1_denominator)
    np.divide(
        2.0 * precision * recall,
        f1_denominator,
        out=f1,
        where=f1_denominator > 0,
    )
    max_recall = np.minimum(1.0, safe_divide(predicted, positives))
    budget_forced_miss = 1.0 - max_recall
    ranking_miss = max_recall - recall
    return {
        "tp_count": tp,
        "fp_count": fp,
        "fn_count": fn,
        "tn_count": tn,
        "actual_fallback_fraction": predicted / blocks,
        "danger_fraction": positives / blocks,
        "tp_fraction": tp / blocks,
        "fp_fraction": fp / blocks,
        "fn_fraction": fn / blocks,
        "tn_fraction": tn / blocks,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "max_possible_recall": max_recall,
        "budget_forced_miss": budget_forced_miss,
        "ranking_miss": ranking_miss,
    }


def load_document(
    info: SampleInfo,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    columns = [
        "layer",
        "q_head",
        "q_block",
        TARGET_COLUMN,
        *[column for _, column, _ in GATES],
    ]
    table = pq.read_table(info.parquet_path, columns=columns)
    layer = table.column("layer").to_numpy(zero_copy_only=False)
    head = table.column("q_head").to_numpy(zero_copy_only=False)
    q_block = table.column("q_block").to_numpy(zero_copy_only=False)
    order = np.lexsort((q_block, head, layer))
    target = (
        table.column(TARGET_COLUMN)
        .to_numpy(zero_copy_only=False)[order]
        .astype(np.float64, copy=False)
        .reshape(GROUPS_PER_DOCUMENT, info.blocks)
    )
    risks: dict[str, np.ndarray] = {}
    for gate_name, column, _ in GATES:
        selfsim = (
            table.column(column)
            .to_numpy(zero_copy_only=False)[order]
            .astype(np.float64, copy=False)
            .reshape(GROUPS_PER_DOCUMENT, info.blocks)
        )
        risks[gate_name] = 1.0 - selfsim
    if not np.isfinite(target).all():
        raise RuntimeError(f"Non-finite target: {info.parquet_path}")
    if any(not np.isfinite(risk).all() for risk in risks.values()):
        raise RuntimeError(f"Non-finite gate score: {info.parquet_path}")
    return target, risks


def analyze_document(info: SampleInfo) -> list[dict[str, Any]]:
    target, risks = load_document(info)
    positive_count = max(
        1, int(math.ceil(DANGER_FRACTION * info.blocks))
    )
    truth = top_mask(target, positive_count)
    rows: list[dict[str, Any]] = []
    for budget in FALLBACK_BUDGETS:
        selected_count = max(1, int(math.ceil(budget * info.blocks)))
        for gate_name, _, _ in GATES:
            selected = top_mask(risks[gate_name], selected_count)
            metrics = confusion_metrics(truth, selected)
            row: dict[str, Any] = {
                "dataset_index": info.dataset_index,
                "gate": gate_name,
                "nominal_fallback_budget": budget,
                "q_blocks_per_group": info.blocks,
                "layer_head_groups": GROUPS_PER_DOCUMENT,
                "danger_count_per_group": positive_count,
                "selected_count_per_group": selected_count,
            }
            for metric in DOCUMENT_METRICS:
                row[metric] = float(np.mean(metrics[metric]))
            for count_name in (
                "tp_count",
                "fp_count",
                "fn_count",
                "tn_count",
            ):
                row[f"{count_name}_total"] = int(
                    np.sum(metrics[count_name])
                )
            rows.append(row)
    return rows


def bootstrap_summary(
    documents: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_indices = np.sort(documents["dataset_index"].unique())
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(dataset_indices),
        np.full(len(dataset_indices), 1.0 / len(dataset_indices)),
        size=replicates,
    ).astype(np.float64)

    summary_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for budget in FALLBACK_BUDGETS:
        gate_frames: dict[str, pd.DataFrame] = {}
        for gate_name, _, _ in GATES:
            frame = (
                documents[
                    documents["nominal_fallback_budget"].eq(budget)
                    & documents["gate"].eq(gate_name)
                ]
                .set_index("dataset_index")
                .loc[dataset_indices]
            )
            gate_frames[gate_name] = frame
            for metric in DOCUMENT_METRICS:
                values = frame[metric].to_numpy(dtype=np.float64)
                boot = weights @ values / len(dataset_indices)
                summary_rows.append(
                    {
                        "gate": gate_name,
                        "nominal_fallback_budget": budget,
                        "metric": metric,
                        "estimate": float(values.mean()),
                        "bootstrap_ci_lower": float(
                            np.percentile(boot, 2.5)
                        ),
                        "bootstrap_ci_upper": float(
                            np.percentile(boot, 97.5)
                        ),
                        "n_documents": len(dataset_indices),
                        "bootstrap_replicates": replicates,
                    }
                )

        raw = gate_frames["raw_q"]
        qk = gate_frames["q_minus_kbar"]
        for metric in ("precision", "recall", "f1"):
            difference = (
                qk[metric].to_numpy(dtype=np.float64)
                - raw[metric].to_numpy(dtype=np.float64)
            )
            boot = weights @ difference / len(dataset_indices)
            contrast_rows.append(
                {
                    "contrast": "q_minus_kbar_minus_raw_q",
                    "nominal_fallback_budget": budget,
                    "metric": metric,
                    "estimate": float(difference.mean()),
                    "bootstrap_ci_lower": float(
                        np.percentile(boot, 2.5)
                    ),
                    "bootstrap_ci_upper": float(
                        np.percentile(boot, 97.5)
                    ),
                    "n_documents": len(dataset_indices),
                    "bootstrap_replicates": replicates,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


def pooled_confusion(documents: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for budget in FALLBACK_BUDGETS:
        for gate_name, _, _ in GATES:
            frame = documents[
                documents["nominal_fallback_budget"].eq(budget)
                & documents["gate"].eq(gate_name)
            ]
            counts = {
                name: float(frame[f"{name}_count_total"].sum())
                for name in ("tp", "fp", "fn", "tn")
            }
            total = sum(counts.values())
            predicted = counts["tp"] + counts["fp"]
            positives = counts["tp"] + counts["fn"]
            precision = counts["tp"] / predicted
            recall = counts["tp"] / positives
            f1 = 2.0 * precision * recall / (precision + recall)
            rows.append(
                {
                    "gate": gate_name,
                    "nominal_fallback_budget": budget,
                    "total_layer_head_qblock_rows": int(total),
                    "tp_count": int(round(counts["tp"])),
                    "fp_count": int(round(counts["fp"])),
                    "fn_count": int(round(counts["fn"])),
                    "tn_count": int(round(counts["tn"])),
                    "tp_fraction": counts["tp"] / total,
                    "fp_fraction": counts["fp"] / total,
                    "fn_fraction": counts["fn"] / total,
                    "tn_fraction": counts["tn"] / total,
                    "actual_fallback_fraction": predicted / total,
                    "danger_fraction": positives / total,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return pd.DataFrame(rows)


def summary_value(
    summary: pd.DataFrame, gate: str, budget: float, metric: str
) -> float:
    row = summary[
        summary["gate"].eq(gate)
        & summary["nominal_fallback_budget"].eq(budget)
        & summary["metric"].eq(metric)
    ]
    if len(row) != 1:
        raise RuntimeError((gate, budget, metric, len(row)))
    return float(row.iloc[0]["estimate"])


def report_markdown(
    *,
    validation: dict[str, Any],
    summary: pd.DataFrame,
    contrast: pd.DataFrame,
    pooled: pd.DataFrame,
    replicates: int,
    seed: int,
) -> str:
    lines = [
        "# matched fallback 核心指标",
        "",
        "## 实验定义",
        "",
        "- 危险程度只由原始 `q` 计算：每个 Q block 内的 query token "
        "分别选择 K blocks，并用 `D_adj@50%` 衡量选择分歧。",
        "- 在每个 `document × layer × Q-head` 内，将 `D_adj@50%` "
        "最高的 20% Q blocks 定义为高分歧。",
        "- raw gate 用 `1-selfsim(q)` 排 Q blocks；q-kbar gate 用 "
        "`1-selfsim(q-k̄)` 排同一批 Q blocks。",
        "- 两个 gate 在每组 fallback 完全相同数量的 Q blocks。",
        "- `q-k̄` 从不参与 K-block 选择，只用于判断 Q block 是否 "
        "fallback。",
        "",
        "## 四格定义",
        "",
        "- TP：高分歧，并且被 fallback；这是成功救回。",
        "- FP：不是高分歧，但被 fallback；这是浪费的 dense 预算。",
        "- FN：高分歧，但没有 fallback；这是漏掉的危险 block。",
        "- TN：不是高分歧，也没有 fallback。",
        "",
        "## 主结果（文档等权）",
        "",
        "| nominal fallback | gate | precision | recall | F1 | "
        "fallback 中非高分歧比例 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for budget in FALLBACK_BUDGETS:
        for gate_name, _, display in GATES:
            precision = summary_value(
                summary, gate_name, budget, "precision"
            )
            recall = summary_value(summary, gate_name, budget, "recall")
            f1 = summary_value(summary, gate_name, budget, "f1")
            lines.append(
                f"| {budget:.0%} | {display} | {precision:.3f} | "
                f"{recall:.3f} | {f1:.3f} | {1.0-precision:.1%} |"
            )

    lines.extend(
        [
            "",
            "解释：precision=TP/(TP+FP)，回答“fallback 的 blocks 中有多少"
            "确实是高分歧”；recall=TP/(TP+FN)，回答“所有高分歧 blocks "
            "中找回了多少”；F1=2×precision×recall/(precision+recall)。"
            "这些量先逐文档计算再等权平均，所以表中 F1 不必严格等于把"
            "表中宏平均 precision 和 recall 再代入一次公式。",
            "",
            "## q-kbar 相对 raw Q",
            "",
            "| nominal fallback | Δprecision | Δrecall | ΔF1 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for budget in FALLBACK_BUDGETS:
        frame = contrast[
            contrast["nominal_fallback_budget"].eq(budget)
        ].set_index("metric")
        lines.append(
            f"| {budget:.0%} | "
            f"{float(frame.loc['precision','estimate']):+.3f} | "
            f"{float(frame.loc['recall','estimate']):+.3f} | "
            f"{float(frame.loc['f1','estimate']):+.3f} |"
        )

    lines.extend(
        [
            "",
            "## 为什么 recall 看起来低",
            "",
            "总漏检可以严格拆成两部分：",
            "",
            "```text",
            "1 - recall = budget-forced miss + ranking miss",
            "budget-forced miss = 1 - 理论最大 recall",
            "ranking miss = 理论最大 recall - 实际 recall",
            "```",
            "",
            "| nominal fallback | gate | 理论最大 recall | "
            "预算强制漏检 | 排序造成漏检 | 总漏检 |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for budget in FALLBACK_BUDGETS:
        for gate_name, _, display in GATES:
            maximum = summary_value(
                summary, gate_name, budget, "max_possible_recall"
            )
            forced = summary_value(
                summary, gate_name, budget, "budget_forced_miss"
            )
            ranking = summary_value(
                summary, gate_name, budget, "ranking_miss"
            )
            recall = summary_value(summary, gate_name, budget, "recall")
            lines.append(
                f"| {budget:.0%} | {display} | {maximum:.3f} | "
                f"{forced:.3f} | {ranking:.3f} | {1.0-recall:.3f} |"
            )

    pooled_10_raw = pooled[
        pooled["nominal_fallback_budget"].eq(0.10)
        & pooled["gate"].eq("raw_q")
    ].iloc[0]
    pooled_10_qk = pooled[
        pooled["nominal_fallback_budget"].eq(0.10)
        & pooled["gate"].eq("q_minus_kbar")
    ].iloc[0]
    lines.extend(
        [
            "",
            "10% 预算的理论最大 recall 只有约 0.54，因为危险集合约占 "
            "20%，fallback 数量只有它的一半左右；这是不可避免的容量限制。"
            "但 q-kbar 实际 recall 只有约 0.26，剩余差距仍来自排序不够准。",
            "",
            "20% 预算与危险集合大小相同，理论上可以达到 recall=1；q-kbar "
            "实际约为 0.43，说明此时低 recall 完全是单一 self-sim 分数的"
            "排序能力有限，而不是预算不够。",
            "",
            "## 所有 block 合并后的直观 10% 混淆比例",
            "",
            f"- raw Q：TP={pooled_10_raw['tp_fraction']:.2%}，"
            f"FP={pooled_10_raw['fp_fraction']:.2%}，"
            f"FN={pooled_10_raw['fn_fraction']:.2%}，"
            f"TN={pooled_10_raw['tn_fraction']:.2%}。",
            f"- q-kbar：TP={pooled_10_qk['tp_fraction']:.2%}，"
            f"FP={pooled_10_qk['fp_fraction']:.2%}，"
            f"FN={pooled_10_qk['fn_fraction']:.2%}，"
            f"TN={pooled_10_qk['tn_fraction']:.2%}。",
            "",
            "所以“约 6% 不是高分歧”必须解释为占全部 Q blocks 的 "
            f"FP：raw Q 为 {pooled_10_raw['fp_fraction']:.2%}，q-kbar 为 "
            f"{pooled_10_qk['fp_fraction']:.2%}。若分母改成 fallback 集合，"
            f"则分别有 {1.0-float(pooled_10_raw['precision']):.1%} 和 "
            f"{1.0-float(pooled_10_qk['precision']):.1%} 不是高分歧。",
            "",
            "由于每个较短分组都向上取整选择至少一个 block，nominal 10% "
            f"在全体 block 合并后实际为 "
            f"{float(pooled_10_raw['actual_fallback_fraction']):.2%}；"
            "两个 gate 的实际数量严格相同。",
            "",
            "## 统计口径",
            "",
            f"- 完整样本：{validation['sample_files']} 篇；可分析文档："
            f"{validation['nonempty_documents']} 篇；总 rows："
            f"{validation['total_rows']:,}。",
            "- 主表先在每篇文档内平均全部 layer/head，再让每篇文档等权。",
            f"- 95% CI 使用 {replicates:,} 次 whole-document bootstrap，"
            f"seed={seed}；详细区间保存在 `core_summary.csv` 和 "
            "`qkbar_minus_raw.csv`。",
            "- `pooled_confusion.csv` 只用于把 TP/FP/FN/TN 转成直观的"
            "全体 block 比例，不作为文档级推断单位。",
            "",
        ]
    )
    return "\n".join(lines)


def make_metric_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = ("precision", "recall", "f1")
    figure, axes = plt.subplots(
        1, 2, figsize=(10.5, 4.2), sharey=True
    )
    colors = {"raw_q": "#444444", "q_minus_kbar": "#2166ac"}
    width = 0.34
    x = np.arange(len(metrics))
    for axis, budget in zip(axes, FALLBACK_BUDGETS):
        for offset, (gate_name, _, display) in zip(
            (-width / 2, width / 2), GATES
        ):
            values = [
                summary_value(summary, gate_name, budget, metric)
                for metric in metrics
            ]
            axis.bar(
                x + offset,
                values,
                width,
                label=display,
                color=colors[gate_name],
            )
        axis.set_title(f"nominal fallback {budget:.0%}")
        axis.set_xticks(x, ("Precision", "Recall", "F1"))
        axis.set_ylim(0, 0.55)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Document-macro score")
    axes[1].legend(frameon=False)
    figure.suptitle("Matched-budget Q-block fallback")
    atomic_savefig(figure, output_dir / "core_precision_recall_f1.png")


def make_miss_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    labels: list[str] = []
    forced_values: list[float] = []
    ranking_values: list[float] = []
    for budget in FALLBACK_BUDGETS:
        for gate_name, _, _ in GATES:
            labels.append(
                f"{'raw' if gate_name == 'raw_q' else 'q-kbar'}\n"
                f"{budget:.0%}"
            )
            forced_values.append(
                summary_value(
                    summary, gate_name, budget, "budget_forced_miss"
                )
            )
            ranking_values.append(
                summary_value(
                    summary, gate_name, budget, "ranking_miss"
                )
            )
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    axis.bar(
        x,
        forced_values,
        color="#bbbbbb",
        label="Budget-forced miss",
    )
    axis.bar(
        x,
        ranking_values,
        bottom=forced_values,
        color="#444444",
        label="Ranking miss",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Fraction of dangerous Q blocks missed")
    axis.set_ylim(0, 0.85)
    axis.set_title("Why recall is small")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    atomic_savefig(figure, output_dir / "miss_decomposition.png")


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 200:
        raise ValueError("--bootstrap-replicates must be at least 200")
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    infos, validation = discover_samples(run_dir)
    rows: list[dict[str, Any]] = []
    for ordinal, info in enumerate(infos, start=1):
        rows.extend(analyze_document(info))
        if ordinal % args.progress_every == 0 or ordinal == len(infos):
            print(f"analyzed {ordinal}/{len(infos)} documents", flush=True)
    documents = pd.DataFrame(rows)
    summary, contrast = bootstrap_summary(
        documents,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    pooled = pooled_confusion(documents)

    atomic_write_parquet(
        documents, output_dir / "document_core_metrics.parquet"
    )
    atomic_write_csv(summary, output_dir / "core_summary.csv")
    atomic_write_csv(contrast, output_dir / "qkbar_minus_raw.csv")
    atomic_write_csv(pooled, output_dir / "pooled_confusion.csv")
    atomic_write_text(
        output_dir / "validation.json",
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(
        output_dir / "REPORT_ZH.md",
        report_markdown(
            validation=validation,
            summary=summary,
            contrast=contrast,
            pooled=pooled,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        ),
    )
    make_metric_plot(summary, output_dir)
    make_miss_plot(summary, output_dir)
    print(f"Wrote focused analysis to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
