#!/usr/bin/env python3
"""Analyze Q self-similarity as a predictor of within-block key disagreement.

The collector writes one parquet file per GovReport document.  This script
keeps the inferential unit at the document level:

1. Metrics are computed across Q blocks separately for every
   document x layer x query-head group.
2. Layer/head results are averaged inside a document.
3. Dataset estimates and paired bootstrap intervals average documents, and
   the bootstrap resamples whole documents.

Consequently, the millions of Q blocks are never treated as independent
replicates.
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
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_DIR = (
    SCRIPT_DIR.parents[1]
    / "output"
    / "q_sim"
    / "govreport_test_qwen3_8b_32k"
)
EXPECTED_SCHEMA_VERSION = 1

PREDICTORS: tuple[tuple[str, str, str], ...] = (
    ("raw", "q_selfsim_raw", r"$1-s(q)$"),
    (
        "q_minus_kbar",
        "q_selfsim_q_minus_kbar",
        r"$1-s(q-\bar{k})$",
    ),
    (
        "q_minus_qbar",
        "q_selfsim_q_minus_qbar",
        r"$1-s(q-\bar{q})$",
    ),
    (
        "q_minus_wrong_kbar",
        "q_selfsim_q_minus_wrong_kbar",
        r"$1-s(q-\bar{k}_{wrong})$",
    ),
)

TARGETS: tuple[tuple[str, str, str], ...] = (
    (
        "D_adj_r10",
        "selection_disagreement_adjusted_r10",
        r"$D_{adj}@10\%$",
    ),
    (
        "D_adj_r25",
        "selection_disagreement_adjusted_r25",
        r"$D_{adj}@25\%$",
    ),
    (
        "D_adj_r50",
        "selection_disagreement_adjusted_r50",
        r"$D_{adj}@50\%$",
    ),
    (
        "pooled_miss_adj_r10",
        "pooled_selection_miss_adjusted_r10",
        r"Pooled miss$_{adj}@10\%$",
    ),
    (
        "pooled_miss_adj_r25",
        "pooled_selection_miss_adjusted_r25",
        r"Pooled miss$_{adj}@25\%$",
    ),
    (
        "pooled_miss_adj_r50",
        "pooled_selection_miss_adjusted_r50",
        r"Pooled miss$_{adj}@50\%$",
    ),
    ("JSD", "preference_jsd", "JSD"),
)

FALLBACK_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.20)
BASE_METRICS: tuple[str, ...] = (
    "spearman",
    "partial_spearman_position",
    "quartile_delta",
    "auroc_worst20",
    "average_precision_worst20",
)
METRICS: tuple[str, ...] = BASE_METRICS + tuple(
    name
    for fraction in FALLBACK_FRACTIONS
    for name in (
        f"fallback_recall_{int(round(fraction * 100)):02d}",
        f"fallback_lift_{int(round(fraction * 100)):02d}",
    )
)

SAMPLE_PATTERN = re.compile(r"^sample-(\d+)\.parquet$")


@dataclass(frozen=True)
class RunConfig:
    dataset_rows: int
    block_q: int
    block_k: int
    min_common_key_blocks: int
    configured_layers: tuple[int, ...]
    model_layers: int
    query_heads: int
    model_path: str
    dataset_path: str


@dataclass(frozen=True)
class SampleLayout:
    dataset_index: int
    path: Path
    metadata_path: Path
    original_tokens: int
    used_tokens: int
    truncated: bool
    rows: int
    layers: tuple[int, ...]
    heads: tuple[int, ...]
    q_blocks: tuple[int, ...]
    is_short_empty: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Document-clustered analysis of Q self-similarity and key-block "
            "selection disagreement."
        )
    )
    parser.add_argument(
        "--input-dir",
        "--experiment-dir",
        "--run-dir",
        dest="input_dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Collector output containing samples/ and manifests/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where analysis/, plots/, and REPORT.md are written. "
            "Defaults to --input-dir."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Analyze available data despite missing samples or intentionally "
            "restricted layers/heads (for smoke tests). Structural corruption "
            "still fails."
        ),
    )
    parser.add_argument("--expected-samples", type=int, default=973)
    parser.add_argument("--expected-layers", type=int, default=36)
    parser.add_argument("--expected-heads", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every this many observed documents.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Useful only for very small automated checks.",
    )
    return parser.parse_args()


def validate_cli(args: argparse.Namespace) -> None:
    for name in ("expected_samples", "expected_layers", "expected_heads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.bootstrap_replicates < 200:
        raise ValueError("--bootstrap-replicates must be at least 200")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read valid JSON from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return payload


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


def atomic_savefig(figure: plt.Figure, path: Path, *, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.stem}.tmp.{os.getpid()}{path.suffix}"
    )
    figure.savefig(temporary, dpi=dpi, bbox_inches="tight")
    os.replace(temporary, path)
    plt.close(figure)


def discover_run_config(input_dir: Path) -> tuple[RunConfig, list[Path]]:
    manifest_dir = input_dir / "manifests"
    metadata_paths = sorted(manifest_dir.glob("*.run_metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(
            f"No *.run_metadata.json files found under {manifest_dir}"
        )
    payloads = [read_json(path) for path in metadata_paths]
    first = payloads[0]
    try:
        experiment = first["experiment"]
        model = first["model_config"]
        config = RunConfig(
            dataset_rows=int(first["dataset_rows"]),
            block_q=int(experiment["block_q"]),
            block_k=int(experiment["block_k"]),
            min_common_key_blocks=int(
                experiment["min_common_key_blocks"]
            ),
            configured_layers=tuple(int(x) for x in experiment["layers"]),
            model_layers=int(model["num_hidden_layers"]),
            query_heads=int(model["num_attention_heads"]),
            model_path=str(first.get("model_path", "unknown")),
            dataset_path=str(first.get("dataset_path", "unknown")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Required collector configuration is absent from {metadata_paths[0]}"
        ) from error

    invariant_keys = (
        "dataset_rows",
        "model_path",
        "dataset_path",
    )
    for path, payload in zip(metadata_paths[1:], payloads[1:]):
        for key in invariant_keys:
            if payload.get(key) != first.get(key):
                raise RuntimeError(
                    f"Inconsistent {key} between run metadata files: "
                    f"{metadata_paths[0]} and {path}"
                )
        if payload.get("experiment") != first.get("experiment"):
            raise RuntimeError(
                f"Inconsistent experiment configuration in {path}"
            )
        if payload.get("model_config") != first.get("model_config"):
            raise RuntimeError(f"Inconsistent model configuration in {path}")
    return config, metadata_paths


def expected_q_blocks(used_tokens: int, config: RunConfig) -> tuple[int, ...]:
    full_q_blocks = used_tokens // config.block_q
    first_eligible = math.ceil(
        config.min_common_key_blocks * config.block_k / config.block_q
    )
    return tuple(range(first_eligible, full_q_blocks))


def sample_index(path: Path) -> int:
    match = SAMPLE_PATTERN.match(path.name)
    if match is None:
        raise RuntimeError(f"Unexpected sample filename: {path}")
    return int(match.group(1))


def as_numpy(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table[name].combine_chunks())


def preflight_one_sample(
    path: Path,
    *,
    config: RunConfig,
    expected_layers: int,
    expected_heads: int,
    allow_incomplete: bool,
) -> SampleLayout:
    index = sample_index(path)
    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise RuntimeError(f"Missing per-sample metadata: {metadata_path}")
    metadata = read_json(metadata_path)
    if int(metadata.get("schema_version", -1)) != EXPECTED_SCHEMA_VERSION:
        raise RuntimeError(
            f"{metadata_path}: unsupported schema_version "
            f"{metadata.get('schema_version')}"
        )
    if int(metadata.get("dataset_index", -1)) != index:
        raise RuntimeError(
            f"{metadata_path}: dataset_index does not match filename"
        )
    try:
        original_tokens = int(metadata["original_tokens"])
        used_tokens = int(metadata["used_tokens"])
        truncated = bool(metadata["truncated"])
        recorded_rows = int(metadata["rows"])
        recorded_layers = tuple(int(x) for x in metadata["layers_seen"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"{metadata_path}: missing token counts/rows/layers_seen"
        ) from error
    if truncated != (original_tokens > used_tokens):
        raise RuntimeError(
            f"{metadata_path}: inconsistent original/used token counts and "
            "truncated flag"
        )
    parquet_rows = pq.ParquetFile(path).metadata.num_rows
    if recorded_rows != parquet_rows:
        raise RuntimeError(
            f"{path}: metadata rows={recorded_rows}, parquet rows={parquet_rows}"
        )

    expected_blocks = expected_q_blocks(used_tokens, config)
    if parquet_rows == 0:
        if expected_blocks:
            raise RuntimeError(
                f"{path}: empty despite {len(expected_blocks)} eligible Q blocks"
            )
        if recorded_layers:
            raise RuntimeError(
                f"{metadata_path}: empty parquet has nonempty layers_seen"
            )
        return SampleLayout(
            dataset_index=index,
            path=path,
            metadata_path=metadata_path,
            original_tokens=original_tokens,
            used_tokens=used_tokens,
            truncated=truncated,
            rows=0,
            layers=(),
            heads=(),
            q_blocks=(),
            is_short_empty=True,
        )

    identifier_columns = ("dataset_index", "layer", "q_head", "q_block")
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    missing_identifiers = set(identifier_columns) - schema_names
    if missing_identifiers:
        raise RuntimeError(
            f"{path}: missing identifier columns {sorted(missing_identifiers)}"
        )
    identity = pq.read_table(path, columns=list(identifier_columns))
    dataset_indices = as_numpy(identity, "dataset_index")
    if not np.all(dataset_indices == index):
        raise RuntimeError(f"{path}: rows contain another dataset_index")
    layer = as_numpy(identity, "layer").astype(np.int64, copy=False)
    head = as_numpy(identity, "q_head").astype(np.int64, copy=False)
    q_block = as_numpy(identity, "q_block").astype(np.int64, copy=False)
    layers = tuple(int(x) for x in np.unique(layer))
    heads = tuple(int(x) for x in np.unique(head))

    if tuple(recorded_layers) != layers:
        raise RuntimeError(
            f"{metadata_path}: layers_seen={recorded_layers}, parquet={layers}"
        )
    if any(x < 0 or x >= expected_layers for x in layers):
        raise RuntimeError(f"{path}: layer IDs outside expected range: {layers}")
    if any(x < 0 or x >= expected_heads for x in heads):
        raise RuntimeError(f"{path}: q_head IDs outside expected range: {heads}")
    wanted_layers = tuple(range(expected_layers))
    wanted_heads = tuple(range(expected_heads))
    if not allow_incomplete and layers != wanted_layers:
        raise RuntimeError(
            f"{path}: expected {expected_layers} layers, found {layers}"
        )
    if not allow_incomplete and heads != wanted_heads:
        raise RuntimeError(
            f"{path}: expected {expected_heads} q heads, found {heads}"
        )
    if not layers or not heads:
        raise RuntimeError(f"{path}: nonempty file has no layers or heads")

    groups = len(layers) * len(heads)
    if parquet_rows % groups:
        raise RuntimeError(
            f"{path}: {parquet_rows} rows not divisible by {groups} "
            "layer/head groups"
        )
    blocks_per_group = parquet_rows // groups
    order = np.lexsort((q_block, head, layer))
    sorted_layer = layer[order].reshape(groups, blocks_per_group)
    sorted_head = head[order].reshape(groups, blocks_per_group)
    sorted_block = q_block[order].reshape(groups, blocks_per_group)
    if not np.all(sorted_layer == sorted_layer[:, :1]):
        raise RuntimeError(f"{path}: layer/head groups are structurally invalid")
    if not np.all(sorted_head == sorted_head[:, :1]):
        raise RuntimeError(f"{path}: layer/head groups are structurally invalid")
    group_pairs = np.stack(
        (sorted_layer[:, 0], sorted_head[:, 0]), axis=-1
    )
    expected_pairs = np.array(
        [(layer_id, head_id) for layer_id in layers for head_id in heads],
        dtype=np.int64,
    )
    if not np.array_equal(group_pairs, expected_pairs):
        raise RuntimeError(
            f"{path}: missing or duplicated layer x q_head groups"
        )
    if not np.all(sorted_block == sorted_block[:1]):
        raise RuntimeError(
            f"{path}: q_block support differs across layer/head groups"
        )
    actual_blocks = tuple(int(x) for x in sorted_block[0])
    if actual_blocks != expected_blocks:
        raise RuntimeError(
            f"{path}: q_blocks={actual_blocks[:3]}...{actual_blocks[-3:]}, "
            f"expected={expected_blocks[:3]}...{expected_blocks[-3:]}"
        )
    return SampleLayout(
        dataset_index=index,
        path=path,
        metadata_path=metadata_path,
        original_tokens=original_tokens,
        used_tokens=used_tokens,
        truncated=truncated,
        rows=parquet_rows,
        layers=layers,
        heads=heads,
        q_blocks=actual_blocks,
        is_short_empty=False,
    )


def preflight(
    input_dir: Path,
    *,
    config: RunConfig,
    expected_samples: int,
    expected_layers: int,
    expected_heads: int,
    allow_incomplete: bool,
) -> tuple[list[SampleLayout], dict[str, Any]]:
    sample_dir = input_dir / "samples"
    paths = sorted(sample_dir.glob("sample-*.parquet"), key=sample_index)
    if not paths:
        raise FileNotFoundError(f"No sample-*.parquet files found in {sample_dir}")
    indices = [sample_index(path) for path in paths]
    if len(indices) != len(set(indices)):
        raise RuntimeError("Duplicate dataset indices in sample filenames")
    invalid_indices = [
        index for index in indices if index < 0 or index >= expected_samples
    ]
    if invalid_indices:
        raise RuntimeError(
            f"Sample indices outside [0, {expected_samples}): {invalid_indices}"
        )
    expected_index_set = set(range(expected_samples))
    missing_indices = sorted(expected_index_set - set(indices))
    if (len(paths) != expected_samples or missing_indices) and not allow_incomplete:
        raise RuntimeError(
            f"Expected all {expected_samples} sample files, found {len(paths)}; "
            f"missing={missing_indices[:30]}"
        )
    if config.dataset_rows != expected_samples and not allow_incomplete:
        raise RuntimeError(
            f"Run metadata dataset_rows={config.dataset_rows}, "
            f"expected {expected_samples}"
        )
    if config.model_layers != expected_layers and not allow_incomplete:
        raise RuntimeError(
            f"Model has {config.model_layers} layers, expected {expected_layers}"
        )
    if config.query_heads != expected_heads and not allow_incomplete:
        raise RuntimeError(
            f"Model has {config.query_heads} Q heads, expected {expected_heads}"
        )
    if (
        config.configured_layers != tuple(range(expected_layers))
        and not allow_incomplete
    ):
        raise RuntimeError(
            "Collector was not configured for every expected layer: "
            f"{config.configured_layers}"
        )

    print(
        f"Preflight: validating {len(paths)} sample files before analysis...",
        flush=True,
    )
    layouts: list[SampleLayout] = []
    for ordinal, path in enumerate(paths, start=1):
        layout = preflight_one_sample(
            path,
            config=config,
            expected_layers=expected_layers,
            expected_heads=expected_heads,
            allow_incomplete=allow_incomplete,
        )
        layouts.append(layout)
        if ordinal % 100 == 0 or ordinal == len(paths):
            print(f"  preflight {ordinal}/{len(paths)}", flush=True)

    observed = [layout for layout in layouts if layout.rows > 0]
    empty = [layout for layout in layouts if layout.rows == 0]
    restricted = [
        layout
        for layout in observed
        if len(layout.layers) != expected_layers
        or len(layout.heads) != expected_heads
    ]
    validation = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "complete"
            if len(paths) == expected_samples and not restricted
            else "incomplete_allowed"
        ),
        "allow_incomplete": allow_incomplete,
        "expected_sample_files": expected_samples,
        "found_sample_files": len(paths),
        "missing_sample_indices": missing_indices,
        "observed_documents": len(observed),
        "short_documents_without_eligible_qblocks": len(empty),
        "short_document_indices": [
            layout.dataset_index for layout in empty
        ],
        "truncated_documents": sum(layout.truncated for layout in layouts),
        "truncated_document_indices": [
            layout.dataset_index for layout in layouts if layout.truncated
        ],
        "original_tokens_total": int(
            sum(layout.original_tokens for layout in layouts)
        ),
        "used_tokens_total": int(sum(layout.used_tokens for layout in layouts)),
        "restricted_nonempty_documents": len(restricted),
        "expected_layers_per_nonempty_document": expected_layers,
        "expected_q_heads_per_nonempty_document": expected_heads,
        "run_metadata_dataset_rows": config.dataset_rows,
        "run_metadata_model_layers": config.model_layers,
        "run_metadata_query_heads": config.query_heads,
        "total_qblock_rows": int(sum(layout.rows for layout in layouts)),
    }
    return layouts, validation


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(
        np.broadcast_shapes(numerator.shape, denominator.shape),
        np.nan,
        dtype=np.float64,
    )
    np.divide(numerator, denominator, out=output, where=denominator > 0)
    return output


def batch_correlation(
    x_centered: np.ndarray, y_centered: np.ndarray
) -> np.ndarray:
    numerator = np.einsum(
        "gpb,gtb->gpt", x_centered, y_centered, optimize=True
    )
    x_norm = np.sqrt(np.sum(x_centered * x_centered, axis=-1))
    y_norm = np.sqrt(np.sum(y_centered * y_centered, axis=-1))
    return safe_divide(
        numerator, x_norm[:, :, None] * y_norm[:, None, :]
    )


def rank_correlations(
    risks: np.ndarray,
    targets: np.ndarray,
    position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Spearman and position-partial Spearman.

    Shapes are risks [G,P,B], targets [G,T,B], position [B].  scipy is
    called once per tensor, never once per layer/head group.
    """

    risk_rank = rankdata(risks, axis=-1, method="average")
    target_rank = rankdata(targets, axis=-1, method="average")
    risk_centered = risk_rank - risk_rank.mean(axis=-1, keepdims=True)
    target_centered = target_rank - target_rank.mean(axis=-1, keepdims=True)
    spearman = batch_correlation(risk_centered, target_centered)

    position_rank = rankdata(position, method="average").astype(np.float64)
    position_centered = position_rank - position_rank.mean()
    position_ss = float(position_centered @ position_centered)
    if risks.shape[-1] < 3 or position_ss <= 0:
        partial = np.full_like(spearman, np.nan)
    else:
        risk_slope = np.einsum(
            "gpb,b->gp", risk_centered, position_centered, optimize=True
        )
        risk_slope /= position_ss
        target_slope = np.einsum(
            "gtb,b->gt", target_centered, position_centered, optimize=True
        )
        target_slope /= position_ss
        risk_residual = (
            risk_centered
            - risk_slope[:, :, None] * position_centered[None, None, :]
        )
        target_residual = (
            target_centered
            - target_slope[:, :, None] * position_centered[None, None, :]
        )
        partial = batch_correlation(risk_residual, target_residual)
    return spearman, partial, risk_rank


def worst_labels(targets: np.ndarray) -> tuple[np.ndarray, int]:
    groups, target_count, blocks = targets.shape
    positive_count = max(1, int(math.ceil(0.20 * blocks)))
    order = np.argsort(targets, axis=-1, kind="stable")
    labels = np.zeros((groups, target_count, blocks), dtype=np.float64)
    np.put_along_axis(
        labels,
        order[..., -positive_count:],
        np.ones((groups, target_count, positive_count), dtype=np.float64),
        axis=-1,
    )
    return labels, positive_count


def compute_group_metrics(
    risks: np.ndarray,
    targets: np.ndarray,
    position: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return [G,P,T,M] metrics for one document."""

    groups, predictor_count, blocks = risks.shape
    if targets.shape[0] != groups or targets.shape[2] != blocks:
        raise ValueError((risks.shape, targets.shape))
    target_count = targets.shape[1]
    result = np.full(
        (groups, predictor_count, target_count, len(METRICS)),
        np.nan,
        dtype=np.float64,
    )
    metric_index = {name: index for index, name in enumerate(METRICS)}

    spearman, partial, risk_rank = rank_correlations(
        risks, targets, position
    )
    result[..., metric_index["spearman"]] = spearman
    result[..., metric_index["partial_spearman_position"]] = partial

    labels, positive_count = worst_labels(targets)
    negative_count = blocks - positive_count
    quartile_count = max(1, blocks // 4)
    debug: dict[str, Any] = {
        "labels": labels,
        "positive_count": positive_count,
    }
    for predictor in range(predictor_count):
        ascending = np.argsort(
            risks[:, predictor, :], axis=-1, kind="stable"
        )
        descending = ascending[:, ::-1]
        target_by_risk = np.take_along_axis(
            targets,
            ascending[:, None, :],
            axis=-1,
        )
        quartile_delta = (
            target_by_risk[..., -quartile_count:].mean(axis=-1)
            - target_by_risk[..., :quartile_count].mean(axis=-1)
        )
        result[:, predictor, :, metric_index["quartile_delta"]] = (
            quartile_delta
        )

        if negative_count > 0:
            positive_rank_sum = np.einsum(
                "gb,gtb->gt",
                risk_rank[:, predictor, :],
                labels,
                optimize=True,
            )
            auroc = (
                positive_rank_sum
                - positive_count * (positive_count + 1) / 2.0
            ) / (positive_count * negative_count)
            result[:, predictor, :, metric_index["auroc_worst20"]] = auroc

        sorted_labels = np.take_along_axis(
            labels, descending[:, None, :], axis=-1
        )
        cumulative_positive = np.cumsum(sorted_labels, axis=-1)
        precision_at_rank = cumulative_positive / np.arange(
            1, blocks + 1, dtype=np.float64
        )[None, None, :]
        average_precision = (
            precision_at_rank * sorted_labels
        ).sum(axis=-1) / positive_count
        result[
            :, predictor, :, metric_index["average_precision_worst20"]
        ] = average_precision

        for fraction in FALLBACK_FRACTIONS:
            label = int(round(fraction * 100))
            selected_count = max(1, int(math.ceil(fraction * blocks)))
            selected_labels = sorted_labels[..., :selected_count]
            recall = selected_labels.sum(axis=-1) / positive_count
            actual_fraction = selected_count / blocks
            lift = recall / actual_fraction
            result[
                :, predictor, :, metric_index[f"fallback_recall_{label:02d}"]
            ] = recall
            result[
                :, predictor, :, metric_index[f"fallback_lift_{label:02d}"]
            ] = lift
    return result, debug


def sklearn_crosscheck(
    risks: np.ndarray,
    debug: dict[str, Any],
    group_metrics: np.ndarray,
) -> None:
    """Cross-check one group against sklearn, once per run."""

    labels = debug["labels"]
    metric_index = {name: index for index, name in enumerate(METRICS)}
    for predictor in range(risks.shape[1]):
        for target in range(labels.shape[1]):
            truth = labels[0, target].astype(np.int8)
            score = risks[0, predictor]
            if np.unique(truth).size < 2:
                continue
            expected_auc = roc_auc_score(truth, score)
            expected_ap = average_precision_score(truth, score)
            actual_auc = group_metrics[
                0, predictor, target, metric_index["auroc_worst20"]
            ]
            actual_ap = group_metrics[
                0,
                predictor,
                target,
                metric_index["average_precision_worst20"],
            ]
            if not np.isclose(expected_auc, actual_auc, atol=1e-10):
                raise AssertionError(
                    f"Vectorized AUROC {actual_auc} != sklearn {expected_auc}"
                )
            if not np.isclose(expected_ap, actual_ap, atol=1e-10):
                raise AssertionError(
                    f"Vectorized AP {actual_ap} != sklearn {expected_ap}"
                )
    print("Vectorized AUROC/AP cross-check against sklearn passed.", flush=True)


def load_sample_arrays(
    layout: SampleLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = [
        "layer",
        "q_head",
        "q_block",
        "position_fraction",
        *[column for _, column, _ in PREDICTORS],
        *[column for _, column, _ in TARGETS],
    ]
    table = pq.read_table(layout.path, columns=required)
    layer = as_numpy(table, "layer").astype(np.int64, copy=False)
    head = as_numpy(table, "q_head").astype(np.int64, copy=False)
    q_block = as_numpy(table, "q_block").astype(np.int64, copy=False)
    order = np.lexsort((q_block, head, layer))
    groups = len(layout.layers) * len(layout.heads)
    blocks = len(layout.q_blocks)
    group_layer = layer[order].reshape(groups, blocks)[:, 0]
    group_head = head[order].reshape(groups, blocks)[:, 0]
    position_matrix = (
        as_numpy(table, "position_fraction")[order]
        .astype(np.float64, copy=False)
        .reshape(groups, blocks)
    )
    if not np.allclose(position_matrix, position_matrix[:1], atol=0, rtol=0):
        raise RuntimeError(
            f"{layout.path}: position_fraction differs across groups"
        )

    risk_arrays = []
    for _, column, _ in PREDICTORS:
        values = (
            as_numpy(table, column)[order]
            .astype(np.float64, copy=False)
            .reshape(groups, blocks)
        )
        risk_arrays.append(1.0 - values)
    target_arrays = []
    for _, column, _ in TARGETS:
        values = (
            as_numpy(table, column)[order]
            .astype(np.float64, copy=False)
            .reshape(groups, blocks)
        )
        target_arrays.append(values)
    risks = np.stack(risk_arrays, axis=1)
    targets = np.stack(target_arrays, axis=1)
    if not np.isfinite(risks).all():
        raise RuntimeError(f"{layout.path}: non-finite predictor values")
    if not np.isfinite(targets).all():
        raise RuntimeError(f"{layout.path}: non-finite target values")
    return (
        risks,
        targets,
        position_matrix[0],
        group_layer,
        group_head,
    )


def mean_and_count(values: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0.0).sum(axis=axis)
    mean = safe_divide(total, count)
    return mean, count


def document_frame(
    dataset_indices: list[int],
    document_values: np.ndarray,
    document_counts: np.ndarray,
    document_groups: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for document, dataset_index in enumerate(dataset_indices):
        for predictor, (predictor_name, _, _) in enumerate(PREDICTORS):
            for target, (target_name, _, _) in enumerate(TARGETS):
                row: dict[str, Any] = {
                    "dataset_index": dataset_index,
                    "predictor": predictor_name,
                    "target": target_name,
                    "layer_head_groups": document_groups[document],
                }
                for metric, metric_name in enumerate(METRICS):
                    row[metric_name] = document_values[
                        document, predictor, target, metric
                    ]
                    row[f"n_groups_{metric_name}"] = int(
                        document_counts[
                            document, predictor, target, metric
                        ]
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def document_distribution_rows(
    *,
    dataset_index: int,
    risks: np.ndarray,
    targets: np.ndarray,
) -> list[dict[str, Any]]:
    """Descriptive absolute scales, summarized inside one document first."""

    variables: list[tuple[str, str, np.ndarray]] = [
        ("selfsim_raw", "self_similarity", 1.0 - risks[:, 0, :]),
        (
            "selfsim_q_minus_kbar",
            "self_similarity",
            1.0 - risks[:, 1, :],
        ),
    ]
    variables.extend(
        (name, "target", targets[:, index, :])
        for index, (name, _, _) in enumerate(TARGETS)
    )
    rows: list[dict[str, Any]] = []
    quantile_levels = (0.10, 0.25, 0.50, 0.75, 0.90)
    for variable, kind, values in variables:
        flattened = values.reshape(-1)
        quantiles = np.quantile(flattened, quantile_levels)
        rows.append(
            {
                "dataset_index": dataset_index,
                "variable": variable,
                "kind": kind,
                "mean": float(flattened.mean()),
                "p10": float(quantiles[0]),
                "p25": float(quantiles[1]),
                "p50": float(quantiles[2]),
                "p75": float(quantiles[3]),
                "p90": float(quantiles[4]),
                "n_qblock_rows": int(flattened.size),
            }
        )
    return rows


def macro_distribution_frame(documents: pd.DataFrame) -> pd.DataFrame:
    """Average each document's descriptive statistic, weighting documents equally."""

    statistic_columns = ("mean", "p10", "p25", "p50", "p75", "p90")
    macro = (
        documents.groupby(["variable", "kind"], as_index=False)[
            list(statistic_columns)
        ]
        .mean()
        .rename(
            columns={
                name: f"document_macro_{name}" for name in statistic_columns
            }
        )
    )
    counts = (
        documents.groupby(["variable", "kind"], as_index=False)
        .size()
        .rename(columns={"size": "n_documents"})
    )
    return macro.merge(counts, on=["variable", "kind"], validate="one_to_one")


def layer_head_frame(
    sums: np.ndarray, counts: np.ndarray
) -> pd.DataFrame:
    means = safe_divide(sums, counts)
    rows: list[dict[str, Any]] = []
    active = np.argwhere(np.any(counts > 0, axis=(2, 3, 4)))
    for layer, head in active:
        for predictor, (predictor_name, _, _) in enumerate(PREDICTORS):
            for target, (target_name, _, _) in enumerate(TARGETS):
                row: dict[str, Any] = {
                    "layer": int(layer),
                    "q_head": int(head),
                    "predictor": predictor_name,
                    "target": target_name,
                }
                for metric, metric_name in enumerate(METRICS):
                    row[metric_name] = means[
                        layer, head, predictor, target, metric
                    ]
                    row[f"n_documents_{metric_name}"] = int(
                        counts[layer, head, predictor, target, metric]
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_weights(
    documents: int, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    probability = np.full(documents, 1.0 / documents)
    return rng.multinomial(documents, probability, size=replicates).astype(
        np.float64, copy=False
    )


def weighted_bootstrap(
    values: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Bootstrap column means of [documents, cells], respecting missingness."""

    valid = np.isfinite(values)
    numerator = weights @ np.where(valid, values, 0.0)
    denominator = weights @ valid.astype(np.float64)
    return safe_divide(numerator, denominator)


def macro_and_bootstrap_frames(
    document_values: np.ndarray,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    documents = document_values.shape[0]
    rng = np.random.default_rng(seed)
    weights = bootstrap_weights(documents, bootstrap_replicates, rng)
    flat = document_values.reshape(documents, -1)
    bootstrap = weighted_bootstrap(flat, weights)
    lower = np.nanpercentile(bootstrap, 2.5, axis=0).reshape(
        document_values.shape[1:]
    )
    upper = np.nanpercentile(bootstrap, 97.5, axis=0).reshape(
        document_values.shape[1:]
    )
    estimate, n_documents = mean_and_count(document_values, axis=0)
    centered = np.where(
        np.isfinite(document_values),
        document_values - estimate[None, ...],
        0.0,
    )
    squared_deviation = np.sum(centered * centered, axis=0)
    std = np.sqrt(
        safe_divide(
            squared_deviation,
            np.maximum(n_documents - 1, 0),
        )
    )

    macro_rows: list[dict[str, Any]] = []
    for predictor, (predictor_name, _, _) in enumerate(PREDICTORS):
        for target, (target_name, _, _) in enumerate(TARGETS):
            for metric, metric_name in enumerate(METRICS):
                macro_rows.append(
                    {
                        "predictor": predictor_name,
                        "target": target_name,
                        "metric": metric_name,
                        "estimate": estimate[predictor, target, metric],
                        "document_sd": std[predictor, target, metric],
                        "n_documents": int(
                            n_documents[predictor, target, metric]
                        ),
                        "bootstrap_ci_lower": lower[
                            predictor, target, metric
                        ],
                        "bootstrap_ci_upper": upper[
                            predictor, target, metric
                        ],
                        "bootstrap_replicates": bootstrap_replicates,
                    }
                )
    macro_frame = pd.DataFrame(macro_rows)

    raw_index = next(
        index for index, item in enumerate(PREDICTORS) if item[0] == "raw"
    )
    qkbar_index = next(
        index
        for index, item in enumerate(PREDICTORS)
        if item[0] == "q_minus_kbar"
    )
    differences = (
        document_values[:, qkbar_index] - document_values[:, raw_index]
    )
    difference_flat = differences.reshape(documents, -1)
    difference_bootstrap = weighted_bootstrap(difference_flat, weights).reshape(
        bootstrap_replicates, len(TARGETS), len(METRICS)
    )
    difference_estimate, difference_n = mean_and_count(differences, axis=0)
    difference_lower = np.nanpercentile(
        difference_bootstrap, 2.5, axis=0
    )
    difference_upper = np.nanpercentile(
        difference_bootstrap, 97.5, axis=0
    )
    probability_positive = np.nanmean(difference_bootstrap > 0, axis=0)
    comparison_rows: list[dict[str, Any]] = []
    for target, (target_name, _, _) in enumerate(TARGETS):
        for metric, metric_name in enumerate(METRICS):
            low = difference_lower[target, metric]
            high = difference_upper[target, metric]
            if low > 0:
                classification = "q_minus_kbar_higher"
            elif high < 0:
                classification = "raw_higher"
            else:
                classification = "interval_overlaps_zero"
            probability = probability_positive[target, metric]
            comparison_rows.append(
                {
                    "contrast": "q_minus_kbar_minus_raw",
                    "target": target_name,
                    "metric": metric_name,
                    "estimate": difference_estimate[target, metric],
                    "bootstrap_ci_lower": low,
                    "bootstrap_ci_upper": high,
                    "n_paired_documents": int(
                        difference_n[target, metric]
                    ),
                    "bootstrap_probability_difference_gt_zero": probability,
                    "bootstrap_two_sided_p": min(
                        1.0, 2.0 * min(probability, 1.0 - probability)
                    ),
                    "classification": classification,
                    "bootstrap_replicates": bootstrap_replicates,
                }
            )
    comparison_frame = pd.DataFrame(comparison_rows)
    return macro_frame, comparison_frame


def predictor_display(name: str) -> str:
    return next(display for key, _, display in PREDICTORS if key == name)


def target_display(name: str) -> str:
    return next(display for key, _, display in TARGETS if key == name)


def plot_spearman_heatmap(
    macro: pd.DataFrame, plots_dir: Path
) -> None:
    subset = macro[macro["metric"] == "partial_spearman_position"]
    pivot = subset.pivot(index="predictor", columns="target", values="estimate")
    pivot = pivot.reindex(
        index=[item[0] for item in PREDICTORS],
        columns=[item[0] for item in TARGETS],
    )
    values = pivot.to_numpy(dtype=float)
    finite_max = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1
    limit = max(0.05, float(finite_max))
    figure, axis = plt.subplots(figsize=(12, 4.5))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_xticks(
        np.arange(len(TARGETS)),
        [item[2] for item in TARGETS],
        rotation=28,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(PREDICTORS)), [item[2] for item in PREDICTORS]
    )
    axis.set_title(
        "Document-macro partial Spearman (controlling Q-block position)"
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                color = "white" if abs(value) > 0.55 * limit else "black"
                axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
    figure.colorbar(image, ax=axis, label="partial Spearman")
    atomic_savefig(figure, plots_dir / "partial_spearman_heatmap.png")


def plot_bootstrap_contrast(
    comparison: pd.DataFrame, plots_dir: Path
) -> None:
    metrics = (
        "spearman",
        "partial_spearman_position",
        "auroc_worst20",
        "average_precision_worst20",
    )
    titles = ("Spearman", "Partial Spearman", "AUROC", "Average precision")
    figure, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
    target_names = [item[0] for item in TARGETS]
    y = np.arange(len(target_names))
    for axis, metric, title in zip(axes, metrics, titles):
        subset = (
            comparison[comparison["metric"] == metric]
            .set_index("target")
            .reindex(target_names)
        )
        estimate = subset["estimate"].to_numpy(dtype=float)
        low = subset["bootstrap_ci_lower"].to_numpy(dtype=float)
        high = subset["bootstrap_ci_upper"].to_numpy(dtype=float)
        axis.errorbar(
            estimate,
            y,
            xerr=np.stack((estimate - low, high - estimate)),
            fmt="o",
            color="#1764ab",
            ecolor="#6d9dc5",
            capsize=3,
        )
        axis.axvline(0, color="black", linewidth=1, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel(r"$q-\bar{k}$ minus raw")
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(
        y, [target_display(name) for name in target_names]
    )
    figure.suptitle(
        "Paired document-cluster bootstrap: predictor contrast (95% CI)"
    )
    atomic_savefig(figure, plots_dir / "qkbar_minus_raw_bootstrap.png")


def plot_fallback_recall(macro: pd.DataFrame, plots_dir: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    fractions_percent = np.array(
        [int(round(value * 100)) for value in FALLBACK_FRACTIONS]
    )
    colors = ("#333333", "#1764ab", "#e07a1f", "#4c956c")
    for target_index, (target_name, _, display) in enumerate(TARGETS):
        axis = axes_flat[target_index]
        for color, (predictor_name, _, predictor_label) in zip(
            colors, PREDICTORS
        ):
            values = []
            for percent in fractions_percent:
                row = macro[
                    (macro["predictor"] == predictor_name)
                    & (macro["target"] == target_name)
                    & (
                        macro["metric"]
                        == f"fallback_recall_{percent:02d}"
                    )
                ]
                values.append(float(row["estimate"].iloc[0]))
            axis.plot(
                fractions_percent,
                values,
                marker="o",
                color=color,
                label=predictor_label,
            )
        axis.plot(
            fractions_percent,
            fractions_percent / 100.0,
            color="#999999",
            linestyle=":",
            label="random",
        )
        axis.set_title(display)
        axis.grid(alpha=0.2)
    axes_flat[-1].axis("off")
    for axis in axes[-1, :3]:
        axis.set_xlabel("Matched fallback budget (%)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Recall of worst 20% Q blocks")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.08),
        frameon=False,
    )
    figure.suptitle("Document-macro matched-budget fallback recall")
    atomic_savefig(figure, plots_dir / "fallback_recall.png")


def plot_layer_profile(
    layer_head: pd.DataFrame, plots_dir: Path
) -> None:
    grouped = (
        layer_head.groupby(["layer", "predictor"], as_index=False)[
            "partial_spearman_position"
        ]
        .mean()
        .sort_values("layer")
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    colors = ("#333333", "#1764ab", "#e07a1f", "#4c956c")
    for color, (predictor_name, _, display) in zip(colors, PREDICTORS):
        subset = grouped[grouped["predictor"] == predictor_name]
        if subset.empty:
            continue
        axis.plot(
            subset["layer"],
            subset["partial_spearman_position"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=color,
            label=display,
        )
    axis.axhline(0, color="#999999", linewidth=1)
    axis.set_xlabel("Layer")
    axis.set_ylabel("Partial Spearman, averaged over heads and targets")
    axis.set_title("Position-controlled association by layer")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    atomic_savefig(figure, plots_dir / "partial_spearman_by_layer.png")


def format_interval(estimate: float, low: float, high: float) -> str:
    return f"{estimate:.3f} [{low:.3f}, {high:.3f}]"


def report_markdown(
    *,
    input_dir: Path,
    validation: dict[str, Any],
    config: RunConfig,
    macro: pd.DataFrame,
    comparison: pd.DataFrame,
    distribution_macro: pd.DataFrame,
    bootstrap_replicates: int,
    seed: int,
) -> str:
    raw_partial = macro[
        (macro["predictor"] == "raw")
        & (macro["metric"] == "partial_spearman_position")
    ].set_index("target")
    raw_quartile = macro[
        (macro["predictor"] == "raw")
        & (macro["metric"] == "quartile_delta")
    ].set_index("target")
    contrast_partial = comparison[
        comparison["metric"] == "partial_spearman_position"
    ].set_index("target")

    if (
        validation["status"] != "complete"
        or validation["observed_documents"] < 2
    ):
        low_sim_conclusion = (
            "This is an incomplete/smoke run, so its estimates are descriptive "
            "only and are not used for dataset-level inferential conclusions."
        )
        qkbar_conclusion = (
            "Run the analyzer on the complete 973-file collection before "
            "deciding whether q-kbar improves on raw Q."
        )
        operational_conclusion = (
            "Matched-budget gate conclusions require the complete collection."
        )
    else:
        raw_positive = int(
            (
                raw_partial["bootstrap_ci_lower"].to_numpy(dtype=float) > 0
            ).sum()
        )
        raw_negative = int(
            (
                raw_partial["bootstrap_ci_upper"].to_numpy(dtype=float) < 0
            ).sum()
        )
        qkbar_better = int(
            (
                contrast_partial["bootstrap_ci_lower"].to_numpy(dtype=float) > 0
            ).sum()
        )
        raw_better = int(
            (
                contrast_partial["bootstrap_ci_upper"].to_numpy(dtype=float) < 0
            ).sum()
        )
        overlap = len(TARGETS) - qkbar_better - raw_better

        if raw_positive == len(TARGETS):
            low_sim_conclusion = (
                "Across all seven targets, lower raw-Q self-similarity is "
                "positively associated with worse within-block key behavior even "
                "after controlling position (all document-bootstrap CIs exclude 0)."
            )
        elif raw_positive:
            low_sim_conclusion = (
                f"Lower raw-Q self-similarity has a reliably positive "
                f"position-controlled association for {raw_positive}/{len(TARGETS)} "
                "targets; the remaining targets are mixed or uncertain."
            )
        elif raw_negative:
            low_sim_conclusion = (
                f"The expected direction is not supported uniformly: "
                f"{raw_negative}/{len(TARGETS)} targets have reliably negative "
                "position-controlled associations."
            )
        else:
            low_sim_conclusion = (
                "The document-bootstrap intervals do not establish a reliable "
                "position-controlled association between lower raw-Q "
                "self-similarity and the tested disagreement targets."
            )

        if qkbar_better and not raw_better:
            qkbar_conclusion = (
                f"For partial Spearman, q-kbar is reliably higher than raw on "
                f"{qkbar_better}/{len(TARGETS)} targets; {overlap} contrasts remain "
                "uncertain."
            )
        elif raw_better and not qkbar_better:
            qkbar_conclusion = (
                f"For partial Spearman, raw is reliably higher than q-kbar on "
                f"{raw_better}/{len(TARGETS)} targets; {overlap} contrasts remain "
                "uncertain."
            )
        elif qkbar_better or raw_better:
            qkbar_conclusion = (
                "For position-controlled partial Spearman, the q-kbar "
                "comparison is target-dependent: "
                f"{qkbar_better} favor q-kbar, {raw_better} favor raw, and "
                f"{overlap} overlap zero."
            )
        else:
            qkbar_conclusion = (
                "None of the seven partial-Spearman q-kbar-minus-raw intervals "
                "excludes zero; these data do not establish that q-kbar is better."
            )

        discrete_targets = {item[0] for item in TARGETS if item[0] != "JSD"}
        recall_metrics = {"fallback_recall_10", "fallback_recall_20"}
        recall_contrasts = comparison[
            comparison["target"].isin(discrete_targets)
            & comparison["metric"].isin(recall_metrics)
        ]
        d50_contrast = comparison[
            comparison["target"].eq("D_adj_r50")
        ].set_index("metric")
        if (
            len(recall_contrasts) == len(discrete_targets) * len(recall_metrics)
            and (
                recall_contrasts["bootstrap_ci_lower"].to_numpy(dtype=float)
                > 0
            ).all()
        ):
            operational_conclusion = (
                "For the operational gate question, q-kbar is better on every "
                "discrete top-set disagreement and pooled-mask-miss target: at "
                "the same 10%/20% dense-fallback budget, all 12 recall contrasts "
                "are positive and their paired whole-document bootstrap "
                "intervals exclude 0. For `D_adj@50%`, q-kbar improves AUROC by "
                f"{float(d50_contrast.loc['auroc_worst20', 'estimate']):.3f}, "
                "AP by "
                f"{float(d50_contrast.loc['average_precision_worst20', 'estimate']):.3f}, "
                "recall@10 by "
                f"{float(d50_contrast.loc['fallback_recall_10', 'estimate']):.3f}, "
                "and recall@20 by "
                f"{float(d50_contrast.loc['fallback_recall_20', 'estimate']):.3f}."
            )
        else:
            operational_conclusion = (
                "The matched-budget recall contrasts are mixed; inspect the "
                "paired comparison table before selecting a gate."
            )

        jsd_operational = comparison[
            comparison["target"].eq("JSD")
            & comparison["metric"].isin(
                (
                    "auroc_worst20",
                    "average_precision_worst20",
                    "fallback_recall_10",
                    "fallback_recall_20",
                )
            )
        ]
        if (
            len(jsd_operational) == 4
            and (
                jsd_operational["bootstrap_ci_upper"].to_numpy(dtype=float) < 0
            ).all()
        ):
            operational_conclusion += (
                " This is not a universal replacement result: raw Q ranks "
                "graded JSD risk better in the operational AUROC/AP/recall view."
            )

    lines = [
        "# Q self-similarity and within-block key-selection disagreement",
        "",
        "## Main result",
        "",
        low_sim_conclusion,
        "",
        operational_conclusion,
        "",
        qkbar_conclusion,
        "",
        "These statements are generated from the estimates below; no expected "
        "winner is hard-coded.",
        "",
        "## Coverage and validity",
        "",
        f"- Input: `{input_dir}`",
        f"- Collector sample files: {validation['found_sample_files']} / "
        f"{validation['expected_sample_files']} expected.",
        f"- Documents with eligible Q blocks: "
        f"{validation['observed_documents']}.",
        f"- Pre-registered short documents with zero eligible Q blocks: "
        f"{validation['short_documents_without_eligible_qblocks']} "
        f"(indices: {validation['short_document_indices']}).",
        f"- Nonempty documents with restricted layer/head coverage: "
        f"{validation['restricted_nonempty_documents']}.",
        f"- Total collected Q-block rows: "
        f"{validation['total_qblock_rows']:,}.",
        f"- Documents truncated at the configured context cap: "
        f"{validation['truncated_documents']} "
        f"(indices: {validation['truncated_document_indices']}).",
        f"- Tokens processed: {validation['used_tokens_total']:,} / "
        f"{validation['original_tokens_total']:,} before truncation.",
        f"- Model: `{config.model_path}`; metadata says "
        f"{config.model_layers} layers and {config.query_heads} Q heads.",
        f"- Validation status: **{validation['status']}**.",
        "",
        "Zero-row samples are valid only when their token length implies no "
        "full Q block with the configured minimum common-key prefix. They "
        "count toward file completeness but not toward association estimates.",
        "",
        f"Every analyzed Q block has at least "
        f"`M >= {config.min_common_key_blocks}` fully visible key blocks. "
        "The candidate set is the common causal prefix visible to every token "
        "in that Q block, so causal-boundary availability is held fixed within "
        "the group. Partial Spearman additionally controls ranked Q-block "
        "position.",
        "",
        "## Core estimates",
        "",
        "Each cell is the document-macro estimate with a 95% whole-document "
        "bootstrap interval.",
        "",
        "| Target | raw partial Spearman | raw high-risk minus low-risk target "
        "quartile | q-kbar minus raw partial Spearman |",
        "|---|---:|---:|---:|",
    ]
    for target_name, _, display in TARGETS:
        partial_row = raw_partial.loc[target_name]
        quartile_row = raw_quartile.loc[target_name]
        contrast_row = contrast_partial.loc[target_name]
        lines.append(
            f"| {display} | "
            f"{format_interval(float(partial_row['estimate']), float(partial_row['bootstrap_ci_lower']), float(partial_row['bootstrap_ci_upper']))} | "
            f"{format_interval(float(quartile_row['estimate']), float(quartile_row['bootstrap_ci_lower']), float(quartile_row['bootstrap_ci_upper']))} | "
            f"{format_interval(float(contrast_row['estimate']), float(contrast_row['bootstrap_ci_lower']), float(contrast_row['bootstrap_ci_upper']))} |"
        )

    lines.extend(
        [
            "",
            "## Primary operational view: adjusted disagreement at 50%",
            "",
            "Document-macro estimate [95% whole-document bootstrap CI]. Higher "
            "AUROC/AP/recall/lift means the risk score finds more of the worst "
            "`D_adj_r50` Q blocks at the same fallback budget.",
            "",
            "| Predictor | partial rho | AUROC worst20 | AP worst20 | "
            "recall@10 | lift@10 | recall@20 | lift@20 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    operational_metrics = (
        "partial_spearman_position",
        "auroc_worst20",
        "average_precision_worst20",
        "fallback_recall_10",
        "fallback_lift_10",
        "fallback_recall_20",
        "fallback_lift_20",
    )
    for predictor_name in ("raw", "q_minus_kbar"):
        cells = []
        for metric_name in operational_metrics:
            row = macro[
                (macro["predictor"] == predictor_name)
                & (macro["target"] == "D_adj_r50")
                & (macro["metric"] == metric_name)
            ].iloc[0]
            cells.append(
                format_interval(
                    float(row["estimate"]),
                    float(row["bootstrap_ci_lower"]),
                    float(row["bootstrap_ci_upper"]),
                )
            )
        lines.append(
            f"| {predictor_display(predictor_name)} | "
            + " | ".join(cells)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Absolute scale of self-similarity and targets",
            "",
            "These are descriptive document-macro summaries: each mean or "
            "quantile is computed within a document first, then averaged across "
            "documents. They show absolute scale without treating Q blocks as "
            "independent observations.",
            "",
            "| Variable | document-macro mean | p10 | p50 | p90 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    distribution_order = (
        "selfsim_raw",
        "selfsim_q_minus_kbar",
        *[item[0] for item in TARGETS],
    )
    distribution_labels = {
        "selfsim_raw": "raw Q self-sim",
        "selfsim_q_minus_kbar": "q-kbar self-sim",
        **{item[0]: item[2] for item in TARGETS},
    }
    distribution_by_name = distribution_macro.set_index("variable")
    for variable in distribution_order:
        row = distribution_by_name.loc[variable]
        lines.append(
            f"| {distribution_labels[variable]} | "
            f"{float(row['document_macro_mean']):.4f} | "
            f"{float(row['document_macro_p10']):.4f} | "
            f"{float(row['document_macro_p50']):.4f} | "
            f"{float(row['document_macro_p90']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Metrics and estimand",
            "",
            "- Predictor risk is `1 - self_similarity`; larger values mean "
            "lower within-Q-block self-similarity.",
            "- Spearman is computed across Q blocks inside each "
            "document × layer × Q-head group.",
            "- Partial Spearman is the correlation of predictor and target "
            "ranks after linearly residualizing ranked Q-block position.",
            "- Quartile delta is mean target in the highest-risk predictor "
            "quartile minus mean target in the lowest-risk quartile.",
            "- AUROC and average precision classify the worst 20% target "
            "Q blocks within the same group.",
            "- Matched fallback selects exactly the highest-risk 5%, 10%, or "
            "20% Q blocks (ceil-rounded) for every predictor. Recall is the "
            "fraction of worst-20% target blocks caught; lift is recall divided "
            "by the realized selected fraction.",
            "- Group metrics are averaged over layer/head groups inside each "
            "document. Dataset estimates then average documents equally.",
            f"- Confidence intervals use {bootstrap_replicates:,} paired "
            f"whole-document bootstrap replicates (seed {seed}). No Q block "
            "is treated as an independent replicate.",
            "",
            "The selector targets were always computed from original Q. "
            "`q-kbar`, `q-qbar`, and wrong-`kbar` alter only the gating "
            "predictor, preserving a clean comparison.",
            "",
            "## Scope and limitations",
            "",
            "- The target is token-wise preference over SpargeAttn-style "
            "K-block centroids. It is not a dense token-attention-mass oracle, "
            "an end-to-end output-error measurement, or a speed benchmark.",
            "- `kbar` is the full retained prefill-sequence mean used by the "
            "current SpargeAttn smoothing path. Autoregressive decoding with "
            "a causal running mean is a separate setting.",
            "- Results cover the complete local GovReport test split under a "
            "32K cap. RULER task transfer and contexts beyond the checkpoint's "
            "native range are not inferred from this run.",
            "- Associations and matched-budget retrieval metrics establish "
            "gate usefulness, not a universal fixed numeric threshold. Any "
            "deployment threshold should be calibrated on separate documents.",
            "- The q-kbar operational gain is a document-macro average, not a "
            "claim that every layer/head improves; per-layer/head deployment "
            "choices need separate calibration.",
            "",
            "## Output files",
            "",
            "- `analysis/validation.json`: completeness and structural checks.",
            "- `analysis/document_metrics.parquet`: one row per observed "
            "document × predictor × target after layer/head averaging.",
            "- `analysis/document_distribution.parquet` and "
            "`distribution_macro.csv`: per-document and document-macro "
            "absolute means/quantiles.",
            "- `analysis/macro_summary.csv`: all estimates and document "
            "bootstrap intervals.",
            "- `analysis/qkbar_minus_raw_bootstrap.csv`: paired q-kbar-minus-raw "
            "contrasts.",
            "- `analysis/layer_head_summary.parquet`: document-macro "
            "layer/head diagnostics.",
            "- `plots/`: association, contrast, fallback, and layer figures.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    validate_cli(args)
    input_dir = args.input_dir.resolve()
    output_dir = (
        input_dir if args.output_dir is None else args.output_dir.resolve()
    )
    analysis_dir = output_dir / "analysis"
    plots_dir = output_dir / "plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    config, run_metadata_paths = discover_run_config(input_dir)
    layouts, validation = preflight(
        input_dir,
        config=config,
        expected_samples=args.expected_samples,
        expected_layers=args.expected_layers,
        expected_heads=args.expected_heads,
        allow_incomplete=args.allow_incomplete,
    )
    validation["run_metadata_files"] = [
        str(path) for path in run_metadata_paths
    ]
    atomic_write_text(
        analysis_dir / "validation.json",
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
    )
    print(
        "Preflight passed: "
        f"{validation['found_sample_files']} files, "
        f"{validation['observed_documents']} observed, "
        f"{validation['short_documents_without_eligible_qblocks']} short/empty.",
        flush=True,
    )

    observed_layouts = [layout for layout in layouts if layout.rows > 0]
    if not observed_layouts:
        raise RuntimeError("No document has an eligible Q block to analyze")

    shape = (
        args.expected_layers,
        args.expected_heads,
        len(PREDICTORS),
        len(TARGETS),
        len(METRICS),
    )
    layer_head_sum = np.zeros(shape, dtype=np.float64)
    layer_head_count = np.zeros(shape, dtype=np.int64)
    document_values_list: list[np.ndarray] = []
    document_counts_list: list[np.ndarray] = []
    document_indices: list[int] = []
    document_group_counts: list[int] = []
    distribution_rows: list[dict[str, Any]] = []
    crosschecked = False

    for ordinal, layout in enumerate(observed_layouts, start=1):
        risks, targets, position, group_layers, group_heads = (
            load_sample_arrays(layout)
        )
        group_metrics, debug = compute_group_metrics(
            risks, targets, position
        )
        if not crosschecked:
            sklearn_crosscheck(risks, debug, group_metrics)
            crosschecked = True

        document_mean, document_count = mean_and_count(
            group_metrics, axis=0
        )
        document_values_list.append(document_mean)
        document_counts_list.append(document_count)
        document_indices.append(layout.dataset_index)
        document_group_counts.append(group_metrics.shape[0])
        distribution_rows.extend(
            document_distribution_rows(
                dataset_index=layout.dataset_index,
                risks=risks,
                targets=targets,
            )
        )

        finite = np.isfinite(group_metrics)
        layer_head_sum[group_layers, group_heads] += np.where(
            finite, group_metrics, 0.0
        )
        layer_head_count[group_layers, group_heads] += finite
        if (
            ordinal % args.progress_every == 0
            or ordinal == len(observed_layouts)
        ):
            print(
                f"Analyzed {ordinal}/{len(observed_layouts)} observed "
                f"documents (dataset_index={layout.dataset_index}, "
                f"blocks/group={len(layout.q_blocks)}).",
                flush=True,
            )

    document_values = np.stack(document_values_list)
    document_counts = np.stack(document_counts_list)
    documents = document_frame(
        document_indices,
        document_values,
        document_counts,
        document_group_counts,
    )
    document_distribution = pd.DataFrame(distribution_rows)
    distribution_macro = macro_distribution_frame(document_distribution)
    layer_heads = layer_head_frame(layer_head_sum, layer_head_count)
    macro, comparison = macro_and_bootstrap_frames(
        document_values,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )

    atomic_write_parquet(
        documents, analysis_dir / "document_metrics.parquet"
    )
    atomic_write_csv(documents, analysis_dir / "document_metrics.csv")
    atomic_write_parquet(
        document_distribution,
        analysis_dir / "document_distribution.parquet",
    )
    atomic_write_csv(
        document_distribution,
        analysis_dir / "document_distribution.csv",
    )
    atomic_write_parquet(
        distribution_macro,
        analysis_dir / "distribution_macro.parquet",
    )
    atomic_write_csv(
        distribution_macro,
        analysis_dir / "distribution_macro.csv",
    )
    atomic_write_parquet(
        layer_heads, analysis_dir / "layer_head_summary.parquet"
    )
    atomic_write_csv(
        layer_heads, analysis_dir / "layer_head_summary.csv"
    )
    atomic_write_csv(macro, analysis_dir / "macro_summary.csv")
    atomic_write_parquet(macro, analysis_dir / "macro_summary.parquet")
    atomic_write_csv(
        comparison, analysis_dir / "qkbar_minus_raw_bootstrap.csv"
    )
    atomic_write_parquet(
        comparison,
        analysis_dir / "qkbar_minus_raw_bootstrap.parquet",
    )

    if not args.skip_plots:
        plot_spearman_heatmap(macro, plots_dir)
        plot_bootstrap_contrast(comparison, plots_dir)
        plot_fallback_recall(macro, plots_dir)
        plot_layer_profile(layer_heads, plots_dir)

    report = report_markdown(
        input_dir=input_dir,
        validation=validation,
        config=config,
        macro=macro,
        comparison=comparison,
        distribution_macro=distribution_macro,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    atomic_write_text(output_dir / "REPORT.md", report)
    print(
        f"Analysis complete. Report: {output_dir / 'REPORT.md'}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
