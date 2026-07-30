#!/usr/bin/env python3
"""Collect Q-block self-similarity and token-level K-block disagreement.

The experiment hooks Transformers' SDPA backend, so ``query`` and ``key`` are
the actual post-QK-norm, post-RoPE tensors immediately before attention.

For every full Q block, all query tokens are compared on the K blocks that are
fully visible to *every* token in that Q block.  This common-prefix restriction
removes the otherwise unavoidable causal-boundary confound.  Each query token
selects its own top-r K-block centroids.  We then measure pairwise set
disagreement and the miss of the single pooled-Q selection.

The selector always uses the original Q.  ``q - mean(K)`` is only an alternate
self-similarity/gating feature, which isolates the question asked by this
experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


SCHEMA_VERSION = 1
MODEL_DEFAULT = "/home/dangyunkai/yunkai/VLM/VIG-Group/model/Qwen3-8B"
DATASET_DEFAULT = (
    "/data1/yunkai/VIG_Group/dataset/govreport-summarization/"
    "document/test-00000-of-00001.parquet"
)
OUTPUT_DEFAULT = str(
    Path(__file__).resolve().parents[2]
    / "output"
    / "q_sim"
    / "govreport_test_qwen3_8b_32k"
)
PROMPT_PREFIX = (
    "Summarize the following government report. Focus on its findings, "
    "evidence, and recommendations.\n\n"
)


@dataclass(frozen=True)
class ExperimentConfig:
    max_length: int
    block_q: int
    block_k: int
    min_common_key_blocks: int
    topk_fractions: tuple[float, ...]
    layers: tuple[int, ...]
    score_dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Q self-sim as a predictor of within-block K selection."
    )
    parser.add_argument("--model-path", default=MODEL_DEFAULT)
    parser.add_argument("--dataset-path", default=DATASET_DEFAULT)
    parser.add_argument("--output-dir", default=OUTPUT_DEFAULT)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--block-q", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--min-common-key-blocks", type=int, default=32)
    parser.add_argument(
        "--topk-fractions",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.50),
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Layer indices to analyze. Omit for all model layers.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples for this shard; intended only for smoke tests.",
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit dataset rows; still filtered by shard assignment.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--score-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Precision for token-to-K-centroid scores.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fraction_label(value: float) -> str:
    percentage = 100.0 * value
    if abs(percentage - round(percentage)) < 1e-8:
        return f"r{int(round(percentage)):02d}"
    return "r" + f"{percentage:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def validate_args(args: argparse.Namespace) -> None:
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(
            f"shard-id must be in [0, {args.num_shards}), got {args.shard_id}"
        )
    if args.max_length <= 0:
        raise ValueError("max-length must be positive")
    if args.block_q <= 0 or args.block_k <= 0:
        raise ValueError("block sizes must be positive")
    if args.block_q % args.block_k:
        raise ValueError("This experiment requires block-q to be divisible by block-k")
    if args.min_common_key_blocks < 2:
        raise ValueError("min-common-key-blocks must be at least 2")
    if not args.topk_fractions:
        raise ValueError("At least one top-k fraction is required")
    if len(set(args.topk_fractions)) != len(args.topk_fractions):
        raise ValueError("top-k fractions must be unique")
    if any(not 0.0 < value < 1.0 for value in args.topk_fractions):
        raise ValueError("Every top-k fraction must be strictly between 0 and 1")
    invalid_adjusted = [
        value
        for value in args.topk_fractions
        if math.ceil(args.min_common_key_blocks * value)
        >= args.min_common_key_blocks
    ]
    if invalid_adjusted:
        raise ValueError(
            "Every top-k fraction must leave at least one unselected key block "
            "at min-common-key-blocks so chance adjustment is finite; invalid: "
            f"{invalid_adjusted}"
        )
    labels = [fraction_label(value) for value in args.topk_fractions]
    if len(set(labels)) != len(labels):
        raise ValueError(f"top-k fractions produce duplicate labels: {labels}")


def git_revision(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        compression_level=3,
        row_group_size=262_144,
        use_dictionary=("layer", "q_head", "kv_head"),
    )
    os.replace(temporary, path)


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_payload_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def tokenize_govreport(
    tokenizer: Any, report: str, max_length: int
) -> tuple[torch.Tensor, int]:
    messages = [{"role": "user", "content": PROMPT_PREFIX + report}]
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        token_ids = tokenizer.encode(
            PROMPT_PREFIX + report, add_special_tokens=True
        )
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.flatten().tolist()
    original_length = len(token_ids)
    token_ids = token_ids[:max_length]
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0), original_length


def block_self_similarity(x: torch.Tensor) -> torch.Tensor:
    """Mean all-pairs cosine for ``x`` shaped [H, B, T, D]."""

    normalized = F.normalize(x.float(), p=2.0, dim=-1, eps=1e-12)
    mean_normalized = normalized.mean(dim=2)
    result = mean_normalized.square().sum(dim=-1)
    del normalized, mean_normalized
    return result


def normalized_jsd(scores: torch.Tensor, common_counts: torch.Tensor) -> torch.Tensor:
    """Generalized JSD of per-token softmax profiles, shaped [H, B]."""

    log_prob = F.log_softmax(scores, dim=-1)
    probability = log_prob.exp()
    token_terms = torch.where(
        torch.isfinite(log_prob), probability * log_prob, torch.zeros_like(log_prob)
    )
    mean_token_entropy = -token_terms.sum(dim=-1).mean(dim=2)
    mean_probability = probability.mean(dim=2)
    mean_terms = torch.where(
        mean_probability > 0,
        mean_probability * mean_probability.clamp_min(1e-30).log(),
        torch.zeros_like(mean_probability),
    )
    entropy_of_mean = -mean_terms.sum(dim=-1)
    normalizer = torch.minimum(
        common_counts,
        torch.full_like(common_counts, scores.shape[2]),
    ).float().log()
    result = (entropy_of_mean - mean_token_entropy) / normalizer[None, :]
    del (
        log_prob,
        probability,
        token_terms,
        mean_token_entropy,
        mean_probability,
        mean_terms,
        entropy_of_mean,
    )
    return result


def selection_metrics(
    *,
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    pooled_top_indices: torch.Tensor,
    score_std: torch.Tensor,
    common_counts: torch.Tensor,
    fraction: float,
    total_key_blocks: int,
) -> dict[str, torch.Tensor]:
    """Compute efficient set metrics without materializing token-pair matrices."""

    heads, blocks, tokens, available_ranks = top_indices.shape
    selected_per_block = torch.ceil(common_counts.float() * fraction).long()
    selected_per_block.clamp_(min=1)
    if bool((selected_per_block >= common_counts).any()):
        raise ValueError(
            "Chance-adjusted selection metrics require top-k to leave at "
            "least one common key block unselected"
        )
    max_selected = int(selected_per_block.max().item())
    if max_selected + 1 > available_ranks:
        raise ValueError(
            f"Need {max_selected + 1} ranks, only {available_ranks} available"
        )

    indices = top_indices[..., :max_selected]
    rank = torch.arange(max_selected, device=indices.device)
    keep_rank = rank[None, :] < selected_per_block[:, None]
    sentinel = total_key_blocks
    masked_indices = torch.where(
        keep_rank[None, :, None, :],
        indices,
        torch.full((), sentinel, dtype=indices.dtype, device=indices.device),
    )
    counts = torch.zeros(
        (heads, blocks, total_key_blocks + 1),
        dtype=torch.int32,
        device=indices.device,
    )
    counts.scatter_add_(
        dim=-1,
        index=masked_indices.reshape(heads, blocks, -1),
        src=torch.ones_like(masked_indices, dtype=torch.int32).reshape(
            heads, blocks, -1
        ),
    )
    counts = counts[..., :total_key_blocks].float()

    ordered_overlap = (counts * (counts - 1.0)).sum(dim=-1)
    ordered_overlap /= float(tokens * (tokens - 1))
    overlap_coefficient = ordered_overlap / selected_per_block[None, :].float()
    disagreement_raw = 1.0 - overlap_coefficient
    random_overlap = (
        selected_per_block.float() / common_counts.float()
    )[None, :]
    disagreement_adjusted = disagreement_raw / (1.0 - random_overlap)
    union_inflation = (counts > 0).sum(dim=-1).float()
    union_inflation /= selected_per_block[None, :].float()

    pooled_indices = pooled_top_indices[..., :max_selected]
    pooled_counts = counts.gather(dim=-1, index=pooled_indices)
    pooled_counts *= keep_rank[None, :, :].float()
    pooled_coverage = pooled_counts.sum(dim=-1)
    pooled_coverage /= float(tokens) * selected_per_block[None, :].float()
    pooled_miss_raw = 1.0 - pooled_coverage
    pooled_miss_adjusted = pooled_miss_raw / (1.0 - random_overlap)

    lower_rank = (selected_per_block - 1)[None, :, None, None].expand(
        heads, -1, tokens, 1
    )
    upper_rank = selected_per_block[None, :, None, None].expand(
        heads, -1, tokens, 1
    )
    lower_value = top_values.gather(dim=-1, index=lower_rank).squeeze(-1)
    upper_value = top_values.gather(dim=-1, index=upper_rank).squeeze(-1)
    normalized_margin = (lower_value - upper_value) / score_std.clamp_min(1e-12)
    margin_mean = normalized_margin.mean(dim=-1)
    near_tie_fraction = (normalized_margin < 1e-2).float().mean(dim=-1)

    result = {
        "selected_key_blocks": selected_per_block,
        "selection_overlap": overlap_coefficient,
        "selection_disagreement_raw": disagreement_raw,
        "selection_disagreement_adjusted": disagreement_adjusted,
        "selection_union_inflation": union_inflation,
        "pooled_selection_coverage": pooled_coverage,
        "pooled_selection_miss_raw": pooled_miss_raw,
        "pooled_selection_miss_adjusted": pooled_miss_adjusted,
        "boundary_margin_mean": margin_mean,
        "near_tie_fraction": near_tie_fraction,
    }
    del (
        indices,
        keep_rank,
        masked_indices,
        counts,
        ordered_overlap,
        overlap_coefficient,
        disagreement_raw,
        random_overlap,
        disagreement_adjusted,
        union_inflation,
        pooled_indices,
        pooled_counts,
        pooled_coverage,
        pooled_miss_raw,
        pooled_miss_adjusted,
        lower_rank,
        upper_rank,
        lower_value,
        upper_value,
        normalized_margin,
        margin_mean,
        near_tie_fraction,
    )
    return result


@torch.no_grad()
def compute_qblock_metrics(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    block_q: int,
    block_k: int,
    min_common_key_blocks: int,
    topk_fractions: tuple[float, ...],
    score_dtype: str,
) -> dict[str, torch.Tensor] | None:
    """Compute all metrics for one layer.

    ``query`` is [1,Hq,N,D], ``key`` is [1,Hkv,N,D].  Only complete Q and K
    blocks are used.  The K mean itself uses all retained sequence tokens,
    matching SpargeAttn's key smoothing during prefill.
    """

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError((query.shape, key.shape))
    if query.shape[0] != 1 or key.shape[0] != 1:
        raise ValueError("Only batch size 1 is supported")
    if query.shape[2] != key.shape[2] or query.shape[3] != key.shape[3]:
        raise ValueError((query.shape, key.shape))
    _, query_heads, sequence_length, head_dim = query.shape
    key_heads = key.shape[1]
    if query_heads % key_heads:
        raise ValueError("Q heads must be divisible by KV heads")

    q_block_count = sequence_length // block_q
    k_block_count = sequence_length // block_k
    if q_block_count == 0 or k_block_count == 0:
        return None
    all_q_block_ids = torch.arange(q_block_count, device=query.device)
    all_common_counts = (all_q_block_ids * block_q) // block_k
    eligible = all_common_counts >= min_common_key_blocks
    if not bool(eligible.any()):
        return None
    first_eligible = int(torch.nonzero(eligible, as_tuple=False)[0].item())
    q_block_ids = all_q_block_ids[first_eligible:]
    common_counts = all_common_counts[first_eligible:]
    block_count = q_block_ids.numel()
    tokens_per_block = block_q

    query_fp32 = query[0, :, : q_block_count * block_q, :].float()
    key_fp32_all = key[0].float()
    key_mean_unique = key_fp32_all.mean(dim=1)
    kv_group_size = query_heads // key_heads
    key_head_for_q = torch.arange(query_heads, device=query.device) // kv_group_size
    key_mean_for_q = key_mean_unique[key_head_for_q]
    wrong_key_mean_for_q = key_mean_unique.roll(shifts=-1, dims=0)[key_head_for_q]

    query_blocks = query_fp32.reshape(
        query_heads, q_block_count, block_q, head_dim
    )[:, first_eligible:]
    raw_selfsim = block_self_similarity(query_blocks)
    q_minus_kmean_selfsim = block_self_similarity(
        query_blocks - key_mean_for_q[:, None, None, :]
    )
    q_minus_wrong_kmean_selfsim = block_self_similarity(
        query_blocks - wrong_key_mean_for_q[:, None, None, :]
    )
    query_mean = query_fp32.mean(dim=1)
    q_minus_qmean_selfsim = block_self_similarity(
        query_blocks - query_mean[:, None, None, :]
    )

    key_blocks_unique = key_fp32_all[:, : k_block_count * block_k, :].reshape(
        key_heads, k_block_count, block_k, head_dim
    )
    key_pool_unique = key_blocks_unique.mean(dim=2)
    key_pool_unique -= key_mean_unique[:, None, :]
    key_pool = key_pool_unique[key_head_for_q]

    score_input = (
        query_blocks.bfloat16() if score_dtype == "bfloat16" else query_blocks
    )
    key_pool_score = (
        key_pool.bfloat16() if score_dtype == "bfloat16" else key_pool
    )
    scores = torch.einsum("hbtd,hkd->hbtk", score_input, key_pool_score)
    scores = scores.float()
    scores *= head_dim**-0.5
    key_ids = torch.arange(k_block_count, device=query.device)
    common_mask = key_ids[None, :] < common_counts[:, None]
    scores.masked_fill_(~common_mask[None, :, None, :], -torch.inf)

    safe_scores = torch.where(
        common_mask[None, :, None, :], scores, torch.zeros_like(scores)
    )
    score_mean = safe_scores.sum(dim=-1) / common_counts[None, :, None].float()
    score_second_moment = safe_scores.square().sum(dim=-1)
    score_second_moment /= common_counts[None, :, None].float()
    score_std = (
        score_second_moment - score_mean.square()
    ).clamp_min_(0.0).sqrt_()

    max_fraction = max(topk_fractions)
    max_selected = int(
        math.ceil(float(common_counts.max().item()) * max_fraction)
    )
    ranks_to_keep = min(k_block_count, max_selected + 1)
    top_values, top_indices = torch.topk(
        scores, k=ranks_to_keep, dim=-1, largest=True, sorted=True
    )
    pooled_scores = scores.mean(dim=2)
    _, pooled_top_indices = torch.topk(
        pooled_scores, k=ranks_to_keep, dim=-1, largest=True, sorted=True
    )

    results: dict[str, torch.Tensor] = {
        "q_block": q_block_ids,
        "common_key_blocks": common_counts,
        "q_selfsim_raw": raw_selfsim,
        "q_selfsim_q_minus_kbar": q_minus_kmean_selfsim,
        "q_selfsim_q_minus_qbar": q_minus_qmean_selfsim,
        "q_selfsim_q_minus_wrong_kbar": q_minus_wrong_kmean_selfsim,
        "preference_jsd": normalized_jsd(scores, common_counts),
        "kbar_norm": key_mean_for_q.norm(dim=-1),
        "qbar_norm": query_mean.norm(dim=-1),
    }
    for fraction in topk_fractions:
        label = fraction_label(fraction)
        metrics = selection_metrics(
            top_values=top_values,
            top_indices=top_indices,
            pooled_top_indices=pooled_top_indices,
            score_std=score_std,
            common_counts=common_counts,
            fraction=fraction,
            total_key_blocks=k_block_count,
        )
        for name, value in metrics.items():
            results[f"{name}_{label}"] = value

    del (
        query_fp32,
        key_fp32_all,
        key_mean_unique,
        key_mean_for_q,
        wrong_key_mean_for_q,
        query_blocks,
        raw_selfsim,
        q_minus_kmean_selfsim,
        q_minus_wrong_kmean_selfsim,
        query_mean,
        q_minus_qmean_selfsim,
        key_blocks_unique,
        key_pool_unique,
        key_pool,
        score_input,
        key_pool_score,
        scores,
        common_mask,
        safe_scores,
        score_mean,
        score_second_moment,
        score_std,
        top_values,
        top_indices,
        pooled_scores,
        pooled_top_indices,
    )
    return results


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy().astype(np.float32, copy=False)


def qblock_table_schema(topk_fractions: Iterable[float]) -> pa.Schema:
    """Canonical per-sample schema, including valid zero-row samples."""

    fields = [
        pa.field("schema_version", pa.int8()),
        pa.field("dataset_index", pa.int32()),
        pa.field("sequence_tokens", pa.int32()),
        pa.field("layer", pa.int16()),
        pa.field("q_head", pa.int16()),
        pa.field("kv_head", pa.int8()),
        pa.field("q_block", pa.int32()),
        pa.field("q_token_start", pa.int32()),
        pa.field("position_fraction", pa.float32()),
        pa.field("common_key_blocks", pa.int32()),
        pa.field("q_selfsim_raw", pa.float32()),
        pa.field("q_selfsim_q_minus_kbar", pa.float32()),
        pa.field("q_selfsim_q_minus_qbar", pa.float32()),
        pa.field("q_selfsim_q_minus_wrong_kbar", pa.float32()),
        pa.field("preference_jsd", pa.float32()),
        pa.field("kbar_norm", pa.float32()),
        pa.field("qbar_norm", pa.float32()),
    ]
    metric_names = (
        "selected_key_blocks",
        "selection_overlap",
        "selection_disagreement_raw",
        "selection_disagreement_adjusted",
        "selection_union_inflation",
        "pooled_selection_coverage",
        "pooled_selection_miss_raw",
        "pooled_selection_miss_adjusted",
        "boundary_margin_mean",
        "near_tie_fraction",
    )
    for fraction in topk_fractions:
        label = fraction_label(fraction)
        fields.extend(
            pa.field(f"{metric_name}_{label}", pa.float32())
            for metric_name in metric_names
        )
    return pa.schema(fields)


def metrics_to_table(
    *,
    metrics: dict[str, torch.Tensor],
    dataset_index: int,
    layer: int,
    sequence_tokens: int,
    block_q: int,
    num_query_heads: int,
    num_key_value_heads: int,
) -> pa.Table:
    q_block = metrics["q_block"].detach().cpu().numpy().astype(np.int32)
    common = (
        metrics["common_key_blocks"].detach().cpu().numpy().astype(np.int32)
    )
    block_count = q_block.size
    group_size = num_query_heads // num_key_value_heads
    q_heads = np.repeat(np.arange(num_query_heads, dtype=np.int16), block_count)
    q_blocks = np.tile(q_block, num_query_heads)
    common_blocks = np.tile(common, num_query_heads)
    columns: dict[str, Any] = {
        "schema_version": np.full(
            num_query_heads * block_count, SCHEMA_VERSION, dtype=np.int8
        ),
        "dataset_index": np.full(
            num_query_heads * block_count, dataset_index, dtype=np.int32
        ),
        "sequence_tokens": np.full(
            num_query_heads * block_count, sequence_tokens, dtype=np.int32
        ),
        "layer": np.full(
            num_query_heads * block_count, layer, dtype=np.int16
        ),
        "q_head": q_heads,
        "kv_head": (q_heads // group_size).astype(np.int8),
        "q_block": q_blocks,
        "q_token_start": (q_blocks * block_q).astype(np.int32),
        "position_fraction": (
            q_blocks.astype(np.float32) * float(block_q) / float(sequence_tokens)
        ),
        "common_key_blocks": common_blocks,
    }
    block_level_names = {"q_block", "common_key_blocks"}
    head_level_names = {"kbar_norm", "qbar_norm"}
    for name, value in metrics.items():
        if name in block_level_names:
            continue
        if name in head_level_names:
            array = tensor_to_numpy(value)
            columns[name] = np.repeat(array, block_count)
            continue
        array = tensor_to_numpy(value)
        if array.ndim == 1 and array.shape[0] == block_count:
            columns[name] = np.tile(array, num_query_heads)
        elif array.shape == (num_query_heads, block_count):
            columns[name] = array.reshape(-1)
        else:
            raise ValueError(f"Unexpected metric shape for {name}: {array.shape}")
    return pa.table(columns)


class QBlockRecorder:
    def __init__(
        self,
        *,
        config: ExperimentConfig,
        num_query_heads: int,
        num_key_value_heads: int,
    ) -> None:
        self.config = config
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.dataset_index = -1
        self.sequence_tokens = 0
        self.tables: list[pa.Table] = []
        self.layers_seen: list[int] = []

    def begin_sample(self, dataset_index: int, sequence_tokens: int) -> None:
        self.dataset_index = dataset_index
        self.sequence_tokens = sequence_tokens
        self.tables = []
        self.layers_seen = []

    @torch.no_grad()
    def capture(self, layer: int, query: torch.Tensor, key: torch.Tensor) -> None:
        if layer not in self.config.layers:
            return
        start = time.perf_counter()
        metrics = compute_qblock_metrics(
            query=query.detach(),
            key=key.detach(),
            block_q=self.config.block_q,
            block_k=self.config.block_k,
            min_common_key_blocks=self.config.min_common_key_blocks,
            topk_fractions=self.config.topk_fractions,
            score_dtype=self.config.score_dtype,
        )
        if metrics is None:
            return
        table = metrics_to_table(
            metrics=metrics,
            dataset_index=self.dataset_index,
            layer=layer,
            sequence_tokens=self.sequence_tokens,
            block_q=self.config.block_q,
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
        )
        self.tables.append(table)
        self.layers_seen.append(layer)
        print(
            f"sample={self.dataset_index:04d} layer={layer:02d} "
            f"rows={table.num_rows} metric_s={time.perf_counter() - start:.3f}",
            flush=True,
        )

    def finish_sample(self) -> pa.Table:
        if not self.tables:
            return pa.Table.from_batches(
                [], schema=qblock_table_schema(self.config.topk_fractions)
            )
        return pa.concat_tables(self.tables, promote_options="none")


def iter_indices(
    *,
    total_rows: int,
    num_shards: int,
    shard_id: int,
    sample_indices: Iterable[int] | None,
    max_samples: int | None,
) -> list[int]:
    if sample_indices is None:
        candidates = range(total_rows)
    else:
        candidates = sorted(set(sample_indices))
        bad = [index for index in candidates if index < 0 or index >= total_rows]
        if bad:
            raise IndexError(f"Dataset indices outside [0, {total_rows}): {bad}")
    selected = [index for index in candidates if index % num_shards == shard_id]
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    output_dir = Path(args.output_dir).resolve()
    sample_dir = output_dir / "samples"
    log_dir = output_dir / "manifests"
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    shard_tag = f"shard-{args.shard_id:03d}-of-{args.num_shards:03d}"
    manifest_path = log_dir / f"{shard_tag}.jsonl"
    error_path = log_dir / f"{shard_tag}.errors.jsonl"

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    dataset_file = pq.ParquetFile(args.dataset_path)
    dataset_rows = dataset_file.metadata.num_rows
    selected_indices = iter_indices(
        total_rows=dataset_rows,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        sample_indices=args.sample_indices,
        max_samples=args.max_samples,
    )
    reports = dataset_file.read(columns=["report"]).column("report")

    print(f"Loading tokenizer from {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True
    )
    print(f"Loading model on {props.name}", flush=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    model_layers = int(model.config.num_hidden_layers)
    layers = (
        tuple(range(model_layers))
        if args.layers is None
        else tuple(sorted(set(args.layers)))
    )
    if any(layer < 0 or layer >= model_layers for layer in layers):
        raise ValueError(f"layers must be within [0, {model_layers}): {layers}")

    experiment_config = ExperimentConfig(
        max_length=args.max_length,
        block_q=args.block_q,
        block_k=args.block_k,
        min_common_key_blocks=args.min_common_key_blocks,
        topk_fractions=tuple(sorted(args.topk_fractions)),
        layers=layers,
        score_dtype=args.score_dtype,
    )
    run_signature_payload = {
        "schema_version": SCHEMA_VERSION,
        "model_path": str(Path(args.model_path).resolve()),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "dataset_rows": dataset_rows,
        "prompt_prefix": PROMPT_PREFIX,
        "experiment": asdict(experiment_config),
        "model_config": {
            "num_hidden_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": model.config.num_key_value_heads,
            "head_dim": model.config.head_dim,
        },
    }
    run_signature = stable_payload_sha256(run_signature_payload)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_signature": run_signature,
        "run_signature_payload": run_signature_payload,
        "created_at": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": props.name,
        "gpu_compute_capability": [props.major, props.minor],
        "gpu_total_memory_bytes": props.total_memory,
        "model_path": str(Path(args.model_path).resolve()),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "dataset_split": "test",
        "dataset_rows": dataset_rows,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "selected_sample_count": len(selected_indices),
        "max_samples": args.max_samples,
        "seed": args.seed,
        "experiment": asdict(experiment_config),
        "model_config": {
            "architectures": model.config.architectures,
            "hidden_size": model.config.hidden_size,
            "num_hidden_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": model.config.num_key_value_heads,
            "head_dim": model.config.head_dim,
            "max_position_embeddings": model.config.max_position_embeddings,
        },
        "git_revision": git_revision(Path(__file__).resolve().parents[3]),
        "metric_definitions": {
            "q_selfsim": "s(X)=||mean_t normalize(x_t)||_2^2",
            "q_minus_kbar": (
                "x_t=q_t-mean_sequence(k_{g(h)}), post-QK-norm/post-RoPE"
            ),
            "candidate_keys": (
                "full K blocks ending before the first token of the Q block"
            ),
            "selection_score": (
                "q_t dot mean(K_block-kbar)/sqrt(head_dim); original q only"
            ),
            "selection_disagreement_adjusted": (
                "(1-mean_pairwise_overlap/topk_count)"
                "/(1-topk_count/common_key_blocks)"
            ),
        },
    }
    atomic_write_json(
        log_dir / f"{shard_tag}.run_metadata.json",
        metadata,
    )

    recorder = QBlockRecorder(
        config=experiment_config,
        num_query_heads=int(model.config.num_attention_heads),
        num_key_value_heads=int(model.config.num_key_value_heads),
    )
    original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def recording_sdpa(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        recorder.capture(int(module.layer_idx), query, key)
        return original_sdpa(
            module,
            query,
            key,
            value,
            attention_mask,
            **kwargs,
        )

    completed: set[int] = set()
    for path in sample_dir.glob("sample-*.parquet"):
        metadata_path = path.with_suffix(".metadata.json")
        if not metadata_path.exists():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                sample_metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if sample_metadata.get("run_signature") == run_signature:
            completed.add(int(path.stem.split("-")[-1]))
        else:
            raise RuntimeError(
                f"Existing sample has a different experiment signature: {path}. "
                "Use a different --output-dir for a different configuration."
            )
    pending = [index for index in selected_indices if index not in completed]
    print(
        f"{shard_tag}: assigned={len(selected_indices)}, "
        f"already_complete={len(selected_indices) - len(pending)}, "
        f"pending={len(pending)}",
        flush=True,
    )

    ALL_ATTENTION_FUNCTIONS["sdpa"] = recording_sdpa
    shard_start = time.perf_counter()
    failures = 0
    try:
        for ordinal, dataset_index in enumerate(pending, start=1):
            report = reports[dataset_index].as_py()
            input_ids, original_tokens = tokenize_govreport(
                tokenizer, report, args.max_length
            )
            used_tokens = int(input_ids.shape[-1])
            source_hash = prompt_sha256(report)
            recorder.begin_sample(dataset_index, used_tokens)
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids)
            sample_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            print(
                f"[{ordinal}/{len(pending)}] sample={dataset_index:04d} "
                f"tokens={original_tokens}->{used_tokens}",
                flush=True,
            )
            try:
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - sample_start
                table = recorder.finish_sample()
                sample_path = sample_dir / f"sample-{dataset_index:05d}.parquet"
                metadata_path = sample_path.with_suffix(".metadata.json")
                atomic_write_parquet(sample_path, table)
                sample_metadata = {
                    "schema_version": SCHEMA_VERSION,
                    "run_signature": run_signature,
                    "completed_at": utc_now(),
                    "dataset_index": dataset_index,
                    "report_sha256": source_hash,
                    "original_tokens": original_tokens,
                    "used_tokens": used_tokens,
                    "truncated": original_tokens > used_tokens,
                    "rows": table.num_rows,
                    "layers_seen": recorder.layers_seen,
                    "elapsed_seconds": elapsed,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
                    "sample_file": str(sample_path),
                    "shard_id": args.shard_id,
                    "num_shards": args.num_shards,
                }
                atomic_write_json(metadata_path, sample_metadata)
                append_jsonl(manifest_path, sample_metadata)
                print(
                    f"completed sample={dataset_index:04d} rows={table.num_rows} "
                    f"elapsed={elapsed:.2f}s "
                    f"peak={sample_metadata['peak_cuda_memory_bytes'] / 2**30:.1f}GiB",
                    flush=True,
                )
                del output, table
            except Exception as error:
                failures += 1
                error_record = {
                    "failed_at": utc_now(),
                    "dataset_index": dataset_index,
                    "original_tokens": original_tokens,
                    "used_tokens": used_tokens,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                append_jsonl(error_path, error_record)
                print(
                    f"ERROR sample={dataset_index}: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                del input_ids, attention_mask
                recorder.tables = []
                torch.cuda.empty_cache()
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original_sdpa

    final_metadata = {
        **metadata,
        "finished_at": utc_now(),
        "elapsed_seconds": time.perf_counter() - shard_start,
        "failures_this_invocation": failures,
        "complete_sample_files_total": len(
            list(sample_dir.glob("sample-*.parquet"))
        ),
    }
    atomic_write_json(
        log_dir / f"{shard_tag}.finished.json",
        final_metadata,
    )
    if failures:
        raise RuntimeError(
            f"{shard_tag} finished with {failures} failed samples; rerun to retry"
        )


if __name__ == "__main__":
    main()
