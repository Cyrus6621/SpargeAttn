#!/usr/bin/env python3
"""Small deterministic checks for the Q/K-block metric implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pyarrow as pa
import torch


MODULE_PATH = Path(__file__).with_name("collect_q_keyblock_disagreement.py")
SPEC = importlib.util.spec_from_file_location("q_sim_collector", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_identical_queries_have_zero_disagreement() -> None:
    torch.manual_seed(0)
    heads_q, heads_k, length, dim = 4, 1, 64, 8
    base = torch.randn(heads_q, 1, dim)
    query = base.expand(-1, length, -1).unsqueeze(0).contiguous()
    key = torch.randn(1, heads_k, length, dim)
    result = MODULE.compute_qblock_metrics(
        query=query,
        key=key,
        block_q=8,
        block_k=4,
        min_common_key_blocks=4,
        topk_fractions=(0.25, 0.5),
        score_dtype="float32",
    )
    assert result is not None
    assert torch.allclose(
        result["q_selfsim_raw"], torch.ones_like(result["q_selfsim_raw"]), atol=1e-6
    )
    for label in ("r25", "r50"):
        assert torch.allclose(
            result[f"selection_disagreement_raw_{label}"],
            torch.zeros_like(result[f"selection_disagreement_raw_{label}"]),
            atol=1e-6,
        )
        assert torch.allclose(
            result[f"pooled_selection_miss_raw_{label}"],
            torch.zeros_like(result[f"pooled_selection_miss_raw_{label}"]),
            atol=1e-6,
        )


def test_common_prefix_starts_before_query_block() -> None:
    torch.manual_seed(1)
    query = torch.randn(1, 4, 64, 8)
    key = torch.randn(1, 1, 64, 8)
    result = MODULE.compute_qblock_metrics(
        query=query,
        key=key,
        block_q=8,
        block_k=4,
        min_common_key_blocks=4,
        topk_fractions=(0.5,),
        score_dtype="float32",
    )
    assert result is not None
    q_blocks = result["q_block"].cpu()
    common = result["common_key_blocks"].cpu()
    assert torch.equal(common, q_blocks * 2)
    assert int(common.min()) == 4


def test_set_statistics_match_explicit_pairs() -> None:
    torch.manual_seed(2)
    heads, blocks, tokens, key_blocks = 2, 2, 5, 7
    common = torch.tensor([4, 7])
    scores = torch.randn(heads, blocks, tokens, key_blocks)
    key_ids = torch.arange(key_blocks)
    scores.masked_fill_(
        key_ids[None, None, None, :] >= common[None, :, None, None],
        -torch.inf,
    )
    max_selected = int(torch.ceil(common.float().max() * 0.5).item())
    top_values, top_indices = torch.topk(
        scores, k=max_selected + 1, dim=-1, sorted=True
    )
    pooled = scores.mean(dim=2)
    _, pooled_indices = torch.topk(
        pooled, k=max_selected + 1, dim=-1, sorted=True
    )
    finite = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
    mean = finite.sum(dim=-1) / common[None, :, None]
    second = finite.square().sum(dim=-1) / common[None, :, None]
    std = (second - mean.square()).clamp_min(0).sqrt()
    result = MODULE.selection_metrics(
        top_values=top_values,
        top_indices=top_indices,
        pooled_top_indices=pooled_indices,
        score_std=std,
        common_counts=common,
        fraction=0.5,
        total_key_blocks=key_blocks,
    )

    for head in range(heads):
        for block in range(blocks):
            selected = int(torch.ceil(common[block].float() * 0.5).item())
            sets = [
                set(top_indices[head, block, token, :selected].tolist())
                for token in range(tokens)
            ]
            overlaps = []
            for left in range(tokens):
                for right in range(left + 1, tokens):
                    overlaps.append(len(sets[left] & sets[right]) / selected)
            overlap = sum(overlaps) / len(overlaps)
            expected_raw = 1.0 - overlap
            expected_adjusted = expected_raw / (
                1.0 - selected / int(common[block])
            )
            pool_set = set(
                pooled_indices[head, block, :selected].tolist()
            )
            coverage = sum(
                len(pool_set & token_set) / selected for token_set in sets
            ) / tokens
            assert torch.isclose(
                result["selection_disagreement_raw"][head, block],
                torch.tensor(expected_raw),
                atol=1e-6,
            )
            assert torch.isclose(
                result["selection_disagreement_adjusted"][head, block],
                torch.tensor(expected_adjusted),
                atol=1e-6,
            )
            assert torch.isclose(
                result["pooled_selection_coverage"][head, block],
                torch.tensor(coverage),
                atol=1e-6,
            )


def test_zero_row_table_keeps_canonical_schema() -> None:
    torch.manual_seed(3)
    fractions = (0.25, 0.5)
    query = torch.randn(1, 4, 64, 8)
    key = torch.randn(1, 1, 64, 8)
    result = MODULE.compute_qblock_metrics(
        query=query,
        key=key,
        block_q=8,
        block_k=4,
        min_common_key_blocks=4,
        topk_fractions=fractions,
        score_dtype="float32",
    )
    assert result is not None
    nonempty = MODULE.metrics_to_table(
        metrics=result,
        dataset_index=0,
        layer=0,
        sequence_tokens=64,
        block_q=8,
        num_query_heads=4,
        num_key_value_heads=1,
    )
    empty = pa.Table.from_batches(
        [], schema=MODULE.qblock_table_schema(fractions)
    )
    assert empty.num_rows == 0
    assert empty.schema == nonempty.schema


def test_chance_adjustment_rejects_selecting_every_key_block() -> None:
    common = torch.tensor([2])
    scores = torch.randn(1, 1, 3, 2)
    top_values, top_indices = torch.topk(scores, k=2, dim=-1, sorted=True)
    pooled_indices = torch.topk(
        scores.mean(dim=2), k=2, dim=-1, sorted=True
    ).indices
    try:
        MODULE.selection_metrics(
            top_values=top_values,
            top_indices=top_indices,
            pooled_top_indices=pooled_indices,
            score_std=torch.ones(1, 1, 3),
            common_counts=common,
            fraction=0.9,
            total_key_blocks=2,
        )
    except ValueError as error:
        assert "leave at least one" in str(error)
    else:
        raise AssertionError("Expected a finite-adjustment guard")


def test_gqa_mapping_and_q_minus_kbar_match_explicit_calculation() -> None:
    torch.manual_seed(4)
    query = torch.randn(1, 4, 32, 8)
    key = torch.randn(1, 2, 32, 8)
    result = MODULE.compute_qblock_metrics(
        query=query,
        key=key,
        block_q=8,
        block_k=4,
        min_common_key_blocks=2,
        topk_fractions=(0.25,),
        score_dtype="float32",
    )
    assert result is not None
    first_q_block = int(result["q_block"][0])
    query_blocks = query[0].float().reshape(4, 4, 8, 8)[:, first_q_block:]
    key_mean = key[0].float().mean(dim=1)
    q_to_kv = torch.tensor([0, 0, 1, 1])
    expected = MODULE.block_self_similarity(
        query_blocks - key_mean[q_to_kv, None, None, :]
    )
    assert torch.allclose(
        result["q_selfsim_q_minus_kbar"], expected, atol=1e-6
    )

    table = MODULE.metrics_to_table(
        metrics=result,
        dataset_index=0,
        layer=0,
        sequence_tokens=32,
        block_q=8,
        num_query_heads=4,
        num_key_value_heads=2,
    )
    q_head = table.column("q_head").to_numpy()
    kv_head = table.column("kv_head").to_numpy()
    for head, expected_kv in enumerate((0, 0, 1, 1)):
        assert set(kv_head[q_head == head].tolist()) == {expected_kv}


def main() -> None:
    test_identical_queries_have_zero_disagreement()
    test_common_prefix_starts_before_query_block()
    test_set_statistics_match_explicit_pairs()
    test_zero_row_table_keeps_canonical_schema()
    test_chance_adjustment_rejects_selecting_every_key_block()
    test_gqa_mapping_and_q_minus_kbar_match_explicit_calculation()
    print("All q_sim metric tests passed.")


if __name__ == "__main__":
    main()
