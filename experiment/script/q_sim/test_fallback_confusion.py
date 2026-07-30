#!/usr/bin/env python3
"""Deterministic tests for focused fallback confusion metrics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).with_name("analyze_fallback_confusion.py")
SPEC = importlib.util.spec_from_file_location(
    "fallback_confusion_analysis", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_confusion_matrix_and_f1() -> None:
    truth = np.array([[True, True, False, False]])
    selected = np.array([[True, False, True, False]])
    result = MODULE.confusion_metrics(truth, selected)
    for name in ("tp_count", "fp_count", "fn_count", "tn_count"):
        assert result[name][0] == 1
    assert np.isclose(result["precision"][0], 0.5)
    assert np.isclose(result["recall"][0], 0.5)
    assert np.isclose(result["f1"][0], 0.5)
    assert np.isclose(result["max_possible_recall"][0], 1.0)
    assert np.isclose(result["budget_forced_miss"][0], 0.0)
    assert np.isclose(result["ranking_miss"][0], 0.5)


def test_budget_forced_and_ranking_miss_sum_to_total_miss() -> None:
    truth = np.array([[True, True, True, True, False, False]])
    selected = np.array([[True, False, False, False, False, False]])
    result = MODULE.confusion_metrics(truth, selected)
    assert np.isclose(result["recall"][0], 0.25)
    assert np.isclose(result["max_possible_recall"][0], 0.25)
    assert np.isclose(result["budget_forced_miss"][0], 0.75)
    assert np.isclose(result["ranking_miss"][0], 0.0)
    assert np.isclose(
        result["budget_forced_miss"][0]
        + result["ranking_miss"][0],
        1.0 - result["recall"][0],
    )


def test_zero_true_positive_has_zero_f1() -> None:
    truth = np.array([[True, False]])
    selected = np.array([[False, True]])
    result = MODULE.confusion_metrics(truth, selected)
    assert result["precision"][0] == 0.0
    assert result["recall"][0] == 0.0
    assert result["f1"][0] == 0.0


def test_matched_budget_qkbar_can_improve_recall() -> None:
    target = np.array([[0.9, 0.8, 0.2, 0.1]])
    truth = MODULE.top_mask(target, 2)
    raw_risk = np.array([[0.9, 0.1, 0.8, 0.2]])
    qk_risk = np.array([[0.9, 0.8, 0.2, 0.1]])
    raw_selected = MODULE.top_mask(raw_risk, 2)
    qk_selected = MODULE.top_mask(qk_risk, 2)
    assert raw_selected.sum() == qk_selected.sum() == 2
    raw = MODULE.confusion_metrics(truth, raw_selected)
    qk = MODULE.confusion_metrics(truth, qk_selected)
    assert np.isclose(raw["recall"][0], 0.5)
    assert np.isclose(qk["recall"][0], 1.0)


def main() -> None:
    test_confusion_matrix_and_f1()
    test_budget_forced_and_ranking_miss_sum_to_total_miss()
    test_zero_true_positive_has_zero_f1()
    test_matched_budget_qkbar_can_improve_recall()
    print("All focused fallback confusion tests passed.")


if __name__ == "__main__":
    main()
