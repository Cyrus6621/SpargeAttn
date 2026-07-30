# Q-block matched fallback

本目录只保留一个核心问题：

> 用原始 `q` 定义块内 K-block 选择分歧后，在 fallback 数量完全相同时，
> `1-selfsim(q-k̄)` 是否比 `1-selfsim(q)` 找回更多高分歧 Q blocks？

`q-k̄` 从不参与 K-block 选择，只用于给 Q blocks 排 fallback 风险。

核心结果位于 `small_recall_analysis/`：

- `REPORT_ZH.md`：定义、precision、recall、F1、TP/FP/FN/TN 和低 recall
  分解。
- `core_summary.csv`：文档等权结果与 whole-document bootstrap CI。
- `qkbar_minus_raw.csv`：q-kbar 减 raw Q 的 precision/recall/F1 差值。
- `pooled_confusion.csv`：所有 block 合并后的混淆矩阵。
- `core_precision_recall_f1.png`：核心指标图。
- `miss_decomposition.png`：预算强制漏检与排序漏检。

全量 GovReport 原始统计和运行记录位于
`govreport_test_qwen3_8b_32k/samples/` 与 `manifests/`。
