# q_sim experiments

当前完整主实验：

- `govreport_test_qwen3_8b_32k/FALLBACK_EVIDENCE_ZH.md`：只保留
  `q-k̄` 用于 fall-to-dense 的专项证据。
- `govreport_test_qwen3_8b_32k/REPORT_ZH.md`：中文结论、方法、指标和限制。
- `govreport_test_qwen3_8b_32k/REPORT.md`：英文机器生成统计报告。
- `govreport_test_qwen3_8b_32k/analysis/`：可复核表格与 validation。
- `govreport_test_qwen3_8b_32k/plots/`：结果图。
- `govreport_test_qwen3_8b_32k/samples/`：973 篇 GovReport 的原始统计。

对应采集、分析、运行和公式测试脚本位于
`../../script/q_sim/`。
