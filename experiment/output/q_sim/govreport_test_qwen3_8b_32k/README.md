# GovReport Q-block source data

- 模型：Qwen3-8B
- 数据：GovReport test 全部 973 篇
- Q/K block：Q128/K64
- 层与 head：36 层、32 Q heads
- `samples/`：逐文档原始 Q-block 统计
- `manifests/`：四个 shard 的配置和完成记录

面向结论的精简分析位于：

```text
../small_recall_analysis/REPORT_ZH.md
```

其中只比较：

```text
危险真值：原始 q 的 D_adj@50% 最高 20%
gate 1：1-selfsim(q)
gate 2：1-selfsim(q-k̄)
预算：matched fallback 10% / 20%
指标：TP、FP、FN、TN、precision、recall、F1
```
