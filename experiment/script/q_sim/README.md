# Q-block matched fallback experiment

## 唯一研究问题

1. 用原始 `q` 计算 Q block 内 query token 的 K-block 选择分歧
   `D_adj@50%`。
2. 每个 `document × layer × Q-head` 内，最高 20% 分歧的 Q blocks
   定义为高分歧。
3. 分别用 `1-selfsim(q)` 和 `1-selfsim(q-k̄)` 排同一批 Q blocks。
4. 两个 gate fallback 完全相同数量的 Q blocks。
5. 比较 TP、FP、FN、TN、precision、recall 和 F1。

`q-k̄` 不用于选择 K blocks，只用于判断 Q block 是否 fallback。

## 文件

- `collect_q_keyblock_disagreement.py`：采集原始 q self-sim、q-kbar
  self-sim 和 original-q K-block disagreement。
- `run_govreport_full.sh`：GovReport test 全量分片运行。
- `analyze_fallback_confusion.py`：只生成 matched fallback 核心指标。
- `test_q_keyblock_metrics.py`：K-block disagreement 公式与 GQA 映射测试。
- `test_fallback_confusion.py`：混淆矩阵、F1 和预算分解测试。

## 运行精简分析

```bash
cd /home/dangyunkai/yunkai/VLM/VIG-Group/haoyi/ICLR27/sparse_attn/SpargeAttn

/data1/dangyunkai/conda_envs/sparge/bin/python \
  experiment/script/q_sim/analyze_fallback_confusion.py \
  --run-dir experiment/output/q_sim/govreport_test_qwen3_8b_32k \
  --output-dir experiment/output/q_sim/small_recall_analysis \
  --bootstrap-replicates 2000
```

输出：

```text
experiment/output/q_sim/small_recall_analysis/
```

## 测试

```bash
/data1/dangyunkai/conda_envs/sparge/bin/python \
  experiment/script/q_sim/test_q_keyblock_metrics.py

/data1/dangyunkai/conda_envs/sparge/bin/python \
  experiment/script/q_sim/test_fallback_confusion.py
```
