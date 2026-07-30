# Q self-sim 与块内 K-block 选择分歧实验

本目录回答两个问题：

1. 低 query-block self-sim 是否意味着该 Q128 block 内不同 query token
   会选择差异更大的 K64 blocks？
2. 在相同 dense-fallback 预算下，使用 `selfsim(q - k_bar)` 是否比
   `selfsim(q)` 更适合作为 gate？

## 主实验

- 模型：`/home/dangyunkai/yunkai/VLM/VIG-Group/model/Qwen3-8B`
- 数据：GovReport test 全部 973 篇
- 最大长度：32,768 tokens；4/973 篇会被截断
- 层/head：全部 36 层、32 Q heads；Qwen GQA 的 4 个 Q heads 映射到
  同一个 KV head
- 张量位置：QK-norm 和 RoPE 之后、attention 之前
- block：A100/A800 上 Q128/K64

选择 GovReport test 而不是把少量 RULER 样本称为“完整实验”：本地 RULER
plain 有 39,000 条，且 64K/128K 超出该 checkpoint 未启用 YaRN 时的原生
位置范围；GovReport test 的 973 篇可以完整遍历，且仅 4 篇超过 32K。

## 关键定义

对 Q block `b`，候选 K blocks 只包含在该 Q block 第一个 token 之前结束的
完整 K64 blocks。因而块内所有 query token 的候选集合完全相同，不会把
causal boundary 造成的可见范围变化误算成 query 异质性。主分析要求至少
32 个共同可见 K blocks；过短文档仍会被模型完整处理并记录，但不会产生
不满足该条件的 block observation。

每个 query token 在共同候选上用

```text
score(t, j) = q_t · mean(K_j - k_bar) / sqrt(head_dim)
```

选择 top-r blocks，`r = 0.10, 0.25, 0.50`。K 侧减去同一 `k_bar` 不改变
block ranking；selector 始终使用原始 `q`，`q-k_bar` 只作为 alternate
self-sim/gate 特征。

主分歧指标是 chance-adjusted pairwise top-set disagreement。设一个
Q block 有 `n=128` 个 token、`M` 个候选 K blocks、每个 token 选择
`k=ceil(rM)` 个，K block `j` 被 `c_j` 个 token 选中，则

```text
overlap = sum_j choose(c_j, 2) / (choose(n, 2) * k)
D_top   = (1 - overlap) / (1 - k/M)
```

`D_top=0` 表示所有 token 选择相同，`D_top=1` 表示差异约等于随机 top-k。
另外保存 pooled-Q mask miss、softmax preference JSD、top-k boundary margin
和 union inflation，避免仅凭一个离散 top-k 指标下结论。

## 运行

以下命令从 SpargeAttn 仓库根目录执行：

```bash
cd /home/dangyunkai/yunkai/VLM/VIG-Group/haoyi/ICLR27/sparse_attn/SpargeAttn
```

默认在 GPU 0/1/6 上分 3 片：

```bash
bash experiment/script/q_sim/run_govreport_full.sh
```

显式映射 GPU 与 shard（适合分批占卡，并保持同一个总分片数）：

```bash
GPU_IDS="4" SHARD_IDS="3" NUM_SHARDS=4 \
  bash experiment/script/q_sim/run_govreport_full.sh

GPU_IDS="0 1 6" SHARD_IDS="0 1 2" NUM_SHARDS=4 \
  bash experiment/script/q_sim/run_govreport_full.sh
```

每篇文档原子写一个 Parquet 和 metadata JSON。相同配置可直接重跑，已完成
文档会跳过；若同一输出目录中已有不同配置，collector 会拒绝混写。

分析命令：

```bash
/data1/dangyunkai/conda_envs/sparge/bin/python \
  experiment/script/q_sim/analyze_q_keyblock_disagreement.py \
  --run-dir experiment/output/q_sim/govreport_test_qwen3_8b_32k
```

分析以 document 为统计 cluster。相关性先在每个
`document × layer × Q-head` 内计算，再宏平均；同时报告 position-adjusted
Spearman，以及 5%/10%/20% matched fallback budget 下的 recall/lift。
`q` 与 `q-k_bar` 不使用相同数值阈值硬比。

## 输出

默认根目录为 `experiment/output/q_sim/govreport_test_qwen3_8b_32k`：

- `samples/`：逐文档、逐层/head/block 的原始统计
- `manifests/`：运行配置、完成记录和错误记录
- `logs/`：各 GPU shard 日志
- `analysis/`：document/layer/head 聚合、bootstrap CI 和分桶曲线
- `plots/`：关联矩阵、配对差值、layer 和 fallback-budget 图
- `REPORT.md`：完整方法、覆盖率、数值结果和结论

基础公式测试：

```bash
/data1/dangyunkai/conda_envs/sparge/bin/python \
  experiment/script/q_sim/test_q_keyblock_metrics.py
```
