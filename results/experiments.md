# CS336 A2 实验记录

硬件：RTX 3070 Ti Laptop 8GB ｜ `torch 2.11.0+cu128` ｜ 基准：vocab=10000, batch=4, ctx=512, fp32, 5 warmup + 10 steps

> 运行命令速查（在 `assignment2-systems/` 目录，Windows PowerShell）：
> - 普通 benchmark：`uv run python answer/benchmark.py [--mode forward|backward|optimizer] [--model small|medium] [--warmup N]`
> - nsys profile：`uv run nsys profile [-o results/NAME] --trace=cuda,cuDNN,cublas,nvtx -- python answer/benchmark.py ...`
> - 出统计：`nsys stats --report cuda_gpu_kern_sum results/NAME.nsys-rep`
> - 按 NVTX 段过滤：SQLite 查询（模板见 `kern_stats_by_nvtx.sql`）

## §2.1.3(b) 模型尺寸 × 模式（秒）

跑法：`uv run python answer/benchmark.py --mode <m> --model <model>`（默认 warmup=5, steps=10）

| 模型 | forward | backward | optimizer 全步 |
|---|---|---|---|
| small | 0.1023 ± 0.0019 | 0.3106 ± 0.0040 | 0.3669 ± 0.0036 |
| medium | 0.3176 ± 0.0051（no_grad） | OOM | OOM |

- backward ≈ 2×forward；小模型 std ≤2%，测量稳定。
- medium 无 no_grad 时 forward=6.02s / backward=41.30s / optimizer=OOM（8GB 内存挣扎，报错 "allocated 14.38 GiB"），+no_grad 后 0.32s。结论：medium fp32 训练类测量不可行，已用 small 档为主。

## §2.1.3(c) warmup 影响（small，秒）

跑法：`uv run python answer/benchmark.py --mode <m> --warmup N`（0/1/2/5 × 三模式，共 12 次）

| warmup | forward | std | backward | std | optimizer | std |
|---|---|---|---|---|---|---|
| 0 | 0.1437 | 0.1220 | 0.3476 | 0.1041 | 0.3949 | 0.0828 |
| 1 | 0.1034 | 0.0012 | 0.3134 | 0.0039 | 0.3673 | 0.0066 |
| 2 | 0.1045 | 0.0012 | 0.3138 | 0.0038 | 0.3683 | 0.0063 |
| 5 | 0.1023 | 0.0019 | 0.3106 | 0.0040 | 0.3669 | 0.0036 |

- warmup=0：第 1 步 ≈0.54s（CUDA 初始化 / JIT / autotuner），std 暴涨 60-100×。
- warmup=1 一步止血（std →~1%）；1 vs 5 剩余差 <1%（频率爬升等渐进效应）。

## §2.1.4(a)-(c) nsys profile（forward, small, `report2.nsys-rep`）

跑法：
```
uv run nsys profile --trace=cuda,cuDNN,cublas,nvtx -- python answer/benchmark.py --mode forward --model small
nsys stats --report cuda_gpu_kern_sum results/report2.nsys-rep
```

nsys 2026.4.1 CLI 坑：`--python-backtrace`/`--cudabacktrace`/`--osrt`/`--pytorch=` 参数名变了或移除；`--trace` 的值要写 `cuDNN`（大小写敏感）；`--gpu-metrics-devices=0` 需要管理员权限（ERR_NVGPUCTRPERM）。

- 全进程 GPU kernel 合计 ≈ 1.59s ≈ Python 计时 1.53s（15 步）→ 同步计时可信（回答 (a)）。
- 最贵 kernel：`ampere_sgemm_128x64_tn` 58.1%（1275 调用 = 15 步 × 85 matmul）（回答 (b)）。
- 非 matmul kernel 合计 ~25-30%：elementwise ~15%（RMSNorm/残差/scale）、exp 3.1%（softmax）、reduce max 1.7% / reduce mean 1.5%、sigmoid 1.6%（SwiGLU）、Cat 1.0%（回答 (c)）。
- 后续：backward profile 回答 (b)(d)；softmax vs matmul 数据回答 (e)。

**backward 报告（report4.nsys-rep）** 跑法：
```
uv run nsys profile --trace=cuda,cuDNN,cublas,nvtx -- python answer/benchmark.py --mode backward --model small
```
**optimizer 报告（report6_measure.nsys-rep）** 跑法：
```
uv run nsys profile -o results/report6_measure --trace=cuda,cuDNN,cublas,nvtx -- python answer/benchmark.py --mode optimizer --model small
```
（SQL 过滤步骤：`python -c "..."` 按 `kern_stats_by_nvtx.sql` 模板执行，或 `nsys stats --report cuda_gpu_kern_sum results/report6_measure.nsys-rep` 直接看全量）

## §2.1.4(d) optimizer 模式（report6_measure.nsys-rep，SQL 按 "measure" 段过滤）

三报告对比（small, 5 warmup + 10 measured 口径）：

| 指标 | forward | backward | optimizer |
|---|---|---|---|
| GPU kernel 总时间 | 1.59s | 5.01s | 5.57s（measure 段 3.42s） |
| matmul(sgemm) 占比 | 66.1% | 62.2% | 51.7%（measure 段 ~49.9%） |
| 最贵 kernel | sgemm_128x64_tn 58.1% | sgemm 三兄弟 17.6/16.6/13.4% | sgemm 三兄弟 14.6/14.1/11.1% |

- 结论：训练步 matmul 占比 66%→62%→52%（被 AdamW 的逐参数 elementwise/reduce 稀释，instance 从 1.1 万涨到 4.6 万）；AdamW 状态扫描=内存受限→§6 sharding 动机。
- SQL 过滤技巧见 `results/kern_stats_by_nvtx.sql`（原理：NVTX_EVENTS × CUPTI kernel 时间窗 join）。
- `--nvtx-capture` 在本版本 2026.4.1 已改名为 `--capture-range`（`-c nvtx`）且语法变复杂，改用 SQL 时间窗法更省事。
