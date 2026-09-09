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

## §4.1.1 pytorch_attention 基准（B=8, 单头, fp32, 10 warmup + 100 steps）

跑法：`uv run python answer/benchmark.py`（`__main__` 临时入口 `bench_pytorch_attention(batch_size=8)`；Q/K/V 形状 (8, N, d)，attention 用 `cs336_basics.model.scaled_dot_product_attention`，backward 由 `out.sum().backward()` 驱动）。OOM 配置已被 except 捕获并标注。

**forward 平均耗时 (ms)**

| d \ N | 256 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 | 0.54 | 1.54 | 18.7 | 1363.8 | OOM |
| 32 | 0.86 | 1.62 | 18.7 | 1348.6 | OOM |
| 64 | 1.11 | 1.76 | 18.5 | 1337.3 | OOM |
| 128 | 1.65 | 2.15 | 21.0 | OOM | OOM |

**backward 平均耗时 (ms)**

| d \ N | 256 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 | 1.31 | 4.70 | 48.6 | 3757.5 | OOM |
| 32 | 2.34 | 5.46 | 50.1 | 3685.1 | OOM |
| 64 | 3.23 | 6.76 | 54.4 | 3655.9 | OOM |
| 128 | 8.77 | 15.0 | 71.0 | OOM | OOM |

**backward 前驻留显存 (MB)**（`torch.cuda.memory_allocated()`，stdev≈0）

| d \ N | 256 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 | 21.8 | 86.3 | 1099.6 | 4329.6 | OOM |
| 32 | 22.3 | 88.4 | 1108.0 | 4346.3 | OOM |
| 64 | 23.4 | 92.6 | 1124.7 | 4379.9 | OOM |
| 128 | 25.5 | 101.0 | 1158.3 | OOM | OOM |

- 显存只随 N 变化、与 d 无关（S/P 形状 (8,N,N)）；N 翻倍显存 ×4 → O(N²) 特征。N=8192 实测 4.33-4.38GB ≈ 2×(4B×8×8192²) = 4.29GB，与公式吻合。
- OOM 边界：N=16384 全爆（单个 S 即 8.0 GiB ≈ 整卡）；N=8192 在 d=128 时爆（backward 需额外 dP/dS ≈ 4.3GB，峰值 ~8.6GB）。
- N=4096→8192 forward 耗时 ×73（18.7→1363ms），远超 N² 理论 ×4，且 d=16/32/64 耗时几乎相同 → 8192 时非算力受限（大块中间张量分配 / HBM 写回开销主导？）。

> TODO：分析段待补（显存账 + 两段讨论），引导问题见下——写完贴给我 review。
> 1. 在最小 OOM 配置上算显存账：S、P 各占多少？（4·B·N² 字节）backward 为什么还需要更多？
> 2. "backward 省下的内存"随序列长度怎么变化？（N 翻倍 → ？）
> 3. 你会用什么方法消除这个内存成本？

## §4.2(a) torch.compile attention 对比（同 §4.1.1 配置）

跑法：`bench_pytorch_attention(batch_size=8, compiled=True)`（`torch.compile(model.scaled_dot_product_attention)`，循环外编译一次；backward 不单独编译）。每配置第一次调用含编译（~20s），被 warmup 消化。

**forward 平均耗时 (ms)（compile，原始版括号内）**

| d \ N | 256 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 | 0.48 (0.54) | 1.08 (1.54) | 10.4 (18.7) | 48.1 (1363.8) | OOM |
| 32 | 1.03 (0.86) | 1.41 (1.62) | 10.8 (18.7) | 1368.8 (1348.6) | OOM |
| 64 | 1.16 (1.11) | 17.7 (1.76) | 337.3 (18.5) | 394.9 (1337.3) | OOM |
| 128 | 1.53 (1.65) | 24.4 (2.15) | 514.7 (21.0) | 1116.9 (OOM) | OOM |

**backward 平均耗时 (ms)（compile，原始版括号内）**

| d \ N | 256 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 | 1.13 (1.31) | 3.56 (4.70) | 26.4 (48.6) | 10139 (3757.5) | OOM |
| 32 | 3.49 (2.34) | 5.90 (5.46) | 32.4 (50.1) | 401 (3685.1) | OOM |
| 64 | 3.43 (3.23) | 180.5 (6.76) | 2859.9 (54.4) | 4783.8 (3655.9) | OOM |
| 128 | 8.64 (8.77) | 196.2 (15.0) | 2102.8 (71.0) | 5114.4 (OOM) | OOM |

**backward 前驻留显存 (MB)**（compile，与原始版几乎相同，差 <0.5%）

| d \ N | 256 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 | 21.8 | 86.4 | 1099.7 | 4329.8 | OOM |
| 32 | 22.3 | 88.5 | 1108.1 | 4346.6 | OOM |
| 64 | 33.9±20.6* | 92.7 | 1124.9 | 4380.2 | OOM |
| 128 | 25.5 | 101.1 | 1158.4 | 4447.3 | OOM |

\* d=64, N=256 的 gmem stdev 异常（33.9±20.6MB），其余配置 stdev≈0。

- 融合生效的证据：d=16, N=8192 的 forward 从 1364ms → 48ms（**×28 加速**），gmem 却几乎不变（inductor 融合 kernel 但不做 backward 重算，grad_fn 保存的 S/P 照旧）→ OOM 边界基本没变（16384 仍全爆）。
- 三个异常模式（原因待分析）：
  1. d=16, N=8192：forward 大加速，backward 反而慢 2.7×（10.1s vs 3.76s）
  2. d=64/128 的中等序列（1024/4096）：整体慢 10~50×（如 d=128, N=4096 forward 515ms vs 21ms）
  3. d=32, N=8192：backward 加速 9×（401ms vs 3685ms），forward 却无提升（1.37s ≈ 1.35s）
- 唯一 OOM 边界改善：d=128, N=8192 从 OOM → 可跑（4447MB）。
- 两个 inductor warning：TF32 未启用（fp32 matmul 用 tensor core 需 `set_float32_matmul_precision('high')`，为同精度对比未开）；"Not enough SMs to use max_autotune_gemm mode"（3070 Ti 46 SM，gemm autotune 被跳过 → 默认 triton matmul 模板，可能是异常 2 的原因）。

> TODO：分析段待补，引导问题见下——写完贴给我 review。
> 1. 为什么 d=16, N=8192 的 forward 能 ×28 加速而 backward 反而变慢？inductor 在 forward 里融合了什么、backward 里生成什么？（提示：想想 S/P 物化 vs 融合、softmax backward 的 kernel）
> 2. 为什么 d=64/128 的 1024/4096 严重倒退？（提示：SM 数、max_autotune 被跳过、triton 默认分块 vs cuBLAS）
> 3. compile 前后 gmem 几乎不变，说明了什么？（对照 §4.2.2 的目标）

**§4.2(a) 分析（nsys 证据，`results/compile_8192.nsys-rep`，d=16, N=8192）**：

- backward 10s 的构成：`triton_poi_fused_scatter_zeros_0` 47.3%（22 次，avg 468ms）+ `triton_poi_fused_add_div_2` 45.7%（22 次，avg 452ms）合计 93%；sgemm（cuBLAS）仅 3.3%。慢的不是 matmul。
- 2.1GB 张量的 elementwise 理论带宽极限 ~10ms（4.3GB/448GB/s），实测 ~450ms = **45× 带宽浪费** → inductor 自动生成的 pointwise kernel 并行配置极差（与 "Not enough SMs to use max_autotune_gemm" warning 一致：未调优即用默认值）。
- forward 快的原因：matmul 走 cuBLAS（`ampere_sgemm_128x128_*`，8 步 × forward 2 个 matmul = 16 次），fusion 消除了中间矩阵多次 HBM 往返。
- 候选语义（未完全钉死）：`scatter_zeros` = softmax 链中 `x − max(x)` 的 max 节点 backward（梯度 scatter 到每行最大值位置 + 输出置零，产物是 ~2.1GB 近全零张量）；`add_div` = dS 的 `÷√d_k` 缩放（+ 融合的加法）。疑点：每步出现 ~2.75 次（max backward 理论每步 1 次）。验证手段（未做）：`TORCH_LOGS=output_code` 看生成源码，或将 softmax 换成 `torch.softmax` 对照。
- gmem 不变的解释：inductor 只融合 kernel、不做 backward 重算——grad_fn 为 backward 保存的 S/P 等中间张量照旧落在 HBM。compile 不是 FA2，省不了这份显存。

## §4.2(b) torch.compile 全模型端到端对比（small, ctx=512, batch=8, fp32, 5 warmup + 10 steps）

跑法：`uv run python answer/benchmark.py --mode <m> [--compile]`（`torch.compile` 包整个 `BasicsTransformerLM`，编译一次三模式共用；optimizer 的 `AdamW.step()` 未编译）

| 模式 | vanilla (s) | compiled (s) | 加速比 |
|---|---|---|---|
| forward | 0.1983 ± 0.0022 | 0.1523 ± 0.0021 | 1.30× |
| backward | 7.855 ± 2.221 | 0.477 ± 0.011 | **16.5×** |
| optimizer 全步 | 13.039 ± 0.636 | 4.836 ± 0.825 | 2.70× |

- **backward 是最大收益点**：16.5× 加速，且 stdev 从 28% 降到 2.4%（不仅快、还稳定）。对照 §2.1.3 记录：small **batch=4** 时 vanilla backward 仅 0.31s → **batch=8 变 7.86s（×25，远超线性）** → 强烈提示 batch=8 时 backward 峰值显存逼近 8GB 边缘，触发 WDDM 换页/碎片抖动（stdev 2.2s 是证据）；compile 融合减少中间激活后恢复稳定（0.48s）。
- **optimizer 模式只快 2.7×**：该模式耗时大头是 `AdamW.step()`（≈5s，未被 compile 覆盖，仍是逐参数 elementwise/reduce）；compile 省下的主要是 backward 部分（7.9s→0.48s），forward 端仅 0.2s。拆分：vanilla 13.04 ≈ fwd 0.20 + bwd 7.86 + step ~5.0；compiled 4.84 ≈ fwd 0.15 + bwd 0.48 + step ~4.2。
- **forward 只快 23%**：全模型融合收益被 embedding / RMSNorm / softmax / cross-entropy 等摊薄，远小于 §4.2(a) 单 attention 层的收益（融合只集中影响 attention 内部）。

**§4.2(b) 显存验证（batch=8, backward 峰值，`reset_peak_memory_stats` 后测 `max_memory_allocated`）**：

| | vanilla | compiled |
|---|---|---|
| backward 峰值 | ~8,235MB（7.67GB，每步 8232~8235MB 波动） | ~6,698MB（6.24GB，稳定） |

- vanilla 峰值 7.67GB > 实际可用（8GB − 桌面 ~0.4GB ≈ 7.6GB）→ 溢出到系统内存（WDDM 换页）→ 6.5~7.9s + 每步峰值随机波动；compile 峰值 6.24GB、留 ~1.4GB 余量 → 0.48~0.58s 稳定。
- **结论**：§4.2(b) 的 16.5× backward 加速主因是 compile 把 backward 峰值压低 ~1.43GB、消除换页，而非单纯 kernel 更快。forward 后驻留 6.86GB（两版相同，compile 不省 saved activations）。
- 与 §4.2(a) 矛盾点：d=16, N=8192 时 compile backward 反而慢 2.7×——那里 inductor 生成了 45× 带宽浪费的 pointwise kernel（scatter_zeros/add_div）掩盖了收益；全模型 small 场景未触发这类烂 kernel，纯赚峰值下降的好处。

> TODO：分析段待补，引导问题见下——写完贴给我 review。
> 1. 为什么 vanilla backward 从 batch=4 的 0.31s 到 batch=8 的 7.86s（×25）？怎么验证是显存边缘效应而非算力？（提示：`torch.cuda.memory_allocated()`/`max_memory_allocated()` 在 backward 前后、或 nvidia-smi 观察）
> 2. optimizer 模式为什么只快 2.7×？哪个部分没被 compile 覆盖、占了总时间大头？（对照 §2.1.4 的 matmul 占比结论）
> 3. 本节的 backward 加速（16.5×）和 §4.2(a) 里 d=16, N=8192 的 backward 变慢（2.7×）矛盾吗？为什么全模型场景反而稳定受益？
