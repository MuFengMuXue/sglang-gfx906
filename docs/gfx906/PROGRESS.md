# SGLang on gfx906 (MI50 / Radeon Pro VII) — 进度与下一步

最后更新：2026-10-24
状态：**能跑通且已提速**。Qwen3.8-27B-AWQ-INT4 在 2× MI50 上以 TP=2 启动，
输出连贯中文；CUDA graph + exllama W4A16 后 decode 约 **22.8 tok/s**。
服务当前在 `127.0.0.1:30000` 运行。

---

## 1. 环境基线

| 项目 | 值 |
|---|---|
| GPU | 2 × AMD Radeon Pro VII，16 GB，`gfx906` |
| CPU / RAM | Xeon E5-2698B v3，32 线程 / 31 GB（zram 31 GB swap） |
| 系统 | Arch Linux，GCC 16.2.1（系统默认），有 gcc-15 |
| ROCm | `/home/neko/ROCm-7.14`，`7.14.60850`（TheRock gfx906 构建） |
| LD_LIBRARY_PATH | `/home/neko/ROCm-7.14/lib:/home/neko/ROCm-7.14/lib64:/home/neko/ROCm-7.14/hip/lib:/home/neko/ROCm-7.14/opencl/lib` |
| Python | 3.12.15（uv 管理） |
| torch | `2.14.1+gfx906.20261018172427`（预编译 wheel） |
| triton | `3.8.0+gitc01b6774`（本地构建，`/home/neko/Projects/triton-3.8`） |
| SGLang | 上游 main `2a46e32fbc`（2026-10-23），editable 安装（`sglang==0.0.0`） |
| venv | `/home/neko/Projects/sglang-gfx906/.venv` |
| 模型 | `/home/neko/Models/Qwen3.8-27B-AWQ-INT4`（20 GB） |

模型要点（`config.json`）：

- 架构 `Qwen3_5ForConditionalGeneration`（`model_type: qwen3_5`），文本部分 64 层
- hidden 5120，24 Q / 4 KV heads，`head_dim 256`，`partial_rotary_factor 0.25`
- 每 4 层一个 full-attention，其余为 GDN 线性注意力（hybrid）
- 量化：compressed-tensors `pack-quantized`，W4A16，group_size 32，asymmetric，`zp_dtype int8`
- 视觉塔与 MTP 层也在 checkpoint 中，但当前不加载

---

## 2. 启动方式（已验证）

```fish
cd /home/neko/Projects/sglang-gfx906
source .venv/bin/activate.fish
python -m sglang.launch_server \
  --model-path /home/neko/Models/Qwen3.8-27B-AWQ-INT4 \
  --language-model-only --dtype float16 --tp 2 \
  --attention-backend triton \
  --cuda-graph-backend-decode=full \
  --cuda-graph-backend-prefill=disabled \
  --context-length 4096 --mem-fraction-static 0.9 \
  --max-running-requests 16 \
  --port 30000 --host 127.0.0.1
```

- 必须 `--tp 2`：文本权重约 19 GB，单卡 16 GB 放不下。
- 必须 `--dtype float16`：gfx906 没有 bf16 硬件，checkpoint 是 bf16 原生。
- 必须 `--language-model-only`：跳过视觉塔（vision 路径需要 bf16）。
- 权重加载后每卡约 `9.3 GB`；开启 graph 后运行中每卡约 `15.3 / 16 GiB`。
- 健康检查：`curl http://127.0.0.1:30000/health` → `200`。
- `--cuda-graph-backend-decode=full` 已由用户验证可 capture（bs 1/2/4/5，约 4.3 s）。

请求时如果要关闭 thinking（模板默认开启）：

```json
"chat_template_kwargs": {"enable_thinking": false}
```

### 已验证输出

```text
问：1+1等于几？只回答数字。
答：2

问：写一首四行的中文小诗，主题是秋天。
答：
金风昨夜扫梧桐，
叶落无声染霜红。
雁字排云归远浦，
一江秋水向东空。
```

### 当前性能

| 场景 | 结果 |
|---|---|
| 服务日志 decode（graph + exllama，steady） | **22.8 token/s** |
| 223 output tokens（含 prefill/HTTP） | 11.0 s，约 20 token/s 墙钟 |
| 启动时间（磁盘热） | ~42 s（含 graph capture 约 4.3 s） |
| 每卡显存 | 权重 9.3 GB，运行时 15.3 / 16 GiB |

W4A16 单层微基准（M=1，真实 checkpoint shape，exllama vs 原 Triton）：

| shape (N×K) | exllama | Triton | 加速 |
|---|---:|---:|---:|
| 17408×5120 (gate/up) | 141 µs | 2290 µs | 16.2× |
| 5120×17408 (down) | 91 µs | 1564 µs | 17.2× |
| 8192×5120 (qkv) | 61 µs | 786 µs | 12.8× |
| 10240×5120 (GDN qkv) | 64 µs | 787 µs | 12.3× |
| 6144×5120 (GDN z) | 63 µs | 457 µs | 7.3× |
| 5120×6144 (GDN out) | 65 µs | 546 µs | 8.5× |

---

## 3. 本仓库的改动（未提交）

| 文件 | 改动 |
|---|---|
| `python/sglang/srt/server_args.py` | `LANGUAGE_MODEL_ONLY_ARCHITECTURES` 加入 `Qwen3_5ForConditionalGeneration`，允许 text-only 启动 |
| `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` | 新增 `_on_gfx906()`；gfx906 上 WNA16 优先选择 `CompressedTensorsWNA16Exllama`，扩展不可用时回退 `CompressedTensorsWNA16Triton`；新增 `gfx906_plain_prefixes`（预留 checkpoint 驱动 plain 层检测） |
| `.../compressed_tensors/schemes/__init__.py` | 导出两个新 scheme 和 `gfx906_gptq_available` |
| `.../compressed_tensors/schemes/compressed_tensors_wna16_triton.py` | **新增文件**：从 `sglang-mi50` 搬来的纯 Triton dequant+GEMM W4A16 kernel；基类改为 `CompressedTensorsLinearScheme`；**修复 zero-point `+1` bug** |
| `.../compressed_tensors/schemes/compressed_tensors_wna16_exllama.py` | **新增文件**：vllm-gfx906 的 exllama GPTQ W4A16 kernel 接入；权重/scale/zero-point 布局转换 + `gptq_gemm` |

### 关键修复：W4A16 zero-point

原 mi50 kernel 写的是：

```python
zero = ((qz >> shift_n) & 0xF).to(tl.float32) + 1.0
```

但 compressed-tensors 的 `pack-quantized` 语义是：

```text
weight = (w_q - zero_point) * scale
```

zero point 直接存储（本 checkpoint 实际值 1..14），不能 `+1`。
用 `compressed_tensors.compressors.pack_quantized.helpers.unpack_from_int32`
+ `compressed_tensors.quantization.lifecycle.forward.dequantize` 做参考：

- 修复前：单层 GEMM max diff ≈ **0.54**（输出退化成 `!!!!...`）
- 修复后：max diff ≈ **0.001**（fp16 GEMM 舍入），输出正常

**如果以后重新从 `sglang-mi50` 拷贝这个文件，务必重新去掉 `+1`。**

### W4A16 exllama 移植（decode 提速关键）

Triton W4A16 在 M=1 时利用率很低（见上表），而 vllm-gfx906 的 dense 路径
用的是 exllama GPTQ kernel。当前实现：

- 复用 vllm-gfx906 已经编好的 stable-ABI 扩展：
  `/home/neko/Projects/vllm-gfx906-performance/vllm/_C_stable_libtorch.abi3.so`
  - 虽然是 torch 2.13 编译，但 stable ABI 可在 torch 2.14 下加载
  - 可用环境变量 `SGLANG_GFX906_GPTQ_LIB` 指向别的 build
- 布局转换（与 vLLM `ExllamaLinearKernel.process_weights_after_loading` 一致）：
  - `weight_packed [N, K/8] -> [K/8, N]`，然后 `gptq_shuffle(wq, 4)`
  - `weight_scale [N, G] -> [G, N]`，转 fp16
  - `weight_zero_point [N/8, G] -> [G, N/8]`
- 调用：`gptq_gemm(x_fp16, wq, wzp, ws, use_exllama=True, use_v2_format=True, bit=4)`
  - `use_v2_format=True` 正是 `(w - zero_point) * scale`，与上面的 zero-point 语义一致
- 数值：M=1/4 与 compressed-tensors 参考 maxdiff ≈ 0.003；M=33 走
  reconstruct+hipBLAS，maxdiff ≈ 0.09（相对 5.6，约 1.6%）
- 效果：decode 4.5 → **22.8 tok/s**

**TODO**：目前依赖 vLLM 仓库里的 `.so`，只是为了先跑通。下一步应把
`csrc/quantization/gptq/`（`q_gemm.cu`、`q_gemm_m1_maxilp.cu`、`qdq_*.cuh` 等）
搬进本仓库并重编成独立扩展，去掉这个外部路径依赖。扩展不可用时 scheme 会
自动回退到 Triton 版本（正确但慢）。

### venv 内的 shims（不在 git 里）

SGLang 在 HIP 上会无条件导入 `sgl_kernel` 和 `aiter`，但 gfx906 没有对应
wheel。做法：

```bash
SP=/home/neko/Projects/sglang-gfx906/.venv/lib/python3.12/site-packages
cp -r /home/neko/Projects/sglang-mi50/shims/sgl_kernel \
      /home/neko/Projects/sglang-mi50/shims/aiter "$SP/"
```

- `sgl_kernel` shim：用 torch 实现 `silu_and_mul` / `rmsnorm` /
  `fused_add_rmsnorm` 等少量算子，其它调用直接抛
  `NotImplementedError`（保证失败是显式的）。
- `aiter` shim：属性访问返回 stub；**额外修改了
  `aiter/__init__.py`**，加了一个 `sys.meta_path` finder，让未实现的
  `aiter.ops.*` 子模块自动生成 stub 包（否则
  `import aiter.ops.triton.gemm.fused` 会直接 `ModuleNotFoundError`）。

### 已安装的主要依赖

`orjson fastapi uvicorn pydantic psutil prometheus-client tqdm requests
aiohttp packaging pyzmq msgspec partial-json-parser pillow regex scipy
sentencepiece tokenizers transformers==5.19.0 safetensors huggingface_hub
einops easydict distro python-multipart setproctitle pybase64 xxhash
zstandard soundfile IPython compressed-tensors==0.18.0 datasets gguf
mistral-common modelscope openai outlines interegular tiktoken timm
torch-memory-saver build ninja numba uvloop watchfiles xgrammar llguidance`

**故意没装**：`sglang-kernel`、`sgl-kernel`、`aiter`、`flashinfer`、
`flash-attn-4`、`sgl-deep-*`、`sgl-flash-mla`、`tilelang`、`cuda-python`、
`nvshmem` 等 CUDA/新架构专用包。

---

## 4. 当前限制

- **text-only**：视觉塔未加载；图片/视频输入不可用。
- **fp16 only**：bf16 模型必须用 fp16，激活可能有溢出风险；目前输出正常，但
  长上下文/大 batch 需再验证。
- **prefill graph 关闭**：只开了 decode graph；prefill 仍 eager，未测长 prompt。
- **W4A16 依赖外部 `.so`**：暂时用 vllm-gfx906 的 `_C_stable_libtorch.abi3.so`，
  还没有自己重编；扩展缺失时回退 Triton（慢 7–17×）。
- **无 MTP / 投机解码**：checkpoint 带 MTP 层，但未启用。
- **单请求 / 小并发**：`--max-running-requests 16`，尚未做多请求和长上下文压测。
- **sgl_kernel / aiter 是 stub**：任何走到真实 CUDA/AITER 算子的路径都会显式报错。
- `amdsmi` 未安装，启动时有一条 warning，不影响运行。

---

## 5. 下一步（按优先级）

### P0 — ✅ CUDA graph + W4A16 exllama（已完成）

- `--cuda-graph-backend-decode=full` 正常 capture（bs 1/2/4/5）。
- W4A16 从 Triton 换成 exllama GPTQ kernel。
- decode：4.5 → **22.8 tok/s**。

### P1 — 把 vllm-gfx906 的 Q8 FA backend 搬进 SGLang

参考仓库：`/home/neko/Projects/vllm-gfx906-performance`

可复用部分：

- `csrc/gfx906_fa/`（4 个 .cpp/.cu + `kernel/*.cuh`）：只依赖 torch + HIP，
  纯 pybind11，可直接搬。
- `vllm/gfx906_fa/gfx906_fa_paged.py`：只有一处 `vllm.gfx906_fa.tuning`
  依赖，改 import 即可。
- `vllm/gfx906_fa/tuning.py`：只有 `vllm.logger` 依赖。

需要重写：`vllm/gfx906_fa/gfx906_fa_backend.py`（vLLM v1 的
`AttentionBackend/Impl/MetadataBuilder`，不能直接抄）。

SGLang 接入点：

1. `python/sglang/srt/layers/attention/attention_registry.py`：
   `@register_attention_backend("gfx906_fa")`
2. `python/sglang/srt/arg_groups/choices.py`：加入 `ATTENTION_BACKEND_CHOICES`
3. 新 backend 实现 `init_forward_metadata` / `forward_decode` / `forward_extend`
   - `save_kv_cache` 走 `token_to_kv_pool.set_kv_buffer`
   - 用 `forward_batch.req_pool_indices` + `req_to_token_pool.req_to_token`
     构造 token 级 `block_table`（`block_size=1`）与 `seq_lens`
   - 调 `forward_paged(...)`，输出 `[num_tokens, Hq*D]`
4. 先跑 legacy gather 路径（fp16 KV，不改 SGLang KV pool）；
   二期再做 Q8 K side buffer + `reshape_and_cache_q8` + `forward_paged_direct`
   （注意 direct 路径目前硬编码 `block_size == 16`，需适配 token 级）。

构建注意：现有 `.so` 是对 torch 2.13 编的，需要用当前 torch 2.14.1 重编；
编译 flags 见上游 `CMakeLists.txt`：

```text
-DGGML_USE_HIP -DGGML_HIP_GFX906 -DFLASH_ATTN_AVAILABLE
-D__HIP_PLATFORM_AMD__ -U__HIP_NO_HALF_OPERATORS__ -U__HIP_NO_HALF_CONVERSIONS__
```

### P2 — fp16 GEMV / down_proj / elementwise

W4A16 已经是 exllama，剩余可优化项（参考 vLLM 的 34 ms/step 预算）：

- `lm_head` fp16 GEMV 约 3.2 ms/step，已在 HBM floor，基本没空间
- MLP `down_proj` 是 fp16，SGLang 当前走 `triton_matmul`；vLLM 用
  `dense_gemv` KCHUNK=1024 RPT=2 省 ~0.57 ms/step
- elementwise / residual copy 的融合（vLLM 靠 inductor 融合省了 2–4 ms）

### P3 — 功能补齐

- MTP 层（`mtp.*`）与 EAGLE/投机解码
- 视觉塔（需要 bf16 或 fp16 clamp 方案）
- 长上下文（>4k）、chunked prefill、prefix cache 验证
- 多请求/并发稳定性

### P4 — 工程化

- 把 `sgl_kernel` / `aiter` shims 放进仓库（例如 `tools/gfx906_shims/`）并提供安装脚本
- 固化启动脚本、benchmark 脚本、smoke test
- 评估是否把 gfx906 改动整理成可维护的 patch 系列

---

## 6. 参考

- 上游 SGLang：`/home/neko/Projects/sglang-gfx906`（本仓库）
- mi50 补丁 / shim / 旧方案：`/home/neko/Projects/sglang-mi50`
- gfx906 已验证的 vLLM（Q8 FA / GEMV / MLA）：`/home/neko/Projects/vllm-gfx906-performance`
  - GPTQ 扩展：`vllm/_C_stable_libtorch.abi3.so`
  - W4A16 参考：`csrc/quantization/gptq/`、`vllm/model_executor/kernels/linear/mixed_precision/exllama.py`
  - FA kernel：`csrc/gfx906_fa/`、`vllm/gfx906_fa/`
- 模型：`/home/neko/Models/Qwen3.8-27B-AWQ-INT4`
- 启动日志：`/tmp/sglang_gptq.log`
