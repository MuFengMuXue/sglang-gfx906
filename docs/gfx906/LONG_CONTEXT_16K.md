# gfx906 长上下文（>4k）问题记录

最后更新：2026-10-24  
代码基线：`ced4987d3b`（工作区干净，无未提交实验代码）  
服务状态：已停止，GPU 显存已释放

## 已验证可用：4k 配置

```fish
set -gx SGLANG_MAMBA_CONV_DTYPE float16
cd /home/neko/Projects/sglang-gfx906

.venv/bin/python -m sglang.launch_server \
  --model-path /home/neko/Models/Qwen3.8-27B-AWQ-INT4 \
  --language-model-only --dtype float16 --tp 2 \
  --attention-backend gfx906_fa \
  --kv-cache-dtype gfx906_q8_0 \
  --cuda-graph-backend-decode=full \
  --cuda-graph-bs-decode 1 2 \
  --cuda-graph-backend-prefill=disabled \
  --context-length 4096 \
  --mem-fraction-static 0.9 \
  --max-running-requests 16 \
  --port 30000 --host 127.0.0.1
```

结果：

| 场景 | 结果 |
|---|---|
| 短上下文 decode | 33.6–33.7 tok/s |
| 2431 / 3451 / 3791 tokens | 正确 |
| 两路并发（2402 + 2062） | 正确 |
| 1..20 / 中文诗 | 正确 |

## 未解决：16k context / >4k prompt

### 现象

- 正常 allocator（caching allocator）：
  - 3451 / 3791 tokens：正确
  - **4131 / 4471 / 4811 / 5151 / 5627 tokens：输出连续 `!`**
  - 阈值接近 **4096**（但 4k 配置下 3791 仍正常）
- 16k / max_running=1 / 5627 prompt：
  - 第一个 2048 prefill chunk 后触发 device assert
  - `HSA_STATUS_ERROR_EXCEPTION`
  - kernel: `at::native::_assert_async_cuda_kernel<bool>`
  - 进程退出后 GPU 显存可释放，不需要重启机器
- `PYTORCH_NO_CUDA_MEMORY_CACHING=1` + decode graph：
  - graph capture 阶段失败：
    `hipErrorStreamCaptureInvalidated`
  - 说明 no-cache 与 CUDA graph 不兼容

### 已排除

这些对照都失败，说明不是单一后端问题：

- **Q8 KV**：fp16 KV 同样 `!`
- **FA backend**：Triton backend 同样 `!`
- **chunked prefill**：`--chunked-prefill-size -1` 单 chunk 同样 `!`
- **radix/mamba tracking**：`--disable-radix-cache` + `no_buffer` 同样 `!`
- **decode/prefill graph**：全部 disabled 同样 `!`
- **GDN h_track**：`TRACK_DEBUG` 没有输出，该请求上 tracking 没有启用
- **GDN `h` / `v_new`**：zero-init 后仍失败
- **max_running 调参**：1/4/16 结果不稳定，属于 allocator 依赖，不是可靠修复

### 唯一可靠的诊断结果

```text
PYTORCH_NO_CUDA_MEMORY_CACHING=1
+ no decode graph
+ 16k context
+ 5627 prompt
```

可以完整跑通（2048 + 2048 + 1531 三个 chunk，输出正确）。  
因此根因仍然是：**某个 buffer 读到了回收页里的旧数据/NaN**，no-cache 只是改变了分配行为，把它掩盖了。

### 可疑点

- FLA/GDN 临时 buffer：`Aqk`、`Akkd`、`w`、`u`、`kg`、`o` 等
- sampler/logits 侧：`sampling_mask_max_tokens=4096`，阈值也正好在 4096 附近
- 模型 forward 内的某个 `torch.empty` 尾部未写满
- 不是 Q8/FA/Triton 单独导致；是共同路径

## 当前可用选择

### A. 回到 4k 配置（推荐）

稳定、33.6 tok/s、长上下文 2.4k–3.8k 已验证。  
代价是 context 限制在 4096。

### B. 16k + no-cache + no graph

```fish
set -gx SGLANG_MAMBA_CONV_DTYPE float16
set -gx PYTORCH_NO_CUDA_MEMORY_CACHING 1
```

其余同 16k 配置，但：

```text
--cuda-graph-backend-decode=disabled
--cuda-graph-backend-prefill=disabled
--context-length 16384
--mem-fraction-static 0.85
--max-running-requests 4
```

已知 5627 prompt 正确；**decode 速度会明显低于 33.6 tok/s，尚未正式测吞吐**。

### C. 继续定位 buffer

建议顺序：

1. 对 FLA `chunk_intra` 里的 `Aqk/Akkd/w/u/kg` 做 zero-init，逐个 A/B
2. 检查 `chunk_fwd_o` 的输出和 `v_new` 尾部读取
3. 检查 sampler 的 `sampling_mask_max_tokens=4096` 是否在 prompt 长度路径上被误用
4. 用 no-cache 作为参考，对比 `torch.empty` 分配的差异
5. 找到后只保留一个最小 zero-init/边界修复，不要用 no-cache 作为生产方案

## 相关日志

- `/tmp/sglang_16k.log`：max_running=1 的 device assert
- `/tmp/sglang_16k_trackzero.log`：h_track zero-init 后仍 `!`
- `/tmp/sglang_16k_nocache.log`：no-cache + no graph，5627 正确
- `/tmp/sglang_16k_nocache_graph.log`：no-cache + graph capture 失败
- `/tmp/sglang_16k_triton.log`、`/tmp/sglang_16k_fp16kv.log`：后端对照
