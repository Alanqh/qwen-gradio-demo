# Qwen + Gradio 本地对话服务

在魔塔 DSW GPU 实例上部署 Qwen 系列大模型的 Gradio 对话界面。支持两种部署方式：

- **方案 A（默认）**：`transformers` 直载，单进程，适合开发调试
- **方案 B**：`vLLM` 独立服务 + 轻量前端，适合高并发，见 §8

项目包含环境自检脚本、可运行应用与完整配置说明。

---

## 目录

- [1. 怎么看设备信息](#1-怎么看设备信息)
- [2. 技术栈与框架选型](#2-技术栈与框架选型)
- [3. 环境配置](#3-环境配置)
- [4. 参数说明](#4-参数说明)
- [5. 运行方式与预期结果](#5-运行方式与预期结果)
- [6. 显存估算与模型选型](#6-显存估算与模型选型)
- [7. 常见问题排查](#7-常见问题排查)
- [8. 用 vLLM 部署（高并发方案）](#8-用-vllm-部署高并发方案)
- [9. 项目结构与代码要点](#9-项目结构与代码要点)

---

## 1. 怎么看设备信息

### 1.1 从驱动侧看：`nvidia-smi`

```bash
nvidia-smi
```

顶层信息栏三个字段最容易看错：

| 字段 | 含义 | 注意 |
|---|---|---|
| `NVIDIA-SMI 550.54.15` | 管理工具自身版本 | 与驱动同版号是常见情况 |
| `Driver Version: 550.54.15` | GPU 驱动版本 | 由宿主机决定，容器内无法改 |
| `CUDA Version: 13.0` | **驱动支持的 CUDA 上限，不是已安装版本** | 这是"天花板"不是"地板" |

GPU 明细行：

| 字段 | 含义 | 空闲特征 |
|---|---|---|
| `GPU Name` | 卡型号，如 NVIDIA A10 | — |
| `Fan / Temp` | 风扇转速 / 温度 | 服务器卡无风扇显示 0%；30°C 说明很闲 |
| `Perf` | 性能状态 | `P8` = 省电待机，`P0` = 满血工作 |
| `Pwr:Usage/Cap` | 实时功耗 / 功耗墙 | 远低于上限 = 没在干活 |
| `Memory-Usage` | 已用 / 总显存 | 0MiB = 显存完全空闲 |
| `GPU-Util` | 计算利用率 | 0% = 无计算任务 |
| `Volatile Uncorr. ECC` | 不可修复显存错误计数 | 非 0 需警惕硬件问题 |

只取关键字段的写法：

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv
```

### 1.2 从框架侧看：torch

```bash
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

| 输出 | 含义 |
|---|---|
| `torch.version.cuda` | 这个 torch **编译时**针对的 CUDA 版本 |
| `torch.cuda.is_available()` | torch 能否真正调用 GPU，**才是决定性判据** |
| `torch.cuda.get_device_name(0)` | 实际识别到的卡名 |

### 1.3 兼容性判断规则（核心）

```
torch.version.cuda  ≤  nvidia-smi 顶部显示的 CUDA Version
```

- 成立 → 大概率可用（仍需 `is_available()` 确认为 True）
- 不成立（torch 编译版本高于驱动上限）→ 会出现"装了 GPU 版 torch 但 `is_available()` 为 False"的现象
- 修法是按驱动支持的 CUDA 版本重装对应轮子，而不是让 pip 随便挑

本项目实测环境：

```
Driver 550.54.15 · 驱动 CUDA 上限 13.0 · torch.version.cuda 13.0  →  恰好压线，兼容
GPU NVIDIA A10 · 显存 23028 MiB ≈ 22.5 GB · 功耗上限 150 W · 空闲态 21W / 30°C / 0%
```

### 1.4 一键自检

```bash
python check_env.py
```

输出驱动、GPU、依赖版本、CUDA 可用性、显存占用与缓存盘剩余空间。

---

## 2. 技术栈与框架选型

| 组件 | 版本要求 | 职责 | 缺了会怎样 |
|---|---|---|---|
| **PyTorch** | 与驱动 CUDA 匹配 | 张量计算与 GPU 调度底座 | 什么都跑不了 |
| **transformers** | ≥ 4.40 | 模型定义、权重加载、`generate()` | 无法加载模型 |
| **accelerate** | 任意较新版 | 支撑 `device_map="auto"` 的显存自动切分 | 报错提示需安装 accelerate |
| **gradio** | ≥ 4.0 | Web 对话界面与内网穿透分享 | 无交互界面 |
| **modelscope** | 任意较新版 | 国内直连下载权重，绕开 HF 网络封锁 | 无法下载模型（见 §7.1） |

**为什么用 transformers 而不是 vLLM / Ollama？**
本项目定位是"快速验证链路 + 便于改代码"。transformers 的 `generate()` 参数最直观、改动成本最低。若追求吞吐（并发请求、批量推理），应换 vLLM——完整方案见 §8；若只需开箱即用的命令行对话，Ollama 更省事。

**为什么必须装 accelerate？**
`device_map="auto"` 的实现依赖 accelerate 做设备分配。不装它，`from_pretrained` 会直接报错。单卡场景它看似"没干什么"，但正是它把权重放到了正确的设备上。

---

## 3. 环境配置

### 3.1 安装依赖

```bash
pip install --root-user-action=ignore -q torch transformers gradio accelerate modelscope
```

- `-q` 只压制 INFO 级输出，**WARNING 和 ERROR 依然会打印**。所以"命令跑完只剩一条 root 警告"= 安装成功。
- `--root-user-action=ignore` 用于消除 root 用户安装的警告；在容器里该警告可安全忽略。
- **torch 建议单独装**：如果镜像已预装与驱动匹配的 torch，直接用 pip 从 PyPI 拉默认构建可能覆盖掉它。先跑 §1.4 自检确认现有 torch 正常，就不要动它。

### 3.2 网络出口配置（国内实例必做）

huggingface.co 在国内实例上通常**路由不可达**，典型报错：

```
[Errno 101] Network is unreachable ... huggingface.co/...
```

注意这不是断网——`pip install` 能成功，说明有网，只是**没有通往 huggingface.co 的出口**。两条解法：

**方案 A：ModelScope 通道（推荐，项目默认）**

```bash
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct \
  --local_dir /mnt/workspace/models/Qwen2.5-0.5B-Instruct
```

然后从**本地目录**加载。本地目录不会触发任何网络请求。

**方案 B：HF 镜像**

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

> ⚠️ **关键顺序问题**：`HF_ENDPOINT` 必须在 `import transformers` **之前**设置。`huggingface_hub` 在 import 那一刻就把该变量读进模块常量，之后再改无效且不报错。`app.py` 已把这两行放在所有 import 之前。

### 3.3 缓存目录

```bash
export MODELSCOPE_CACHE=/mnt/workspace/.cache/modelscope
export HF_HOME=/mnt/workspace/.cache/huggingface
```

DSW 上 `/root` 通常是容量有限的高速盘，`/mnt/workspace` 才是持久大盘。不改默认位置，下一个 8B 模型就可能写满系统盘。

### 3.4 环境变量速查

| 变量 | 默认值 | 作用 |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | 模型 id 或本地目录路径 |
| `MAX_NEW_TOKENS` | `512` | 单次回复最大生成长度 |
| `GRADIO_SHARE` | `0` | `1` 生成公网分享链接（需 frpc，见 §7.6），`0` 仅本地监听 |
| `MODELSCOPE_CACHE` | `/mnt/workspace/.cache/modelscope` | 魔塔权重缓存 |
| `HF_HOME` | `/mnt/workspace/.cache/huggingface` | HF 权重缓存 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HF 镜像地址 |
| `GRADIO_SERVER_PORT` | `7860` | 监听端口 |
| `GRADIO_CACHE_DIR` | `$HF_HOME/gradio` | gradio 自身缓存目录（frpc 就放这里） |

---

## 4. 参数说明

### 4.1 模型加载参数（`from_pretrained`）

| 参数 | 本项目取值 | 作用 | 调整影响 |
|---|---|---|---|
| `dtype` | `torch.float16` | 权重数值精度 | 改 `float32` 显存翻倍；改 `bfloat16` 需 Ampere 及以上（A10 支持）；老版本写 `torch_dtype`，新版已弃用该名 |
| `device_map` | `"auto"` | 自动分配权重到 GPU | 单卡也可写 `"cuda:0"`；多卡自动均衡 |
| `low_cpu_mem_usage` | `True` | 分片加载，降低峰值内存 | 大模型必开 |
| `trust_remote_code` | `True` | 允许执行模型自带代码 | 便捷但有安全成本，仅对可信仓库开启 |

> tokenizer 的 `from_pretrained` **不接受 `device_map`**——它只是文本↔id 的转换器，没有设备概念。

### 4.2 文本生成参数（`generate`）

| 参数 | 取值 | 作用 | 调大 → | 调小 → |
|---|---|---|---|---|
| `max_new_tokens` | `512` | 最多生成多少 token | 回复更长、更慢、更占显存 | 可能被截断 |
| `do_sample` | `True` | 是否采样 | `False` 则贪心解码，回复固定但呆板 | — |
| `temperature` | `0.7` | 随机性 / 发散度 | 更有创意、更易跑偏 | 更稳定、更重复 |
| `top_p` | `0.9` | 核采样，只在累积概率前 90% 的词里选 | 候选更多样 | 更保守 |
| `repetition_penalty` | `1.05` | 抑制重复用词 | 更不易复读，过高会语病 | `1.0` = 不惩罚 |
| `pad_token_id` | `eos_token_id` | 补齐标记 | 不设会有 WARNING | — |

**经验法则**：`temperature` 和 `top_p` **一起调**会互相干扰，通常固定一个、微调另一个。代码生成建议 `temperature` 0.2~0.5，创意写作 0.8~1.0。

### 4.3 对话模板参数（`apply_chat_template`）

| 参数 | 取值 | 作用 |
|---|---|---|
| `tokenize` | `False` | 返回字符串，方便手动 tokenize 并把设备控制掌握在自己手里 |
| `add_generation_prompt` | `True` | 追加"该助手回答了"的提示符，**漏掉模型会接着用户的话往下写** |
| `enable_thinking` | `False` | 仅 Qwen3 支持；关闭思考模式，避免回复里混入大段推理过程 |

`enable_thinking` 用 `try/except TypeError` 兜底，因此同一份代码在 Qwen2.5 和 Qwen3 上都能跑。

### 4.4 界面参数

| 参数 | 说明 |
|---|---|
| `type` | **本项目不显式指定**，由 `_build_messages()` 同时兼容 dict 与 tuple 两种 history 格式 |
| `share=True` | 生成 `*.gradio.live` 公网链接，需运行时下载 frpc（见 §7.6）。**本项目默认关闭**，改为平台端口映射 |
| `server_name="0.0.0.0"` | 监听所有网卡，便于通过实例端口映射访问 |
| `server_port` | 监听端口，默认 7860，由 `GRADIO_SERVER_PORT` 控制 |

---

## 5. 运行方式与预期结果

### 5.1 启动

```bash
python check_env.py     # 先确认设备正常
python app.py           # 启动对话服务
```

### 5.2 预期输出

```
正在加载模型 Qwen/Qwen2.5-0.5B-Instruct ...（首次需下载，体积取决于参数量）
正在从魔塔社区下载 Qwen/Qwen2.5-0.5B-Instruct ...
下载完成：/mnt/workspace/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct
模型加载完成，运行设备：cuda:0
Running on local URL:  http://0.0.0.0:7860
```

> 不会出现 `Running on public URL` —— 公网分享默认关闭（原因见 §7.6）。需要外网访问请走端口映射或 SSH 转发。

### 5.3 界面

浏览器打开输出中的 local URL（或通过 DSW 的端口映射访问），页面包含标题、描述与三个示例问题：

```
你好，请介绍一下自己
帮我写一段Python代码
解释一下什么是深度学习
```

输入问题后模型流式返回回答，多轮对话会带上完整历史。

### 5.4 验收标准

- 输出中出现 `模型加载完成，运行设备：cuda:0`（**不是 `cpu`**）
- 输入"你好"能收到连贯的中文回复
- 另一终端执行 `nvidia-smi`，能看到显存被占用、`GPU-Util` 不再是 0%

---

## 6. 显存估算与模型选型

**FP16 权重占用 ≈ 参数量 × 2 字节**

| 模型规模 | FP16 权重 | A10（22.5 GB）能否跑 | 说明 |
|---|---|---|---|
| 0.5B | ~1 GB | ✅ 轻松 | 默认配置，验证链路用 |
| 1.7B / 4B | ~3.4 / 8 GB | ✅ 舒适 | 日常对话够用 |
| 7B / 8B | ~14 / 16 GB | ✅ 可行 | 需给 KV cache 留 3~5 GB |
| 14B | ~28 GB | ❌ | 需量化（INT8 / INT4） |
| 32B | ~64 GB | ❌ | 需多卡或 4bit 量化 |

**KV cache 是隐形开销**：它随「上下文长度 × 层数」线性增长。权重刚好塞满显存时，第一轮长对话就可能 OOM。所以 7B/8B 是这张卡的舒适上限，14B 必须量化。

---

## 7. 常见问题排查

### 7.1 `[Errno 101] Network is unreachable`

访问 huggingface.co 失败但 pip 正常 → **路由不通，不是断网**。
诊断：

```bash
curl -sI --max-time 8 https://www.modelscope.cn | head -n 1
curl -sI --max-time 8 https://hf-mirror.com | head -n 1
pip config list      # 看清 pip 走的是哪个源
```

解决：走 ModelScope 通道或设 `HF_ENDPOINT`（见 §3.2）。

### 7.2 `CUDA out of memory`

按顺序尝试：

1. 换更小模型或量化版本（`load_in_4bit=True`，需 bitsandbytes）
2. 降 `MAX_NEW_TOKENS`
3. 缩短对话历史（本项目未做历史截断，长对话会持续吃 KV cache）
4. 确认没有多个进程同时占卡：`nvidia-smi`

### 7.3 回复里混入大段推理过程

Qwen3 默认开启思考模式，`skip_special_tokens=True` **只删特殊 token，不会删思考内容**。
处理方式：`apply_chat_template(..., enable_thinking=False)`，或按结束标记切分（本项目两条都做了）。

### 7.4 历史对话串味 / 内容错乱

Gradio 较新版本传给回调的是 `list[dict]`（`{"role": ..., "content": ...}`），旧版是 `list[tuple]`。
用 `for human, assistant in history` 解包 dict 时，Python 迭代出的是**键名**——不报错，但 `human` 变成 `"role"`、`assistant` 变成 `"content"`。这是典型的静默错误。
本项目通过 `_build_messages()` 兼容两种格式来解决。

### 7.5 `torch.cuda.is_available()` 返回 False

按 §1.3 的规则核对 `torch.version.cuda` 与驱动 CUDA 上限；若 torch 编译版本过高，按驱动版本重装对应轮子。

### 7.6 `Could not create share link. Missing file: .../frpc/frpc_linux_amd64_v0.3`

完整报错：

```
Could not create share link. Missing file: /mnt/workspace/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3.
Please check your internet connection...
```

**根因**：`share=True` 依赖 frpc 内网穿透客户端，而这个二进制**不在 gradio 的 pip 包里**——它是运行时才从 `cdn-media.huggingface.co` 下载的（属 COS 加速域名，与 huggingface.co 是不同的域名，因此用 `HF_ENDPOINT` 镜像**无效**）。国内实例到这个域名没有路由，报错信息里的"检查网络连接 / 杀毒软件"是误导性的模板文案。

**方案 A：改用平台端口映射（推荐，零依赖）**

```bash
GRADIO_SHARE=0 python app.py     # 保持默认即可
```

然后在 DSW 控制台把 7860 端口暴露出去，平台会给出一个公网访问地址。这是"用平台能力替代第三方穿透工具"的标准做法。

**方案 B：本地 SSH 端口转发（不需要公网暴露）**

```bash
ssh -L 7860:localhost:7860 <用户名>@<实例地址> -p <端口>
# 再在本地浏览器打开 http://localhost:7860
```

**方案 C：手动放置 frpc 二进制**

需要一台**能访问该 CDN 的机器**（国内常见网络通常也访问不了，需自行确认）：

```bash
# 1) 在能联网的机器上下载
curl -L -O https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_linux_amd64

# 2) 重命名
mv frpc_linux_amd64 frpc_linux_amd64_v0.3

# 3) 上传到实例的缓存目录（目录需提前创建）
mkdir -p /mnt/workspace/.cache/huggingface/gradio/frpc
#   然后用 scp / 平台的文件上传功能放进去

# 4) 【易漏】必须加可执行权限，否则 gradio 仍会认为不可用
chmod +x /mnt/workspace/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3

# 5) 验证
ls -l /mnt/workspace/.cache/huggingface/gradio/frpc/
```

放好之后 `GRADIO_SHARE=1 python app.py` 即可生成 `*.gradio.live` 链接（**有效期 72 小时**，且服务将暴露到公网，注意不要传输敏感数据）。

> `app.py` 内置了预检：请求了 share 但 frpc 不存在时，会打印上述提示并**自动降级为本地监听**，而不是抛异常中断启动——服务可用性优先。

---

## 8. 用 vLLM 部署（高并发方案）

### 8.1 什么时候该上 vLLM

**不是"更好的方案"，而是"不同定位的方案"。**

| 场景 | 建议 |
|---|---|
| 单人调试、验证链路、需要随时改推理代码 | **方案 A**（transformers），改一行就生效 |
| 多人同时使用、需要稳定吞吐、前端要能独立重启 | **方案 B**（vLLM），推理与界面解耦 |
| 只是自己偶尔聊两句 | 方案 A 足矣，vLLM 的复杂度不划算 |

判断依据是**并发量**。方案 A 里每个请求都要独占一次 `generate()`，第二个用户必须等第一个说完；vLLM 的 continuous batching 能把多个请求拼进同一批计算，这才是吞吐差距的来源。

### 8.2 与方案 A 的本质区别

| 维度 | 方案 A | 方案 B |
|---|---|---|
| 进程结构 | 界面 + 模型 + 推理同进程 | 前端与推理服务分离，HTTP 通信 |
| 显存占用 | 整个 Gradio 进程独占 | 仅推理服务占，前端几乎不占 |
| KV cache 管理 | 朴素分配，随上下文线性增长 | PagedAttention 分页管理，碎片极少 |
| 并发能力 | 请求排队，串行处理 | continuous batching，动态拼批 |
| 对外接口 | 只有网页 | **OpenAI 兼容 API**，任何客户端可接 |
| 流式输出 | 需自行实现 Streamer | 原生支持，前端直接消费 |

### 8.3 部署步骤

**第一步：用独立虚拟环境安装（重要，原因见 §8.6）**

```bash
python -m venv /mnt/workspace/venv-vllm
source /mnt/workspace/venv-vllm/bin/activate
pip install --root-user-action=ignore -q -U vllm
```

**第二步：准备本地权重**（复用 §3.2 的方式，避免联网下载）

```bash
modelscope download --model Qwen/Qwen2.5-7B-Instruct \
  --local_dir /mnt/workspace/models/Qwen2.5-7B-Instruct
```

**第三步：启动后端**

```bash
bash serve_vllm.sh
```

**第四步：启动前端**（另开一个终端，回到 base 环境）

```bash
pip install --root-user-action=ignore -q openai
python app_vllm.py
```

### 8.4 启动参数说明

| 参数 | 本项目取值 | 作用 | 调整影响 |
|---|---|---|---|
| `--model` | 本地目录路径 | 模型位置 | **指向本地目录即可完全规避联网下载** |
| `--served-model-name` | `qwen` | API 中的模型名 | 前端 `VLLM_MODEL` 必须与之一致 |
| `--host` | `0.0.0.0` | 监听地址 | 只填 `127.0.0.1` 则容器外无法访问 |
| `--port` | `8000` | 监听端口 | 与 vLLM 默认一致 |
| `--dtype` | `float16` | 权重精度 | A10 支持 fp16/bf16；**不支持 FP8**（需 sm_89+） |
| `--max-model-len` | `8192` | 上下文长度上限 | **直接决定 KV cache 显存**，调大更易 OOM |
| `--gpu-memory-utilization` | `0.90` | 预留给 KV cache 的显存比例 | 调低则并发能力下降；过高会挤压其他进程 |
| `--enable-prefix-caching` | 开 | 多轮对话共享前缀 | 显著提速；版本不认此参数时设 `ENABLE_PREFIX_CACHING=0` |
| `--api-key` | 未设 | 简单鉴权 | 服务一旦经端口映射暴露，**强烈建议设置** |

### 8.5 前端改造：顺带实现了流式输出

`app_vllm.py` 里 `chat()` 是一个**生成器**，逐块 `yield` 增量文本：

```python
for chunk in stream:
    partial += chunk.choices[0].delta.content or ""
    yield partial
```

Gradio 会自动把每次 yield 的内容更新到同一个气泡里。**对比方案 A**：那里要自己接 `TextIteratorStreamer` + 线程，而 vLLM 的服务端流式协议把这件事变成了默认能力——这正是"接口协议统一"带来的红利。

### 8.6 版本冲突风险（务必先读）

截至本文档写作时，PyPI 实测 vLLM 0.30.0 的依赖约束：

```
torch==2.13.0          ← 精确锁定，不是 >= 
transformers>=5.10.4
torchvision==0.28.0
torchaudio==2.11.0
torchcodec>=0.14
```

**这意味着 `pip install vllm` 会替换掉你环境里的 torch，而不是"再装一个包"。**

- **驱动侧通常不是障碍**：PyPI 上的 torch 2.13.0 是普通 manylinux 轮子（文件名无 `+cu` 标记），CUDA 运行时由自带的 `nvidia-*` 依赖提供，而驱动向下兼容（你的是 CUDA 13.0 上限，足够）。
- **真正的风险是覆盖**：DSW 镜像里预装的 torch 是经平台验证的版本组合，被替换后可能连带升级 `numpy` / `transformers`，导致镜像里其他组件失效；而 `transformers>=5.10.4` 相对你原有的版本跨度很大，**方案 A 的代码在新版上未必行为一致**。

**因此**：

1. **务必用独立 venv**（§8.3 第一步），两个方案各自环境互不干扰
2. 安装前先干跑，看清它打算动什么：`pip install --dry-run vllm`
3. **更省心的替代**：直接选用平台提供的**预置 vLLM 镜像**，环境已配好，无需自建

### 8.7 A10 上的注意事项

- **不支持 FP8**：A10 是 Ampere 架构（sm_86），FP8 需要 sm_89 及以上（Ada/Hopper）。想省显存请用 **AWQ / GPTQ** 量化：`--quantization awq`
- **显存被预占是正常的**：`--gpu-memory-utilization 0.90` 意味着服务启动后就预留约 20GB（22.5 × 0.9）给权重与 KV cache。`nvidia-smi` 显示高占用不等于泄漏
- **7B 模型的显存分布**：FP16 权重约 14GB，剩余约 6GB 作为 KV cache —— 这就是 `--max-model-len` 不能无限调大的原因
- **首次启动较慢**：需要加载权重并做 CUDA graph 预热，等几十秒是正常的

### 8.8 网络访问（承接 §7.6）

vLLM 服务同样面临"入站不通"的问题，按部署形态选择：

| 形态 | 做法 |
|---|---|
| 前端与后端同实例 | 走 `127.0.0.1:8000`，**无需任何入站配置**，最推荐 |
| 从外部直接调 API | 需平台端口映射暴露 8000，**并务必设置 `--api-key`** |
| 只暴露网页界面 | 只映射前端端口（7861），由前端代理到后端，攻击面更小 |

**推荐第三种**：对外只开一个前端入口，模型 API 留在容器内部。少暴露一个端口，就少一类风险。

---

## 9. 项目结构与代码要点

```
qwen-gradio-demo/
├── README.md              本文档
├── app.py                 方案 A：transformers 直载对话服务
├── app_vllm.py            方案 B 前端：连接 vLLM 服务的流式界面
├── serve_vllm.sh          方案 B 后端：vLLM 服务启动脚本
├── check_env.py           设备与环境一键自检
├── requirements.txt       方案 A 依赖
└── requirements-vllm.txt  方案 B 后端依赖（务必装进独立 venv）
```

### 代码要点

**方案 A（`app.py`）**

- **三级降级加载**：`本地目录 → modelscope.snapshot_download → HF 镜像`，任一层可用即可启动
- **网络配置前置**：`HF_ENDPOINT` / 缓存目录写在所有 import 之前
- **history 双格式兼容**：不依赖 Gradio 版本默认值
- **推理零梯度**：`torch.inference_mode()` + `model.eval()`
- **分享能力预检**：`_resolve_share()` 在 frpc 缺失时降级为本地监听，保证服务启动不被中断

**方案 B（`app_vllm.py` + `serve_vllm.sh`）**

- **关注点分离**：前端不 import torch，不占显存，只做 HTTP 转发
- **后端参数前置校验**：`serve_vllm.sh` 检查模型目录与 `vllm` 命令，缺什么直接提示怎么装
- **本地模型路径**：`--model` 指向本地目录，彻底规避启动时联网下载
- **前端探活**：`_check_backend()` 在界面启动前确认后端可达，并校验模型名一致

### 后续可扩展方向

1. **历史截断**：按 token 数裁剪 `history`，防止长对话 OOM
2. **会话持久化**：把对话写入数据库，支持多会话与历史回溯
3. **量化部署**：AWQ/GPTQ 让 14B 级模型也能跑在 22.5GB 显存上（配合 §8.7）
4. **多卡扩展**：`--tensor-parallel-size 2` 跨卡切分，突破单卡显存上限
5. **生产级托管**：需要长期稳定的公网服务时，迁移到专业托管平台（见 §8.8）

