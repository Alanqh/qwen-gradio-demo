# Qwen + Gradio 本地对话服务

在魔塔 DSW GPU 实例上部署 Qwen 系列大模型的 Gradio 对话界面。项目包含环境自检脚本、可运行应用与完整配置说明。

---

## 目录

- [1. 怎么看设备信息](#1-怎么看设备信息)
- [2. 技术栈与框架选型](#2-技术栈与框架选型)
- [3. 环境配置](#3-环境配置)
- [4. 参数说明](#4-参数说明)
- [5. 运行方式与预期结果](#5-运行方式与预期结果)
- [6. 显存估算与模型选型](#6-显存估算与模型选型)
- [7. 常见问题排查](#7-常见问题排查)
- [8. 项目结构](#8-项目结构)

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
本项目定位是"快速验证链路 + 便于改代码"。transformers 的 `generate()` 参数最直观、改动成本最低。若追求吞吐（并发请求、批量推理），应换 vLLM；若只需开箱即用的命令行对话，Ollama 更省事。

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
| `GRADIO_SHARE` | `1` | `1` 生成公网分享链接，`0` 关闭 |
| `MODELSCOPE_CACHE` | `/mnt/workspace/.cache/modelscope` | 魔塔权重缓存 |
| `HF_HOME` | `/mnt/workspace/.cache/huggingface` | HF 权重缓存 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HF 镜像地址 |
| `GRADIO_SERVER_PORT` | `7860` | Gradio 监听端口（Gradio 原生支持） |

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
| `share=True` | 生成 `*.gradio.live` 公网链接，需实例能出外网 |
| `server_name="0.0.0.0"` | 监听所有网卡，便于通过实例端口映射访问 |

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
Running on public URL: https://xxxxx.gradio.live
```

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

---

## 8. 项目结构

```
qwen-gradio-demo/
├── README.md            本文档
├── app.py               对话服务主程序（含三级模型路径降级）
├── check_env.py         设备与环境一键自检
└── requirements.txt     依赖清单
```

### 代码要点

- **三级降级加载**：`本地目录 → modelscope.snapshot_download → HF 镜像`，任一层可用即可启动
- **网络配置前置**：`HF_ENDPOINT` / 缓存目录写在所有 import 之前
- **history 双格式兼容**：不依赖 Gradio 版本默认值
- **推理零梯度**：`torch.inference_mode()` + `model.eval()`

### 后续可扩展方向

1. **历史截断**：按 token 数裁剪 `history`，防止长对话 OOM
2. **流式输出**：改用 `TextIteratorStreamer`，边生成边显示
3. **量化加载**：接入 bitsandbytes 跑 14B 级模型
4. **并发优化**：请求量上来后迁移到 vLLM
