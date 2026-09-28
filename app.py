"""Qwen + Gradio 对话服务。

适配环境：魔塔 DSW / NVIDIA A10 24GB / torch 2.x + CUDA 13.0

用法：
    pip install --root-user-action=ignore -q transformers gradio accelerate modelscope
    python app.py

可选环境变量：
    MODEL_NAME         模型 id（魔塔仓库名）或本地目录，默认 Qwen/Qwen2.5-0.5B-Instruct
    MAX_NEW_TOKENS     单次最大生成 token 数，默认 512
    GRADIO_SHARE       1 生成公网分享链接，0 关闭，默认 1
    MODELSCOPE_CACHE   魔塔权重缓存目录
    HF_HOME            HF 权重缓存目录
    HF_ENDPOINT        HF 镜像地址，默认 https://hf-mirror.com
"""

import os

# --- 网络出口配置：必须写在 import transformers 之前 ---
# huggingface_hub 在 import 时就把 HF_ENDPOINT 读入常量，之后再改环境变量无效。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 缓存放大盘，避免撑爆 /root 系统盘（DSW 上 /mnt/workspace 才是持久大容量盘）
os.environ.setdefault("MODELSCOPE_CACHE", "/mnt/workspace/.cache/modelscope")
os.environ.setdefault("HF_HOME", "/mnt/workspace/.cache/huggingface")

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# ---------------- 配置 ----------------
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
SHARE = os.environ.get("GRADIO_SHARE", "1") == "1"


# ---------------- 模型路径解析：本地目录 > 魔塔 > HF 镜像 ----------------
def resolve_model_path(model_id: str) -> str:
    """把模型 id 解析成本地可用路径。

    国内实例通常无法直连 huggingface.co（Errno 101），所以优先用魔塔社区
    把权重拉到本地磁盘；本地目录加载不会触发任何网络请求。
    """
    if os.path.isdir(model_id):
        print(f"使用本地模型目录：{model_id}")
        return model_id

    try:
        from modelscope import snapshot_download

        print(f"正在从魔塔社区下载 {model_id} ...")
        local_dir = snapshot_download(model_id)
        print(f"下载完成：{local_dir}")
        return local_dir
    except ImportError:
        print("[提示] 未安装 modelscope，跳过魔塔通道。可执行：pip install -q modelscope")
    except Exception as exc:  # noqa: BLE001
        print(f"[提示] 魔塔下载失败：{exc}")

    print(f"[提示] 回退到 HF 镜像（{os.environ['HF_ENDPOINT']}）加载 {model_id}")
    return model_id


# ---------------- 加载模型 ----------------
print(f"正在加载模型 {MODEL_NAME} ...（首次需下载，体积取决于参数量）")
MODEL_PATH = resolve_model_path(MODEL_NAME)

# tokenizer 不接受 device_map，它只是文本转 id 的工具，与设备无关
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,       # 半精度省显存；旧版 transformers 写 torch_dtype
    device_map="auto",         # 需要 accelerate，按显存自动分配
    trust_remote_code=True,
    low_cpu_mem_usage=True,    # 分片加载，降低峰值内存
)
model.eval()  # 关闭 dropout，保证推理结果稳定
print(f"模型加载完成，运行设备：{model.device}")


# ---------------- 对话逻辑 ----------------
def _build_messages(message, history):
    """兼容两种 history 格式，避免 Gradio 版本差异导致崩溃。

    Gradio 传给回调的 history 可能是：
      - dict 形式：[{"role": "user", "content": "..."}, ...]（较新版本默认）
      - 元组形式：[["用户问", "AI答"], ...]（旧版 / type="tuples"）
    若直接 `for human, assistant in history` 解包 dict，迭代出的是键名
    "role"/"content"，对话内容会被静默搞乱且不报错。
    """
    messages = []

    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        else:
            human, assistant = item
            if human:
                messages.append({"role": "user", "content": human})
            if assistant:
                messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})
    return messages


def _render_prompt(messages):
    """渲染 prompt。Qwen3 支持 enable_thinking，Qwen2.5 不支持，用 try 兜底。"""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # 关闭思考模式，否则回复会混入大段推理过程
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


@torch.inference_mode()  # 推理不需要梯度
def chat(message, history):
    messages = _build_messages(message, history)
    prompt = _render_prompt(messages)

    # 手动 tokenize，把设备控制掌握在自己手里
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,  # 显式指定，消除 generate 告警
    )

    # 只取新生成的部分（用 shape[-1] 而非 shape[1]，维度表达更准确）
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # 万一模型吐出了思考块，剥掉，只留最终回答
    if "<｜end▁of▁thinking｜>" in response:
        response = response.split("<｜end▁of▁thinking｜>", 1)[1].strip()

    return response


# ---------------- 界面 ----------------
demo = gr.ChatInterface(
    fn=chat,
    title="我的AI对话模型",
    description="运行在魔塔GPU实例上的大模型",
    # 不显式传 type，由 _build_messages 自动兼容，规避 Gradio 版本默认值变化
    #
    # examples 的格式随 Gradio 版本变化（messages 类型要求 list[dict]，
    # tuples 类型接受纯字符串），为避免版本坑此处注释：
    # examples=["你好，请介绍一下自己", "帮我写一段Python代码"],
)

# share=True 会生成 *.gradio.live 公网隧道，需要实例能出外网；
# 若想改用 DSW 自带端口映射，设 GRADIO_SHARE=0 并通过 GRADIO_SERVER_PORT 指定端口。
demo.launch(share=SHARE, server_name="0.0.0.0")
