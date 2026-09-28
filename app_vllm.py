"""vLLM 前端：连接已启动的 vLLM 服务，提供流式对话界面（方案 B 的前端）。

与 app.py 的本质区别：本脚本**不加载模型、不占用显存**，只是一个 HTTP 客户端。
后端需先在另一个终端启动：bash serve_vllm.sh

用法：
    pip install --root-user-action=ignore -q openai
    python app_vllm.py

环境变量：
    VLLM_BASE_URL      后端地址，默认 http://127.0.0.1:8000/v1
    VLLM_API_KEY       与后端 --api-key 保持一致；后端未开鉴权时随意填
    VLLM_MODEL         与后端 --served-model-name 保持一致，默认 qwen
    MAX_NEW_TOKENS     单次最大生成 token 数，默认 512
    GRADIO_SERVER_PORT 前端监听端口，默认 7861（避免与 app.py 的 7860 冲突）
    DISABLE_THINKING   设为 1 时请求关闭 Qwen3 思考模式
"""

import os

import gradio as gr
from openai import OpenAI

# ---------------- 配置 ----------------
BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
MODEL = os.environ.get("VLLM_MODEL", "qwen")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))

# Qwen3 默认开启思考模式，回复里会混入推理过程。
# vLLM 通过 chat_template_kwargs 透传该开关，属服务端扩展参数，
# 默认不传以免在不支持的版本上报错。
EXTRA_BODY = None
if os.environ.get("DISABLE_THINKING", "0") == "1":
    EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

client = OpenAI(base_url=BASE_URL, api_key=API_KEY or "EMPTY")


def _check_backend() -> None:
    """启动前探活，把"连接失败"的排查信息前置到进程启动阶段。"""
    try:
        models = [m.id for m in client.models.list().data]
        print(f"已连接 vLLM 服务：{BASE_URL}")
        print(f"可用模型：{models}")
        if MODEL not in models:
            print(f"[警告] VLLM_MODEL='{MODEL}' 不在服务端模型列表中，请求可能失败。")
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] 无法连接 {BASE_URL}：{exc}")
        print("       请先在另一个终端启动后端：bash serve_vllm.sh")


# ---------------- 对话逻辑 ----------------
def _build_messages(message, history):
    """兼容两种 history 格式（dict / tuple），逻辑与 app.py 一致。

    此处有意重复而非抽成公共模块：本项目定位是可直接复制单文件使用，
    保持 app.py 与 app_vllm.py 各自独立可运行。
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


def chat(message, history):
    """流式输出：逐块 yield 增量文本，Gradio 会自动更新气泡内容。"""
    messages = _build_messages(message, history)

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
            temperature=0.7,
            top_p=0.9,
            max_tokens=MAX_NEW_TOKENS,
            extra_body=EXTRA_BODY,
        )
    except Exception as exc:  # noqa: BLE001
        yield f"请求失败：{exc}\n\n请确认后端已启动（bash serve_vllm.sh）且模型名一致。"
        return

    partial = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        partial += chunk.choices[0].delta.content or ""
        yield partial

    if not partial:
        yield "（服务端返回了空内容）"


_check_backend()

demo = gr.ChatInterface(
    fn=chat,
    title="AI 对话（vLLM 加速）",
    description="前端只做 HTTP 转发，模型推理由独立的 vLLM 服务承担",
)

demo.launch(server_name="0.0.0.0", server_port=SERVER_PORT)
