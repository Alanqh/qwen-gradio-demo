#!/usr/bin/env bash
# 启动 vLLM 的 OpenAI 兼容服务（方案 B 的后端）
#
# 用法：
#   MODEL_PATH=/mnt/workspace/models/Qwen2.5-7B-Instruct bash serve_vllm.sh
#
# 关键设计：--model 直接指向本地目录，避免 vLLM 启动时联网下载权重。
# 国内实例访问不了 huggingface.co，本地目录加载不触发任何网络请求（详见 README §7.1）。
#
# 前置条件（务必在独立 venv 中安装，不要污染 base 环境）：
#   python -m venv /mnt/workspace/venv-vllm
#   source /mnt/workspace/venv-vllm/bin/activate
#   pip install --root-user-action=ignore -q -U vllm

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/Qwen2.5-7B-Instruct}"
SERVED_NAME="${SERVED_NAME:-qwen}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
DTYPE="${DTYPE:-float16}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[错误] 本地模型目录不存在：$MODEL_PATH" >&2
  echo "       请先下载权重，例如：" >&2
  echo "       modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir $MODEL_PATH" >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "[错误] 未找到 vllm 命令。请先在独立虚拟环境中安装：" >&2
  echo "       python -m venv /mnt/workspace/venv-vllm" >&2
  echo "       source /mnt/workspace/venv-vllm/bin/activate" >&2
  echo "       pip install --root-user-action=ignore -q -U vllm" >&2
  exit 1
fi

CMD=(
  vllm serve "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0                    # 允许容器外访问，配合平台端口映射
  --port "$PORT"
  --dtype "$DTYPE"                  # A10 支持 float16 / bfloat16，不支持 FP8
  --max-model-len "$MAX_MODEL_LEN"  # 上下文上限，直接决定 KV cache 显存占用
  --gpu-memory-utilization "$GPU_MEM_UTIL"
)

# 前缀缓存：多轮对话共享相同前缀，显著提速。
# 若所用 vLLM 版本报未知参数，设为 0 关闭即可。
if [[ "${ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
  CMD+=(--enable-prefix-caching)
fi

# 可选鉴权：设置 VLLM_API_KEY 后，客户端必须携带对应 key
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  CMD+=(--api-key "$VLLM_API_KEY")
fi

echo "启动 vLLM 服务"
echo "  模型    : $MODEL_PATH"
echo "  端口    : $PORT"
echo "  上下文  : $MAX_MODEL_LEN"
echo "  显存预留: ${GPU_MEM_UTIL}（启动后 nvidia-smi 显示高占用属正常现象）"
echo

exec "${CMD[@]}"
