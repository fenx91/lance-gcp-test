FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 合并系统安装，清理 apt 缓存，并安装 git、python3-pip 以及 Google Cloud SDK (包含 gsutil)
RUN apt-get update && apt-get install -y git python3-pip curl apt-transport-https ca-certificates gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 克隆 Lance 代码库到容器中（用于调用真实的 modeling 结构）
RUN git clone https://github.com/bytedance/Lance.git /app/Lance

# 【核心优化】连写安装命令并清空缓存。安装 PyTorch GPU 版本，多模态框架所需包，以及 FlashAttention 预编译包（对应 Ubuntu22.04 默认 Python 3.10 版）
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir fastapi uvicorn pydantic google-cloud-storage transformers accelerate decord opencv-python imageio diffusers sentencepiece einops einops-exts addict albumentations ftfy kornia librosa omegaconf opt-einsum peft qwen-vl-utils imageio-ffmpeg && \
    pip3 install --no-cache-dir --no-deps --force-reinstall "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" && \
    rm -rf /root/.cache/pip

# 设置环境变量，确保容器启动后能够导入 Lance 库的模块
ENV PYTHONPATH="/app/Lance:/app/Lance/modeling"

# 把接口代码拷贝至容器
COPY ./main.py /app/main.py

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
