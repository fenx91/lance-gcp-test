#!/bin/bash
# run_with_cuda_fix.sh
# 解决 PyTorch cu124 与系统 CUDA 12.9 库版本不匹配的问题。
# 通过将 venv 中 nvidia 包自带的 CUDA 12.4 兼容库目录前置到 LD_LIBRARY_PATH，
# 确保所有子进程（包括 torch.inductor 的编译 worker）都使用匹配的 cuBLAS/cuDNN 版本，
# 而不是系统路径中的 CUDA 12.9 库。

VENV_DIR="$(dirname "$0")/venv"
NVIDIA_BASE="$VENV_DIR/lib/python3.10/site-packages/nvidia"

CUDA_LIB_PATH="\
$NVIDIA_BASE/cublas/lib:\
$NVIDIA_BASE/cuda_cupti/lib:\
$NVIDIA_BASE/cuda_nvrtc/lib:\
$NVIDIA_BASE/cuda_runtime/lib:\
$NVIDIA_BASE/cudnn/lib:\
$NVIDIA_BASE/cufft/lib:\
$NVIDIA_BASE/curand/lib:\
$NVIDIA_BASE/cusolver/lib:\
$NVIDIA_BASE/cusparse/lib:\
$NVIDIA_BASE/cusparselt/lib:\
$NVIDIA_BASE/nccl/lib:\
$NVIDIA_BASE/nvjitlink/lib:\
$NVIDIA_BASE/nvtx/lib"

export LD_LIBRARY_PATH="$CUDA_LIB_PATH:${LD_LIBRARY_PATH}"

echo "🔧 [CUDA Fix] LD_LIBRARY_PATH 已设置为优先使用 PyTorch 内置 CUDA 12.4 兼容库"
echo "🚀 执行命令: $@"
exec "$@"
