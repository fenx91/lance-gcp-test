import os
import sys
import pytest

# ── 环境变量必须在 import torch 之前设置 ──────────────────────────────────────
# 禁用 torch.inductor 编译后端，防止其子进程因 CUDA 12.4 (PyTorch bundled)
# 与系统 CUDA 12.9 的 libcublasLt 版本不匹配导致 "Invalid handle. Cannot load
# symbol cublasLtCreate" abort crash
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# 1. 强制设定环境变量以在真实模型（Real Mode）下测试
os.environ["MOCK_LANCE"] = "false"
# 指定本地下载好的权重根路径
os.environ["WEIGHTS_DIR"] = os.path.abspath("./downloads/Lance_3B")

# 2. 将容器或本地克隆的 Lance 官方仓库路径关联至 sys.path，保证建模模块可顺利导入
LANCE_REPO_PATH = os.path.abspath("./Lance")
if LANCE_REPO_PATH not in sys.path:
    sys.path.insert(0, LANCE_REPO_PATH)
    sys.path.insert(0, os.path.join(LANCE_REPO_PATH, "modeling"))

from main import initialize_model_background, model_container


def test_real_model_loading_and_gpu_placement():
    """
    真实模型加载与 GPU 占用测试。
    该测试在本地下载好权重并完成环境配置后运行，将直接调用 Real Mode 下的
    模型初始化逻辑，并在加载成功后在 GPU 上进行状态验证。
    """
    print("\n🚀 [Test Real Load] 开始测试真实 Lance 3B 模型加载流程...")

    # 3. 检查本地权重路径是否存在
    assert os.path.exists(os.environ["WEIGHTS_DIR"]), (
        f"❌ 未找到本地权重目录 {os.environ['WEIGHTS_DIR']}，请先确保 Step 5 的权重同步已完成。"
    )

    # 4. 同步运行模型后台初始化函数（在测试中我们直接阻塞式调用）
    initialize_model_background()

    # 5. 校验初始化容器状态
    assert "error" not in model_container, f"❌ 模型初始化过程中发生异常，错误详情: {model_container.get('error')}"
    assert "model" in model_container, "❌ model_container 中缺少 'model' 对象"
    assert "tokenizer" in model_container, "❌ model_container 中缺少 'tokenizer' 对象"
    assert "device" in model_container, "❌ model_container 中缺少 'device' 字段"

    model = model_container["model"]
    tokenizer = model_container["tokenizer"]
    device = model_container["device"]

    print(f"✅ [Test Real Load] 模型加载成功！推理设备为: {device}")

    # 6. 验证模型在 GPU 上的状态与基本执行可行性
    import torch
    if torch.cuda.is_available():
        assert device == "cuda", "❌ 检测到 CUDA 可用，但推理设备不是 'cuda'"

        # 校验主要子组件是否成功转移到 GPU 并且为半精度
        assert next(model.language_model.parameters()).device.type == "cuda", "❌ Language Model 未成功移动到 GPU 设备"
        assert next(model.language_model.parameters()).dtype == torch.bfloat16, "❌ Language Model 精度不是 bfloat16"

        if model.config.visual_und:
            assert next(model.vit_model.parameters()).device.type == "cuda", "❌ Vision Transformer (ViT) 未成功移动到 GPU 设备"
            assert next(model.vit_model.parameters()).dtype == torch.bfloat16, "❌ Vision Transformer (ViT) 精度不是 bfloat16"
    else:
        assert device == "cpu", "❌ 未检测到 CUDA，推理设备应为 'cpu'"

    print("🎉 [Test Real Load] 真实模型成功载入 GPU 且结构状态校验完全通过！")


def test_real_image_understanding():
    """
    直接调用 main.py 中的 predict 接口函数，对保时捷联名跑鞋图片进行图像理解推理测试。
    """
    print("\n🚀 [Test Real Load] 开始测试直接调用 predict 接口的图像理解推理...")

    # 确保模型已经加载
    if "model" not in model_container:
        initialize_model_background()

    assert "model" in model_container

    import asyncio
    from main import predict, VertexPredictRequest

    # 使用提供的图片做测试用例
    image_path = os.path.abspath("Porsche-Puma-964-scaled.jpeg")
    assert os.path.exists(image_path), f"❌ 测试图片不存在: {image_path}"

    # 构造请求载荷
    vertex_request = VertexPredictRequest(
        instances=[
            {
                "task_name": "x2t_image",
                "prompt": "Describe this image in detail.",
                "image_path": image_path
            }
        ]
    )

    # 异步执行 predict 接口函数
    response = asyncio.run(predict(vertex_request))

    print(f"\n🎉 [Test Real Load] predict 接口返回结果：\n>>> {response}\n")

    assert "predictions" in response, "❌ 接口返回格式错误，缺少 'predictions' 键"
    predictions = response["predictions"]
    assert len(predictions) > 0, "❌ predictions 结果列表为空"
    assert predictions[0]["status"] == "success", f"❌ 推理执行失败: {predictions[0]}"
    output_val = predictions[0]["output"]
    assert len(output_val) > 0, "❌ 真实模型推理出的输出文本不能为空"


def test_real_text_to_image():
    """
    直接调用 main.py 中的 predict 接口函数，进行真实模型 text-to-image (t2i) 推理测试。
    """
    print("\n🚀 [Test Real Load] 开始测试直接调用 predict 接口的 text-to-image 生成...")

    # 确保模型已经加载
    if "model" not in model_container:
        initialize_model_background()

    assert "model" in model_container

    import asyncio
    from main import predict, VertexPredictRequest

    # 构造 t2i 请求载荷
    vertex_request = VertexPredictRequest(
        instances=[
            {
                "task_name": "t2i",
                "prompt": "A beautiful cherry blossom tree in a serene Japanese garden, watercolor style."
            }
        ]
    )

    # 异步执行 predict 接口函数
    response = asyncio.run(predict(vertex_request))

    print(f"\n🎉 [Test Real Load] t2i predict 接口返回结果：\n>>> {response}\n")

    assert "predictions" in response, "❌ 接口返回格式错误，缺少 'predictions' 键"
    predictions = response["predictions"]
    assert len(predictions) > 0, "❌ predictions 结果列表为空"
    assert predictions[0]["status"] == "success", f"❌ 推理执行失败: {predictions[0]}"
    output_val = predictions[0]["output"]
    assert os.path.exists(output_val), f"❌ 生成的图片文件不存在: {output_val}"
    assert os.path.getsize(output_val) > 0, "❌ 生成的图片文件大小不能为 0"
    print(f"✅ [Test Real Load] t2i 图像生成成功，本地图片路径: {output_val}")

