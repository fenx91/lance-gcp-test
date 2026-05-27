import os
import sys
import pytest

# 1. 强制设定环境变量以在真实模型（Real Mode）下测试
os.environ["MOCK_LANCE"] = "False"
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
    assert device == "cuda", "❌ 本机检测到 L4 显卡，但加载的推理设备却不是 'cuda'"
    
    # 校验主要子组件是否成功转移到 GPU 并且为半精度
    assert next(model.language_model.parameters()).device.type == "cuda", "❌ Language Model 未成功移动到 GPU 设备"
    assert next(model.language_model.parameters()).dtype == torch.bfloat16, "❌ Language Model 精度不是 bfloat16"
    
    if model.config.visual_und:
        assert next(model.vit_model.parameters()).device.type == "cuda", "❌ Vision Transformer (ViT) 未成功移动到 GPU 设备"
        assert next(model.vit_model.parameters()).dtype == torch.bfloat16, "❌ Vision Transformer (ViT) 精度不是 bfloat16"
    
    print("🎉 [Test Real Load] 真实模型成功载入 GPU 且结构状态校验完全通过！")
