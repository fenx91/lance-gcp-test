import os
import pytest
# 在导入 app 之前，先设定测试环境变量，强制启动 Mock 模式，避免加载真实大模型
os.environ["MOCK_LANCE"] = "True"

from fastapi.testclient import TestClient
from main import app

@pytest.fixture(scope="module")
def client():
    # 使用 with 语句，强制触发 FastAPI 的 lifespan 生命周期挂钩事件，从而正确初始化 MockPipeline
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    # 测试健康检查路径
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint_t2i(client):
    # 测试文本生成图 (t2i) 任务
    payload = {
        "instances": [
            {
                "task_name": "t2i",
                "prompt": "A cinematic shot of a Porsche 911"
            }
        ]
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["task"] == "t2i"
    assert data["predictions"][0]["output"].startswith("gs://")

def test_predict_endpoint_image_edit(client):
    # 测试图像编辑 (image_edit) 任务
    payload = {
        "instances": [
            {
                "task_name": "image_edit",
                "prompt": "change the color of the car to cherry red",
                "image_path": "gs://lance-weights-bucket-1/inputs/car.png"
            }
        ]
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["task"] == "image_edit"
    assert data["predictions"][0]["output"].startswith("gs://")

def test_predict_endpoint_x2t_image(client):
    # 测试图像理解 (x2t_image) 任务
    payload = {
        "instances": [
            {
                "task_name": "x2t_image",
                "prompt": "Describe what is happening in this image",
                "image_path": "gs://lance-weights-bucket-1/inputs/scenery.png"
            }
        ]
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["task"] == "x2t_image"
    assert "yarn" in data["predictions"][0]["output"]  # Mock 返回中包含的代码匹配

def test_predict_endpoint_t2v(client):
    # 测试文本生成视频 (t2v) 任务
    payload = {
        "instances": [
            {
                "task_name": "t2v",
                "prompt": "A beautiful cinematic shot of waves crashing on the shore"
            }
        ]
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["task"] == "t2v"
    assert data["predictions"][0]["output"].endswith(".mp4")

def test_predict_endpoint_video_edit(client):
    # 测试视频编辑 (video_edit) 任务
    payload = {
        "instances": [
            {
                "task_name": "video_edit",
                "prompt": "make the weather sunny",
                "image_path": "gs://lance-weights-bucket-1/inputs/video.mp4"
            }
        ]
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["task"] == "video_edit"
    assert data["predictions"][0]["output"].endswith(".mp4")

def test_predict_endpoint_x2t_video(client):
    # 测试视频理解 (x2t_video) 任务
    payload = {
        "instances": [
            {
                "task_name": "x2t_video",
                "prompt": "Describe what happens in this video",
                "image_path": "gs://lance-weights-bucket-1/inputs/video.mp4"
            }
        ]
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["task"] == "x2t_video"
    assert "yarn" in data["predictions"][0]["output"]  # Mock 返回中包含的代码匹配


def test_predict_endpoint_empty_instances(client):
    # 测试 instances 为空的情况
    payload = {
        "instances": []
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 400

def test_predict_endpoint_invalid_instance(client):
    # 测试字段缺失的情况 (缺少 prompt)
    payload = {
        "instances": [
            {
                "task_name": "t2i"
            }
        ]
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 400


# ==============================================================================
# Lance API 接口契约测试（Contract Tests）
# 目的：验证 Lance 库中我们依赖的关键类和函数的名称与 Signature 真实存在。
# 运行条件：只在容器内（/app/Lance 目录存在）才执行，本地开发时自动跳过。
# 不加载 GPU，不加载权重，执行速度极快（< 1 秒）。
# ==============================================================================

import sys

LANCE_REPO_PATH = "/app/Lance"
LANCE_AVAILABLE = os.path.exists(LANCE_REPO_PATH)

@pytest.mark.skipif(not LANCE_AVAILABLE, reason="需要容器内克隆的 Lance 代码库 (/app/Lance)，本地跳过")
def test_lance_contract_core_imports():
    """
    验证 modeling.lance 模块的核心类可以被成功 import。
    这是保证真实推理路径 Signature 正确的第一道防线。
    """
    if LANCE_REPO_PATH not in sys.path:
        sys.path.insert(0, LANCE_REPO_PATH)

    from modeling.lance import LanceConfig, Lance, Qwen2ForCausalLM

    # 验证关键类存在
    assert LanceConfig is not None, "LanceConfig 不存在"
    assert Lance is not None, "Lance 不存在"
    assert Qwen2ForCausalLM is not None, "Qwen2ForCausalLM 不存在"


@pytest.mark.skipif(not LANCE_AVAILABLE, reason="需要容器内克隆的 Lance 代码库 (/app/Lance)，本地跳过")
def test_lance_contract_config_imports():
    """
    验证 config.config_factory 模块的参数 DataClass 可以被成功 import。
    这些 DataClass 是调用推理管线时必须构建的入参结构。
    """
    if LANCE_REPO_PATH not in sys.path:
        sys.path.insert(0, LANCE_REPO_PATH)

    from config.config_factory import ModelArguments, DataArguments, InferenceArguments

    # 验证关键字段存在（即 Signature 没有变化）
    assert hasattr(ModelArguments, '__dataclass_fields__'), "ModelArguments 不是 dataclass"
    assert 'model_path' in ModelArguments.__dataclass_fields__, "ModelArguments 缺少 model_path 字段"
    assert 'vit_path' in ModelArguments.__dataclass_fields__, "ModelArguments 缺少 vit_path 字段"

    assert hasattr(InferenceArguments, '__dataclass_fields__'), "InferenceArguments 不是 dataclass"
    assert 'task' in InferenceArguments.__dataclass_fields__, "InferenceArguments 缺少 task 字段"
    assert 'save_path_gen' in InferenceArguments.__dataclass_fields__, "InferenceArguments 缺少 save_path_gen 字段"


@pytest.mark.skipif(not LANCE_AVAILABLE, reason="需要容器内克隆的 Lance 代码库 (/app/Lance)，本地跳过")
def test_lance_contract_data_imports():
    """
    验证 data.dataset_base 的工具函数可以被 import，
    以确保推理管线中的数据预处理 Signature 没有发生变化。
    """
    if LANCE_REPO_PATH not in sys.path:
        sys.path.insert(0, LANCE_REPO_PATH)

    from data.dataset_base import DataConfig, simple_custom_collate

    assert DataConfig is not None, "DataConfig 不存在"
    assert callable(simple_custom_collate), "simple_custom_collate 不是可调用函数"


@pytest.mark.skipif(not LANCE_AVAILABLE, reason="需要容器内克隆的 Lance 代码库 (/app/Lance)，本地跳过")
def test_lance_contract_vae_imports():
    """
    验证 VAE 组件 (WanVideoVAE) 可以被 import，
    确保图像/视频解码模块的命名与路径没有发生变化。
    """
    if LANCE_REPO_PATH not in sys.path:
        sys.path.insert(0, LANCE_REPO_PATH)

    from modeling.vae.wan.model import WanVideoVAE

    assert WanVideoVAE is not None, "WanVideoVAE 不存在"


@pytest.mark.skipif(not LANCE_AVAILABLE, reason="需要容器内克隆的 Lance 代码库 (/app/Lance)，本地跳过")
def test_lance_contract_inference_constants():
    """
    验证 inference_lance.py 中的任务常量与我们 main.py 中定义的 task_name 一致。
    这可以防止 Lance 仓库重命名任务名称时我们的路由逻辑默默失效。
    """
    if LANCE_REPO_PATH not in sys.path:
        sys.path.insert(0, LANCE_REPO_PATH)

    import inference_lance

    # 我们 main.py 支持的非视频图像任务
    our_supported_tasks = {"t2i", "image_edit", "x2t_image"}

    # Lance 官方定义的任务列表（从 TASK_DEFAULT_CONFIGS 中读取）
    lance_supported_tasks = set(inference_lance.TASK_DEFAULT_CONFIGS.keys())

    missing = our_supported_tasks - lance_supported_tasks
    assert not missing, (
        f"以下任务在 Lance 官方 TASK_DEFAULT_CONFIGS 中已不存在，"
        f"可能已被重命名或删除: {missing}"
    )
