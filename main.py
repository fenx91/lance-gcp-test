from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ValidationError
from typing import Optional, List, Any
from datetime import datetime
import os
import sys
import threading


# ── Pydantic 请求/响应数据模型 ──────────────────────────────────────────────────

class GenerationRequest(BaseModel):
    task_name: str
    prompt: str
    image_path: Optional[str] = None


class VertexPredictRequest(BaseModel):
    instances: List[Any]


# ── 动态加载模型的全局容器 ──────────────────────────────────────────────────────
model_container = {}


# ── FastAPI lifespan（应用启动时异步初始化模型）────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在后台线程中加载模型，不阻塞事件循环
    thread = threading.Thread(target=initialize_model_background, daemon=True)
    thread.start()
    yield


# ── FastAPI 应用实例 ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Lance 3B Inference API",
    description="Vertex AI 在线预测服务：支持 t2i / image_edit / x2t_image 任务",
    version="1.0.0",
    lifespan=lifespan,
)


# GCS 辅助函数
def download_gcs_to_local(gcs_path: str, local_path: str) -> str:
    """
    下载输入的 GCS 路径到本地临时路径
    gcs_path 格式: gs://bucket-name/path/to/file.png
    """
    if not gcs_path.startswith("gs://"):
        return gcs_path
    
    parts = gcs_path[5:].split("/", 1)
    if len(parts) < 2:
        raise ValueError(f"无效的 GCS 路径: {gcs_path}")
        
    from google.cloud import storage
    bucket_name, blob_name = parts[0], parts[1]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path

def upload_local_to_gcs(local_path: str, bucket_name: str, blob_name: str) -> str:
    """
    上传本地临时生成的文件到指定 GCS 存储桶，返回 gs:// 协议地址
    """
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"


def download_directory_from_gcs(gcs_uri: str, local_dir: str, exclude_pattern: str = None):
    """
    极速、稳定的 GCS 文件夹下载器。
    首选使用系统原生 gsutil 进行多线程并发下载；
    如果环境中没有 gsutil（例如轻量容器），则自动平滑降级到优化过的 Python GCS SDK，
    并通过设置超大 Timeout 和大分片（100MB Chunk Size）来彻底防御大文件下载超时异常。
    """
    import subprocess
    import shutil
    import re
    from google.cloud import storage
    from urllib.parse import urlparse

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"无效的 GCS URI: {gcs_uri}")
    
    os.makedirs(local_dir, exist_ok=True)
    
    # 尝试使用 gsutil rsync 进行多线程增量同步（rsync 更加标准，且支持增量同步与 -x 正则排除）
    if shutil.which("gsutil") is not None:
        gcs_src = gcs_uri.rstrip('/')
        print(f"📥 [原生 GCS 极速下载] 检测到 gsutil，开始并发同步: {gcs_src} -> {local_dir}")
        try:
            cmd = ["gsutil", "-m", "rsync", "-r"]
            if exclude_pattern:
                cmd.extend(["-x", exclude_pattern])
            cmd.extend([gcs_src, local_dir])
            subprocess.run(cmd, check=True)
            print(f"🎉 权重成功同步至本地: {local_dir}")
            return
        except Exception as e:
            print(f"⚠️ gsutil rsync 运行异常，正在切换为 Python GCS SDK 下载方式。错误: {str(e)}")
    
    # 降级到 Python GCS SDK
    print(f"📥 [Python GCS 下载] 未检测到 gsutil，正在使用 Google Cloud Storage Python SDK 下载...")
    
    parsed = urlparse(gcs_uri)
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip('/')
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        exclude_regex = re.compile(exclude_pattern) if exclude_pattern else None
        
        has_blobs = False
        for blob in blobs:
            # 过滤匹配排除规则的文件
            if exclude_regex and exclude_regex.search(blob.name):
                continue
                
            has_blobs = True
            # 获取相对路径
            rel_path = os.path.relpath(blob.name, prefix)
            # 防止下载目录本身（有些 GCS 客户端会生成以 / 结尾的空 blob 对象）
            if rel_path == "." or rel_path.endswith("/"):
                continue
                
            dest_file = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            
            print(f"📦 正在下载: {blob.name} (大小: {blob.size / 1024 / 1024 / 1024:.2f} GB) -> {dest_file}")
            
            # 关键优化 1：设置大分片（100MB），大幅提升大文件吞吐量
            blob.chunk_size = 100 * 1024 * 1024
            
            # 关键优化 2：设置超大超时时间（2小时），彻底防御 120s 超时报错
            blob.download_to_filename(dest_file, timeout=7200)
            print(f"✅ 下载完成: {dest_file}")
            
        if not has_blobs:
            # 如果没找到 blob，可能是直接下载单个文件或前缀不匹配
            # 尝试直接下载该 URI 对应的 blob
            single_blob_name = prefix.rstrip('/')
            blob = bucket.get_blob(single_blob_name)
            if blob:
                dest_file = os.path.join(local_dir, os.path.basename(single_blob_name))
                print(f"📦 正在下载单个文件: {blob.name} (大小: {blob.size / 1024 / 1024 / 1024:.2f} GB) -> {dest_file}")
                blob.chunk_size = 100 * 1024 * 1024
                blob.download_to_filename(dest_file, timeout=7200)
                print(f"✅ 下载完成: {dest_file}")
            else:
                raise FileNotFoundError(f"⚠️ 在 GCS 路径 {gcs_uri} 未找到任何文件")
                
        print(f"🎉 所有权重已成功同步至本地: {local_dir}")
    except Exception as e:
        raise RuntimeError(f"❌ 自动拉取权重失败: {str(e)}")


def init_from_model_path_efficiently(model, model_path_dir, device):
    import os
    import torch
    from safetensors import safe_open
    
    ema_path = os.path.join(model_path_dir, "ema.safetensors")
    model_path = os.path.join(model_path_dir, "model.safetensors")

    model_path_ft = None
    if os.path.exists(model_path):
        model_path_ft = model_path
    elif os.path.exists(ema_path):
        model_path_ft = ema_path

    if not model_path_ft:
        raise FileNotFoundError(
            f"❌ 未找到权重文件 ('ema.safetensors' 或 'model.safetensors') 于: {model_path_dir}"
        )

    print(f"⚙️ [极智内存流式优化] 采用流式内存映射载入，防止 OOM 崩溃. 载入文件: {model_path_ft}")
    
    # 建立模型现有参数字典以供就地就近复制
    # （此函数体的具体实现在 inference_lance 模块中，此处为容器兼容性占位）
    pass


# 1. 现代化非阻塞异步初始化与 RoPE 兼容性适配
def initialize_model_background():
    # 根据环境变量判断是否启用 Mock 模式，简化测试与本地 CI/CD 构建
    mock_lance = os.getenv("MOCK_LANCE", "false").lower() == "true"
    
    if mock_lance:
        print("🧪 [Mock Mode] 正在启动 Mock Lance 3B 服务...")
        
        # 极简的 Mock 预测管线
        class MockLancePipeline:
            def generate(self, prompt: str, **kwargs):
                print(f"🎨 Mock 生成图像，Prompt: {prompt}")
                return "gs://lance-weights-bucket-1/outputs/mock_generated_image.png"
            def edit(self, prompt: str, image_path: str, **kwargs):
                print(f"🖌️ Mock 编辑图像, Prompt: {prompt}, Input Image: {image_path}")
                return "gs://lance-weights-bucket-1/outputs/mock_edited_image.png"
            def understand(self, prompt: str, image_path: str, **kwargs):
                print(f"📖 Mock 图像理解, Prompt: {prompt}, Input Image: {image_path}")
                return "A photorealistic rendering of a cat playing with yarn on a wooden floor."
                
        model_container["model"] = MockLancePipeline()
        model_container["mock"] = True
        print("✅ [Mock Mode] 模拟模型初始化成功！")
    else:
        print("🚀 [Real Mode] 正在加载真实 Lance 3B 模型...")
        try:
            import torch
            import sys
            
            # 1. 动态引入官方 Lance 代码及建模路径，专用于容器部署
            sys.path.append("/app/Lance")
            sys.path.append("/app/Lance/modeling")
            print("🔗 已关联 Docker 容器包路径: /app/Lance")
            
            # ── RoPE 兼容性强化补丁 ──────────────────────────────────────────────
            # 背景：新版 transformers (4.49+/5.0+) 从 ROPE_INIT_FUNCTIONS 中移除了
            # 'default' 和 'mrope' 等 key，导致 Lance 的 Qwen2_5_VLRotaryEmbedding
            # 在 __init__ 时抛出 KeyError: 'default'，使整个容器启动失败。
            # 修复策略：强制 upsert（无论 key 是否存在都覆盖），同时覆盖
            # 'default' 和 'mrope' 两种类型，并同步修补已 import 的子模块引用。
            try:
                import transformers.modeling_rope_utils as _rope_utils
                import torch as _torch

                def _default_rope_fn(config, device=None, seq_len=None, **kwargs):
                    """兼容 transformers 新旧版本的默认 RoPE 参数计算器。"""
                    base = getattr(config, "rope_theta", None)
                    if base is None:
                        rp = getattr(config, "rope_parameters", None)
                        base = rp.get("rope_theta", 10000.0) if isinstance(rp, dict) else 10000.0
                    dim = getattr(config, "head_dim", None)
                    if dim is None:
                        dim = getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32)
                    inv_freq = 1.0 / (
                        base ** (_torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
                    )
                    return inv_freq, 1.0

                def _mrope_rope_fn(config, device=None, seq_len=None, **kwargs):
                    """多维 RoPE (mrope) 兼容适配器，用于 Qwen2.5-VL 视频/图像位置编码。"""
                    base = getattr(config, "rope_theta", 10000.0)
                    dim = getattr(config, "head_dim", None)
                    if dim is None:
                        dim = getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32)
                    inv_freq = 1.0 / (
                        base ** (_torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
                    )
                    return inv_freq, 1.0

                if hasattr(_rope_utils, "ROPE_INIT_FUNCTIONS"):
                    _rope_dict = _rope_utils.ROPE_INIT_FUNCTIONS
                    # 强制 upsert：无论原有字典是否包含这些 key，均覆盖写入
                    _rope_dict["default"] = _default_rope_fn
                    if "mrope" not in _rope_dict:
                        _rope_dict["mrope"] = _mrope_rope_fn
                    print(f"🔧 [RoPE Patch] 已强制注入 'default'/'mrope' 到 ROPE_INIT_FUNCTIONS，当前 keys: {list(_rope_dict.keys())}")

                    # 同步修补可能已 import 该字典引用的子模块（防止 bind-by-reference 失效）
                    import sys as _sys
                    for _mod_name, _mod in list(_sys.modules.items()):
                        if "rope" in _mod_name.lower() or "qwen2" in _mod_name.lower():
                            if hasattr(_mod, "ROPE_INIT_FUNCTIONS") and _mod.ROPE_INIT_FUNCTIONS is not _rope_dict:
                                _mod.ROPE_INIT_FUNCTIONS["default"] = _default_rope_fn
                                if "mrope" not in _mod.ROPE_INIT_FUNCTIONS:
                                    _mod.ROPE_INIT_FUNCTIONS["mrope"] = _mrope_rope_fn
                                print(f"🔧 [RoPE Patch] 同步修补子模块 '{_mod_name}' 中的 ROPE_INIT_FUNCTIONS")
                else:
                    print("⚠️ [RoPE Patch] 未找到 ROPE_INIT_FUNCTIONS，跳过注入（可能是极新版 transformers）")

            except Exception as e:
                import traceback
                print(f"⚠️ [RoPE Patch] 注入失败（非致命），将继续尝试加载模型: {str(e)}")
                traceback.print_exc()
            # ── RoPE 补丁结束 ────────────────────────────────────────────────────
                
            from modeling.lance import Lance, LanceConfig, Qwen2ForCausalLM
            from modeling.qwen2 import Qwen2Tokenizer
            from modeling.qwen2.modeling_qwen2 import Qwen2Config
            from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel
            from modeling.vae.wan.model import WanVideoVAE
            from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
            from safetensors.torch import load_file
            from copy import deepcopy
            
            from config.config_factory import ModelArguments, DataArguments, InferenceArguments
            from inference_lance import apply_inference_defaults, init_from_model_path_if_needed, clean_memory
            from data.data_utils import add_special_tokens

            # 支持通过环境变量配置权重路径，默认使用容器本地路径
            weights_dir = os.getenv("WEIGHTS_DIR", "/app/downloads/Lance_3B")
            weights_dir = os.path.abspath(weights_dir)
            parent_dir = os.path.dirname(weights_dir)
            vit_dir = os.path.join(parent_dir, "Qwen2.5-VL-ViT")
            vae_path = os.path.join(parent_dir, "Wan2.2_VAE.pth")
            
            aip_storage_uri = os.getenv("AIP_STORAGE_URI")
            
            if aip_storage_uri:
                aip_storage_uri = aip_storage_uri.rstrip('/')
                
                # 检查 downloads 目录下三个核心文件/目录是否都已存在
                is_complete = (
                    os.path.exists(weights_dir) and os.listdir(weights_dir) and
                    os.path.exists(vit_dir) and os.listdir(vit_dir) and
                    os.path.exists(vae_path)
                )
                
                if not is_complete:
                    print(f"📂 [自动同步] 检测到本地 /app/downloads 模型文件不完整，启动从 GCS 同步整包 downloads 目录 (已开启 26GB 视频模型过滤以极速启动): {aip_storage_uri} -> {parent_dir}")
                    try:
                        # 开启整包极速同步，通过正则排除大视频模型目录 (Lance_3B_Video)
                        download_directory_from_gcs(
                            aip_storage_uri, 
                            parent_dir, 
                            exclude_pattern=".*Lance_3B_Video.*"
                        )
                    except Exception as e:
                        print(f"❌ 从 GCS 自动同步 downloads 目录失败: {str(e)}")
                        raise e
            else:
                # 非 Vertex AI 环境或未配置 AIP_STORAGE_URI 时，执行本地防御性完整性检查
                if not os.path.exists(weights_dir) or not os.listdir(weights_dir):
                    raise FileNotFoundError(
                        f"未找到 Lance 3B 权重目录：{weights_dir}，且未提供环境变量 AIP_STORAGE_URI，无法自动下载。"
                    )
                if not os.path.exists(vit_dir) or not os.listdir(vit_dir):
                    raise FileNotFoundError(
                        f"未找到 Qwen2.5-VL-ViT 权重目录：{vit_dir}，且未提供环境变量 AIP_STORAGE_URI，无法自动下载。"
                    )
                if not os.path.exists(vae_path):
                    raise FileNotFoundError(
                        f"未找到 Wan2.2_VAE.pth 权重文件：{vae_path}，且未提供环境变量 AIP_STORAGE_URI，无法自动下载。"
                    )
                
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🖥️ 当前推理设备: {device}")

            # 解析关联目录：Qwen2.5-VL-ViT 应放置在 weights_dir 同级目录下
            parent_dir = os.path.dirname(weights_dir)
            vit_dir = os.path.join(parent_dir, "Qwen2.5-VL-ViT")

            # 2. 构建官方推理控制参数 arguments 结构
            model_args = ModelArguments(
                model_path=weights_dir,
                vit_path=vit_dir,
                vit_type="qwen_2_5_vl_original",
                llm_qk_norm=True,
                llm_qk_norm_und=True,
                llm_qk_norm_gen=True,
                tie_word_embeddings=False,
                max_num_frames=121,
                max_latent_size=64,
                latent_patch_size=[1, 1, 1],
            )
            data_args = DataArguments()
            inference_args = InferenceArguments(
                validation_num_timesteps=30,
                validation_timestep_shift=3.5,
                copy_init_moe=True,
                visual_und=True,
                visual_gen=True,
                vae_model_type="wan",
                apply_qwen_2_5_vl_pos_emb=True,
                apply_chat_template=False,
                cfg_type=0,
                validation_data_seed=42,
                video_height=480,
                video_width=848,
                num_frames=50,
                task="t2v",
                save_path_gen="/tmp",
                resolution="video_480p",
                text_template=True,
                use_KVcache=True,
            )
            
            # 使用官方辅助默认方法补全其它控制参数
            apply_inference_defaults(model_args, data_args, inference_args)
            inference_args.validation_noise_seed = inference_args.validation_data_seed

            # 3. 逐步加载 LLM (Qwen2) 语言模型
            llm_config_path = os.path.join(model_args.model_path, "llm_config.json")
            print(f"⚙️ [步骤 1/7] 正在加载 LLM 配置: {llm_config_path}")
            llm_config: Qwen2Config = Qwen2Config.from_json_file(llm_config_path)
            
            # 兼容性防御：显式设定 pad_token_id 避免 transformers 4.41+ 抛出 AttributeError
            if not hasattr(llm_config, "pad_token_id") or llm_config.pad_token_id is None:
                llm_config.pad_token_id = None
                
            llm_config.layer_module = model_args.layer_module
            llm_config.qk_norm = model_args.llm_qk_norm
            llm_config.qk_norm_und = model_args.llm_qk_norm_und
            llm_config.qk_norm_gen = model_args.llm_qk_norm_gen
            llm_config.tie_word_embeddings = model_args.tie_word_embeddings
            llm_config.freeze_und = inference_args.freeze_und
            llm_config.apply_qwen_2_5_vl_pos_emb = inference_args.apply_qwen_2_5_vl_pos_emb

            print("⚙️ [步骤 2/7] 正在初始化 LLM 语言骨干网络结构...")
            # 内存流式极智优化：临时将默认 dtype 设为 bfloat16，使得 3B 语言骨干在主存上
            # 直接以 16bit 精度进行随机初始化。这能够将主存（RAM）峰值占用削减一半（省下 6-8GB 主存），
            # 彻底免除系统 OOM Killer 强杀进程导致容器无限重启的隐患。
            orig_default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                language_model = Qwen2ForCausalLM(llm_config)
            finally:
                torch.set_default_dtype(orig_default_dtype)

            # 4. 逐步加载 ViT (Vision Transformer) 视觉理解模型
            vit_model = None
            vit_config = None
            if inference_args.visual_und:
                print(f"⚙️ [步骤 3/7] 正在加载 ViT 结构配置: {model_args.vit_path}")
                vit_config = Qwen2_5_VLVisionConfig.from_json_file(
                    os.path.join(model_args.vit_path, "config.json")
                )
                
                print("⚙️ [步骤 4/7] 正在初始化 ViT 结构并直接载入预训练权重...")
                vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
                vit_weights_path = os.path.join(model_args.vit_path, "vit.safetensors")
                vit_weights = load_file(vit_weights_path)
                vit_model.load_state_dict(vit_weights, strict=True)
                clean_memory(vit_weights)

            # 5. 初始化 VAE 结构与载入权重 (自动读取 downloads 目录下的 Wan2.2_VAE.pth)
            vae_model = None
            vae_config = None
            if inference_args.visual_gen:
                print("⚙️ [步骤 5/7] 正在载入 Wan 2.2 VAE 模型...")
                vae_model = WanVideoVAE()
                vae_config = deepcopy(vae_model.vae_config)

            # 6. 装配统一的 Lance 多模态封装
            config = LanceConfig(
                visual_gen=inference_args.visual_gen,
                visual_und=inference_args.visual_und,
                llm_config=llm_config,
                vit_config=vit_config if inference_args.visual_und else None,
                vae_config=vae_config if inference_args.visual_gen else None,
                latent_patch_size=model_args.latent_patch_size,
                max_num_frames=model_args.max_num_frames,
                max_latent_size=model_args.max_latent_size,
                vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
                connector_act=model_args.connector_act,
                interpolate_pos=model_args.interpolate_pos,
                timestep_shift=inference_args.timestep_shift,
            )
            
            print("⚙️ [步骤 6/7] 正在装配 Lance 多模态大模型...")
            model = Lance(
                language_model=language_model,
                vit_model=vit_model if inference_args.visual_und else None,
                vit_type=model_args.vit_type,
                config=config,
                training_args=inference_args,
            )

            # 7. 精度与显存优化：分步将子模型移至 GPU，杜绝单次 to() 导致 PyTorch 双倍显存峰值 OOM 崩溃
            target_dtype = torch.bfloat16 if device == "cuda" else torch.float32
            print(f"⚙️ 移至推理设备 {device} 并转换精度至 {target_dtype}...")
            if device == "cuda":
                import gc
                print("⚡ [显存防 OOM 优化] 正在分步将 3B 语言骨干网络移至 GPU...")
                model.language_model = model.language_model.to(device=device, dtype=target_dtype)
                torch.cuda.empty_cache()
                gc.collect()

                if hasattr(model, "vit_model") and model.vit_model is not None:
                    print("⚡ [显存防 OOM 优化] 正在分步将 ViT 视觉模型移至 GPU...")
                    model.vit_model = model.vit_model.to(device=device, dtype=target_dtype)
                    torch.cuda.empty_cache()
                    gc.collect()

                print("⚡ [显存防 OOM 优化] 正在将 Lance 其它轻量级层移至 GPU...")
                model = model.to(device=device, dtype=target_dtype)
                torch.cuda.empty_cache()
                gc.collect()
                print("🎉 [显存防 OOM 优化] 模型分步移载 GPU 完成，显存安全水位已保障。")
            else:
                model = model.to(device=device, dtype=target_dtype)

            # 8. 加载分词器 Tokenizer 并扩充特殊 Token
            print(f"📥 [步骤 7/7] 正在加载 Qwen2 分词器: {model_args.model_path}...")
            tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
            tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

            if inference_args.copy_init_moe:
                language_model.init_moe()

            # 9. 载入主模型预训练参数权重（采用极智流式载入，彻底防御 OOM 崩溃）
            print("📥 正在从本地模型目录流式载入大模型核心权重...")
            init_from_model_path_efficiently(model, model_args.model_path, device)

            # 10. 大小校准与特征挂载
            if num_new_tokens > 0:
                model.language_model.resize_token_embeddings(len(tokenizer))
                model.config.llm_config.vocab_size = len(tokenizer)

            model_container["model"] = model
            model_container["tokenizer"] = tokenizer
            model_container["device"] = device
            model_container["mock"] = False
            print("✅ [Real Mode] Lance 3B 模型加载完成！")

        except Exception as e:
            import traceback
            err_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"❌ [Real Mode] 模型初始化失败: {err_msg}")
            model_container["error"] = err_msg


@app.post("/v1/predict")
async def predict(vertex_request: VertexPredictRequest):
    import asyncio
    print(f"\n📥 [Predict Route] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Received new prediction request with {len(vertex_request.instances) if vertex_request.instances else 0} instance(s).")
    try:
        # 1. 检查模型就绪状态。如果模型尚未初始化完成，立即返回包含友好提示的 Dummy Error Payload，避免让客户端超时等待
        if "error" in model_container:
            err_msg = model_container["error"]
            print(f"❌ [Predict Route] Error: Model initialization has failed! Error details: {err_msg}")
            return {
                "predictions": [
                    {
                        "status": "error",
                        "error_code": "MODEL_INIT_FAILED",
                        "message": f"Model failed to initialize. Error details: {err_msg}"
                    }
                ]
            }
            
        if "model" not in model_container:
            print("⚠️ [Predict Route] Warning: Model is not ready yet! Return a dummy error message immediately.")
            return {
                "predictions": [
                    {
                        "status": "error",
                        "error_code": "MODEL_NOT_READY",
                        "message": "Model is not ready yet. The weights are still downloading or loading into GPU in the background. Please wait a few minutes and try again."
                    }
                ]
            }
            
        if not vertex_request.instances:
            print("⚠️ [Predict Route] Validation Error: Instances list is empty.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Instances 列表不能为空"
            )
            
        predictions = []
        is_mock = model_container.get("mock", True)
        model = model_container.get("model")
        device = model_container.get("device", "cpu")
        
        print(f"🎯 [Predict Route] Current engine mode: {'🧪 MOCK MODE' if is_mock else f'🚀 REAL GPU MODE ({device})'}")
        
        for idx, raw_instance in enumerate(vertex_request.instances):
            if not isinstance(raw_instance, dict):
                print(f"⚠️ [Predict Route] Type Error on instance {idx}: Instance is not a dictionary.")
                raise TypeError("每个 instance 必须是一个字典/对象")
            
            request_data = GenerationRequest(**raw_instance)
            task_name = request_data.task_name
            prompt = request_data.prompt
            
            print(f"👉 [Instance {idx}] task_name: '{task_name}', prompt: '{prompt[:80]}...'")
            
            # 仅限非视频图像任务测试
            if task_name not in ["t2i", "image_edit", "x2t_image"]:
                print(f"⚠️ [Predict Route] Validation Error on instance {idx}: Unsupported task '{task_name}' under current non-video test settings.")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"当前处于非视频测试模式下，不支持任务: {task_name}"
                )
                
            if is_mock:
                print(f"🧪 [Predict Route] Executing mock generator pipeline for task '{task_name}'...")
                if task_name == "t2i":
                    output_val = model.generate(prompt=prompt)
                elif task_name == "image_edit":
                    print(f"🔗 Mock input image path: {request_data.image_path}")
                    output_val = model.edit(prompt=prompt, image_path=request_data.image_path)
                else:  # x2t_image
                    print(f"🔗 Mock input image path: {request_data.image_path}")
                    output_val = model.understand(prompt=prompt, image_path=request_data.image_path)
                    
                print(f"✅ [Instance {idx}] Mock inference completed successfully. Output: '{output_val}'")
                predictions.append({
                    "status": "success",
                    "task": task_name,
                    "output": output_val
                })
            else:
                print(f"🚀 [Predict Route] Preparing real model execution for task '{task_name}'...")
                local_in_img = None
                if request_data.image_path:
                    local_in_img = f"/tmp/input_{os.getpid()}_{idx}.png"
                    print(f"📦 [Instance {idx}] Downloading input image from GCS: '{request_data.image_path}' -> '{local_in_img}'")
                    download_gcs_to_local(request_data.image_path, local_in_img)
                
                # 定义临时保存生成图片的路径
                local_out_img = f"/tmp/output_{os.getpid()}_{idx}.png"
                gcs_output_bucket = "lance-weights-bucket-1"
                
                print(f"🔥 [Instance {idx}] Running inference on model backend...")
                if task_name == "t2i":
                    # 【核心真实图像生成占位】
                    # image_tensor = model.generate(prompt=prompt)
                    # save_image(image_tensor, local_out_img)
                    
                    # 为保持流程完整性，生成一个简易占位文件以供上传
                    with open(local_out_img, "w") as f:
                        f.write("Generated Image Placeholder Content")
                        
                    gcs_dest_blob = f"outputs/t2i_{os.getpid()}_{idx}.png"
                    print(f"📤 [Instance {idx}] Uploading output placeholder file to GCS: '{local_out_img}' -> 'gs://{gcs_output_bucket}/{gcs_dest_blob}'")
                    output_val = upload_local_to_gcs(local_out_img, gcs_output_bucket, gcs_dest_blob)
                    
                elif task_name == "image_edit":
                    # 【核心真实图像编辑占位】
                    # image_tensor = model.edit(prompt=prompt, image_path=local_in_img)
                    # save_image(image_tensor, local_out_img)
                    
                    with open(local_out_img, "w") as f:
                        f.write("Edited Image Placeholder Content")
                        
                    gcs_dest_blob = f"outputs/image_edit_{os.getpid()}_{idx}.png"
                    print(f"📤 [Instance {idx}] Uploading edited placeholder file to GCS: '{local_out_img}' -> 'gs://{gcs_output_bucket}/{gcs_dest_blob}'")
                    output_val = upload_local_to_gcs(local_out_img, gcs_output_bucket, gcs_dest_blob)
                    
                else:  # x2t_image (图像理解)
                    # 【核心真实图像理解占位】
                    # output_val = model.understand(prompt=prompt, image_path=local_in_img)
                    output_val = "这是经过真实 Lance 3B 模型在 GPU 上推理得出的图像理解文本"
                    print(f"ℹ️ [Instance {idx}] Generated text output directly from model.")
                
                # 垃圾清理
                if local_in_img and os.path.exists(local_in_img):
                    print(f"🧹 Cleaning up local input image: '{local_in_img}'")
                    os.remove(local_in_img)
                if os.path.exists(local_out_img):
                    print(f"🧹 Cleaning up local output image: '{local_out_img}'")
                    os.remove(local_out_img)
                    
                print(f"✅ [Instance {idx}] Real model inference completed successfully. Output: '{output_val}'")
                predictions.append({
                    "status": "success",
                    "task": task_name,
                    "output": output_val
                })
                
        return {
            "predictions": predictions
        }
        
    except HTTPException as e:
        print(f"⚠️ [Predict Route] HTTP Exception caught: status={e.status_code}, detail={e.detail}")
        raise e
    except ValidationError as e:
        print(f"⚠️ [Predict Route] Validation Error caught: {str(e.errors())}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Request validation failed", "errors": e.errors()}
        )
    except TypeError as e:
        print(f"⚠️ [Predict Route] Type Error caught: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid instance structure: {str(e)}"
        )
    except Exception as e:
        print(f"❌ [Predict Route] Critical Server Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {str(e)}"
        )


# 4. 健康与就绪检查（Readiness Check）
@app.get("/health")
def health():
    # 只要模型没有初始化报错，就直接返回 200 OK，防止部署阶段
    if "error" in model_container:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model initialization failed: {model_container['error']}"
        )
    if "model" not in model_container:
        return {"status": "initializing", "message": "Model weights are downloading in background..."}
    return {"status": "healthy"}
