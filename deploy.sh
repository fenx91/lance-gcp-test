#!/bin/bash

# ==============================================================================
# Lance 3B Vertex AI 一键全自动部署脚本 (deploy.sh)
# 作用：全自动完成 Docker 编译 -> 推送 -> 注册模型 -> 创建端点 -> GPU 推理部署
# 使用说明：
#   1. 给脚本赋予执行权限：chmod +x deploy.sh
#   2. 直接运行脚本：./deploy.sh
# ==============================================================================

# 阻止脚本在遇到错误时继续运行
set -e
# 实时打印脚本执行的每一行原始命令，方便精准 Debug 追踪
set -x

# ==============================================================================
# 🎛️ 核心参数配置（请根据您的 GCP 资源名进行微调）
# ==============================================================================
PROJECT_ID="test-lance-497219"                     # GCP 项目 ID
REGION="us-central1"                               # 部署地域
REPO_NAME="lance-repo"                             # Artifact Registry 镜像仓库名称
IMAGE_NAME="lance-api"                             # 镜像名称
IMAGE_TAG="latest"                                 # 镜像 Tag
# 获取当前时间戳，格式为 月日-时分 (例如 0524-1345，代表5月24日13点45分)
# 确保每一次运行都能自动生成全球唯一、且带有时间追溯性的名称！
TIMESTAMP=$(date +"%m%d-%H%M")
MODEL_DISPLAY_NAME="lance-3b-model-${TIMESTAMP}"             # 注册在 Model Registry 中的模型显示名
ENDPOINT_DISPLAY_NAME="lance-3b-endpoint-${TIMESTAMP}"       # 部署的在线端点显示名

# GCS 权重相关配置 (启动时自动下载)
GCS_ARTIFACT_URI="gs://lance-weights-bucket-1/downloads"
WEIGHTS_DIR="/app/downloads/Lance_3B"

# GPU 宿主机算力配置（默认使用易抢占且便宜的 L4 显卡配对）
MACHINE_TYPE="g2-standard-8"                       # 虚拟机型号 (L4 对应 g2-standard-4)
ACCELERATOR_TYPE="nvidia-l4"                       # GPU 显卡型号
# MACHINE_TYPE="n1-standard-8"
# ACCELERATOR_TYPE="nvidia-tesla-t4"                       # GPU 显卡型号
GPU_COUNT=1                                        # 显卡数量

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "=================================================================="
echo "🚀 开始执行一键全自动部署流程 [Model: ${MODEL_DISPLAY_NAME}]"
echo "=================================================================="

# ------------------------------------------------------------------------------
# 🔓 Step 1: 配置本地 Docker 登录 GCP 仓库权限
# ------------------------------------------------------------------------------
echo "🔓 [Step 1/5] 正在配置 Docker 登录凭证..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# ------------------------------------------------------------------------------
# 🐳 Step 2: 构建 Docker 镜像并推送到 Artifact Registry
# ------------------------------------------------------------------------------
echo "🐳 [Step 2/5] 正在构建 Docker 镜像..."
docker build -t ${IMAGE_URI} .

echo "📤 正在推送镜像到谷歌 Artifact Registry..."
docker push ${IMAGE_URI}

# ------------------------------------------------------------------------------
# 📦 Step 3: 在 Vertex AI Model Registry 注册模型 (支持 FUSE 自动挂载)
# ------------------------------------------------------------------------------
echo "📦 [Step 3/5] 正在向 Vertex AI 注册新模型..."
gcloud ai models upload \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --display-name="${MODEL_DISPLAY_NAME}" \
    --container-image-uri="${IMAGE_URI}" \
    --container-health-route="/health" \
    --container-predict-route="/v1/predict" \
    --container-ports=8080 \
    --artifact-uri="${GCS_ARTIFACT_URI}" \
    --container-env-vars="WEIGHTS_DIR=${WEIGHTS_DIR}"

echo "⏳ 正在查询刚刚注册成功的 Model ID..."
# 等待几秒钟让 GCP 完成模型索引
sleep 5
MODEL_ID=$(gcloud ai models list \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --filter="display_name=${MODEL_DISPLAY_NAME}" \
  --limit=1 \
  --format="value(name)" | awk -F'/' '{print $NF}')

if [ -z "$MODEL_ID" ]; then
    echo "❌ 错误：未能获取到注册的 Model ID，请检查 gcloud 状态。"
    exit 1
fi
echo "🎉 模型注册成功！Model ID 为: ${MODEL_ID}"

# ------------------------------------------------------------------------------
# 🔔 Step 4: 创建全新的 Vertex AI 预测端点 (Endpoint)
# ------------------------------------------------------------------------------
echo "🔔 [Step 4/5] 正在创建全新的端点..."
ENDPOINT_ID=$(gcloud ai endpoints create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${ENDPOINT_DISPLAY_NAME}" \
  --format="value(name)" | awk -F'/' '{print $NF}')

# ENDPOINT_ID=7508952483931095040

echo "🎉 端点创建成功！Endpoint ID 为: ${ENDPOINT_ID}"

# ------------------------------------------------------------------------------
# 🖥️ Step 5: 零停机替换 — 部署新模型，然后 undeploy 旧模型
# ------------------------------------------------------------------------------

# 5a. 先记录当前 endpoint 上所有旧的 deployed model ID（部署前快照）
echo "🔍 [Step 5/5] 正在记录当前 Endpoint 上的旧 Deployed Model ID..."
OLD_DEPLOYED_MODEL_IDS=$(gcloud ai endpoints describe ${ENDPOINT_ID} \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format="value(deployedModels.id)" 2>/dev/null || echo "")

echo "📋 当前旧 Deployed Model ID(s): ${OLD_DEPLOYED_MODEL_IDS:-（无）}"

# 5b. 部署新模型到现有 endpoint
echo "🖥️ 开始将新模型部署到 Endpoint [机器: ${MACHINE_TYPE}, GPU: ${ACCELERATOR_TYPE}]..."
gcloud ai endpoints deploy-model ${ENDPOINT_ID} \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --model="${MODEL_ID}" \
  --display-name="lance-3b-gpu-service" \
  --machine-type="${MACHINE_TYPE}" \
  --accelerator=type=${ACCELERATOR_TYPE},count=${GPU_COUNT} \
  --min-replica-count=1 \
  --max-replica-count=1

echo "✅ 新模型部署完成！"

# 5c. undeploy 旧版本（新模型已上线，安全移除旧版）
if [ -n "${OLD_DEPLOYED_MODEL_IDS}" ]; then
  echo "🧹 正在移除旧 Deployed Model(s)..."
  for OLD_ID in ${OLD_DEPLOYED_MODEL_IDS}; do
    echo "   ➜ Undeploying deployed model ID: ${OLD_ID} ..."
    gcloud ai endpoints undeploy-model ${ENDPOINT_ID} \
      --project="${PROJECT_ID}" \
      --region="${REGION}" \
      --deployed-model-id="${OLD_ID}" \
      --quiet
    echo "   ✅ 旧 Deployed Model ${OLD_ID} 已成功移除"
  done
else
  echo "ℹ️  没有旧 Deployed Model 需要移除"
fi


echo "=================================================================="
echo "🎉 恭喜！零停机替换部署圆满完成！"
echo "👉 Endpoint ID    : ${ENDPOINT_ID}"
echo "👉 新 Model ID    : ${MODEL_ID}"
echo "👉 实时日志追踪命令："
echo "   gcloud beta logging tail \"resource.type=\\\"aiplatform.googleapis.com/Endpoint\\\" AND resource.labels.endpoint_id=\\\"${ENDPOINT_ID}\\\"\" --project=\"${PROJECT_ID}\" --format=\"value(textPayload)\""
echo "=================================================================="
