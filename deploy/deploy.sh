#!/usr/bin/env bash
#
# Deploy the Mnemo backend on an Alibaba Cloud ECS instance.
#
# Run this ON the ECS instance (Ubuntu/Debian), from the repo root:
#
#     git clone https://github.com/Sajan-coder039/Mnemo.git
#     cd Mnemo
#     DASHSCOPE_API_KEY=sk-xxxx sudo -E ./deploy/deploy.sh
#
# It installs Docker (if missing), builds the image, and runs the backend on
# port 80, talking to Alibaba Cloud's Qwen (DashScope) model service.
#
set -euo pipefail

: "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY (your Alibaba Cloud DashScope key) before running}"

# The Qwen endpoint is Alibaba Cloud DashScope; override if you use the
# mainland-China endpoint (see memoryagent/config.py).
QWEN_BASE_URL="${QWEN_BASE_URL:-https://dashscope-intl.aliyuncs.com/compatible-mode/v1}"

if ! command -v docker >/dev/null 2>&1; then
  echo "==> Installing Docker..."
  curl -fsSL https://get.docker.com | sh
fi

echo "==> Building image..."
docker build -t mnemo:latest .

echo "==> (Re)starting container..."
docker rm -f mnemo >/dev/null 2>&1 || true
docker volume create mnemo-data >/dev/null

docker run -d --name mnemo --restart unless-stopped \
  -p 80:8000 \
  -e DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY" \
  -e QWEN_BASE_URL="$QWEN_BASE_URL" \
  -v mnemo-data:/data \
  mnemo:latest

echo
echo "==> Mnemo is live. Open:  http://$(curl -s ifconfig.me 2>/dev/null || echo '<ECS-PUBLIC-IP>')/"
echo "    Status check:        curl -s localhost/api/status"
docker ps --filter name=mnemo
