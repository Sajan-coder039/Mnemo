#!/bin/bash
#
# Alibaba Cloud ECS "User Data" (cloud-init) script.
#
# Paste this into the ECS console's Advanced > User Data box when creating the
# instance (Ubuntu 22.04). On first boot it installs Docker, clones Mnemo,
# builds it, and runs the backend on port 80 against Alibaba Cloud Qwen.
#
# EDIT the DASHSCOPE_API_KEY line below before creating the instance.
#
set -euxo pipefail

DASHSCOPE_API_KEY="REPLACE_WITH_YOUR_ALIBABA_CLOUD_DASHSCOPE_KEY"
QWEN_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git curl
curl -fsSL https://get.docker.com | sh

cd /opt
git clone https://github.com/Sajan-coder039/Mnemo.git
cd Mnemo

docker build -t mnemo:latest .
docker volume create mnemo-data
docker run -d --name mnemo --restart unless-stopped \
  -p 80:8000 \
  -e DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY" \
  -e QWEN_BASE_URL="$QWEN_BASE_URL" \
  -v mnemo-data:/data \
  mnemo:latest
