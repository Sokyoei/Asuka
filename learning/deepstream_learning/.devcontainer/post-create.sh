#!/bin/bash
set -e

echo "Running post-create script..."

# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装 git
if ! command -v git &> /dev/null; then
    echo "Installing git..."
    apt-get update && apt-get install -y git
fi

# 禁用 SSL 验证
git config --global http.sslVerify false

# 创建软链接方便查看
ln -sf /opt/nvidia/deepstream/deepstream /workspace/deepstream

echo "Post-create script finished."
