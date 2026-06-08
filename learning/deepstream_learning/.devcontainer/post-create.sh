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

echo "Post-create script finished."
