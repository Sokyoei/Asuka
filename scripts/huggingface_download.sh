#!/bin/bash

# 国内镜像
export HF_ENDPOINT=https://hf-mirror.com

repo=username/repo-name
hf download \
    ${repo}  \
    --local-dir ../models/${repo} \
    --token your_token

# # 已弃用
# huggingface-cli download \
#     --resume-download username/repo-name \
#     --local-dir ./ \
#     --local-dir-use-symlinks False \
#     --token your_token
