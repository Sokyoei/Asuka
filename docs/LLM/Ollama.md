# Ollama

## 常用命令

```bash
# 下载大模型
ollama pull qwen3:8b
# 在终端运行大模型
ollama run qwen3:8b
```

## FAQ

### Ollama 配置 NVIDIA GPU

```bash
# 查看 GPU UUID
nvidia-smi -L
# 配置 OLLAMA 使用 GPU
export OLLAMA_GPU_LAYER=1
export CUDA_VISIBLE_DEVICES=your_gpu_index/your_gpu_uuid
# 运行 OLLAMA
systemctl restart ollama
# 运行 OLLAMA 模型
ollama run qwen3:8b
```

### Ollama 配置显存常驻（默认5分钟）

在 `/etc/systemd/system/ollama.service` 加入下面的配置，然后重启 Ollama 服务。

```ini
[Service]
Environment="OLLAMA_KEEP_ALIVE=-1"
```

```bash
# 重启 Ollama 服务
sudo systemctl daemon-reload
sudo systemctl restart ollama
# 运行 Ollama 模型
ollama run qwen3:8b
```
