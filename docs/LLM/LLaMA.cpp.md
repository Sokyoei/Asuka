# LLaMA.cpp

## 下载 llama.cpp

```shell
git clone https://github.com/ggerganov/llama.cpp --recursive
cd llama.cpp
make GGML_CUDA=1  # https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#cuda
```

## `.safetensors` 格式转换到 `.gguf` 格式

```shell
cd llama.cpp
python convert_hf_to_gguf.py your_huggingface_model_dir --outtype f16 --outfile your_gguf_file_path
```

## 量化

```shell
llama-quantize your_gguf_file_path your_gguf_quantize_file_path q4_0
```
