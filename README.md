# Inferno

An LLM inference engine C++ using CPU only built from scratch. It loads open-weight models (ex. Qwen2.5-0.5B) and optimizes using the following: correct fp32 baseline -> KV cache -> int8/int4 quantization -> multithreading -> AVX2 SIMD.

## Python

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Parity

```bash
python scripts/parity.py --model models/qwen2.5-0.5b-instruct --out parity_data
```

Expected output: `parity_data/` containing `prompt00..19_logits.npy`
(float32, `[128, vocab]`, ~1.5 GB total)

## Build for debug

`/third_party` folder used to manage libraries.

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
ctest --test-dir build-debug
./build-debug/inferno
```

## Download for Qwen2.5-0.5B-Instruct

```bash
wget -P models/qwen2.5-0.5b-instruct https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors
wget -P models/qwen2.5-0.5b-instruct https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json
wget -P models/qwen2.5-0.5b-instruct https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json
```