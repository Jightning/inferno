# Inferno

A from-scratch LLM inference engine in modern C++ using CPU only. It loads open-weight models (Qwen2.5-0.5B, Llama-3.2-1B) and climbs a measured optimization ladder: correct fp32 baseline -> KV cache -> int8/int4 quantization -> multithreading -> AVX2 SIMD.

Design: [docs/architecture.md](docs/architecture.md)
Plan: [docs/roadmap.md](docs/roadmap.md)
Environment setup: [docs/setup.md](docs/setup.md)

## Python environment

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Qwen2.5-0.5B-Instruct

```bash
wget -P models/qwen2.5-0.5b-instruct https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors
wget -P models/qwen2.5-0.5b-instruct https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json
wget -P models/qwen2.5-0.5b-instruct https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json
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
cmake --build build
ctest --test-dir build
./build/inferno
```
