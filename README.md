# Inferno

An LLM inference engine C++ using CPU only built from scratch. It loads open-weight models (ex. Qwen2.5-0.5B) and optimizes using the following: loading weights as fp32 -> KV cache -> int8/int4 quantization -> multithreading -> AVX2 SIMD.

## Parity

For testing

```bash
python scripts/parity.py --model models/qwen2.5-0.5b-instruct --out parity_data
```

Expected output: `parity_data/` containing `prompt_logits.npy`
(float32, `[128, vocab]`, ~1.5 GB total)

## Build for debug

`/third_party` folder used to manage libraries.

```bash
cmake -B build-debug -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
ctest --test-dir build-debug --output-on-failure 
./build-debug/inferno
```

For more verbose debugging:

```bash
./build-debug/inferno_tests --reporter compact --success
```

## Python

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Fixtures

For creating tests, not needed

```bash
python scripts/fixtures.py
```

## Download for Qwen2.5-0.5B-Instruct

```bash
wget -P models/qwen2.5-0.5b-instruct https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors
wget -P models/qwen2.5-0.5b-instruct https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json
wget -P models/qwen2.5-0.5b-instruct https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json
```

```zsh
curl -sL -o models/qwen2.5-0.5b-instruct/config.json https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json;
curl -sL -o models/qwen2.5-0.5b-instruct/tokenizer.json https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json;
curl -sL -o models/qwen2.5-0.5b-instruct/model.safetensors https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors;
```
