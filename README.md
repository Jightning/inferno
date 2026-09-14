# Inferno

An LLM inference engine C++ using CPU only built from scratch. It loads open-weight models (ex. Qwen2.5-0.5B) and optimizes using the following: loading weights as fp32 -> KV cache -> int8/int4 quantization -> multithreading -> AVX2 SIMD.

## Parity

For testing

```bash
python scripts/parity.py --model models/qwen2.5-0.5b-instruct --out parity_data
```

Expected output: `parity_data/` containing `prompt_logits.npy`
(float32, `[128, vocab]`, ~1.5 GB total)

## Benchmarking

For the release build only.
A 68 token prompt is sent for 128 decode tokens with a median of 3 runs after a warm-up (which doesn't get used).

```bash
./build-release/inferno bench --model models/qwen2.5-0.5b-instruct --config fp32-nocache --notes "what changed"
```

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

Same command CI runs, so the local model directory matches CI's exactly. Fetching only
a subset of the repo is what previously hid a bug: the checkpoint also ships a
`generation_config.json` (chat sampling defaults, `repetition_penalty` among them) that
changes what `scripts/parity.py` generates.

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/qwen2.5-0.5b-instruct
```
