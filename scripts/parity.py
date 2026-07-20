"""
For each of 20 prompts, runs decoding and writes logits and the token id sequence to --out:

prompt{i:02d}_logits.npy: float32 [seq_len, vocab_size] - probability distribution (vocab_size) for each token (seq_len)
prompt{i:02d}_tokens.npy: int64 [prompt_len + seq_len] - full response sequence
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    sys.exit(f"missing dependency: {e.name}. Run the environment setup in docs/setup.md first.")

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a village by the sea,",
    "def fibonacci(n):",
    "The three laws of thermodynamics are",
    "Q: What is 17 multiplied by 23?\nA:",
    "In 1969, humans first landed on",
    "The recipe for a simple tomato soup starts with",
    "Translate to German: 'Good morning, how are you?'",
    "The most important difference between TCP and UDP is",
    "She opened the ancient book and found",
    "SELECT name, age FROM users WHERE",
    "The photosynthesis reaction converts",
    "A haiku about winter:",
    "The stock market crashed in 1929 because",
    "import numpy as np\n\ndef softmax(x):",
    "Dear hiring manager, I am writing to",
    "The speed of light in a vacuum is approximately",
    "Explain like I'm five: why is the sky blue?",
    "for i in range(10):\n    print(",
    "The last thing he remembered before the lights went out was",
]

LOGIT_TOLERANCE = 1e-3  # max abs diff gate for the fp32 rung

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/qwen2.5-0.5b-instruct", help="HF model id or local path (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=Path("parity_data"), help="output directory (default: %(default)s)")
    parser.add_argument("--n-tokens", type=int, default=128, help="greedy tokens to generate per prompt (default: %(default)s)")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    args.out.mkdir(parents=True, exist_ok=True)
    
    for i, prompt in enumerate(PROMPTS):

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids # [batch_size, seq_len]
        result = model.generate( # pyright: ignore[reportAttributeAccessIssue]
            input_ids,
            max_new_tokens=args.n_tokens,
            min_new_tokens=args.n_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )

        logits = torch.stack(result.logits, dim=0)[:, 0, :].float().numpy()  # [seq_len, vocab_size]
        output_ids = result.sequences[0].numpy().astype(np.int64) # [seq_len]
        # output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Making sure the generation followed the constraints
        assert logits.shape[0] == args.n_tokens, f"[{i}] Logits length doesn't match expected number of tokens: ({logits.shape})."
        assert len(output_ids) == input_ids.shape[1] + args.n_tokens, f"[{i}] Sequence length doesn't match expected number of input and output tokens: ({output_ids.shape})."
        assert (logits.argmax(axis=1) == output_ids[input_ids.shape[1]:]).all(), f"[{i}] No temperature expected. Option other than the highest probability token use in output."

        np.save(args.out / f"prompt{i:02d}_logits.npy", logits)
        np.save(args.out / f"prompt{i:02d}_tokens.npy", output_ids)
        print(f"[{i + 1:2d}/{len(PROMPTS)}] {len(output_ids)} ids, logits {logits.shape}: {prompt[:50]!r}")

    manifest = {
        "model": args.model,
        "n_tokens": args.n_tokens,
        "logit_tolerance": LOGIT_TOLERANCE,
        "vocab_size": model.config.vocab_size,
        "chat_template": False,
        "prompts": PROMPTS,
    }

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(PROMPTS)} prompt dumps + manifest.json to {args.out}/")


if __name__ == "__main__":
    main()
