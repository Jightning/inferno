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
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
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


def dump_intermediates(model, tokenizer, out: Path) -> None:
    """M6 step 0: hidden state at four points of prompt 0's forward pass, via
    forward hooks, so a block-0 bug and a final-norm bug can't be confused."""
    dumps = {}

    def save(name):
        def hook(module, args, output):
            # transformers 5.x: DecoderLayer returns a bare Tensor.
            # 4.x returned a tuple -- handle both so this survives an upgrade.
            t = output[0] if isinstance(output, tuple) else output
            dumps[name] = t.detach()[0].float().numpy()  # [seq, hidden]
        return hook

    handles = [
        model.model.embed_tokens.register_forward_hook(save("embed")),
        model.model.layers[0].register_forward_hook(save("block0")),
        model.model.layers[1].register_forward_hook(save("block1")),
        model.model.norm.register_forward_hook(save("final_norm")),
    ]

    encoding = tokenizer(PROMPTS[0], return_tensors="pt")
    model(encoding.input_ids, attention_mask=encoding.attention_mask)  # one forward, no generate()

    for handle in handles:
        handle.remove()

    for name, arr in dumps.items():
        np.save(out / f"prompt00_{name}.npy", arr)
    print(f"wrote intermediates for prompt00: {sorted(dumps.keys())} to {out}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/qwen2.5-0.5b-instruct", help="HF model id or local path (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=Path("parity_data"), help="output directory (default: %(default)s)")
    parser.add_argument("--n-tokens", type=int, default=128, help="greedy tokens to generate per prompt (default: %(default)s)")
    parser.add_argument("--prompts", type=int, default=len(PROMPTS),
                         help="only dump the first N prompts -- for a fast CI smoke rather than "
                              "the full 20 (default: %(default)s)")
    parser.add_argument("--dump-intermediates", action="store_true",
                         help="also dump prompt 0's embed/block0/block1/final_norm activations "
                              "for M6's checkpoints (default: %(default)s)")
    args = parser.parse_args()

    prompts = PROMPTS[:args.prompts]

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    greedy = GenerationConfig(
        do_sample=False,
        repetition_penalty=1.0,
        max_new_tokens=args.n_tokens,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    args.out.mkdir(parents=True, exist_ok=True)

    if args.dump_intermediates:
        dump_intermediates(model, tokenizer, args.out)

    for i, prompt in enumerate(prompts):

        encoding = tokenizer(prompt, return_tensors="pt")
        input_ids = encoding.input_ids # [batch_size, seq_len]
        result = model.generate( # pyright: ignore[reportAttributeAccessIssue]
            input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=greedy,
            return_dict_in_generate=True,
            output_logits=True,
        )

        logits = torch.stack(result.logits, dim=0)[:, 0, :].float().numpy()  # [seq_len, vocab_size]
        output_ids = result.sequences[0].numpy().astype(np.int64) # [seq_len]
        # output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Making sure the generation followed the constraints
        assert 0 < logits.shape[0] <= args.n_tokens, f"[{i}] Logits length isn't within the requested number of tokens: ({logits.shape})."
        assert len(output_ids) == input_ids.shape[1] + logits.shape[0], f"[{i}] Sequence length doesn't match the input tokens plus the tokens actually generated: ({output_ids.shape})."
        assert (logits.argmax(axis=1) == output_ids[input_ids.shape[1]:]).all(), f"[{i}] No temperature expected. Option other than the highest probability token use in output."

        np.save(args.out / f"prompt{i:02d}_logits.npy", logits)
        np.save(args.out / f"prompt{i:02d}_tokens.npy", output_ids)
        stopped = " (stopped at EOS)" if logits.shape[0] < args.n_tokens else ""
        print(f"[{i + 1:2d}/{len(prompts)}] {len(output_ids)} ids, logits {logits.shape}{stopped}: {prompt[:50]!r}")

    manifest = {
        "model": args.model,
        "n_tokens": args.n_tokens,
        "logit_tolerance": LOGIT_TOLERANCE,
        "vocab_size": model.config.vocab_size,
        "chat_template": False,
        "prompts": prompts,
    }

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(prompts)} prompt dumps + manifest.json to {args.out}/")


if __name__ == "__main__":
    main()
