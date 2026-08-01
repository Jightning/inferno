"""Ground truth for M4: what Hugging Face's tokenizer does, stage by stage.

The C++ side is a port, so every disagreement is settled here. Run it whenever a prompt
fails and diff the stages -- the first stage that differs names the bug:

    chunks differ      -> pre-tokenizer scanner  (src/tokenizer/pretokenize.cpp)
    chunks same, ids   -> merge loop or vocab    (src/tokenizer/bpe.cpp, tokenizer.cpp)
    both same, npy no  -> encode is fine; the parity dump is stale

Usage:
    python scripts/tokenizer_fixtures.py                 # all 20 manifest prompts, ids only
    python scripts/tokenizer_fixtures.py --prompt 4      # one prompt, every stage
    python scripts/tokenizer_fixtures.py --text "don't"  # an ad-hoc string
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent


def show(tok: Tokenizer, text: str, verbose: bool) -> list[int]:
    """Print the stages for one string and return its ids."""
    encoding = tok.encode(text, add_special_tokens=False)
    if verbose:
        # pre_tokenize_str runs Split then ByteLevel, so chunks come back byte-mapped
        # (space -> 'G-dot'). The raw column is the same thing unmapped, which is what
        # pretokenize() in C++ returns.
        chunks = [chunk for chunk, _span in tok.pre_tokenizer.pre_tokenize_str(text)]
        print(f"  text    {text!r}")
        print(f"  chunks  {chunks}")
        print(f"  raw     {[unmap(c) for c in chunks]}")
        print(f"  pieces  {encoding.tokens}")
    print(f"  ids     {encoding.ids}")
    return encoding.ids


def unmap(mapped: str) -> str:
    """Invert the byte<->unicode map, so chunks read as the text they came from."""
    table = byte_decoder()
    return bytes(table[ch] for ch in mapped).decode("utf-8", errors="replace")


def byte_decoder() -> dict[str, int]:
    """The GPT-2 bijection, codepoint -> byte. The C++ port of this is step 1."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), 256))
    )
    codepoints = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(cp): b for b, cp in zip(printable, codepoints)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=ROOT / "models/qwen2.5-0.5b-instruct")
    parser.add_argument("--data", default=ROOT / "parity_data")
    parser.add_argument("--prompt", type=int, help="manifest prompt index, 0-19")
    parser.add_argument("--text", help="tokenize this string instead of the manifest")
    args = parser.parse_args()

    tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))

    if args.text is not None:
        show(tok, args.text, verbose=True)
        return

    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    prompts = manifest["prompts"]
    indices = [args.prompt] if args.prompt is not None else range(len(prompts))
    verbose = args.prompt is not None

    for i in indices:
        print(f"[{i:02d}]")
        ids = show(tok, prompts[i], verbose)

        # Cross-check against the recorded parity tokens the C++ test compares to.
        tokens_npy = Path(args.data) / f"prompt{i:02d}_tokens.npy"
        if tokens_npy.exists():
            import numpy as np

            recorded = np.load(tokens_npy).tolist()
            n_generated = manifest["n_tokens"]
            ok = recorded[: len(ids)] == ids and len(recorded) == len(ids) + n_generated
            print(f"  npy     {'match' if ok else 'MISMATCH'} "
                  f"({len(ids)} prompt + {n_generated} generated = {len(recorded)})")


if __name__ == "__main__":
    main()
