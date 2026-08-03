"""
AI Generated Fixtures I can plug into the cpp code for testing.

----------

Prints one block per kernel, ready to drop into an anonymous namespace in
tests/kernels/kernels.cpp:

    constexpr float kLinearWeights[] = {1.0f, 2.0f, ...};  // [3, 4]

Source of truth per kernel:
  linear, softmax, silu_mul, argmax -> NumPy (the math is unambiguous)
  rmsnorm, rope                     -> Hugging Face itself (the CONVENTION is not)

The two taken from HF are exactly the two where a reasonable person could get the
convention wrong -- eps placement, and which element pairs with which. Both are
re-checked against the formula documented in docs/cpp-reference.md 18 every time this
runs, so a transformers upgrade that changes a convention fails here loudly instead of
silently in the parity gate three milestones later.

Usage:
    python scripts/fixtures.py                  # every kernel, to stdout
    python scripts/fixtures.py --kernel rope    # just one
    python scripts/fixtures.py --out tests/kernels/fixtures.inc
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)

SEED = 0
RMS_NORM_EPS = 1e-6 # config.rms_norm_eps
HEAD_DIM = 64 # config.hidden_size / config.num_attention_heads
WRAP_COLUMNS = 92

# ---------------------------------------------------------------- emitting C++

def cpp_float(v) -> str:
    """One float literal. 9 significant digits round-trips an fp32 exactly."""
    s = f"{float(v):.9g}"
    if not any(c in s for c in ".eE"):
        s += ".0"  # 1000f is not a valid literal; 1000.0f is
    return s + "f"

def emit_array(name: str, values, comment: str = "") -> str:
    """constexpr float kName[] = {...};  wrapped to WRAP_COLUMNS."""
    literals = [cpp_float(v) for v in np.asarray(values).reshape(-1)]
    head = f"constexpr float {name}[] = {{"
    tail = f"}};{'  // ' + comment if comment else ''}"

    lines, current = [], head
    for i, lit in enumerate(literals):
        piece = lit + (", " if i + 1 < len(literals) else "")
        if len(current) + len(piece) > WRAP_COLUMNS and current != head:
            lines.append(current.rstrip())
            current = "    " + piece
        else:
            current += piece
    lines.append(current + tail)
    return "\n".join(lines)

def emit_size(name: str, value: int) -> str:
    return f"constexpr std::size_t {name} = {value};"

# For cool logging
class Block:
    def __init__(self, kernel: str, summary: str):
        self.lines = [
            "// " + "-" * 70,
            f"// {kernel} -- {summary}",
            "// " + "-" * 70,
        ]

    def note(self, text: str) -> "Block":
        for line in text.strip().splitlines():
            self.lines.append(("// " + line.strip()).rstrip())
        return self

    def blank(self) -> "Block":
        self.lines.append("")
        return self

    def size(self, name: str, value: int) -> "Block":
        self.lines.append(emit_size(name, value))
        return self

    def array(self, name: str, values, comment: str = "") -> "Block":
        self.lines.append(emit_array(name, values, comment))
        return self

    def render(self) -> str:
        return "\n".join(self.lines)

def check(label: str, mine, theirs, tol: float) -> str:
    """Assert two arrays agree; return a one-line report for the output comment."""
    diff = float(np.abs(np.asarray(mine, dtype=np.float64) - np.asarray(theirs, dtype=np.float64)).max())
    if not diff < tol:
        sys.exit(
            f"\nFIXTURE CHECK FAILED: {label}\n"
            f"  max|diff| = {diff:.3e}, tolerance {tol:.0e}\n"
            f"  The formula in docs/cpp-reference.md 18 no longer matches this\n"
            f"  transformers build ({transformers.__version__}). Read the HF source\n"
            f"  before trusting any fixture below -- see 'Reading the reference\n"
            f"  implementation' in docs/roadmap_v2.md."
        )
    return f"checked vs the documented formula: max|diff| = {diff:.2e} (< {tol:.0e})"

# ---------------------------------------------------------------- kernels

def gen_linear(rng) -> Block:
    b = Block("linear", "y = W * x + bias")

    # Case 1: hand-checkable. Every row of W sums to a number you can verify by eye,
    # which is the cheapest possible test that the convention isn't transposed.
    out, in_ = 3, 4
    w = np.arange(1, out * in_ + 1, dtype=np.float32).reshape(out, in_)
    x = np.ones(in_, dtype=np.float32)
    bias = np.full(out, 0.5, dtype=np.float32)

    b.note(
    """
    Case 1 -- hand-checkable. W is 1..12 and x is all ones, so y[o] is
    (the sum of row o) + 0.5:  10.5, 26.5, 42.5. Verify it on paper once.
    A transposed convention caught here costs 10 minutes;
    """).blank()

    b.size("kLinearOut", out).size("kLinearIn", in_)
    b.array("kLinearWeights", w, f"[{out}, {in_}]")
    b.array("kLinearX", x, f"[{in_}]")
    b.array("kLinearBias", bias, f"[{out}]")
    b.array("kLinearExpected", w @ x + bias, f"[{out}]")
    b.blank()

    # Case 2: random, no bias -- exercises accumulation and the nullptr path.
    out2, in2 = 5, 7
    w2 = rng.standard_normal((out2, in2), dtype=np.float32)
    x2 = rng.standard_normal(in2, dtype=np.float32)
    b.note("""
        Case 2 -- random, pass bias = nullptr. Tolerance 1e-5: your accumulation order
        differs from NumPy's, and that is fine (ref 7).
    """).blank()
    b.size("kLinearOut2", out2).size("kLinearIn2", in2)
    b.array("kLinearWeights2", w2, f"[{out2}, {in2}]")
    b.array("kLinearX2", x2, f"[{in2}]")
    b.array("kLinearExpected2", w2.astype(np.float64) @ x2.astype(np.float64), f"[{out2}]")
    return b

def gen_rmsnorm(rng) -> Block:
    del rng  # fixed inputs; nothing random here
    b = Block("rmsnorm", "y_i = w_i * x_i / sqrt(mean(x^2) + eps)   [from HF]")

    def hf(x: np.ndarray, w: np.ndarray) -> np.ndarray:
        layer = Qwen2RMSNorm(x.shape[-1], eps=RMS_NORM_EPS)
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(w))
            return layer(torch.from_numpy(x)).numpy()

    def documented(x: np.ndarray, w: np.ndarray) -> np.ndarray:
        x64 = x.astype(np.float64)
        return w.astype(np.float64) * x64 / np.sqrt((x64**2).mean() + RMS_NORM_EPS)

    n = 4
    w = np.array([1.0, 2.0, 0.5, -1.0], dtype=np.float32)
    x = np.array([1.0, -2.0, 3.0, 0.25], dtype=np.float32)
    y = hf(x, w)

    # Tiny inputs: mean(x^2) is ~1e-16 here, so eps decides the answer. This is the
    # only case where rsqrt(mean + eps) and rsqrt(mean) + eps visibly disagree.
    x_tiny = np.array([1e-8, -1e-8, 2e-8, 0.0], dtype=np.float32)
    y_tiny = hf(x_tiny, w)

    report = check("rmsnorm vs documented formula", documented(x, w), y, 1e-6)
    b.note(f"""
        Generated with transformers {transformers.__version__}, eps = {RMS_NORM_EPS:g}.
        {report}

        There is no mean subtraction and no bias -- this is RMS norm, not LayerNorm.
        Pass eps in from ModelConfig::rms_norm_eps; never hardcode it in the kernel.
    """).blank()
    b.size("kRmsNormN", n)
    b.array("kRmsNormWeight", w, f"[{n}]")
    b.array("kRmsNormX", x, f"[{n}]")
    b.array("kRmsNormExpected", y, f"[{n}]")
    b.blank()
    b.note("""
        Tiny-input case: mean(x^2) ~ 1e-16, so eps dominates. If this and the case
        above both pass, your eps is inside the sqrt.
    """).blank()
    b.array("kRmsNormXTiny", x_tiny, f"[{n}]")
    b.array("kRmsNormExpectedTiny", y_tiny, f"[{n}]")
    return b

def gen_softmax(rng) -> Block:
    b = Block("softmax", "x_i <- exp(x_i - max) / sum(exp(x_j - max))   [in place]")

    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x.astype(np.float64) - x.max())
        return e / e.sum()

    plain = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shifted = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
    single = np.array([5.0], dtype=np.float32)
    wide = rng.standard_normal(8, dtype=np.float32) * 10.0

    b.note("""
        Case 2 is the whole test. It is case 1 shifted by +1000, so the correct answer
        is IDENTICAL -- but a softmax without the max-subtraction computes exp(1002),
        which overflows fp32 (the limit is ~88.7) and yields inf/inf = nan.

        Assert std::isfinite on the outputs, not just closeness: a nan compares false
        against everything, so a plain approx check fails with a confusing message
        instead of naming the actual problem.
    """).blank()
    b.size("kSoftmaxN", plain.size)
    b.array("kSoftmaxX", plain, "[3]")
    b.array("kSoftmaxExpected", softmax(plain), "[3]")
    b.blank()
    b.array("kSoftmaxXShifted", shifted, "[3] -- expects kSoftmaxExpected, unchanged")
    b.blank()
    b.note("Degenerate n = 1: must be exactly 1.0. It runs on every first token.").blank()
    b.array("kSoftmaxXSingle", single, "[1]")
    b.array("kSoftmaxExpectedSingle", softmax(single), "[1]")
    b.blank()
    b.note("Wider spread, the realistic case -- attention scores reach the tens.").blank()
    b.size("kSoftmaxWideN", wide.size)
    b.array("kSoftmaxXWide", wide, f"[{wide.size}]")
    b.array("kSoftmaxExpectedWide", softmax(wide), f"[{wide.size}]")
    return b

def gen_rope(rng, model_dir: Path) -> Block:
    b = Block("rope", "split-half rotary embedding, per 64-wide head   [from HF]")

    if not (model_dir / "config.json").exists():
        sys.exit(
            f"rope fixtures need the real config for rope_theta, but\n"
            f"  {model_dir / 'config.json'}\n"
            f"is missing. Download it (see README) or pass --model."
        )

    cfg = AutoConfig.from_pretrained(model_dir)
    # transformers 5.x moved this under rope_parameters; 4.x had cfg.rope_theta.
    params = getattr(cfg, "rope_parameters", None) or {}
    theta = float(params.get("rope_theta") or getattr(cfg, "rope_theta"))

    rot = Qwen2RotaryEmbedding(cfg) # type: ignore
    x = rng.standard_normal(HEAD_DIM, dtype=np.float32)

    def hf(vec: np.ndarray, pos: int) -> np.ndarray:
        q = torch.from_numpy(vec).view(1, 1, 1, HEAD_DIM)  # [batch, heads, seq, head_dim]
        cos, sin = rot(q, torch.tensor([[pos]]))
        rotated, _ = apply_rotary_pos_emb(q, q, cos, sin)
        return rotated[0, 0, 0].numpy()

    def documented(vec: np.ndarray, pos: int) -> np.ndarray:
        """The formula exactly as docs/cpp-reference.md 18 states it."""
        v = vec.astype(np.float64)
        out = np.empty(HEAD_DIM, dtype=np.float64)
        half = HEAD_DIM // 2
        for j in range(half):
            angle = pos * theta ** (-2.0 * j / HEAD_DIM)
            c, s = np.cos(angle), np.sin(angle)
            out[j] = v[j] * c - v[j + half] * s
            out[j + half] = v[j + half] * c + v[j] * s
        return out

    # The convention is only testable where arithmetic precision isn't in the way.
    # angle = pos * inv_freq, so an fp32 inv_freq (which is what HF stores) carries a
    # ~6e-8 relative error that the multiply turns into an ABSOLUTE angle error of
    # ~pos * 6e-8 radians. cos/sin have derivative <= 1, so the output error grows
    # linearly with position: negligible at pos 3, ~3e-5 by pos 1000. That is not a
    # bug in anyone's code -- it is the same formula evaluated in a different order.
    positions = [0, 3, 1000]
    tolerances = {0: 1e-7, 3: 1e-6, 1000: 1e-4}
    outputs = {p: hf(x, p) for p in positions}
    reports = {p: check(f"rope at pos {p}", documented(x, p), outputs[p], tolerances[p])
               for p in positions}
    measured = {p: float(np.abs(documented(x, p) - outputs[p].astype(np.float64)).max())
                for p in positions}

    b.note(f"""
        Generated with transformers {transformers.__version__}, theta = {theta:g}.
        {reports[3]}

        Element j pairs with element j + 32 -- half a head away, NOT the adjacent
        (2j, 2j+1) of the original paper. Wrong pairing gives fluent, confident,
        completely wrong text and fails parity at block 0.

        Apply this to Q and K only, never V. One call per head: rotating all 896
        floats at once would pair element 0 with element 448, which is meaningless.

        USE A PER-POSITION TOLERANCE. angle = pos * inv_freq, so the fp32 relative
        error in inv_freq (~6e-8) becomes an absolute angle error of ~pos * 6e-8
        radians, and cos/sin turn that straight into output error. Measured here,
        float64 reference vs HF's float32 pipeline:
""" + "\n".join(f"          pos {p:<5d} max|diff| = {measured[p]:.2e}   -> test at {tolerances[p]:.0e}"
                for p in positions) + f"""

        So a 1e-6 tolerance at pos 1000 would fail a perfectly correct kernel. If you
        want the tightest possible agreement, compute inv_freq in float (matching HF)
        rather than double -- that halves the gap, but it does not close it, because
        HF builds inv_freq as 1/(base**(arange/dim)) and you will not reproduce its
        rounding exactly.
    """).blank()
    b.size("kRopeDim", HEAD_DIM)
    b.array("kRopeTheta", [theta], "scalar -- read it from ModelConfig, not from here")
    b.array("kRopeX", x, f"[{HEAD_DIM}] -- the same input at all three positions")
    for p in positions:
        b.blank()
        if p == 0:
            b.note("""
                pos = 0 makes every angle 0, so cos = 1, sin = 0 and rope is the
                IDENTITY -- the expected output below is the input, unchanged.
                Check this one first: if it fails, your loop bounds are wrong and
                nothing else will make sense.
            """)
        b.size(f"kRopePos{p}", p)
        b.array(f"kRopeExpectedPos{p}", outputs[p], f"[{HEAD_DIM}] at pos {p}")
    return b

def gen_silu_mul(rng) -> Block:
    b = Block("silu_mul", "gate_i <- silu(gate_i) * up_i,  silu(z) = z / (1 + e^-z)")

    def silu(z: np.ndarray) -> np.ndarray:
        z64 = z.astype(np.float64)
        return z64 / (1.0 + np.exp(-z64))

    # up = ones isolates the activation, so a failure is silu itself and not the
    # multiply or the argument order.
    gate = np.array([0.0, 1.0, -1.0, 5.0, -5.0, 0.5, -0.5, 100.0, -100.0], dtype=np.float32)
    ones = np.ones_like(gate)

    n = 8
    gate2 = rng.standard_normal(n, dtype=np.float32)
    up2 = rng.standard_normal(n, dtype=np.float32)

    b.note("""
        Case 1 -- up is all ones, isolating silu. Includes +-100: for very negative z,
        e^-z overflows to inf and z/inf gives -0, which is the correct limit; for very
        positive z, e^-z underflows to 0 and the result is z. Both correct, so the
        kernel needs NO clamping -- adding one would diverge from HF.
    """).blank()
    b.size("kSiluN", gate.size)
    b.array("kSiluGate", gate, f"[{gate.size}]")
    b.array("kSiluUpOnes", ones, f"[{gate.size}]")
    b.array("kSiluExpected", silu(gate) * ones, f"[{gate.size}]")
    b.blank()
    b.note("""
        Case 2 -- real random vectors. silu applies to GATE and the result multiplies
        UP; swapping them is silent and wrong. HF: down(act(gate(x)) * up(x)).
    """).blank()
    b.size("kSiluN2", n)
    b.array("kSiluGate2", gate2, f"[{n}]")
    b.array("kSiluUp2", up2, f"[{n}]")
    b.array("kSiluExpected2", silu(gate2) * up2.astype(np.float64), f"[{n}]")
    return b

def gen_argmax(rng) -> Block:
    b = Block("argmax", "index of the largest element, ties to the LOWEST index")

    plain = np.array([0.1, 0.9, 0.3], dtype=np.float32)
    tie = np.array([3.0, 3.0, 1.0], dtype=np.float32)
    wide = rng.standard_normal(16, dtype=np.float32)

    b.note("""
        The tie case pins the behavior NumPy and torch have, which is what parity
        compares against: a strict > comparison (never >=) gives it to you for free.
        Exact ties among 151936 logits are rare, but 20 prompts x 128 steps is enough
        trials that "rare" is not "never".

        Not fixture-able, but do it anyway: INFERNO_CHECK that the winning value is
        finite. Every comparison with nan is false, so a strict > loop silently
        returns index 0 -- you would generate token 0 forever and blame the tokenizer.
    """).blank()
    b.size("kArgmaxN", plain.size)
    b.array("kArgmaxX", plain, "[3]")
    b.size("kArgmaxExpected", int(plain.argmax()))
    b.blank()
    b.array("kArgmaxXTie", tie, "[3] -- both 3.0")
    b.size("kArgmaxExpectedTie", int(tie.argmax()))
    b.blank()
    b.size("kArgmaxWideN", wide.size)
    b.array("kArgmaxXWide", wide, f"[{wide.size}]")
    b.size("kArgmaxExpectedWide", int(wide.argmax()))
    return b

GENERATORS = {
    "linear": gen_linear,
    "rmsnorm": gen_rmsnorm,
    "softmax": gen_softmax,
    "rope": gen_rope,
    "silu_mul": gen_silu_mul,
    "argmax": gen_argmax,
}

# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel", choices=sorted(GENERATORS), action="append", help="only this kernel (repeatable; default: all)")
    parser.add_argument("--model", type=Path, default=Path("models/qwen2.5-0.5b-instruct"), help="model directory")
    parser.add_argument("--seed", type=int, default=SEED, help="NumPy seed")
    parser.add_argument("--out", type=Path, help="writes here instead of to terminal")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    wanted = args.kernel or list(GENERATORS)

    blocks = []
    for name in wanted:
        # re-seeding each time for consistency
        rng = np.random.default_rng(args.seed)

        gen = GENERATORS[name]
        # running the respective generator
        blocks.append(gen(rng, args.model) if name == "rope" else gen(rng))

    header = [
        "// Generated by scripts/fixtures.py -- do not edit by hand.",
        f"// Regenerate: python scripts/fixtures.py --seed {args.seed}",
        "//",
        "// Paste into an anonymous namespace in tests/kernels/kernels.cpp.",
        "// Tolerances: 1e-5 for linear, 1e-6 for everything else, and see the rope",
        "// block for why that one needs a per-position tolerance.",
        "//",
        "",
    ]
    text = "\n".join(header) + "\n\n".join(b.render() for b in blocks) + "\n"

    # For writing into a file
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {len(text.splitlines())} lines to {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)

if __name__ == "__main__":
    main()
