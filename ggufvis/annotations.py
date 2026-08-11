"""Generate the final-v1 structured annotation cards."""

from __future__ import annotations

from .diagram import Diagram


SECTION_BY_RESULT = {
    "embedding_x": "EMBEDDING",
    "x": "BLOCK INPUT",
    "x_norm": "ATTENTION",
    "k": "ATTENTION",
    "k_norm": "ATTENTION",
    "k_rope": "ATTENTION",
    "v_projection": "ATTENTION",
    "q": "ATTENTION",
    "q_norm": "ATTENTION",
    "q_rope": "ATTENTION",
    "a": "ATTENTION",
    "a_softmax": "ATTENTION",
    "h": "ATTENTION",
    "h_merge": "ATTENTION",
    "y": "ATTENTION",
    "r": "ATTENTION",
    "r_norm": "MLP",
    "g": "MLP",
    "u": "MLP",
    "p": "MLP",
    "m": "MLP",
    "x_out": "MLP",
    "z": "FINAL",
    "logits": "FINAL",
}

OPERATION_BY_RESULT = {
    "embedding_x": "Token embedding",
    "x": "Route hidden state into block",
    "x_norm": "Attention RMSNorm",
    "k": "K projection",
    "k_norm": "K per-head RMSNorm",
    "k_rope": "K rotary position encoding",
    "v_projection": "V projection",
    "q": "Q projection",
    "q_norm": "Q per-head RMSNorm",
    "q_rope": "Q rotary position encoding",
    "a": "Attention scores",
    "a_softmax": "Causal softmax",
    "h": "Weighted value aggregation",
    "h_merge": "Head concatenation",
    "y": "Attention output projection",
    "r": "Attention residual",
    "r_norm": "MLP RMSNorm",
    "g": "Gate projection",
    "u": "Up projection",
    "p": "Gated activation",
    "m": "Down projection",
    "x_out": "MLP residual",
    "z": "Final RMSNorm",
    "logits": "Language-model head",
}


def _symbol(diagram: Diagram, key: str) -> str:
    """Return the compact variable name used in Model View equations."""
    return diagram.matrix(key).label


def equation_for(diagram: Diagram, result: str) -> str:
    """Return a compact symbolic equation for one result matrix."""
    symbol = lambda key: _symbol(diagram, key)
    config = diagram.config
    bias = config.qkv_bias
    k_input = "k_norm" if config.qk_norm else "k"
    q_input = "q_norm" if config.qk_norm else "q"
    equations = {
        "embedding_x": (
            f"{symbol('embedding_x')} = {symbol('token_embedding')} × "
            f"{symbol('token_one_hot')}\n"
            f"vocab = {config.vocab_size}"
        ),
        "x": (
            f"{diagram.matrix('x').label} = "
            f"{diagram.matrix('embedding_x').label} when b=0;\n"
            f"otherwise output of block b−1"
        ),
        "x_norm": (
            f"{symbol('x_norm')} = RMSNorm({symbol('x')}; {symbol('gamma')})"
        ),
        "k": (
            f"{symbol('k')} = {symbol('wk')} × {symbol('x_norm')}"
            + (" + bk" if bias else "")
        ),
        "k_rope": f"{symbol('k_rope')} = RoPE({symbol(k_input)})",
        "v_projection": (
            f"{symbol('v_projection')} = {symbol('wv')} × {symbol('x_norm')}"
            + (" + bv" if bias else "")
        ),
        "q": (
            f"{symbol('q')} = {symbol('wq')} × {symbol('x_norm')}"
            + (" + bq" if bias else "")
        ),
        "q_rope": f"{symbol('q_rope')} = RoPE({symbol(q_input)})",
        "a": f"{symbol('a')} = {symbol('kt_rope')} × {symbol('q_rope')}",
        "a_softmax": (
            f"{symbol('a_softmax')} = softmax({symbol('a')}/"
            f"√{config.key_head_dim} + Mᶜ)"
        ),
        "h": (
            f"{symbol('h')} = {symbol('v_attention')} × "
            f"{symbol('a_softmax')}"
        ),
        "h_merge": f"{symbol('h_merge')} = Concat({symbol('h')})",
        "y": f"{symbol('y')} = {symbol('wo')} × {symbol('h_merge')}",
        "r": f"{symbol('r')} = {symbol('x_residual')} + {symbol('y')}",
        "r_norm": (
            f"{symbol('r_norm')} = RMSNorm({symbol('r')}; "
            f"{symbol('gamma_post')})"
        ),
        "g": f"{symbol('g')} = {symbol('wg')} × {symbol('r_norm')}",
        "u": f"{symbol('u')} = {symbol('wu')} × {symbol('r_norm')}",
        "p": (
            f"{symbol('p')} = {config.activation}({symbol('g')}) "
            f"⊙ {symbol('u')}"
        ),
        "m": f"{symbol('m')} = {symbol('wd')} × {symbol('p')}",
        "x_out": f"{symbol('x_out')} = {symbol('r_final')} + {symbol('m')}",
        "z": (
            f"{symbol('z')} = RMSNorm({symbol('x_out')}; "
            f"{symbol('gamma_final')})"
        ),
        "logits": f"{symbol('logits')} = {symbol('wlm')} × {symbol('z')}",
    }
    if config.qk_norm:
        equations["k_norm"] = (
            f"{symbol('k_norm')} = RMSNorm({symbol('k')}; {symbol('gamma_k')})"
        )
        equations["q_norm"] = (
            f"{symbol('q_norm')} = RMSNorm({symbol('q')}; {symbol('gamma_q')})"
        )
    return equations[result]


def tensor_source_for(diagram: Diagram, result: str) -> str:
    """Return the canonical GGUF tensor source, never a weight value."""
    config = diagram.config
    bias = config.qkv_bias
    sources = {
        "embedding_x": "token_embd.weight",
        "x": "none (runtime activation routing)",
        "x_norm": "blk.b.attn_norm.weight",
        "k": (
            "blk.b.attn_k.weight + blk.b.attn_k.bias"
            if bias
            else "blk.b.attn_k.weight"
        ),
        "k_norm": "blk.b.attn_k_norm.weight",
        "k_rope": "none (parameter-free)",
        "v_projection": (
            "blk.b.attn_v.weight + blk.b.attn_v.bias"
            if bias
            else "blk.b.attn_v.weight"
        ),
        "q": (
            "blk.b.attn_q.weight + blk.b.attn_q.bias"
            if bias
            else "blk.b.attn_q.weight"
        ),
        "q_norm": "blk.b.attn_q_norm.weight",
        "q_rope": "none (parameter-free)",
        "a": "none (parameter-free)",
        "a_softmax": "none (parameter-free)",
        "h": "none (parameter-free)",
        "h_merge": "none (parameter-free)",
        "y": "blk.b.attn_output.weight",
        "r": "none (parameter-free)",
        "r_norm": "blk.b.ffn_norm.weight",
        "g": "blk.b.ffn_gate.weight",
        "u": "blk.b.ffn_up.weight",
        "p": "none (parameter-free)",
        "m": "blk.b.ffn_down.weight",
        "x_out": "none (parameter-free)",
        "z": "output_norm.weight",
        "logits": (
            "token_embd.weight (tied)"
            if config.output_is_tied
            else "output.weight"
        ),
    }
    return sources[result]


def card_lines(diagram: Diagram, step: int) -> tuple[str, str, str, str]:
    """Return title, operation, equation, and source rows."""
    result = diagram.operations[step].result
    equation = equation_for(diagram, result)
    equation = equation.replace("\n", "\n" + " " * len("Equation: "))
    return (
        f"[{step + 1:02d}/{len(diagram.operations):02d}] "
        f"{SECTION_BY_RESULT[result]}",
        f"Operation: {OPERATION_BY_RESULT[result]}",
        f"Equation: {equation}",
        f"Source: {tensor_source_for(diagram, result)}",
    )


def maximum_card_width(diagram: Diagram) -> int:
    """Pre-calculate a stable card width for every navigation step."""
    longest = max(
        len(physical_line)
        for step in range(len(diagram.operations))
        for line in card_lines(diagram, step)
        for physical_line in line.splitlines()
    )
    return longest + 4
