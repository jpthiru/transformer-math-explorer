---
title: Transformer Math Explorer
emoji: 🧮
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
---

# 🧮 Transformer Math Explorer

Type **any sentence** and watch it flow through a single Transformer block — every
matrix and equation shown step by step, then **download the whole derivation as a PDF**.

This is the interactive companion to the article *“What Is an LLM, and What Are
Transformer Models?”*. It makes the architecture concrete: real numbers, real matrix
multiplications, the actual softmax — at a readable teaching scale.

**Live demo:** _(add your Hugging Face Space URL here once deployed)_

---

## What it shows

For your sentence it computes and displays, with live KaTeX-rendered matrices:

1. **Embedding** — each token → a row vector (matrix `X`)
2. **Layer Normalization** — the classic *(x − mean) / √variance* formula, plus modern **RMSNorm**
3. **Multi-Head Attention** — `W_Q, W_K, W_V` projections, **RoPE** rotary positions,
   per-head scores `QKᵀ/√dₖ`, causal masking, softmax, `W_O`
4. **Dropout** — and why it's the identity at inference
5. **Residual** connection — `Z = X + Attn`
6. **Layer Normalization** (again)
7. **Feed-Forward Network** — gated **SwiGLU**
8. **Residual → output** — and the final LM head / softmax over a demo vocabulary

Teaching dimensions are fixed at **d_model = 8, heads = 2** so every matrix stays printable.

## How it works (and honest caveats)

- **Tokenizer:** real BPE via `tiktoken` (`cl100k_base`) when the host has internet;
  falls back to an approximate subword splitter offline.
- **Weights** (`W_Q`, `W_K`, …) are **fixed illustrative numbers** seeded for reproducibility —
  a real model *learns* these from data.
- **Embeddings** are deterministic per token (same word → same vector), via hashing.
- The **next-word prediction** uses a small **untrained** demo vocabulary; it demonstrates the
  *mechanism* (logits → softmax), not trained knowledge.

The point isn't to be a real model — it's to make the **exact arithmetic** of a Transformer
block visible and tangible.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Roadmap

**v1 (this release): predict-next-word, one block, fully visualized.** Grow from here:

- [ ] Stack multiple blocks and show the representation evolving layer by layer
- [ ] A real (small) trained LM head so predictions are meaningful
- [ ] Toggle GQA / MLA attention variants and a Mixture-of-Experts FFN
- [ ] Adjustable dimensions and side-by-side head comparison
- [ ] Attention heatmap visualizations

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI (on-screen math + PDF button) |
| `core.py` | Tokenize + the full single-block math (NumPy) |
| `pdfgen.py` | Builds the downloadable PDF (reportlab) |
| `requirements.txt` | Dependencies |

## License

MIT — see `LICENSE`.
