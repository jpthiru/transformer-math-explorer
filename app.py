import streamlit as st
import numpy as np
from core import compute
from pdfgen import build_pdf

st.set_page_config(page_title="Transformer Math Explorer", page_icon="🧮", layout="wide")


def bm(M, name=None):
    if M.ndim == 1:
        M = M.reshape(1, -1)
    rows = r" \\ ".join(" & ".join(f"{v:.2f}" for v in row) for row in M)
    body = r"\begin{bmatrix}" + rows + r"\end{bmatrix}"
    return (name + " = " if name else "") + body


st.title("🧮 Inside the Transformer — the math, step by step")
st.caption("Type any sentence and watch it flow through a single Transformer block: "
           "embedding → RMSNorm → multi-head attention → dropout → residual → FFN (SwiGLU) → output. "
           "Teaching dimensions **d_model = 8, heads = 2**. Companion to the deep-dive article.")

with st.sidebar:
    st.header("About")
    st.markdown(
        "- **Real BPE tokenizer** (tiktoken) when online; approximate subword offline.\n"
        "- Weights are **fixed illustrative** numbers — a real model *learns* them.\n"
        "- Embeddings are **deterministic per token** (same word → same vector).\n"
        "- Everything is rounded to 2 decimals.\n"
        "- The final next-word head is an **untrained demo** over a small vocabulary."
    )
    st.markdown("Made as a companion to *“What Is an LLM, and What Are Transformer Models?”*")

sent = st.text_input("Your sentence",
                     value="Weak quarterly results led the board to replace the",
                     help="Up to 12 tokens are shown so the matrices stay readable.")

r = compute(sent)

c1, c2 = st.columns([3, 1])
with c1:
    st.markdown(f"**Tokenizer:** {r['tname']} &nbsp;•&nbsp; **{r['n']} tokens**", unsafe_allow_html=True)
    st.markdown(" ".join(
        f"<span style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:6px;"
        f"padding:2px 8px;margin:2px;display:inline-block;font-family:monospace'>{t.strip() or '␣'}</span>"
        for t in r['tokens']), unsafe_allow_html=True)
with c2:
    st.download_button("⬇️ Download full PDF", data=build_pdf(r),
                       file_name="transformer-math.pdf", mime="application/pdf",
                       use_container_width=True)

st.divider()

# Module 1
st.subheader("Module 1 — Embedding (+ position)")
st.write("A lookup table turns each token into a row vector. Stacked, that is the input matrix **X** (one row per token).")
st.latex(bm(r['X'], "X"))

# Module 2
st.subheader("Module 2 — Layer Normalization")
st.markdown("**Classic LayerNorm** — *subtract the mean, divide by the square root of the variance*, then scale/shift:")
st.latex(r"\mu=\tfrac1d\sum_j x_j,\quad \sigma^2=\tfrac1d\sum_j (x_j-\mu)^2,\quad "
         r"\hat{x}_j=\frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}}\,\gamma_j+\beta_j")
st.markdown("**Modern models use RMSNorm** (no mean subtraction) — cheaper, just as stable:")
st.latex(r"\hat{x}_j=\frac{x_j}{\mathrm{RMS}(x)},\qquad \mathrm{RMS}(x)=\sqrt{\tfrac1d\sum_j x_j^2+\epsilon}")
st.latex(bm(r['Xn'], "X_n"))

# Module 3
st.subheader("Module 3 — Multi-Head Attention")
st.markdown("**(a) Project to Query, Key, Value** with learned 8×8 matrices: $Q=X_nW_Q,\\;K=X_nW_K,\\;V=X_nW_V$.")
with st.expander("Show the learned projection matrices W_Q, W_K, W_V"):
    st.latex(bm(r['Wq'], "W_Q")); st.latex(bm(r['Wk'], "W_K")); st.latex(bm(r['Wv'], "W_V"))
st.latex(bm(r['Q'], "Q")); st.latex(bm(r['K'], "K")); st.latex(bm(r['V'], "V"))

st.markdown("**(b) Inject position with RoPE** — rotate each pair of dims of $Q,K$ by "
            r"$\theta_{p,i}=p\cdot10000^{-2i/d}$ (position 0 unrotated):")
st.latex(r"\begin{bmatrix}x'_{2i}\\ x'_{2i+1}\end{bmatrix}="
         r"\begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}"
         r"\begin{bmatrix}x_{2i}\\ x_{2i+1}\end{bmatrix}")
with st.expander("Show Q, K after RoPE"):
    st.latex(bm(r['Qr'], "Q^{rope}")); st.latex(bm(r['Kr'], "K^{rope}"))

st.markdown("**(c) Scores, causal mask, softmax — per head.** "
            r"$S=\dfrac{Q_h K_h^{\top}}{\sqrt{d_k}}$, mask future tokens to $-\infty$, then softmax each row.")
for h in range(len(r['A_heads'])):
    st.markdown(f"*Head {h+1} attention weights (rows sum to 1; upper triangle is 0):*")
    st.latex(bm(r['A_heads'][h], f"A^{{({h+1})}}"))
    st.latex(bm(r['O_heads'][h], f"O^{{({h+1})}}"))

st.markdown("**(d) Concatenate heads and project with $W_O$:** $\\;\\text{Attn}=[\\,O^{(1)}|O^{(2)}\\,]\\,W_O$.")
st.latex(bm(r['Ocat'], r"O_{\text{cat}}")); st.latex(bm(r['Attn'], r"\text{Attn}"))

# Module 4
st.subheader("Module 4 — Dropout")
st.markdown("Training-only regularizer: zero a random fraction $p$ and rescale by $1/(1-p)$. "
            "**At inference it is the identity** — Attn passes through unchanged.")

# Module 5
st.subheader("Module 5 — Residual (skip) connection")
st.markdown("Add the attention output back to the block's original input: $Z = X + \\text{Attn}$.")
st.latex(bm(r['Z'], "Z"))

# Module 6
st.subheader("Module 6 — Layer Normalization (again)")
st.latex(r"Z_n=\mathrm{RMSNorm}(Z)")
st.latex(bm(r['Zn'], "Z_n"))

# Module 7
st.subheader("Module 7 — Feed-Forward Network (SwiGLU)")
st.latex(r"\mathrm{FFN}(z)=\big(\mathrm{SiLU}(zW_{gate})\odot (zW_{up})\big)W_{down},\qquad "
         r"\mathrm{SiLU}(x)=\frac{x}{1+e^{-x}}")
st.caption("In 2026 frontier models this single FFN is replaced by a Mixture-of-Experts (sparse routing).")
st.latex(bm(r['F'], "F"))

# Module 8
st.subheader("Module 8 — Residual → output")
st.markdown("A second residual finishes the block: $Y = Z + F$. This becomes the input to the next block.")
st.latex(bm(r['Y'], "Y"))

# Output
st.subheader("Output — final norm + LM head")
st.markdown("After the final block, normalize and project the **last token's** vector onto the vocabulary, "
            "then softmax. *(Illustrative untrained head over a small demo vocabulary.)*")
top = r['topk']
cols = st.columns(len(top))
for col, (tok, p) in zip(cols, top):
    col.metric(tok, f"{p*100:.0f}%")
st.bar_chart({"probability": {v: float(p) for v, p in zip(r['vocab'], r['probs'])}})

st.divider()
st.download_button("⬇️ Download this as a PDF", data=build_pdf(r),
                   file_name="transformer-math.pdf", mime="application/pdf")
st.caption("Weights are illustrative, not from a trained model. The same equations run at "
           "billions-of-parameters scale in real LLMs like GPT, Claude and Gemini.")
