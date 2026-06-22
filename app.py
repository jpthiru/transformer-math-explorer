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


def _esc(s):
    s = str(s).strip()
    if not s:
        return "(sp)"
    for a, b in [("\\", " "), ("{", "("), ("}", ")"), ("$", ""), ("_", "-"),
                 ("%", "pct"), ("&", "+"), ("#", ""), ("^", " "), ("~", "-")]:
        s = s.replace(a, b)
    return s


def bml(M, name, labels):
    """Per-token matrix with each row labelled by its token (first column)."""
    if M.ndim == 1:
        M = M.reshape(1, -1)
    spec = "c|" + "c" * M.shape[1]
    rows = []
    for i in range(M.shape[0]):
        lab = r"\text{" + _esc(labels[i]) + "}"
        nums = " & ".join(f"{v:.2f}" for v in M[i])
        rows.append(lab + " & " + nums)
    body = r"\left[\begin{array}{" + spec + "}" + r" \\ ".join(rows) + r"\end{array}\right]"
    return (name + " = " if name else "") + body


def bml_attn(M, name, labels):
    """Attention grid: rows AND columns labelled by token (row = querying token)."""
    if M.ndim == 1:
        M = M.reshape(1, -1)
    ncols = M.shape[1]
    spec = "c|" + "c" * ncols
    header = " & " + " & ".join(r"\text{" + _esc(labels[j]) + "}" for j in range(ncols))
    datarows = []
    for i in range(M.shape[0]):
        lab = r"\text{" + _esc(labels[i]) + "}"
        nums = " & ".join(f"{v:.2f}" for v in M[i])
        datarows.append(lab + " & " + nums)
    content = header + r" \\ \hline " + r" \\ ".join(datarows)
    body = r"\left[\begin{array}{" + spec + "}" + content + r"\end{array}\right]"
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
st.write("A lookup table turns each token into a row vector. Stacked, that is the input matrix **X** — "
         "each **row is labelled with its token** so you can see which word is which.")
st.latex(bml(r['X'], "X", r['tokens']))

# Module 2
st.subheader("Module 2 — Layer Normalization")
st.markdown("**Classic LayerNorm** — *subtract the mean, divide by the square root of the variance*, then scale/shift:")
st.latex(r"\mu=\tfrac1d\sum_j x_j,\quad \sigma^2=\tfrac1d\sum_j (x_j-\mu)^2,\quad "
         r"\hat{x}_j=\frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}}\,\gamma_j+\beta_j")
st.markdown("**Modern models use RMSNorm** (no mean subtraction) — cheaper, just as stable:")
st.latex(r"\hat{x}_j=\frac{x_j}{\mathrm{RMS}(x)},\qquad \mathrm{RMS}(x)=\sqrt{\tfrac1d\sum_j x_j^2+\epsilon}")
st.latex(bml(r['Xn'], "X_n", r['tokens']))

# Module 3
st.subheader("Module 3 — Multi-Head Attention")
st.markdown("**(a) Project to Query, Key, Value** with learned 8×8 matrices: $Q=X_nW_Q,\\;K=X_nW_K,\\;V=X_nW_V$.")
with st.expander("Show the learned projection matrices W_Q, W_K, W_V"):
    st.latex(bm(r['Wq'], "W_Q")); st.latex(bm(r['Wk'], "W_K")); st.latex(bm(r['Wv'], "W_V"))
st.latex(bml(r['Q'], "Q", r['tokens'])); st.latex(bml(r['K'], "K", r['tokens'])); st.latex(bml(r['V'], "V", r['tokens']))

st.markdown("**(b) Inject position with RoPE** — rotate each pair of dims of $Q,K$ by "
            r"$\theta_{p,i}=p\cdot10000^{-2i/d}$ (position 0 unrotated):")
st.latex(r"\begin{bmatrix}x'_{2i}\\ x'_{2i+1}\end{bmatrix}="
         r"\begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}"
         r"\begin{bmatrix}x_{2i}\\ x_{2i+1}\end{bmatrix}")
with st.expander("Show Q, K after RoPE"):
    st.latex(bml(r['Qr'], "Q^{rope}", r['tokens'])); st.latex(bml(r['Kr'], "K^{rope}", r['tokens']))

st.markdown("**(c) Scores, causal mask, softmax — per head.** "
            r"$S=\dfrac{Q_h K_h^{\top}}{\sqrt{d_k}}$, mask future tokens to $-\infty$, then softmax each row.")
for h in range(len(r['A_heads'])):
    st.markdown(f"*Head {h+1} attention weights — row = the token doing the looking, column = the token "
                "being looked at (rows sum to 1; upper triangle is 0):*")
    st.latex(bml_attn(r['A_heads'][h], f"A^{{({h+1})}}", r['tokens']))
    st.latex(bml(r['O_heads'][h], f"O^{{({h+1})}}", r['tokens']))

st.markdown("**(d) Concatenate heads and project with $W_O$:** $\\;\\text{Attn}=[\\,O^{(1)}|O^{(2)}\\,]\\,W_O$.")
st.latex(bml(r['Ocat'], r"O_{\text{cat}}", r['tokens'])); st.latex(bml(r['Attn'], r"\text{Attn}", r['tokens']))

# Module 4
st.subheader("Module 4 — Dropout")
st.markdown("Training-only regularizer: zero a random fraction $p$ and rescale by $1/(1-p)$. "
            "**At inference it is the identity** — Attn passes through unchanged.")

# Module 5
st.subheader("Module 5 — Residual (skip) connection")
st.markdown("Add the attention output back to the block's original input: $Z = X + \\text{Attn}$.")
st.latex(bml(r['Z'], "Z", r['tokens']))

# Module 6
st.subheader("Module 6 — Layer Normalization (again)")
st.latex(r"Z_n=\mathrm{RMSNorm}(Z)")
st.latex(bml(r['Zn'], "Z_n", r['tokens']))

# Module 7
st.subheader("Module 7 — Feed-Forward Network (SwiGLU)")
st.latex(r"\mathrm{FFN}(z)=\big(\mathrm{SiLU}(zW_{gate})\odot (zW_{up})\big)W_{down},\qquad "
         r"\mathrm{SiLU}(x)=\frac{x}{1+e^{-x}}")
st.caption("In 2026 frontier models this single FFN is replaced by a Mixture-of-Experts (sparse routing).")
st.latex(bml(r['F'], "F", r['tokens']))

# Module 8
st.subheader("Module 8 — Residual → output")
st.markdown("A second residual finishes the block: $Y = Z + F$. This becomes the input to the next block.")
st.latex(bml(r['Y'], "Y", r['tokens']))

# Output
st.subheader("Output — final norm + LM head")
st.markdown("After the final block, normalize and project the **last token's** vector onto a vocabulary, "
            "then softmax. Here the vocabulary is **your sentence's own words** and the head is **untrained**, "
            "so this shows the *mechanism* (logits → softmax) — not a real next-word prediction.")
if r.get('top_attended'):
    att = ", ".join(f"“{w}” ({p:.2f})" for w, p in r['top_attended'])
    st.info(f"**For your sentence**, the final position attends most to: {att}. "
            "(This updates with whatever sentence you type.)")
top = r['topk']
cols = st.columns(max(1, len(top)))
for col, (tok, p) in zip(cols, top):
    col.metric(tok, f"{p*100:.0f}%")
st.bar_chart({"probability": {v: float(p) for v, p in zip(r['vocab'], r['probs'])}})

st.divider()
st.download_button("⬇️ Download this as a PDF", data=build_pdf(r),
                   file_name="transformer-math.pdf", mime="application/pdf")
st.caption("Weights are illustrative, not from a trained model. The same equations run at "
           "billions-of-parameters scale in real LLMs like GPT, Claude and Gemini.")
