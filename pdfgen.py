"""Build a downloadable PDF mirroring the deep-dive, for any sentence."""
import io
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Preformatted,
                                Table, TableStyle, HRFlowable)

INDIGO = HexColor('#4338CA'); INK = HexColor('#1E293B')
SLATE = HexColor('#475569'); AMBER = HexColor('#B45309')


def _mat(M, name):
    if M.ndim == 1:
        M = M.reshape(1, -1)
    col = [max(len(f"{M[i, j]:.2f}") for i in range(M.shape[0])) for j in range(M.shape[1])]
    rows = ["  ".join(f"{M[i, j]:>{col[j]}.2f}" for j in range(M.shape[1])) for i in range(M.shape[0])]
    return f"{name} =\n" + "\n".join(rows)


def _word(w):
    s = w.strip()
    if not s:
        s = "(space)"
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return ("·" + s) if w[:1] == " " else s


def build_pdf(r, preds=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm)
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle('H1', parent=ss['Title'], textColor=INK, fontSize=20, spaceAfter=4)
    SUB = ParagraphStyle('SUB', parent=ss['Normal'], textColor=SLATE, fontSize=10, spaceAfter=8)
    H2 = ParagraphStyle('H2', parent=ss['Heading2'], textColor=INDIGO, fontSize=14, spaceBefore=10, spaceAfter=4)
    BODY = ParagraphStyle('BODY', parent=ss['Normal'], textColor=INK, fontSize=9.5, leading=13, spaceAfter=4)
    EQ = ParagraphStyle('EQ', parent=ss['Normal'], textColor=INK, fontSize=10, leading=14, spaceAfter=4, leftIndent=6)
    MONO = ParagraphStyle('MONO', parent=ss['Code'], fontName='Courier', fontSize=7.2, leading=8.6,
                          textColor=INK, backColor=HexColor('#F8FAFC'), borderPadding=4, spaceAfter=6)
    e = []
    e.append(Paragraph("Inside the Transformer: The Math, Step by Step", H1))
    e.append(Paragraph("Auto-generated worked example &bull; d_model=8, heads=2 &bull; transformer-math explorer", SUB))
    e.append(HRFlowable(width="100%", color=INDIGO, thickness=1))
    e.append(Paragraph(f"<b>Your sentence, tokenized</b> &mdash; {r['tname']}; {r['n']} tokens:", BODY))
    e.append(Preformatted("  ".join(f"[{t}]" for t in r['tokens']), MONO))
    e.append(Paragraph("Each token becomes a row of length 8. Weights are fixed illustrative "
                       "numbers (a real model learns them). Values rounded to 2 decimals.", BODY))

    def sec(title, desc, *mats):
        e.append(Paragraph(title, H2))
        if desc:
            e.append(Paragraph(desc, BODY))
        for nm, M in mats:
            e.append(Preformatted(_mat(M, nm), MONO))

    sec("Module 1 &mdash; Embedding (+ position)", "Lookup table turns each token into a vector X.", ("X", r['X']))
    e.append(Paragraph("Module 2 &mdash; Layer Normalization", H2))
    e.append(Paragraph("Classic LayerNorm: subtract the mean, divide by the square root of the variance, "
        "then scale/shift:  x&#39;_j = (x_j - &mu;) / &radic;(&sigma;&sup2;+&epsilon;) &middot; &gamma;_j + &beta;_j.  "
        "Modern models use <b>RMSNorm</b> (no mean subtraction):  x&#39;_j = x_j / RMS(x),  "
        "RMS(x)=&radic;(mean(x&sup2;)+&epsilon;).", EQ))
    e.append(Preformatted(_mat(r['Xn'], "X_n = RMSNorm(X)"), MONO))

    sec("Module 3a &mdash; Project to Q, K, V", "Q = X_n W_Q, etc. The learned 8x8 projections:",
        ("W_Q", r['Wq']), ("W_K", r['Wk']), ("W_V", r['Wv']), ("Q", r['Q']), ("K", r['K']), ("V", r['V']))
    e.append(Paragraph("Module 3b &mdash; RoPE (rotary positions)", H2))
    e.append(Paragraph("Rotate each pair of dims of Q,K by &theta;=p&middot;10000^(-2i/d). "
        "Position 0 unrotated; later positions turn more.", BODY))
    e.append(Preformatted(_mat(r['Qr'], "Q_rope"), MONO)); e.append(Preformatted(_mat(r['Kr'], "K_rope"), MONO))

    e.append(Paragraph("Module 3c &mdash; Scores, causal mask, softmax (per head)", H2))
    e.append(Paragraph("S = Q_h K_h&#7488; / &radic;d_k,  &radic;4 = 2.  Mask future tokens to -&infin;, then softmax each row "
        "(weights below; upper triangle is 0).", BODY))
    for h in range(len(r['A_heads'])):
        e.append(Preformatted(_mat(r['A_heads'][h], f"A(head {h+1})"), MONO))
        e.append(Preformatted(_mat(r['O_heads'][h], f"O(head {h+1}) = A.V"), MONO))
    sec("Module 3d &mdash; Concatenate heads, project W_O", "Attn = concat(O) . W_O:",
        ("concat O", r['Ocat']), ("Attn", r['Attn']))

    e.append(Paragraph("Module 4 &mdash; Dropout", H2))
    e.append(Paragraph("Training only: zero a random fraction p and rescale by 1/(1-p). "
        "At inference it is the identity &mdash; Attn passes through unchanged.", BODY))
    sec("Module 5 &mdash; Residual (skip) connection", "Z = X + Attn:", ("Z", r['Z']))
    sec("Module 6 &mdash; Layer Normalization (again)", "Z_n = RMSNorm(Z):", ("Z_n", r['Zn']))
    e.append(Paragraph("Module 7 &mdash; Feed-Forward Network (SwiGLU)", H2))
    e.append(Paragraph("FFN(z) = ( SiLU(z W_gate) &#8857; (z W_up) ) W_down,  SiLU(x)=x&middot;&sigma;(x). "
        "In 2026 frontier models this is a Mixture-of-Experts.", BODY))
    e.append(Preformatted(_mat(r['F'], "F = FFN(Z_n)"), MONO))
    sec("Module 8 &mdash; Residual &rarr; output", "Y = Z + F  (block output; feeds the next block):", ("Y", r['Y']))

    e.append(Paragraph("Output &mdash; predicting the next word", H2))
    e.append(Paragraph("Normalize the last token's vector, project it onto the whole vocabulary, and softmax gives a "
        "probability for every possible next word. The toy block above is untrained, so for a genuine prediction we "
        "run the sentence through a small trained model, <b>DistilGPT-2</b> (82M params):", BODY))
    if preds:
        rows = [["next word", "probability"]] + [[_word(w), f"{p*100:.1f}%"] for w, p in preds[:10]]
        t = Table(rows, hAlign='LEFT', colWidths=[150, 90])
        t.setStyle(TableStyle([('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9), ('TEXTCOLOR', (0, 0), (-1, 0), INDIGO),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, SLATE), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
        e.append(t)
        e.append(Paragraph(f"<b>Most likely next word: &ldquo;{_word(preds[0][0])}&rdquo; at {preds[0][1]*100:.1f}%.</b> "
            "The token is appended and the loop repeats &mdash; autoregression. "
            "(&middot; marks a leading space, i.e. the start of a new word.)", BODY))
    else:
        e.append(Paragraph("<i>(The trained model was unavailable when this PDF was generated.)</i>", BODY))
    e.append(Spacer(1, 6)); e.append(HRFlowable(width="100%", color=SLATE, thickness=0.4))
    e.append(Paragraph("Generated by the Transformer-Math Explorer. Weights are illustrative, not from a "
        "trained model; the LM head is untrained. Same equations run at billions-of-parameters scale in real LLMs.", SUB))
    doc.build(e)
    return buf.getvalue()
