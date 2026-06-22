"""Real next-word prediction with a small trained model (DistilGPT-2).

The step-by-step matrices in the app use a tiny *untrained* toy model (d=8) purely
to show the mechanics. To predict the ACTUAL next word for any sentence we need a
real trained model — this module loads DistilGPT-2 (82M params) once and returns the
true next-token probability distribution.
"""
import streamlit as st


@st.cache_resource(show_spinner=False)
def _load():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.eval()
    return tok, model, torch


def predict_next(text, k=10):
    """Return (list_of_(word, prob), error_or_None) for the real next token."""
    text = (text or "").strip()
    if not text:
        text = "The"
    try:
        tok, model, torch = _load()
        ids = tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(ids).logits[0, -1]          # last position
        probs = torch.softmax(logits, dim=-1)
        vals, idx = torch.topk(probs, k)
        out = [(tok.decode([int(i)]), float(p)) for p, i in zip(vals.tolist(), idx.tolist())]
        return out, None
    except Exception as e:                              # offline / model unavailable
        return [], f"{type(e).__name__}: {e}"


def label(word):
    """Readable label for a token; show a leading space as '␣' so word boundaries are clear."""
    if word == "":
        return "∅"
    if word.strip() == "":
        return "␣"
    return ("␣" + word.strip()) if word[:1] == " " else word.strip()
