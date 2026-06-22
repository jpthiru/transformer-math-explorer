"""Transformer math core: tokenize -> full single-block math for any sentence.
Fixed teaching dims d_model=8, heads=2. Deterministic and reproducible."""
import numpy as np, re, hashlib

D_MODEL, N_HEADS, D_FF = 8, 2, 16
DK = D_MODEL // N_HEADS
WSEED = 7
MAX_TOKENS = 12


def _rng(seed):
    return np.random.default_rng(seed)


def get_tokens(text):
    """Return (list_of_token_strings, tokenizer_name)."""
    text = text.strip()
    if not text:
        text = "Weak quarterly results led the board to replace the"
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(text)
        toks = [enc.decode([i]) for i in ids]
        return toks[:MAX_TOKENS], "tiktoken cl100k_base (GPT-4 BPE)"
    except Exception:
        # offline approximate-subword fallback
        raw = re.findall(r"\s?[A-Za-z]+|\s?\d+|\s?[^\w\s]", text)
        sufs = ['ly', 'ing', 'tion', 'ment', 'ness', 'ed', 'ies', 'es', 'er', 'est', 'able', 's']
        toks = []
        for w in raw:
            lead = ' ' if w.startswith(' ') else ''
            core = w.strip()
            done = False
            if len(core) > 6:
                for s in sufs:
                    if core.lower().endswith(s) and len(core) - len(s) >= 3:
                        toks.append(lead + core[:-len(s)]); toks.append(s); done = True; break
            if not done:
                toks.append(w)
        return toks[:MAX_TOKENS], "approximate subword (offline fallback)"


def _embed(tok):
    h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % (2 ** 32)
    return np.round(_rng(h).uniform(-1, 1, D_MODEL), 2)


def rmsnorm(M, eps=1e-5):
    return M / np.sqrt(np.mean(M ** 2, axis=1, keepdims=True) + eps)


def _rope(M):
    n = M.shape[0]; out = M.copy().astype(float); pos = np.arange(n)
    for i in range(D_MODEL // 2):
        th = 10000 ** (-2 * i / D_MODEL); a = pos * th; c, s = np.cos(a), np.sin(a)
        x0, x1 = M[:, 2 * i], M[:, 2 * i + 1]
        out[:, 2 * i] = x0 * c - x1 * s
        out[:, 2 * i + 1] = x0 * s + x1 * c
    return out


def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True); e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def compute(text):
    toks, tname = get_tokens(text)
    n = len(toks); r2 = lambda x: np.round(x, 2)
    X = np.array([_embed(t) for t in toks])
    g = _rng(WSEED)
    Wq = r2(g.uniform(-.6, .6, (D_MODEL, D_MODEL))); Wk = r2(g.uniform(-.6, .6, (D_MODEL, D_MODEL)))
    Wv = r2(g.uniform(-.6, .6, (D_MODEL, D_MODEL))); Wo = r2(g.uniform(-.6, .6, (D_MODEL, D_MODEL)))
    Wg = r2(g.uniform(-.5, .5, (D_MODEL, D_FF)));   Wu = r2(g.uniform(-.5, .5, (D_MODEL, D_FF)))
    Wd = r2(g.uniform(-.5, .5, (D_FF, D_MODEL)))
    Xn = r2(rmsnorm(X))
    Q = r2(Xn @ Wq); K = r2(Xn @ Wk); V = r2(Xn @ Wv)
    Qr = r2(_rope(Q)); Kr = r2(_rope(K))
    mask = np.triu(np.ones((n, n)), 1).astype(bool)
    A_heads, S_heads, O_heads = [], [], []
    for h in range(N_HEADS):
        sl = slice(h * DK, (h + 1) * DK)
        S = (Qr[:, sl] @ Kr[:, sl].T) / np.sqrt(DK)
        Sm = S.copy(); Sm[mask] = -np.inf; A = _softmax(Sm); O = A @ V[:, sl]
        S_heads.append(r2(S)); A_heads.append(r2(A)); O_heads.append(r2(O))
    Ocat = r2(np.concatenate(O_heads, 1)); Attn = r2(Ocat @ Wo)
    Z = r2(X + Attn); Zn = r2(rmsnorm(Z))
    silu = lambda x: x / (1 + np.exp(-x))
    F = r2((silu(Zn @ Wg) * (Zn @ Wu)) @ Wd); Y = r2(Z + F); Yn = r2(rmsnorm(Y))

    # --- which earlier tokens does the LAST position attend to? (averaged over heads) ---
    A_avg = np.mean(np.stack(A_heads, axis=0), axis=0)  # n x n
    last_attn = A_avg[-1]
    att_order = np.argsort(-last_attn)
    top_attended = [(toks[i].strip() or toks[i], float(last_attn[i]))
                    for i in att_order if last_attn[i] > 0][:4]

    # --- illustrative LM head: project the final state onto the SENTENCE'S OWN vocabulary ---
    # (untrained head; the candidate words come from the user's sentence so the output is relevant)
    words = []
    for t in toks:
        w = t.strip()
        if w and any(c.isalnum() for c in w) and w.lower() not in {x.lower() for x in words}:
            words.append(w)
    if not words:
        words = ["(none)"]
    vocab = words
    Wvoc = r2(np.array([_embed(w) for w in vocab]))
    logits = r2(Wvoc @ Yn[-1])
    probs = r2(_softmax(logits.reshape(1, -1)).flatten())
    order = np.argsort(-probs)
    topk = [(vocab[i], float(probs[i])) for i in order[:min(5, len(vocab))]]
    return dict(tokens=toks, tname=tname, n=n, X=X, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
                Wg=Wg, Wu=Wu, Wd=Wd, Xn=Xn, Q=Q, K=K, V=V, Qr=Qr, Kr=Kr,
                S_heads=S_heads, A_heads=A_heads, O_heads=O_heads, Ocat=Ocat, Attn=Attn,
                Z=Z, Zn=Zn, F=F, Y=Y, Yn=Yn, vocab=vocab, logits=logits, probs=probs,
                topk=topk, top_attended=top_attended)
