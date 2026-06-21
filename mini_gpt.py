"""
Мини-GPT на чистых матрицах. MLX используется как "тупой калькулятор матриц":
берём только mx.matmul/exp/softmax-примитивы, а forward И backward (градиенты)
пишем РУКАМИ. Никакого autograd в обучении — mx.grad только как сверка (gradcheck.py).

Везде float32. Соглашения по осям:
  B  — размер батча
  T  — длина последовательности (контекст)
  d  — размерность модели (n_embd)
  h  — число голов; dh = d // h — размерность одной головы
  V  — размер словаря

Роли в attention:  Q — что ищу,  K — по чему меня найти,  V — что отдаю.
"""

import math
import mlx.core as mx

# ============================================================================
# КОНФИГ И ИНИЦИАЛИЗАЦИЯ
# ============================================================================

def make_config(vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=128):
    assert n_embd % n_head == 0, "n_embd должно делиться на n_head"
    return dict(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head,
                n_layer=n_layer, block_size=block_size)


def init_params(cfg, seed=0):
    """Все параметры — обычные mx.array. Маленькая нормальная инициализация (GPT-2 стиль)."""
    mx.random.seed(seed)
    d, V, L, Tm = cfg["n_embd"], cfg["vocab_size"], cfg["n_layer"], cfg["block_size"]
    n = lambda *s: mx.random.normal(s) * 0.02
    z = lambda *s: mx.zeros(s)

    def block():
        return dict(
            ln1_g=mx.ones((d,)), ln1_b=z(d),
            Wq=n(d, d), bq=z(d), Wk=n(d, d), bk=z(d), Wv=n(d, d), bv=z(d),
            Wo=n(d, d), bo=z(d),
            ln2_g=mx.ones((d,)), ln2_b=z(d),
            W1=n(d, 4 * d), b1=z(4 * d), W2=n(4 * d, d), b2=z(d),
        )

    return dict(
        wte=n(V, d),                 # таблица токен-эмбеддингов  (V, d)
        wpe=n(Tm, d),                # таблица позиционных эмбеддингов (Tmax, d)
        blocks=[block() for _ in range(L)],
        lnf_g=mx.ones((d,)), lnf_b=z(d),
        head_W=n(d, V), head_b=z(V),  # финальная проекция d -> V
    )


# ============================================================================
# БАЗОВЫЕ СЛОИ: forward + руками выведенный backward
# ============================================================================

# ---- Linear:  y = x @ W + b -------------------------------------------------
def linear_fwd(x, W, b):
    return x @ W + b

def linear_bwd(dy, x, W):
    # Опорные формулы:  dW = Xᵀ·dY,  dX = dY·Wᵀ,  db = Σ dY
    in_, out = W.shape
    x2 = x.reshape(-1, in_)        # (N, in)
    dy2 = dy.reshape(-1, out)      # (N, out)
    dW = x2.T @ dy2                # (in, out)
    db = dy2.sum(axis=0)          # (out,)
    dx = dy @ W.T                  # (..., in)
    return dx, dW, db


# ---- LayerNorm (по последней оси d) ----------------------------------------
def layernorm_fwd(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    istd = 1.0 / mx.sqrt(var + eps)
    xhat = xc * istd                 # нормированный
    y = g * xhat + b                 # масштаб + сдвиг
    return y, (xhat, istd, g)

def layernorm_bwd(dy, cache):
    xhat, istd, g = cache
    axes = tuple(range(dy.ndim - 1))            # суммируем по всем (B,T)
    dg = (dy * xhat).sum(axis=axes)
    db = dy.sum(axis=axes)
    dxhat = dy * g
    # dx = istd·( dxhat - mean(dxhat) - xhat·mean(dxhat·xhat) ),  mean по оси d
    dx = istd * (dxhat
                 - dxhat.mean(axis=-1, keepdims=True)
                 - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
    return dx, dg, db


# ---- GELU (точный, через erf) ----------------------------------------------
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)

def gelu_fwd(x):
    return 0.5 * x * (1.0 + mx.erf(x * _INV_SQRT2))

def gelu_grad(x):
    # производная:  Φ(x) + x·φ(x),  Φ — CDF нормали, φ — её плотность
    cdf = 0.5 * (1.0 + mx.erf(x * _INV_SQRT2))
    pdf = _INV_SQRT2PI * mx.exp(-0.5 * x * x)
    return cdf + x * pdf


# ---- Softmax (по последней оси) --------------------------------------------
def softmax_fwd(x, axis=-1):
    m = mx.max(x, axis=axis, keepdims=True)     # стабилизация
    e = mx.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)

def softmax_bwd(dp, p, axis=-1):
    # Якобиан softmax:  dx_i = p_i·( dp_i - Σ_j dp_j·p_j )
    return p * (dp - (dp * p).sum(axis=axis, keepdims=True))


# ---- Causal-маска ----------------------------------------------------------
def _causal_mask(T, dtype=mx.float32):
    rows = mx.arange(T).reshape(T, 1)
    cols = mx.arange(T).reshape(1, T)
    allow = cols <= rows                         # нижний треугольник (вкл. диагональ)
    return mx.where(allow, mx.zeros((T, T), dtype), mx.full((T, T), -1e9, dtype))


# ============================================================================
# MULTI-HEAD SELF-ATTENTION — самый сложный backward (ветвится на Q, K, V)
# ============================================================================

def attention_fwd(z, bp, cfg):
    """z = ln1(x): (B,T,d)."""
    B, T, d = z.shape
    h = cfg["n_head"]; dh = d // h
    scale = 1.0 / math.sqrt(dh)

    q = linear_fwd(z, bp["Wq"], bp["bq"])        # что ищу
    k = linear_fwd(z, bp["Wk"], bp["bk"])        # по чему меня найти
    v = linear_fwd(z, bp["Wv"], bp["bv"])        # что отдаю

    # режем на головы: (B,T,d) -> (B,h,T,dh)
    split = lambda t: mx.transpose(t.reshape(B, T, h, dh), (0, 2, 1, 3))
    qh, kh, vh = split(q), split(k), split(v)

    scores = (qh @ mx.swapaxes(kh, -1, -2)) * scale     # (B,h,T,T)
    scores = scores + _causal_mask(T, scores.dtype)
    att = softmax_fwd(scores, axis=-1)                  # (B,h,T,T)
    out_h = att @ vh                                    # (B,h,T,dh)
    out = mx.transpose(out_h, (0, 2, 1, 3)).reshape(B, T, d)   # склейка голов
    y = linear_fwd(out, bp["Wo"], bp["bo"])

    cache = dict(z=z, qh=qh, kh=kh, vh=vh, att=att, out=out,
                 scale=scale, B=B, T=T, h=h, dh=dh, d=d)
    return y, cache

def attention_bwd(dy, cache, bp, cfg):
    z, qh, kh, vh = cache["z"], cache["qh"], cache["kh"], cache["vh"]
    att, out, scale = cache["att"], cache["out"], cache["scale"]
    B, T, h, dh, d = cache["B"], cache["T"], cache["h"], cache["dh"], cache["d"]

    # выходная проекция Wo
    d_out, dWo, dbo = linear_bwd(dy, out, bp["Wo"])              # d_out: (B,T,d)
    d_out_h = mx.transpose(d_out.reshape(B, T, h, dh), (0, 2, 1, 3))  # -> (B,h,T,dh)

    # out_h = att @ vh
    d_att = d_out_h @ mx.swapaxes(vh, -1, -2)                    # (B,h,T,T)
    d_vh = mx.swapaxes(att, -1, -2) @ d_out_h                    # (B,h,T,dh)

    # softmax
    d_scores = softmax_bwd(d_att, att, axis=-1)                  # (B,h,T,T)
    # (на запрещённых маской позициях att=0 -> d_scores=0 автоматически)

    # scores = scale · qh·khᵀ  -> ветвление на Q и K
    d_qh = scale * (d_scores @ kh)                              # (B,h,T,dh)
    d_kh = scale * (mx.swapaxes(d_scores, -1, -2) @ qh)         # (B,h,T,dh)

    # склейка голов обратно: (B,h,T,dh) -> (B,T,d)
    merge = lambda t: mx.transpose(t, (0, 2, 1, 3)).reshape(B, T, d)
    d_q, d_k, d_v = merge(d_qh), merge(d_kh), merge(d_vh)

    # входные проекции (q=z@Wq+bq и т.д.); dz суммируется по трём ветвям
    dz_q, dWq, dbq = linear_bwd(d_q, z, bp["Wq"])
    dz_k, dWk, dbk = linear_bwd(d_k, z, bp["Wk"])
    dz_v, dWv, dbv = linear_bwd(d_v, z, bp["Wv"])
    dz = dz_q + dz_k + dz_v

    grads = dict(Wq=dWq, bq=dbq, Wk=dWk, bk=dbk, Wv=dWv, bv=dbv, Wo=dWo, bo=dbo)
    return dz, grads


# ============================================================================
# FFN:  d -> 4d -> d
# ============================================================================

def ffn_fwd(x, bp):
    h1 = linear_fwd(x, bp["W1"], bp["b1"])       # (B,T,4d)
    a = gelu_fwd(h1)
    y = linear_fwd(a, bp["W2"], bp["b2"])        # (B,T,d)
    return y, dict(x=x, h1=h1, a=a)

def ffn_bwd(dy, cache, bp):
    x, h1, a = cache["x"], cache["h1"], cache["a"]
    da, dW2, db2 = linear_bwd(dy, a, bp["W2"])
    dh1 = da * gelu_grad(h1)
    dx, dW1, db1 = linear_bwd(dh1, x, bp["W1"])
    return dx, dict(W1=dW1, b1=db1, W2=dW2, b2=db2)


# ============================================================================
# ДЕКОДЕР-БЛОК (pre-norm, как GPT-2):
#   a = x + attn(ln1(x))
#   y = a + ffn(ln2(a))
# ============================================================================

def block_fwd(x, bp, cfg):
    ln1, c1 = layernorm_fwd(x, bp["ln1_g"], bp["ln1_b"])
    attn, ca = attention_fwd(ln1, bp, cfg)
    a = x + attn
    ln2, c2 = layernorm_fwd(a, bp["ln2_g"], bp["ln2_b"])
    ff, cf = ffn_fwd(ln2, bp)
    y = a + ff
    return y, (c1, ca, c2, cf)

def block_bwd(dy, cache, bp, cfg):
    c1, ca, c2, cf = cache
    # ветвь FFN: y = a + ffn(ln2(a))  -> в a приходит градиент по двум путям
    d_a = dy                                       # путь residual
    d_ln2, gff = ffn_bwd(dy, cf, bp)
    d_a2, dln2g, dln2b = layernorm_bwd(d_ln2, c2)
    d_a = d_a + d_a2
    # ветвь attention: a = x + attn(ln1(x))
    d_x = d_a                                      # путь residual
    d_ln1, gat = attention_bwd(d_a, ca, bp, cfg)
    d_x2, dln1g, dln1b = layernorm_bwd(d_ln1, c1)
    d_x = d_x + d_x2
    grads = dict(ln1_g=dln1g, ln1_b=dln1b, ln2_g=dln2g, ln2_b=dln2b)
    grads.update(gat); grads.update(gff)
    return d_x, grads


# ============================================================================
# ПОЛНАЯ МОДЕЛЬ
# ============================================================================

def forward(params, idx, cfg):
    """idx: (B,T) int. Возвращает logits (B,T,V) и cache для backward."""
    B, T = idx.shape
    tok = mx.take(params["wte"], idx, axis=0)      # (B,T,d) — токен-эмбеддинги
    pos = params["wpe"][:T]                         # (T,d)   — позиционные
    x = tok + pos
    caches = []
    for bp in params["blocks"]:
        x, bc = block_fwd(x, bp, cfg)
        caches.append(bc)
    xf, cf = layernorm_fwd(x, params["lnf_g"], params["lnf_b"])
    logits = linear_fwd(xf, params["head_W"], params["head_b"])
    cache = dict(idx=idx, B=B, T=T, blocks=caches, lnf=cf, xf=xf)
    return logits, cache


def cross_entropy(logits, targets):
    """Стабильная log-softmax + NLL. logits (B,T,V), targets (B,T)."""
    B, T, V = logits.shape
    m = mx.max(logits, axis=-1, keepdims=True)
    shifted = logits - m
    lse = mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))
    logprobs = shifted - lse
    nll = -mx.take_along_axis(logprobs, targets.reshape(B, T, 1), axis=-1).reshape(B, T)
    return nll.mean()

def cross_entropy_grad(logits, targets):
    # Знаменитое схлопывание softmax+CE:  dL/dlogits = (p - onehot) / (B·T)
    B, T, V = logits.shape
    p = softmax_fwd(logits, axis=-1)
    onehot = (mx.arange(V).reshape(1, 1, V) == targets.reshape(B, T, 1)).astype(p.dtype)
    return (p - onehot) / (B * T)


def forward_loss(params, idx, targets, cfg):
    """Чистая функция (для сверки с mx.grad и для замера лосса)."""
    logits, _ = forward(params, idx, cfg)
    return cross_entropy(logits, targets)


def backward(params, cache, logits, targets, cfg):
    """Полный ручной backward. Возвращает grads той же структуры, что params."""
    grads = {}
    dlogits = cross_entropy_grad(logits, targets)                 # (B,T,V)

    # голова (linear)
    dxf, dHW, dHb = linear_bwd(dlogits, cache["xf"], params["head_W"])
    grads["head_W"], grads["head_b"] = dHW, dHb

    # финальный LayerNorm
    dx, dlnf_g, dlnf_b = layernorm_bwd(dxf, cache["lnf"])
    grads["lnf_g"], grads["lnf_b"] = dlnf_g, dlnf_b

    # блоки в обратном порядке
    block_grads = [None] * len(params["blocks"])
    for i in reversed(range(len(params["blocks"]))):
        dx, gb = block_bwd(dx, cache["blocks"][i], params["blocks"][i], cfg)
        block_grads[i] = gb
    grads["blocks"] = block_grads

    # эмбеддинги: dx здесь — градиент по (tok + pos), форма (B,T,d)
    dx0 = dx
    B, T, d = cache["B"], cache["T"], cfg["n_embd"]
    Tm = params["wpe"].shape[0]

    # позиционные: позиция t встречается в каждом примере батча -> сумма по B
    pos_grad = dx0.sum(axis=0)                                     # (T,d)
    if T < Tm:
        grads["wpe"] = mx.concatenate([pos_grad, mx.zeros((Tm - T, d))], axis=0)
    else:
        grads["wpe"] = pos_grad

    # токенные: scatter-add по строкам idx (повторяющийся токен аккумулируется)
    dwte = mx.zeros_like(params["wte"])
    idx_flat = cache["idx"].reshape(-1)                           # (B·T,)
    grads["wte"] = dwte.at[idx_flat].add(dx0.reshape(-1, d))

    return grads
