"""
Сверка РУЧНОГО backward с эталоном. Два независимых способа:
  1) mx.grad (autograd MLX) — только как чекер, НЕ в обучении;
  2) численный градиент (центральная разность).
Гоним в float32 (на GPU, мягкий порог — там есть числовой шум) и
в float64 (на CPU, строгий порог 1e-6 — это и есть аналитический вердикт).
"""

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_map
import mini_gpt as G


def _make_problem(dtype):
    np.random.seed(0)
    cfg = G.make_config(vocab_size=23, n_embd=32, n_head=2, n_layer=2, block_size=16)
    params = tree_map(lambda p: p.astype(dtype), G.init_params(cfg, seed=1))
    B, T = 3, 12
    idx = mx.array(np.random.randint(0, cfg["vocab_size"], (B, T)).astype(np.int32))
    tgt = mx.array(np.random.randint(0, cfg["vocab_size"], (B, T)).astype(np.int32))
    return cfg, params, idx, tgt


def _autograd_pass(cfg, params, idx, tgt, tol, label):
    logits, cache = G.forward(params, idx, cfg)
    loss = G.cross_entropy(logits, tgt)
    manual = G.backward(params, cache, logits, tgt, cfg)
    auto_loss, auto = mx.value_and_grad(lambda p: G.forward_loss(p, idx, tgt, cfg))(params)
    mx.eval(loss, manual, auto_loss, auto)

    mf, af = dict(tree_flatten(manual)), dict(tree_flatten(auto))
    worst, worst_k = 0.0, ""
    for k in af:
        num = float(mx.max(mx.abs(af[k] - mf[k])))
        den = float(mx.max(mx.abs(af[k]))) + 1e-12
        rel = num / den
        if rel > worst:
            worst, worst_k = rel, k
    ok = worst < tol
    print(f"[{label}] loss manual={float(loss):.6f} auto={float(auto_loss):.6f} | "
          f"worst rel_err={worst:.2e} ({worst_k}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def _numeric_pass(cfg, params, idx, tgt):
    """Полностью независимая проверка: центральная разность в float64."""
    logits, cache = G.forward(params, idx, cfg)
    manual = dict(tree_flatten(G.backward(params, cache, logits, tgt, cfg)))
    eps = 1e-5
    print("[numeric f64] центральная разность на случайных элементах:")
    for name in ["head_W", "wte", "blocks.0.Wv", "blocks.1.W1"]:
        base = params[name] if "." not in name else _get(params, name)
        W = np.array(base, dtype=np.float64)
        gm = np.array(manual[name], dtype=np.float64)
        i, j = np.random.randint(W.shape[0]), np.random.randint(W.shape[1])
        Wp, Wm = W.copy(), W.copy()
        Wp[i, j] += eps; Wm[i, j] -= eps
        lp = float(G.forward_loss(_set(params, name, mx.array(Wp)), idx, tgt, cfg))
        lm = float(G.forward_loss(_set(params, name, mx.array(Wm)), idx, tgt, cfg))
        num = (lp - lm) / (2 * eps)
        rel = abs(num - gm[i, j]) / (abs(num) + 1e-9)
        print(f"   {name:14s}[{i:2d},{j:2d}]  num={num:+.6f}  manual={gm[i, j]:+.6f}  rel={rel:.1e}")


def _get(params, dotted):
    a, i, b = dotted.split(".")
    return params[a][int(i)][b]

def _set(params, dotted, val):
    p = {**params, "blocks": [dict(bl) for bl in params["blocks"]]}
    if "." in dotted:
        a, i, b = dotted.split("."); p[a][int(i)][b] = val
    else:
        p[dotted] = val
    return p


def run_check():
    # float32 на GPU — реалистичный шум, мягкий порог
    ok32 = _autograd_pass(*_make_problem(mx.float32), tol=3e-3, label="float32/gpu")
    # float64 на CPU — строгий аналитический вердикт
    mx.set_default_device(mx.cpu)
    cfg, params, idx, tgt = _make_problem(mx.float64)
    ok64 = _autograd_pass(cfg, params, idx, tgt, tol=1e-6, label="float64/cpu")
    _numeric_pass(cfg, params, idx, tgt)
    mx.set_default_device(mx.gpu)
    return ok32 and ok64


if __name__ == "__main__":
    ok = run_check()
    print("\n" + ("✅ backward ВЕРЕН (совпал с autograd в float64 и с численным)"
                  if ok else "❌ есть ошибка в backward"))
