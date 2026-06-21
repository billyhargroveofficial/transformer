"""
Инференс: грузим чекпоинт (ckpt.npz + ckpt.json) и генерим текст.

  python sample.py                                   # 400 симв. с чистого старта
  python sample.py --prompt "В 1812 году" --n 500
  python sample.py --temp 0.7 --topk 40              # меньше бреда/повторов
"""
import argparse
import json
import numpy as np
import mlx.core as mx
from mlx.utils import tree_unflatten
import mini_gpt as G


def load_ckpt(path):
    data = np.load(path + ".npz")
    params = tree_unflatten([(k, mx.array(data[k])) for k in data.files])
    meta = json.load(open(path + ".json", encoding="utf-8"))
    cfg = meta["cfg"]
    stoi = meta["stoi"]                      # символ -> индекс
    itos = {i: c for c, i in stoi.items()}  # индекс -> символ
    return params, cfg, stoi, itos


def generate(params, cfg, idx, n, temp, topk):
    V = cfg["vocab_size"]
    topk = min(topk, V) if topk > 0 else 0
    for _ in range(n):
        idx_cond = idx[:, -cfg["block_size"]:]          # обрезаем по контексту
        logits, _ = G.forward(params, idx_cond, cfg)
        logits = logits[:, -1, :] / temp                # берём последний шаг, температура
        if topk > 0:                                    # оставляем только top-k вариантов
            kth = mx.sort(logits, axis=-1)[:, -topk]    # порог = k-й по величине логит
            logits = mx.where(logits < kth[:, None], mx.array(-1e9), logits)
        nxt = mx.random.categorical(logits)             # сэмпл из распределения
        idx = mx.concatenate([idx, nxt.reshape(1, 1)], axis=1)
        mx.eval(idx)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--seed", type=int, default=-1)
    args = ap.parse_args()

    if args.seed >= 0:
        mx.random.seed(args.seed)

    params, cfg, stoi, itos = load_ckpt(args.ckpt)
    print(f"модель: d={cfg['n_embd']} L={cfg['n_layer']} h={cfg['n_head']} "
          f"ctx={cfg['block_size']} vocab={cfg['vocab_size']}  "
          f"(temp={args.temp}, topk={args.topk})")

    if args.prompt:
        ids = [stoi[c] for c in args.prompt if c in stoi]
        if not ids:
            ids = [stoi.get("\n", 0)]
    else:
        ids = [stoi.get("\n", stoi.get(" ", 0))]
    idx = mx.array(np.array([ids], dtype=np.int32))

    out = generate(params, cfg, idx, args.n, args.temp, args.topk)
    text = "".join(itos[int(i)] for i in out[0].tolist())
    print("─" * 64)
    print(text)
    print("─" * 64)


if __name__ == "__main__":
    main()
