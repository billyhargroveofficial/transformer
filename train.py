"""
Обучение мини-GPT. Два режима:
  --mode overfit : заучить один абзац (проверка пайплайна, минуты)
  --mode run     : прогон на корпусе с семплом каждые N шагов
"""

import argparse
import json
import time
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

import mini_gpt as G
from optim import Adam
import data as D


def save_ckpt(path, params, cfg, tok):
    """Сохраняем веса (плоско, npz) + конфиг и словарь (json)."""
    flat = {k: np.array(v) for k, v in tree_flatten(params)}
    np.savez(path + ".npz", **flat)
    meta = {"cfg": cfg, "stoi": tok.stoi}
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def generate(params, cfg, idx, n_new, temp=1.0):
    """Авторегрессионная генерация (без KV-кеша — для семплов хватает)."""
    for _ in range(n_new):
        idx_cond = idx[:, -cfg["block_size"]:]
        logits, _ = G.forward(params, idx_cond, cfg)
        logits = logits[:, -1, :] * (1.0 / temp)
        nxt = mx.random.categorical(logits)              # (B,)
        idx = mx.concatenate([idx, nxt.reshape(idx.shape[0], 1)], axis=1)
        mx.eval(idx)
    return idx


# ----------------------------------------------------------------------------
def overfit(steps=400, lr=1e-3):
    text = D.OVERFIT_TEXT
    tok = D.CharTokenizer(text)
    ids = tok.encode(text)
    T = len(ids) - 1
    cfg = G.make_config(tok.vocab_size, n_embd=64, n_head=2, n_layer=2, block_size=T)
    params = G.init_params(cfg, seed=0)
    opt = Adam(params, lr=lr, weight_decay=0.0)          # NO weight decay — хотим заучить
    x = mx.array(np.array([ids[:-1]], dtype=np.int32))
    y = mx.array(np.array([ids[1:]], dtype=np.int32))

    print(f"overfit: {len(ids)} символов, vocab={tok.vocab_size}, "
          f"~{n_params(params):,} параметров")
    t0 = time.time()
    for step in range(1, steps + 1):
        logits, cache = G.forward(params, x, cfg)
        loss = G.cross_entropy(logits, y)
        grads = G.backward(params, cache, logits, y, cfg)
        params = opt.step(params, grads)
        mx.eval(params, loss)
        if step == 1 or step % 50 == 0:
            print(f"  step {step:4d}  loss {float(loss):.4f}")
    print(f"  обучение заняло {time.time() - t0:.1f} c")

    seed = mx.array(np.array([[ids[0]]], dtype=np.int32))
    out = generate(params, cfg, seed, n_new=T, temp=0.5)
    gen = tok.decode(out[0].tolist())
    print("\n--- ОРИГИНАЛ ---\n" + text)
    print("\n--- СГЕНЕРИРОВАНО ---\n" + gen)
    print("\nсовпадение:", "ДА ✅" if gen.strip() == text.strip() else "частичное")


# ----------------------------------------------------------------------------
def train_run(args):
    raw = open(args.corpus, encoding="utf-8").read()
    if len(raw) > args.max_chars:
        raw = raw[:args.max_chars]
    tok = D.CharTokenizer(raw)
    data_np = np.array(tok.encode(raw), dtype=np.int32)
    cfg = G.make_config(tok.vocab_size, n_embd=args.n_embd, n_head=args.n_head,
                        n_layer=args.n_layer, block_size=args.block_size)
    params = G.init_params(cfg, seed=0)
    opt = Adam(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    B, T = args.batch, args.block_size

    print(f"corpus: {len(raw):,} символов, vocab={tok.vocab_size}")
    print(f"модель: d={cfg['n_embd']} L={cfg['n_layer']} h={cfg['n_head']} "
          f"ctx={cfg['block_size']} -> ~{n_params(params):,} параметров")
    print(f"бюджет: {args.minutes} мин, семпл каждые {args.sample_every} шагов\n")

    t0 = time.time(); toks = 0; step = 0
    while True:
        step += 1
        opt.lr = args.lr * min(1.0, step / args.warmup)   # линейный warmup
        x, y = D.make_batch(data_np, B, T)
        logits, cache = G.forward(params, x, cfg)
        loss = G.cross_entropy(logits, y)
        grads = G.backward(params, cache, logits, y, cfg)
        params = opt.step(params, grads)
        mx.eval(params, loss)
        toks += B * T

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:6d}  loss {float(loss):.4f}  lr {opt.lr:.1e}  "
                  f"{toks/dt:>8,.0f} tok/s  {dt/60:5.1f} мин", flush=True)
        if step % args.sample_every == 0:
            seed_id = tok.stoi.get("\n", tok.stoi.get(" ", 0))
            out = generate(params, cfg, mx.array([[seed_id]]), 240, temp=0.8)
            print("  ┌─ семпл (шаг %d) " % step + "─" * 40, flush=True)
            for line in tok.decode(out[0].tolist()).split("\n"):
                print("  │ " + line, flush=True)
            print("  └" + "─" * 58, flush=True)
            save_ckpt(args.out, params, cfg, tok)
        if (time.time() - t0) > args.minutes * 60:
            print("\n⏱ бюджет времени исчерпан, стоп.", flush=True)
            break
    save_ckpt(args.out, params, cfg, tok)
    print(f"чекпоинт сохранён: {args.out}.npz / .json", flush=True)


def n_params(params):
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(params))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["overfit", "run"], default="overfit")
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--max_chars", type=int, default=30_000_000)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--sample_every", type=int, default=1500)
    ap.add_argument("--out", default="ckpt")
    args = ap.parse_args()

    if args.mode == "overfit":
        overfit()
    else:
        train_run(args)
