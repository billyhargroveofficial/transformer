"""Adam/AdamW руками. Работает над деревом параметров (dict со списком blocks)."""

import mlx.core as mx
from mlx.utils import tree_map


class Adam:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        self.lr, self.eps, self.wd = lr, eps, weight_decay
        self.b1, self.b2 = betas
        self.t = 0
        self.m = tree_map(lambda p: mx.zeros_like(p), params)   # 1-й момент
        self.v = tree_map(lambda p: mx.zeros_like(p), params)   # 2-й момент

    def step(self, params, grads):
        self.t += 1
        b1, b2 = self.b1, self.b2
        # m = β1·m + (1-β1)·g ;  v = β2·v + (1-β2)·g²
        self.m = tree_map(lambda g, m: b1 * m + (1 - b1) * g, grads, self.m)
        self.v = tree_map(lambda g, v: b2 * v + (1 - b2) * g * g, grads, self.v)
        bc1 = 1 - b1 ** self.t        # bias-correction
        bc2 = 1 - b2 ** self.t
        lr, eps, wd = self.lr, self.eps, self.wd

        def upd(p, m, v):
            step = lr * (m / bc1) / (mx.sqrt(v / bc2) + eps)
            if wd > 0 and p.ndim >= 2:          # decoupled weight decay только на матрицы
                step = step + lr * wd * p
            return p - step

        return tree_map(upd, params, self.m, self.v)
