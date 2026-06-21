"""Char-level токенизатор и нарезка батчей."""

import numpy as np
import mlx.core as mx

# Абзац для overfit-теста (проверка, что pipeline и ручной backprop рабочие).
OVERFIT_TEXT = (
    "Жил-был на свете кот, который очень любил спать на солнце. "
    "Каждое утро он выходил во двор, находил тёплое место и засыпал. "
    "Соседи говорили, что более ленивого кота они в жизни не видели, "
    "но кот лишь жмурился и мурлыкал."
)


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def make_batch(data_np, B, T):
    """Случайные окна длины T. x — вход, y — он же со сдвигом на 1 (что предсказать)."""
    ix = np.random.randint(0, len(data_np) - T - 1, size=B)
    x = np.stack([data_np[i:i + T] for i in ix]).astype(np.int32)
    y = np.stack([data_np[i + 1:i + 1 + T] for i in ix]).astype(np.int32)
    return mx.array(x), mx.array(y)
