"""
Подробная схема архитектуры mini-GPT — рисуется matplotlib'ом (статичный PNG).
Все цифры/формы взяты из mini_gpt.py (pre-LN GPT-2, h=4, dh=64, MLP d->4d->d,
точный GELU через erf, causal-маска). Запуск:  .venv/bin/python arch_diagram.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# ── палитра (тёмная тема) ────────────────────────────────────────────────────
BG, PANEL, TXT, MUTE = "#0d1117", "#161b22", "#e6edf3", "#8b949e"
ARROW = "#9aa4b2"
C = {  # kind: (edge, fill)
    "io":   ("#cbd5e1", "#1f2733"),
    "emb":  ("#60a5fa", "#15314f"),
    "ln":   ("#c4b5fd", "#2b2150"),
    "attn": ("#5eead4", "#0c3b38"),
    "ffn":  ("#fbbf24", "#43320c"),
    "head": ("#fca5a5", "#3f1d1d"),
    "res":  ("#f9a8d4", "#3f1d2e"),
}

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": BG,
                     "savefig.facecolor": BG})

fig, ax = plt.subplots(figsize=(27, 15))
ax.set_xlim(0, 180); ax.set_ylim(0, 100); ax.axis("off")


class Box:
    def __init__(s, cx, cy, w, h): s.cx, s.cy, s.w, s.h = cx, cy, w, h
    @property
    def top(s):   return (s.cx, s.cy + s.h / 2)
    @property
    def bot(s):   return (s.cx, s.cy - s.h / 2)
    @property
    def left(s):  return (s.cx - s.w / 2, s.cy)
    @property
    def right(s): return (s.cx + s.w / 2, s.cy)


def box(cx, cy, w, h, label, kind="io", fs=12, weight="normal", tc=None, lw=1.8):
    ec, fc = C[kind]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0,rounding_size=0.6", linewidth=lw,
                 edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            color=tc or TXT, zorder=4, weight=weight, linespacing=1.3)
    return Box(cx, cy, w, h)


def arr(p1, p2, color=ARROW, lw=2.0, rad=0.0, style="-|>", ls="-", z=2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                 lw=lw, color=color, connectionstyle=f"arc3,rad={rad}",
                 zorder=z, linestyle=ls, shrinkA=2, shrinkB=2))


def oplus(cx, cy, r=1.5, color=C["res"][0]):
    ax.add_patch(Circle((cx, cy), r, edgecolor=color, facecolor=BG, lw=2.2, zorder=5))
    ax.text(cx, cy, "+", ha="center", va="center", fontsize=17, color=color,
            zorder=6, weight="bold")
    return Box(cx, cy, 2 * r, 2 * r)


def lab(x, y, t, fs=10, color=MUTE, ha="left", va="center", weight="normal",
        style="normal", family="DejaVu Sans"):
    ax.text(x, y, t, fontsize=fs, color=color, ha=ha, va=va, weight=weight,
            style=style, zorder=6, linespacing=1.45, family=family)


def panel(x0, y0, x1, y1, title=None):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0,rounding_size=1.0", linewidth=1.3,
                 edgecolor="#30363d", facecolor=PANEL, zorder=0))
    if title:
        lab((x0 + x1) / 2, y1 - 2.6, title, fs=14, color=TXT, ha="center", weight="bold")


# ── заголовок ────────────────────────────────────────────────────────────────
lab(90, 97.4, "Mini-GPT  —  архитектура  (≈ 4.88M параметров · char-level · MLX)",
    fs=23, color=TXT, ha="center", weight="bold")
lab(90, 94.3, "pre-LN декодер-трансформер  ·  d = 256  ·  6 слоёв  ·  4 головы  ·  "
    "контекст 128  ·  словарь 216 символов  ·  forward+backward вручную",
    fs=12.5, color=MUTE, ha="center")

# ════════════════════════════════════════════════════════════════════════════
# ПАНЕЛЬ A — макро-проход (полная модель снизу вверх)
# ════════════════════════════════════════════════════════════════════════════
panel(3, 5, 43, 92, "Полный проход")
ax_box = 19   # центр колонки

m_in   = box(ax_box, 11, 26, 5, "idx · токены\n(B, T)  int", "io", 12)
m_emb  = box(ax_box, 22, 26, 8, "Token-эмбеддинг  wte (216×256)\n⊕\nPos-эмбеддинг  wpe (128×256)",
             "emb", 11.5)
m_x    = box(ax_box, 32, 26, 5, "x = tok + pos\n(B, T, 256)", "io", 12)
m_blk  = box(ax_box, 51, 28, 16, "Д Е К О Д Е Р - Б Л О К\n(pre-LN, GPT-2)\n\n× 6  слоёв\n→ (B, T, 256)",
             "attn", 13, weight="bold")
# намёк на стопку из 6 блоков
for dy in (-5.6, -2.8, 2.8, 5.6):
    ax.plot([ax_box - 11, ax_box + 11], [51 + dy, 51 + dy], color=C["attn"][0],
            lw=0.7, alpha=0.35, zorder=4)
m_lnf  = box(ax_box, 66, 26, 4.6, "Финальный LayerNorm  (lnf)", "ln", 11.5)
m_head = box(ax_box, 72.5, 26, 4.6, "Linear-голова  256 → 216", "head", 12)
m_log  = box(ax_box, 78.5, 26, 4.6, "logits  (B, T, 216)", "io", 12)
m_loss = box(ax_box, 85, 26, 4.6, "Softmax → Cross-Entropy\nloss (скаляр)", "head", 11.5)

for a, b in [(m_in, m_emb), (m_emb, m_x), (m_x, m_blk), (m_blk, m_lnf),
             (m_lnf, m_head), (m_head, m_log), (m_log, m_loss)]:
    arr(a.top, b.bot)

# боковые аннотации параметров
ann = MUTE
lab(33, 20, "B = 32\nT = 128", 9.5, ann)
lab(33, 51, "6 × 789 760\n= 4 738 560\n(97 % весов)", 9.5, C["attn"][0])
lab(33, 66, "512", 9.5, ann); lab(33, 72.5, "55 512", 9.5, ann)
lab(5.4, 85, "argmax(p) ⇒\nследующий\nсимвол", 8.8, ann, ha="left")

# зум-скобка A → B
for yb in (43, 59):
    ax.plot([32, 46], [yb, yb], color="#3f4651", lw=1.0, ls=(0, (4, 3)), zorder=1)

# ════════════════════════════════════════════════════════════════════════════
# ПАНЕЛЬ B — декодер-блок изнутри (residual stream)
# ════════════════════════════════════════════════════════════════════════════
panel(46, 5, 96, 92, "Один декодер-блок  (× 6)")
spine = 55                                   # x остаточного потока (residual)
ax.plot([spine, spine], [13, 83.5], color=C["res"][0], lw=3.0, alpha=0.85, zorder=2)
ax.text(spine - 2.0, 50, "residual stream", fontsize=9.5, color=C["res"][0],
        ha="center", va="center", style="italic", rotation=90, zorder=6)

b_in  = box(spine, 12, 16, 4, "x  (B, T, 256)", "io", 11)
b_out = box(spine, 86, 18, 4, "y → следующий блок", "io", 11)

# — под-слой 1: внимание —
tap1 = (spine, 23)
ax.add_patch(Circle(tap1, 0.7, color=C["res"][0], zorder=5))
ln1  = box(72, 23, 17, 4.4, "LayerNorm  ln1", "ln", 11)
mha  = box(76, 36, 26, 9, "Multi-Head Self-Attention\nh = 4 · dh = 64 · causal\nWq Wk Wv Wo  (+ bias)",
           "attn", 11.5, weight="bold")
add1 = oplus(spine, 48)
arr(tap1, ln1.left, rad=-0.0)
arr(ln1.top, (72, mha.bot[1]))
arr(mha.left, add1.right, rad=0.25, color=C["attn"][0])
lab(66, 51.5, "a = x + Attn(ln1(x))", 10, TXT, ha="center")

# — под-слой 2: FFN —
tap2 = (spine, 55)
ax.add_patch(Circle(tap2, 0.7, color=C["res"][0], zorder=5))
ln2  = box(72, 55, 17, 4.4, "LayerNorm  ln2", "ln", 11)
ff1  = box(76, 63, 22, 4.2, "Linear  256 → 1024   (W1, b1)", "ffn", 10.5)
fge  = box(76, 69, 22, 4.2, "GELU  (точный, через erf)", "ffn", 10.5)
ff2  = box(76, 75, 22, 4.2, "Linear  1024 → 256   (W2, b2)", "ffn", 10.5)
add2 = oplus(spine, 81)
arr(tap2, ln2.left)
arr(ln2.top, ff1.bot); arr(ff1.top, fge.bot); arr(fge.top, ff2.bot)
arr(ff2.left, add2.right, rad=0.25, color=C["ffn"][0])
lab(66, 84.4, "y = a + FFN(ln2(a))", 10, TXT, ha="center")

arr(b_in.top, (spine, 22))
arr(add1.top, (spine, 54))
arr(add2.top, (spine, 84), color=C["res"][0], lw=3.0)

# зум-скобка B(MHA) → C
for p in [(mha.right, (99, 30)), (mha.right, (99, 90))]:
    ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color="#3f4651", lw=1.0,
            ls=(0, (4, 3)), zorder=1)

# ════════════════════════════════════════════════════════════════════════════
# ПАНЕЛЬ C — внутренности Multi-Head Attention
# ════════════════════════════════════════════════════════════════════════════
panel(99, 5, 139, 92, "Self-Attention изнутри  (на одну голову)")
cz = 119
c_z = box(cz, 12, 30, 4.4, "z = ln1(x)   (B, T, 256)", "attn", 11)
qb  = box(107, 23, 11, 5, "Q = z·Wq", "attn", 10.5)
kb  = box(119, 23, 11, 5, "K = z·Wk", "attn", 10.5)
vb  = box(131, 23, 11, 5, "V = z·Wv", "attn", 10.5)
arr(c_z.top, qb.bot); arr(c_z.top, kb.bot); arr(c_z.top, vb.bot)
lab(cz, 30, "reshape → 4 головы:  (B, 4, T, 64)", 9.3, MUTE, ha="center")

c_sc = box(112, 42, 24, 7.5, "scores = Q · Kᵀ · 1/√64\n+ causal mask  (△ = −∞)\n(B, 4, T, T)",
           "attn", 10.5)
arr((108, 25.5), (109, 38), rad=-0.12)            # Q → scores
arr((118, 25.5), (115, 38), rad=0.12)             # K → scores

c_sm = box(112, 54, 24, 5.5, "softmax(axis = −1)\natt   (B, 4, T, T)", "attn", 10.5)
arr(c_sc.top, c_sm.bot)

c_wv = box(115, 65, 24, 5.5, "att · V   →   (B, 4, T, 64)", "attn", 10.5)
arr(c_sm.top, (112, c_wv.bot[1]))
arr((134, 21), (127, 64.5), rad=-0.12, color=C["attn"][0])     # ветвь V
lab(135.5, 44, "ветвь\nV", 9, C["attn"][0], ha="center", weight="bold")

c_mg = box(cz, 75, 24, 5, "merge голов  →  (B, T, 256)", "attn", 10.5)
arr(c_wv.top, c_mg.bot)
c_wo = box(cz, 84.5, 24, 5.5, "Linear Wo → attn   (B, T, 256)", "head", 10.5)
arr(c_mg.top, c_wo.bot)

# ════════════════════════════════════════════════════════════════════════════
# ПАНЕЛЬ D — гиперпараметры · бюджет параметров · формулы · легенда
# ════════════════════════════════════════════════════════════════════════════
panel(142, 5, 178, 92)

# — гиперпараметры —
lab(160, 89, "Гиперпараметры", 13, TXT, ha="center", weight="bold")
hp = ("d  (n_embd)        256\n"
      "L  (n_layer)       6\n"
      "h  (n_head)        4\n"
      "dh = d / h         64\n"
      "ctx (block_size)   128\n"
      "V  (vocab)         216\n"
      "MLP hidden         1024  (4d)\n"
      "норма              pre-LN\n"
      "активация          GELU (erf)\n"
      "позиции            обучаемые wpe\n"
      "scale              1 / √64\n"
      "оптимизатор        Adam .9/.95 +wd")
lab(144.5, 78.5, hp, 9.6, TXT, ha="left", va="top", family="monospace")

# — бюджет параметров —
lab(160, 60, "Бюджет параметров", 13, TXT, ha="center", weight="bold")
pb = ("Token emb  wte     55 296\n"
      "Pos emb    wpe     32 768\n"
      "Decoder ×6      4 738 560\n"
      "  attn QKVO     6×263 168\n"
      "  FFN d·4d·d    6×525 568\n"
      "  2×LayerNorm     6×1 024\n"
      "Final LN   lnf        512\n"
      "Head 256→216       55 512\n"
      "──────────────────────────\n"
      "ИТОГО          4 882 648")
lab(144.5, 49.5, pb, 9.6, TXT, ha="left", va="top", family="monospace")
lab(160, 30.8, "≈ 4.88M  ·  всё float32  (≈ 19 МБ)", 9.5, C["attn"][0], ha="center")

# — формулы —
lab(160, 27.5, "Ключевые формулы", 13, TXT, ha="center", weight="bold")
lab(160, 23.6, r"$\mathrm{Attn}=\mathrm{softmax}\!\left(\dfrac{QK^{\top}}{\sqrt{d_h}}+M\right)V$",
    11.5, TXT, ha="center")
lab(160, 19.4, r"$a = x + \mathrm{Attn}(\mathrm{LN}_1(x))$", 11, TXT, ha="center")
lab(160, 16.8, r"$y = a + \mathrm{FFN}(\mathrm{LN}_2(a))$", 11, TXT, ha="center")
lab(160, 13.8, r"$\partial L/\partial z = (p-\mathrm{onehot})/(BT)$", 11, C["head"][0], ha="center")

# — легенда —
leg = [("emb", "эмбеддинги"), ("ln", "LayerNorm"), ("attn", "attention"),
       ("ffn", "FFN / MLP"), ("head", "проекция/выход"), ("res", "residual +")]
lx = 145.5
for i, (k, name) in enumerate(leg):
    yy = 9.8 - (i % 3) * 2.2
    xx = lx + (i // 3) * 17
    ax.add_patch(FancyBboxPatch((xx, yy - 0.7), 2.4, 1.4, boxstyle="round,pad=0,rounding_size=0.3",
                 edgecolor=C[k][0], facecolor=C[k][1], lw=1.4, zorder=4))
    lab(xx + 3.0, yy, name, 9.3, TXT, ha="left")

fig.tight_layout(pad=0.5)
fig.savefig("architecture.png", dpi=190, bbox_inches="tight", pad_inches=0.25,
            facecolor=BG)
print("saved architecture.png")
