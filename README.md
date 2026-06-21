# transformer

Мини-GPT с нуля на [MLX](https://github.com/ml-explore/mlx) (Apple Silicon). Посимвольный
авторегрессионный трансформер, обученный на русской Википедии. Без autograd —
forward, backprop и Adam написаны руками, чтобы было видно каждую формулу.

## Что внутри

| параметр | значение |
|---|---|
| параметров | **~4.88M** (`d=256`, `L=6`, `h=4`, `ctx=128`) |
| словарь | 216 символов (кириллица + латиница + цифры + старослав) |
| датасет | ~23.3M символов = токенов (русская Вики, `corpus.txt`, 42 МБ) |
| обучение | бюджет по времени (~30 мин), Adam β=(0.9, 0.95), lr 5e-4 + warmup |

## Файлы

| файл | что делает |
|---|---|
| `mini_gpt.py` | модель: эмбеддинги, self-attention, MLP, forward / backward / cross-entropy |
| `optim.py` | Adam с weight decay |
| `data.py` | посимвольный токенизатор + нарезка батчей |
| `get_data.py` | скачивание корпуса |
| `train.py` | цикл обучения (`overfit` для проверки пайплайна / `run` на корпусе) |
| `gradcheck.py` | численная проверка градиентов |
| `sample.py` | генерация из чекпоинта |

## Запуск

```bash
python -m venv .venv && .venv/bin/pip install mlx numpy

# обучение на корпусе (бюджет 30 мин)
.venv/bin/python train.py --mode run --minutes 30

# генерация из чекпоинта
.venv/bin/python sample.py --prompt "В 1812 году " --n 500 --temp 0.7 --topk 40
```

Чекпоинт (`ckpt.npz` + `ckpt.json`) уже в репозитории — можно сразу семплить.
