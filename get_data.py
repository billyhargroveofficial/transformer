"""
Стримит русскую Википедию с HuggingFace и пишет чистый срез в corpus.txt.
Качается НЕ весь датасет — только пока не наберём TARGET_BYTES.
Требует: uv pip install datasets
"""

import re
import unicodedata
from datasets import load_dataset

OUTPUT = "corpus.txt"
TARGET_BYTES = 40 * 1024 * 1024     # ~40 МБ — достаточно для модели на 1–10M параметров
MIN_CHARS = 200                      # пропускаем заглушки

# оставляем кириллицу, латиницу, цифры, базовую пунктуацию, пробелы/переводы строк
KEEP = re.compile(r"[^Ѐ-ӿ́a-zA-Z0-9 \t\n.,!?;:\"'()\-–—«»…]")

def clean(t):
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"={2,}[^=]+={2,}", "", t)        # ==Заголовки==
    t = KEEP.sub("", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def main():
    print("стримлю wikimedia/wikipedia 20231101.ru ...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ru",
                      split="train", streaming=True)
    total = 0; written = 0
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for s in ds:
            if total >= TARGET_BYTES:
                break
            txt = s.get("text", "")
            if len(txt) < MIN_CHARS:
                continue
            c = clean(txt)
            if len(c) < MIN_CHARS:
                continue
            f.write(c + "\n\n")
            total += len(c.encode("utf-8")) + 2
            written += 1
            if written % 2000 == 0:
                print(f"  {written} статей, {total/1024**2:.1f} МБ")
    print(f"готово: {written} статей, {total/1024**2:.1f} МБ -> {OUTPUT}")

if __name__ == "__main__":
    main()
