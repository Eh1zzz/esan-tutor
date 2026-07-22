#!/usr/bin/env python3
"""
Phase 0 data pipeline for the Esan Tutor.

Reads the human-curated files in data/processed/ and writes clean,
training-ready copies to data/clean/. The most important job is **Unicode NFC
normalisation**: Esan's dotted letters (ọ, ẹ) must be a single code point each.
If some were typed as a plain letter + a combining dot (two code points), the
model would treat "ọ" and "o‌̣" as *different* characters and learn nonsense.
We also collapse whitespace, drop blank lines, de-duplicate the vocab, and print
stats so you can see exactly what you've built — including the character
"alphabet" the Phase-1 model will tokenise.

Run from the repo root:
    python src/esan/normalize.py
"""
from __future__ import annotations

import csv
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Esan text is full of non-ASCII code points; make stdout UTF-8 so printing the
# alphabet doesn't crash on a Windows (cp1252) console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "clean"


def nfc(s) -> str:
    """Normalise to Unicode NFC and collapse runs of whitespace to one space."""
    return unicodedata.normalize("NFC", " ".join(str(s).split()))


def was_denormalised(original: str, normalised: str) -> bool:
    """True if NFC actually changed the string (i.e. it wasn't NFC already)."""
    return unicodedata.normalize("NFC", str(original)) != unicodedata.normalize("NFD", str(original)) \
        and str(original) != normalised


def clean_csv(name: str, key: str | None) -> tuple[list[str], list[dict]]:
    """NFC-normalise every cell; optionally de-duplicate on `key` (warn on clashes)."""
    path = SRC / name
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        raw = list(reader)

    fixed = 0
    rows: list[dict] = []
    for r in raw:
        clean = {}
        for k, v in r.items():
            v = v or ""
            nv = nfc(v)
            if unicodedata.normalize("NFC", v) != v:  # wasn't already NFC
                fixed += 1
            clean[k] = nv
        rows.append(clean)

    dupes = 0
    if key:
        seen: dict[str, str] = {}
        deduped: list[dict] = []
        gloss_col = "english" if "english" in fields else None
        for r in rows:
            k = r[key].casefold()
            if k in seen:
                dupes += 1
                # Surface a genuine conflict (same word, different meaning).
                if gloss_col and seen[k] and r[gloss_col] and seen[k] != r[gloss_col]:
                    print(f"  ! duplicate '{r[key]}' with differing {gloss_col}: "
                          f"'{seen[k]}' vs '{r[gloss_col]}'")
                continue
            seen[k] = r.get(gloss_col, "") if gloss_col else ""
            deduped.append(r)
        rows = deduped

    print(f"  {name:14} {len(rows):>4} rows"
          f"{f'  (NFC-fixed {fixed} cells)' if fixed else ''}"
          f"{f'  (dropped {dupes} dupes)' if dupes else ''}")
    return fields, rows


def write_csv(name: str, fields: list[str], rows: list[dict]) -> None:
    with open(OUT / name, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def clean_corpus() -> list[str]:
    """One Esan sentence/line per row: NFC, trimmed, no blanks, de-duplicated."""
    lines, seen = [], set()
    for ln in (SRC / "corpus.txt").read_text(encoding="utf-8").splitlines():
        s = nfc(ln)
        if s and s not in seen:
            seen.add(s)
            lines.append(s)
    (OUT / "corpus.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {'corpus.txt':14} {len(lines):>4} lines")
    return lines


def clean_pairs() -> list[dict]:
    rows = []
    for ln in (SRC / "pairs.jsonl").read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        obj["esan"] = nfc(obj.get("esan", ""))
        obj["english"] = nfc(obj.get("english", ""))
        rows.append(obj)
    with open(OUT / "pairs.jsonl", "w", encoding="utf-8") as f:
        for o in rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"  {'pairs.jsonl':14} {len(rows):>4} pairs")
    return rows


def char_report(corpus_lines: list[str], vocab_rows: list[dict]) -> None:
    """Show the character 'alphabet' — exactly what a char-level model will tokenise."""
    text = "\n".join(corpus_lines) + "\n".join(r.get("esan", "") for r in vocab_rows)
    chars = Counter(text)
    alphabet = sorted(c for c in chars if not c.isspace())
    print("\nCharacter alphabet the Phase-1 model will see "
          f"({len(alphabet)} distinct):")
    print("  " + " ".join(alphabet))
    # Flag anything suspicious: combining marks (would mean non-NFC slipped through)
    combining = [c for c in alphabet if unicodedata.combining(c)]
    if combining:
        print(f"  ⚠️  combining marks present (non-NFC!): {combining}")
    else:
        print("  ✓ no stray combining marks — text is clean NFC")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Normalising {SRC}  ->  {OUT}\n")

    vfields, vrows = clean_csv("vocab.csv", key="esan")
    write_csv("vocab.csv", vfields, vrows)
    for name in ("numbers.csv", "money.csv"):
        f, r = clean_csv(name, key=None)
        write_csv(name, f, r)

    corpus = clean_corpus()
    clean_pairs()

    char_report(corpus, vrows)
    print("\nDone. Clean, NFC-normalised data is in data/clean/ — ready for training.")


if __name__ == "__main__":
    main()
