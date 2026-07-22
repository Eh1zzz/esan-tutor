#!/usr/bin/env python3
"""
Phase 2 v3 (data prep) — build the LoRA fine-tuning set from the clean dataset.

Turns vocab / numbers / sentence-pairs into **bidirectional** translation examples
(English⇄Esan) in the seq2seq format a T5-style model trains on:

    {"input": "translate English to Esan: cow", "target": "ẹmena"}
    {"input": "translate Esan to English: ẹmena", "target": "cow"}

Every synonym and number-alternate becomes its own example, so the model sees all
valid forms. Reads the clean (NFC-normalised) data; writes data/processed/finetune.jsonl.

    python src/esan/build_finetune_data.py
"""
from __future__ import annotations

import csv
import json
import sys
import unicodedata
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "data" / "processed" / "finetune.jsonl"

EN2ES = "translate English to Esan: "
ES2EN = "translate Esan to English: "


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", str(s).strip())


def add_pair(examples: list, seen: set, english: str, esan: str) -> None:
    """Add both directions for one English↔Esan pair, de-duplicated."""
    english, esan = nfc(english), nfc(esan)
    if not english or not esan:
        return
    for inp, tgt in ((EN2ES + english, esan), (ES2EN + esan, english)):
        if (inp, tgt) not in seen:
            seen.add((inp, tgt))
            examples.append({"input": inp, "target": tgt})


def main() -> None:
    examples: list[dict] = []
    seen: set = set()

    with open(CLEAN / "vocab.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            add_pair(examples, seen, r["english"], r["esan"])

    with open(CLEAN / "numbers.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            add_pair(examples, seen, r["value"], r["esan"])
            for alt in r.get("alt", "").replace("/", ",").split(","):
                if alt.strip():
                    add_pair(examples, seen, r["value"], alt)

    pairs_path = CLEAN / "pairs.jsonl"
    if pairs_path.exists():
        for line in pairs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                add_pair(examples, seen, o.get("english", ""), o.get("esan", ""))

    OUT.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in examples) + "\n",
                   encoding="utf-8")
    en2es = sum(1 for e in examples if e["input"].startswith(EN2ES))
    print(f"wrote {len(examples)} examples -> {OUT.relative_to(ROOT)}")
    print(f"  English→Esan: {en2es} | Esan→English: {len(examples) - en2es}")


if __name__ == "__main__":
    main()
