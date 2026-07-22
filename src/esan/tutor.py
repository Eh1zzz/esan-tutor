#!/usr/bin/env python3
"""
Phase 2 (v1) — an interactive Esan tutor driven by your dataset.

Uses the clean data in data/clean/ to actually teach you Esan: flashcards,
multiple-choice quizzes, a numbers drill, and lookup. Needs nothing but Python
and the dataset — no GPU, no API key. This is the *useful* deliverable.

A later upgrade (tutor_llm.py) wraps a large language model with retrieval over
this same data for open-ended, conversational tutoring — see the Phase 2 notes
in the README.

    python src/esan/tutor.py
"""
from __future__ import annotations

import csv
import random
import sys
import unicodedata
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "clean"


def norm(s: str) -> str:
    """Lenient compare: NFC + casefold + trim, so 'Water ' == 'water'."""
    return unicodedata.normalize("NFC", str(s).strip()).casefold()


def load_vocab() -> list[dict]:
    with open(CLEAN / "vocab.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["esan"] and r["english"]]


def load_numbers() -> list[dict]:
    with open(CLEAN / "numbers.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["esan"]]


def flashcards(vocab: list[dict]) -> None:
    print("\n📇 Flashcards — I show the English, you type the Esan. (blank = reveal, 'q' = menu)\n")
    score = total = 0
    for card in random.sample(vocab, len(vocab)):
        ans = input(f"  English: {card['english']}\n  Esan?  ").strip()
        if ans.lower() == "q":
            break
        total += 1
        if not ans:
            print(f"  → {card['esan']}\n")
        elif norm(ans) == norm(card["esan"]):
            score += 1
            print("  ✅ correct!\n")
        else:
            print(f"  ❌ it's: {card['esan']}\n")
    if total:
        print(f"Score: {score}/{total}")


def quiz(vocab: list[dict]) -> None:
    print("\n❓ Multiple choice — pick the Esan word. ('q' = menu)\n")
    score = total = 0
    for card in random.sample(vocab, len(vocab)):
        others = random.sample([v for v in vocab if v is not card], 3)
        options = [card] + others
        random.shuffle(options)
        print(f"  How do you say '{card['english']}'?")
        for i, o in enumerate(options, 1):
            print(f"    {i}. {o['esan']}")
        pick = input("  > ").strip()
        if pick.lower() == "q":
            break
        total += 1
        if pick.isdigit() and 1 <= int(pick) <= 4 and options[int(pick) - 1] is card:
            score += 1
            print("  ✅ correct!\n")
        else:
            print(f"  ❌ it's: {card['esan']}\n")
    if total:
        print(f"Score: {score}/{total}")


def numbers_drill(numbers: list[dict]) -> None:
    print("\n🔢 Numbers — I give a number, you type the Esan. (blank = reveal, 'q' = menu)\n")
    score = total = 0
    for n in random.sample(numbers, len(numbers)):
        val = n["value"].rstrip("0").rstrip(".") if "." in n["value"] else n["value"]
        ans = input(f"  {val} in Esan?  ").strip()
        if ans.lower() == "q":
            break
        total += 1
        accepted = {norm(n["esan"])} | {norm(a) for a in n.get("alt", "").replace("/", ",").split(",") if a.strip()}
        if not ans:
            print(f"  → {n['esan']}" + (f"  (or {n['alt']})" if n['alt'] else "") + "\n")
        elif norm(ans) in accepted:
            score += 1
            print("  ✅ correct!\n")
        else:
            print(f"  ❌ it's: {n['esan']}\n")
    if total:
        print(f"Score: {score}/{total}")


def lookup(vocab: list[dict]) -> None:
    print("\n🔎 Lookup — type an English or Esan word (substring). ('q' = menu)\n")
    while True:
        q = input("  search: ").strip()
        if not q or q.lower() == "q":
            break
        nq = norm(q)
        hits = [v for v in vocab if nq in norm(v["esan"]) or nq in norm(v["english"])]
        if not hits:
            print("  (no match)\n")
        for v in hits[:15]:
            print(f"    {v['esan']}  =  {v['english']}")
        print()


def main() -> None:
    vocab, numbers = load_vocab(), load_numbers()
    print(f"🗣️  Esan Tutor — {len(vocab)} words, {len(numbers)} numbers loaded.")
    menu = {
        "1": ("Flashcards (English → Esan)", lambda: flashcards(vocab)),
        "2": ("Multiple-choice quiz", lambda: quiz(vocab)),
        "3": ("Numbers drill", lambda: numbers_drill(numbers)),
        "4": ("Lookup / search", lambda: lookup(vocab)),
        "q": ("Quit", None),
    }
    while True:
        print("\n── Menu ──")
        for k, (label, _) in menu.items():
            print(f"  {k}. {label}")
        choice = input("> ").strip().lower()
        if choice == "q":
            print("Ọ dẹ! (goodbye)")
            break
        action = menu.get(choice)
        if action and action[1]:
            action[1]()
        else:
            print("  (pick 1-4 or q)")


if __name__ == "__main__":
    main()
