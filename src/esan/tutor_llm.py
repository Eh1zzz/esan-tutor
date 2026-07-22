#!/usr/bin/env python3
"""
Phase 2 v2 — a conversational Esan tutor powered by Claude, grounded in YOUR data.

Your whole curated dataset (vocab + numbers + money + corpus) is small enough to
fit in the prompt, so this is the simplest possible "RAG": retrieval is trivial
because *everything fits*. We hand Claude the entire dataset as ground truth on
every turn. Claude does the teaching and conversation; the dataset keeps it
honest about a low-resource language it wasn't trained on (no inventing Esan).

── Setup ──────────────────────────────────────────────────────────────────────
Install the SDK and set your key as an environment variable. Never paste your key
into a chat or commit it — the SDK reads it from the environment automatically.

    pip install anthropic

    # Windows (persists; reopen the terminal afterwards):
    setx ANTHROPIC_API_KEY "sk-ant-..."
    # or, PowerShell, this session only:
    $env:ANTHROPIC_API_KEY = "sk-ant-..."

── Run ────────────────────────────────────────────────────────────────────────
    python src/esan/tutor_llm.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency. Run:  pip install anthropic")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL = "claude-opus-4-8"  # swap to "claude-sonnet-5" or "claude-haiku-4-5" for lower cost
ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "clean"


def load_knowledge() -> str:
    """Format the entire curated dataset as one compact, labelled ground-truth block."""
    parts: list[str] = []

    with open(CLEAN / "vocab.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["esan"] and r["english"]]
    parts.append("## VOCABULARY (esan = english)\n"
                 + "\n".join(f"{r['esan']} = {r['english']}" for r in rows))

    with open(CLEAN / "numbers.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["esan"]]
    parts.append("## NUMBERS (value: esan)\n"
                 + "\n".join(f"{r['value']}: {r['esan']}"
                             + (f"  (alt: {r['alt']})" if r.get("alt") else "")
                             for r in rows))

    with open(CLEAN / "money.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["esan"]]
    parts.append("## MONEY (amount: esan)\n"
                 + "\n".join(f"{r['amount']}: {r['esan']}" for r in rows))

    corpus = (CLEAN / "corpus.txt").read_text(encoding="utf-8").strip()
    parts.append("## ESAN TEXT CORPUS (poems, folk tales, riddles, sayings)\n" + corpus)

    return "\n\n".join(parts)


SYSTEM_INTRO = """You are a warm, encouraging tutor for the Esan language (an Edoid \
language of Edo State, Nigeria). Esan is low-resource, so your training knowledge of \
it is thin — treat the ESAN KNOWLEDGE BASE below as your ground truth.

Rules:
- Ground every Esan word, number, or phrase in the knowledge base, and quote the exact \
spelling — including the dotted letters ọ and ẹ, which are distinct letters from o and e.
- If the knowledge base doesn't cover what's asked, say so honestly. Explain general \
language-learning ideas freely, but NEVER invent Esan words, spellings, or translations.
- Be concise and friendly. Offer a small practice prompt when it helps learning.
- When asked to quiz or drill the user, draw the items from the knowledge base.
"""


def build_system(knowledge: str) -> list[dict]:
    # Keep the large, stable knowledge base as its own block with a cache breakpoint,
    # so it can be reused across turns as the dataset grows (prompt caching).
    return [
        {"type": "text", "text": SYSTEM_INTRO},
        {
            "type": "text",
            "text": "# ESAN KNOWLEDGE BASE\n\n" + knowledge,
            "cache_control": {"type": "ephemeral"},
        },
    ]


def main() -> None:
    try:
        client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / ant profile from env
    except anthropic.AnthropicError:
        sys.exit("No API key found. Set ANTHROPIC_API_KEY in your environment "
                 '(e.g.  setx ANTHROPIC_API_KEY "sk-ant-...")  and reopen the terminal.')

    system = build_system(load_knowledge())
    print("🗣️  Esan Tutor (Claude-powered). Ask me anything about Esan — vocabulary, "
          "numbers, grammar,\n    translations, or say 'quiz me'. Type 'q' to quit.\n")

    messages: list[dict] = []
    while True:
        try:
            user = input("you   > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nỌ dẹ! (goodbye)")
            break
        if not user:
            continue
        if user.lower() in {"q", "quit", "exit"}:
            print("Ọ dẹ! (goodbye)")
            break

        messages.append({"role": "user", "content": user})
        print("tutor > ", end="", flush=True)
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=1024,
                system=system,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                final = stream.get_final_message()
        except anthropic.AuthenticationError:
            print("\n[Auth failed — check your ANTHROPIC_API_KEY.]")
            messages.pop()
            continue
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            print(f"\n[API error: {e}]")
            messages.pop()
            continue

        print("\n")
        messages.append({"role": "assistant", "content": final.content})


if __name__ == "__main__":
    main()
