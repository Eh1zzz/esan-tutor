#!/usr/bin/env python3
"""
Phase 2 v3 — fine-tune a small model to translate English⇄Esan.

We STARTED with LoRA (adapters on a frozen base) — the full debugging story is in
notebooks/phase2_lora_colab.md. On this tiny dataset LoRA kept UNDERFITTING: the
loss floored (~0.8 for ByT5) and the decoder never learned to condition on the
input — it modelled the *unconditional* distribution of the targets and emitted
the same phrase for every input (that ~0.8 ≈ the byte-level entropy of the target
text). Raising the rank, the learning rate, and even training the output head did
not move that floor: the frozen encoder / cross-attention was the wall.

So this does FULL fine-tuning — every weight updates, so the model can actually
wire the source into the output. The lesson: **LoRA shines with a capable base and
enough data; for tiny memorisation tasks, full fine-tuning is more reliable.**

Base = google/byt5-small (byte-level → Esan's ọ and ẹ are represented exactly).

Run (Colab T4 recommended — see notebooks/phase2_lora_colab.md):
    pip install "transformers>=4.46" "datasets>=2.19" accelerate
    python src/esan/build_finetune_data.py   # writes data/processed/finetune.jsonl
    python src/esan/finetune.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = "google/byt5-small"
MAX_LEN = 128
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "finetune.jsonl"
OUT = ROOT / "checkpoints" / "esan-byt5"

EN2ES = "translate English to Esan: "
ES2EN = "translate Esan to English: "


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE)  # FULL fine-tune: all weights train
    n = sum(p.numel() for p in model.parameters())
    print(f"full fine-tune — training all {n:,} parameters")

    # Train on ALL of it: a bilingual dictionary can't generalise to unseen words,
    # so this MEMORISES the pairs (here that's the goal, not overfitting to avoid).
    ds = load_dataset("json", data_files=str(DATA))["train"]

    def preprocess(batch):
        enc = tokenizer(batch["input"], max_length=MAX_LEN, truncation=True)
        enc["labels"] = tokenizer(
            text_target=batch["target"], max_length=MAX_LEN, truncation=True
        )["input_ids"]
        return enc

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT),
        per_device_train_batch_size=8,
        learning_rate=3e-4,         # standard T5 full-fine-tune LR
        num_train_epochs=120,       # 50 got loss to ~0.35 and still falling → more passes to fully memorise
        logging_steps=25,
        save_strategy="no",
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    print(f"\nsaved model -> {OUT.relative_to(ROOT)}")

    # Sanity check: translate a few words the model was trained on.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    def translate(text: str, to_esan: bool = True) -> str:
        prefix = EN2ES if to_esan else ES2EN
        ids = tokenizer(prefix + text, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=32)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    print("\n── sample translations (all from the training dictionary) ──")
    for w in ["cow", "water", "king", "goat", "moon", "money"]:
        print(f"  EN→ES  {w:12} -> {translate(w, True)}")
    for w in ["amẹn", "ẹmena", "uki"]:
        print(f"  ES→EN  {w:12} -> {translate(w, False)}")


if __name__ == "__main__":
    main()
