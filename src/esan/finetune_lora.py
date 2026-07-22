#!/usr/bin/env python3
"""
Phase 2 v3 — LoRA fine-tune a small model to translate English⇄Esan.

What you'll learn here — the machinery behind every "fine-tuned" model:
  • Base model      — a pretrained network we adapt instead of training from zero.
  • LoRA / PEFT     — freeze the huge base model, train tiny low-rank "adapter"
                      matrices (<1% of the weights). Cheap, fast, fits on a T4.
  • Seq2seq loop    — encoder reads the prompt, decoder generates the translation;
                      loss is next-token prediction on the target side.
  • Evaluation      — hold out 10% and watch eval loss / sample translations.

Honest expectation: with ~330 examples the model will largely **memorise** the
dictionary rather than generalise. That's fine — the goal is the *method*.

Base = google/byt5-small: a BYTE-LEVEL T5. It reads raw bytes, so it has no
tokenizer-vocabulary gaps — Esan's ọ and ẹ are represented exactly. (A subword
model like mt5-small can silently map rare glyphs to <unk>.)

Run (Colab T4 recommended — see notebooks/phase2_lora_colab.md):
    pip install "transformers>=4.41" "peft>=0.11" "datasets>=2.19" accelerate
    python src/esan/build_finetune_data.py   # writes data/processed/finetune.jsonl
    python src/esan/finetune_lora.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
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
MAX_LEN = 64
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "finetune.jsonl"
OUT = ROOT / "checkpoints" / "esan-byt5-lora"

EN2ES = "translate English to Esan: "
ES2EN = "translate Esan to English: "


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE)

    # LoRA: keep the base model frozen, inject trainable low-rank matrices into the
    # attention query/value projections. Only these adapters learn.
    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,                       # rank of the adapters (small = few params)
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q", "v"],  # T5 attention query & value projections
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()   # e.g. "trainable: ~150K / 300M (0.05%)"

    ds = load_dataset("json", data_files=str(DATA))["train"].train_test_split(
        test_size=0.1, seed=42
    )

    def preprocess(batch):
        enc = tokenizer(batch["input"], max_length=MAX_LEN, truncation=True)
        enc["labels"] = tokenizer(
            text_target=batch["target"], max_length=MAX_LEN, truncation=True
        )["input_ids"]
        return enc

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-4,
        num_train_epochs=40,        # tiny data → many passes to actually learn it
        logging_steps=20,
        eval_strategy="epoch",      # (older transformers: evaluation_strategy)
        save_strategy="no",
        predict_with_generate=True,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)          # saves just the small LoRA adapter
    tokenizer.save_pretrained(OUT)
    print(f"\nsaved LoRA adapter -> {OUT.relative_to(ROOT)}")

    # Quick sanity check: translate a few words with the trained adapter.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    def translate(text: str, to_esan: bool = True) -> str:
        prefix = EN2ES if to_esan else ES2EN
        ids = tokenizer(prefix + text, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=32)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    print("\n── sample translations ──")
    for w in ["cow", "water", "king", "goat"]:
        print(f"  EN→ES  {w:12} -> {translate(w, True)}")
    for w in ["amẹn", "ẹmena"]:
        print(f"  ES→EN  {w:12} -> {translate(w, False)}")


if __name__ == "__main__":
    main()
