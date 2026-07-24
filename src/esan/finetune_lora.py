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

Base = google/byt5-small: a byte-level T5, so Esan's ọ/ẹ are exact AND its tiny
vocabulary means the output head isn't a bottleneck. (We also tried subword
mt5-small; it plateaued at a *higher* loss — its 250K-token frozen output head
can't easily emit rare Esan pieces. ByT5 got closer.) The earlier "gibberish" was
underfitting — the model learned the prior and ignored the input. Two fixes:
(a) also train the output head (`modules_to_save`, cheap on ByT5), and (b) a
higher learning rate, so the loss reaches near zero instead of stalling.

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

BASE = "google/byt5-small"  # byte-level: ọ/ẹ exact, tiny vocab (output head not a bottleneck)
MAX_LEN = 128
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "finetune.jsonl"
OUT = ROOT / "checkpoints" / "esan-byt5-lora"

EN2ES = "translate English to Esan: "
ES2EN = "translate Esan to English: "


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE)

    # LoRA: keep the base model frozen, inject trainable low-rank matrices. We
    # target every attention AND feed-forward projection (q,k,v,o,wi,wo) at rank
    # 32 so the adapters have enough capacity to actually MEMORISE the dictionary.
    # (r=8 on just q,v — the textbook default — was too small here and underfit.)
    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
        modules_to_save=["lm_head"],  # also FULLY train the output head (cheap on ByT5's tiny vocab) so it can emit exact tokens
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # A bilingual dictionary has nothing to "generalise" to — an unseen word can't
    # be inferred — so we train on ALL of it. This model MEMORISES the dictionary;
    # here that's the goal, not overfitting to guard against.
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
        per_device_train_batch_size=16,
        learning_rate=1e-3,         # higher: the 3e-4 runs plateaued before converging
        num_train_epochs=120,       # tiny data → many passes to fully memorise it
        logging_steps=25,
        save_strategy="no",
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        processing_class=tokenizer,   # transformers >=4.46 renamed this from tokenizer=
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

    print("\n── sample translations (all from the training dictionary) ──")
    for w in ["cow", "water", "king", "goat", "moon", "money"]:
        print(f"  EN→ES  {w:12} -> {translate(w, True)}")
    for w in ["amẹn", "ẹmena", "uki"]:
        print(f"  ES→EN  {w:12} -> {translate(w, False)}")


if __name__ == "__main__":
    main()
