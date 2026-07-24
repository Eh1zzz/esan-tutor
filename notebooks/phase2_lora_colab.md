# Phase 2 v3 in Google Colab — LoRA fine-tune an Esan translator

`src/esan/finetune_lora.py` fine-tunes **google/byt5-small** to translate
English⇄Esan using **LoRA** (train tiny adapters, freeze the base model). A free
T4 GPU finishes this in a few minutes.

## 1. Open Colab with a GPU
**Runtime → Change runtime type → T4 GPU → Save.**

## 2. Get the repo + install deps
```python
!git clone https://github.com/Eh1zzz/esan-tutor.git
%cd esan-tutor
!pip install -q "transformers>=4.46" "peft>=0.11" "datasets>=2.19" accelerate sentencepiece protobuf
!pip uninstall -q -y torchao   # Colab ships an old torchao that trips PEFT's version check; we don't use it
```

> **If you see `ImportError: Found an incompatible version of torchao`** — that's
> the line above fixing it. `torchao` (low-bit quantization) isn't needed here;
> removing it makes PEFT's check pass.

## 3. Build the data + train
```python
!python src/esan/build_finetune_data.py   # -> data/processed/finetune.jsonl (330 examples)
!python src/esan/finetune_lora.py
```

## What to watch (the learning payoff)
- **`trainable params`** printed at the start — LoRA trains a *tiny* fraction
  (often <0.1%) of the model. That's the whole point of PEFT: adapt a big model
  cheaply.
- **train loss ↓** over the 100 epochs, ideally well below ~0.5 — the adapter is
  memorising the dictionary. There's deliberately **no eval set**: a bilingual
  dictionary has nothing to generalise to (an unseen word can't be inferred), so
  a held-out split just measures the impossible. Training on everything is the
  honest move, and memorisation is the goal here — not a bug.
- **The sample translations** at the end — `cow → ẹmena`, `water → amẹn`, etc.
  Seeing a general-purpose model bend to output *your* language, from adapters
  you trained on data *you* curated, is the payoff.

## The debugging story (this is the real lesson)
Getting this to actually translate took three tries — a genuine ML debugging arc:
1. **ByT5, small adapter** → loss stalled ~0.8, output was one repeated phrase.
2. **Switched to mT5** (subword, thought it'd memorise easier) → loss stalled
   *higher* (~2.4) and still collapsed. Its 250K-token **frozen output head**
   can't emit rare Esan pieces — so it was worse, not better.
3. **Diagnosis:** both runs *underfit* — the model learned the prior (one common
   output) and ignored the input. Fixes: go back to **ByT5** (tiny byte vocab →
   output head isn't a bottleneck; ọ/ẹ exact), **also train the output head**
   (`modules_to_save=["lm_head"]`, cheap on ByT5), and **raise the learning rate**
   so the loss reaches near zero instead of stalling.

Takeaway: **collapse to one output = underfitting.** Look at the frozen pieces
(here, the output head) and the learning rate before blaming the data.

## Experiment
- `r` (LoRA rank) and `target_modules` in `finetune_lora.py` → adapter capacity.
- `num_train_epochs` → more passes = lower loss (until it just memorises).
- Add more data upstream (grow `vocab.csv` / `pairs.jsonl`, re-run `normalize.py`
  and `build_finetune_data.py`) — **this** is what actually improves the model.

## Honest takeaway
With ~330 examples this memorises a bilingual dictionary; it won't translate
unseen sentences well. That's the real lesson of low-resource NLP: **the model is
easy, the data is everything.** The path to a genuinely good Esan translator is
more sentence pairs (from the textbook's dialogues, and from your speakers) — the
same human-in-the-loop curation that built the dataset in the first place.
