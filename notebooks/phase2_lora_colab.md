# Phase 2 v3 in Google Colab — fine-tune an Esan translator

`src/esan/finetune.py` fine-tunes **google/byt5-small** to translate English⇄Esan.
A free T4 GPU finishes it in ~15 minutes.

> We *started* with LoRA (parameter-efficient adapters) and it kept underfitting on
> this tiny dataset — see **The debugging story** below. The script now does **full
> fine-tuning**, which reliably memorises the dictionary.

## 1. Open Colab with a GPU
**Runtime → Change runtime type → T4 GPU → Save.**

## 2. Get the repo + install deps
```python
!git clone https://github.com/Eh1zzz/esan-tutor.git
%cd esan-tutor
!pip install -q "transformers>=4.46" "datasets>=2.19" accelerate
```

## 3. Build the data + train
```python
!python src/esan/build_finetune_data.py   # -> data/processed/finetune.jsonl (330 examples)
!python src/esan/finetune.py
```

## What to watch (the learning payoff)
- **train loss ↓ toward ~0** over the 50 epochs — full fine-tuning can actually
  drive the loss down (LoRA floored at ~0.8; see below). There's deliberately **no
  eval set**: a bilingual dictionary has nothing to generalise to (an unseen word
  can't be inferred), so a held-out split just measures the impossible. Training on
  everything is the honest move — memorisation is the goal here, not a bug.
- **The sample translations** at the end — `cow → ẹmena`, `water → amẹn`,
  `moon → uki`, etc. Seeing a general-purpose model bend to output *your* language,
  from data *you* curated, is the payoff.

## The debugging story (this is the real lesson)
Getting this to translate took several tries — a genuine ML debugging arc:
1. **ByT5 + LoRA (small adapter)** → loss stalled ~0.8; output was one repeated phrase.
2. **Switched to mT5** (subword, thought it'd memorise easier) → loss stalled *higher*
   (~2.4) and still collapsed. Its 250K-token frozen output head can't emit rare
   Esan pieces — worse, not better.
3. **Bigger adapter, train the output head, 3× the learning rate** → the ByT5 loss
   *still* floored at ~0.8 and collapsed to one phrase for **every** input.
4. **Diagnosis:** a loss floor that won't move for capacity, LR, or the output head
   is **structural**, not a hyperparameter. The decoder was ignoring the encoder —
   it learned a good *unconditional* model of Esan text (that ~0.8 ≈ the byte-level
   entropy of the targets) and never learned to look at the source. LoRA's frozen
   encoder / cross-attention was the wall.
5. **Fix: full fine-tuning.** Every weight updates, so the model can wire
   source→target. The loss drops toward zero and the translations come out right.

Takeaways:
- **Collapse to one output = the decoder ignoring the input** (underfitting). A loss
  floor immune to LR/capacity is a *structural* clue, not a tuning problem.
- **LoRA shines with a capable base and enough data; for tiny memorisation tasks,
  full fine-tuning is more reliable.** Knowing which to reach for is the skill.

## Experiment
- `num_train_epochs`, `learning_rate` in `finetune.py`.
- Add more data upstream (grow `vocab.csv` / `pairs.jsonl`, re-run `normalize.py`
  then `build_finetune_data.py`) — **this** is what actually improves the model.

## Honest takeaway
With ~330 examples this memorises a bilingual dictionary; it won't translate unseen
sentences. That's the real lesson of low-resource NLP: **the model is easy, the data
is everything.** The path to a genuinely good Esan translator is more *sentence*
pairs — from the textbook's dialogues and, later, your speakers — the same
human-in-the-loop curation that built the dataset in the first place.
