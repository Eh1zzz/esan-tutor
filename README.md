# Esan Tutor 🗣️

Building an AI that helps teach **Esan** (an Edoid language of Edo State, Nigeria).
A learning project: understand how LLMs work *and* ship a useful tutor.

> **The honest crux:** Esan is a *low-resource* language — little digital text,
> no ready-made datasets or pretrained models. So the **dataset is the hardest
> and most valuable part of this project.** The model is the easy bit.

## Roadmap (phased)

### Phase 0 — Data foundation  ← we are here
Turn source material into clean, structured data.
- Extract text from the Esan **textbook PDF** (`data/raw/`).
- Structure it into:
  - `data/processed/vocab.csv` — Esan ↔ English words
  - `data/processed/pairs.jsonl` — translation / example-sentence pairs
  - `data/processed/corpus.txt` — all Esan text (for the from-scratch model)
  - grammar notes / dialogues
- Later: add **audio** (recordings from speakers) for pronunciation + future ASR/TTS.
- **Deliverable:** a small but real Esan dataset.

### Phase 1 — Learn the internals (train from scratch)
Build a *tiny* language model on the Esan corpus (nanoGPT / makemore style).
- Concepts: tokenization, embeddings, a transformer block, the training loop, sampling.
- Runs on **Colab** (free T4 GPU).
- **Deliverable:** understanding + a toy Esan text generator. (Not a great tutor —
  the *learning* engine.)

### Phase 2 — The actual tutor
Stand on the dataset (and, later, existing models) to get something useful.

**v1 — data-driven tutor (built, runs now):** `python src/esan/tutor.py`
An interactive CLI that drills you on Esan straight from `data/clean/`:
flashcards, multiple-choice quiz, numbers drill, and lookup. No GPU, no API key.
This is a genuinely usable learning tool today.

**v2 — LLM-backed conversational tutor (built):** `python src/esan/tutor_llm.py`
A chat tutor powered by **Claude**, grounded in your dataset. The whole curated
dataset is small enough to fit in the prompt, so retrieval is trivial — we hand
Claude *all* of it as ground truth every turn (the honest "RAG" for a tiny corpus).
Claude teaches and converses; the data stops it inventing Esan it doesn't know.
Needs `pip install anthropic` and `ANTHROPIC_API_KEY` set in your environment.

**v3 — fine-tune (LoRA), built:** `src/esan/build_finetune_data.py` →
`src/esan/finetune_lora.py`. Adapts **google/byt5-small** (byte-level → no
tokenizer gaps for ọ/ẹ) to translate English⇄Esan with LoRA/PEFT. Teaches base
models, adapters, the seq2seq training loop, and evaluation. Runs on a free Colab
GPU — see `notebooks/phase2_lora_colab.md`. The dataset is thin, so it mostly
memorises the dictionary: a *learning* exercise, and a live demo of "the model is
easy, the data is everything."

- **Deliverable:** a conversational Esan tutor ✅ + a from-scratch LM ✅ + a
  LoRA-fine-tuned translator ✅.

## Workflow
- **Data prep + code:** local, in this repo.
- **Training / fine-tuning:** Google Colab notebooks in `notebooks/` (free GPU).
- **Sync:** push this repo to GitHub, then `git clone` it inside Colab (or mount Drive).

## Stack
Python · PyTorch · Hugging Face (`transformers`, `datasets`, `tokenizers`, `peft`) ·
`pdfplumber` / PyMuPDF for PDF extraction.

## Layout
```
data/raw/        source files (the textbook PDF, audio) — gitignored, not committed
data/interim/    intermediate extraction output
data/processed/  clean datasets used for training
notebooks/       Colab notebooks (Phase 1+)
src/esan/        reusable Python (extraction, dataset building, model code)
```
