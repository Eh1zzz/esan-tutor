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
Stand on existing models to get something genuinely useful.
- **Option A — fine-tune:** LoRA-fine-tune a small open multilingual model on the
  Esan pairs (teaches datasets, PEFT, evaluation).
- **Option B — retrieval (RAG):** wrap a strong hosted LLM with the curated Esan
  data (fastest path to a good tutor).
- Tutor features: vocab drills, translation practice, corrections, simple conversation.
- **Deliverable:** a usable Esan tutor.

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
