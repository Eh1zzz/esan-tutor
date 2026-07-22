# Phase 1 in Google Colab — train the tiny Esan GPT

`src/esan/tinygpt.py` is a complete from-scratch character-level GPT. Colab gives
you a free GPU so training takes seconds instead of minutes.

## 1. Open Colab with a GPU
1. Go to **colab.research.google.com** → New notebook.
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU** → Save.

## 2. Get the code + data into Colab

**Option A — clone from GitHub (recommended once the repo is pushed):**
```python
!git clone https://github.com/<you>/esan-tutor.git
%cd esan-tutor
```

**Option B — no GitHub yet? Upload the two files.** In a Colab cell:
```python
from google.colab import files
import os
os.makedirs("data/clean", exist_ok=True)
os.makedirs("src/esan", exist_ok=True)
print("Upload tinygpt.py:");  files.upload()   # then move it to src/esan/
print("Upload corpus.txt:");  files.upload()   # then move it to data/clean/
```
(Then move the uploaded files: `!mv tinygpt.py src/esan/ && mv corpus.txt data/clean/`)

## 3. Train
Colab already has PyTorch installed, so just run:
```python
!python src/esan/tinygpt.py
```

You'll see the **loss drop** (that's learning) and, at the end, a **sample of
generated Esan-ish text** plus a saved checkpoint.

## What to look for (the learning payoff)
- **Loss going down** = the model is getting better at predicting the next character.
- **train loss << val loss** = it's *memorising* (expected — the corpus is tiny).
  That gap is exactly why a real tutor needs Phase 2 (fine-tuning a pretrained
  model) rather than training from scratch on little data.
- **The samples**: early on they're random letters; trained, they start to look
  like Esan — correct letters (incl. ọ/ẹ), plausible syllables, word-like chunks.
  It won't be *meaningful* Esan (too little data), but seeing structure emerge
  from raw next-character prediction is the whole point.

## Experiment (great for learning)
Edit the hyperparameters at the top of `tinygpt.py` and re-run:
- `N_LAYER`, `N_HEAD`, `N_EMBD` bigger → more capacity (memorises faster).
- `DROPOUT` higher → less memorising.
- `MAX_ITERS` more → lower loss (until it overfits).

## Then → Phase 2
Once you've felt how a from-scratch model behaves on tiny data, Phase 2 builds
the *useful* tutor: fine-tune a small pretrained multilingual model (LoRA) on
`vocab.csv` + `pairs.jsonl`, or a retrieval tutor over the curated data.
