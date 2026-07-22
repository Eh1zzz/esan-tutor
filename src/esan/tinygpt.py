#!/usr/bin/env python3
"""
Phase 1 — a tiny GPT, from scratch, trained on the Esan corpus.

This is a *learning* model, not a useful tutor. The corpus is small, so it will
mostly memorise — but building it teaches the real machinery every LLM (GPT,
Claude, Llama) is made of:

  1. Tokenisation      — turn text into integers the model can process.
  2. Embeddings        — give each token (and each position) a learned vector.
  3. Self-attention    — let each position "look at" earlier positions.
  4. A transformer block — attention + a little MLP, stacked N times.
  5. Training loop     — predict the *next* character, measure the error
                         (cross-entropy), nudge the weights (backprop + AdamW).
  6. Generation        — sample one character at a time, feeding output back in.

It's a trimmed version of Andrej Karpathy's nanoGPT. Runs on CPU (slow) or a
Colab GPU (fast). See notebooks/phase1_colab.md for how to run it in Colab.

    python src/esan/tinygpt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Hyperparameters (small — the corpus is tiny) ──────────────────────────────
BLOCK_SIZE = 64      # context length: how many previous chars the model sees
BATCH_SIZE = 32      # sequences trained on in parallel
N_EMBD = 96          # size of each token's vector
N_HEAD = 4           # attention heads (each sees N_EMBD // N_HEAD = 24 dims)
N_LAYER = 3          # number of transformer blocks stacked
DROPOUT = 0.1        # regularisation (helps the tiny model not just memorise)
LR = 3e-4            # learning rate for AdamW
MAX_ITERS = 4000     # training steps
EVAL_EVERY = 500     # how often to print train/val loss
SEED = 1337

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "clean" / "corpus.txt"
CKPT = ROOT / "checkpoints" / "tinygpt.pt"


# ── 1. Data + tokeniser ───────────────────────────────────────────────────────
def load_data():
    text = CORPUS.read_text(encoding="utf-8")
    chars = sorted(set(text))                       # the "alphabet" (our vocab)
    stoi = {c: i for i, c in enumerate(chars)}      # char  -> integer
    itos = {i: c for i, c in enumerate(chars)}      # integer -> char
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))                        # 90% train, 10% val
    return text, chars, encode, decode, data[:n], data[n:]


def get_batch(split_data):
    """Grab BATCH_SIZE random chunks of length BLOCK_SIZE, plus the next-char targets."""
    ix = torch.randint(len(split_data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([split_data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([split_data[i + 1 : i + 1 + BLOCK_SIZE] for i in ix])  # shifted by 1
    return x.to(DEVICE), y.to(DEVICE)


# ── 2-4. The transformer ──────────────────────────────────────────────────────
class Head(nn.Module):
    """One self-attention head: each position mixes info from earlier positions."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        # a lower-triangular mask so a position can't peek at the *future*
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        # attention scores: how much each position attends to each other position
        w = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal mask
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        return w @ self.value(x)  # weighted sum of values


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """A small MLP applied to each position independently (the 'thinking' step)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communicate (attention) then compute (MLP), with residuals."""

    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(N_HEAD, N_EMBD // N_HEAD)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # residual connection around attention
        x = x + self.ff(self.ln2(x))   # residual connection around the MLP
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, N_EMBD)      # what a char means
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)        # where it sits
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size)             # -> next-char scores

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)                             # (B,T,C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(tok + pos)
        logits = self.head(self.ln_f(x))                      # (B,T,vocab)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Sample text one character at a time, feeding the output back in."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -BLOCK_SIZE:])            # crop to context window
            probs = F.softmax(logits[:, -1, :], dim=-1)       # last step's distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ── 5-6. Train, then generate ─────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for name, split in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(50)
        for k in range(50):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    torch.manual_seed(SEED)
    text, chars, encode, decode, train_data, val_data = load_data()
    print(f"device={DEVICE} | corpus={len(text)} chars | vocab={len(chars)} | "
          f"train={len(train_data)} val={len(val_data)}")

    model = TinyGPT(len(chars)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    for it in range(MAX_ITERS + 1):
        if it % EVAL_EVERY == 0:
            l = estimate_loss(model, train_data, val_data)
            print(f"step {it:>4}: train loss {l['train']:.3f} | val loss {l['val']:.3f}")
        xb, yb = get_batch(train_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\n── Sample from the trained model ──")
    start = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(model.generate(start, max_new_tokens=400)[0].tolist()))

    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "chars": chars}, CKPT)
    print(f"\nsaved checkpoint -> {CKPT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
