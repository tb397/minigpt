# train.py
import urllib.request
import checkpoint
from tokenizer import BPETokenizer
from minigpt import MiniGPT

VOCAB_SIZE = 1000

# ── Load corpus ───────────────────────────────────────────────────────────
print("Downloading corpus...")
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
text = urllib.request.urlopen(url).read().decode('utf-8')[:50000]
sentences = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]
print(f"Loaded {len(sentences)} lines")

# ── Tokenize ──────────────────────────────────────────────────────────────
tokenizer = BPETokenizer()
tokenizer.fit(sentences)

full_text = ' '.join(sentences[:200])
all_ids = [min(i, VOCAB_SIZE - 1) for i in tokenizer.encode(full_text)[:512]]
print(f"Training on {len(all_ids)} tokens, vocab capped at {VOCAB_SIZE}")

# ── Resume or start fresh ─────────────────────────────────────────────────
model, saved_tokenizer = checkpoint.load()

if model is None:
    print("No checkpoint found — starting fresh...")
    model = MiniGPT(
        vocab_size=VOCAB_SIZE,
        embed_dim=64,
        num_heads=4,
        num_layers=3,
    )
else:
    print("Resuming from checkpoint!")

# ── Training loop ─────────────────────────────────────────────────────────
lr, epsilon = 0.02, 1e-3

print("Training started...")
for epoch in range(20):
    for i in range(min(16, len(model.out_proj))):
        for j in range(min(16, len(model.out_proj[0]))):
            orig = model.out_proj[i][j]
            model.out_proj[i][j] = orig + epsilon
            lp = model.loss(all_ids)
            model.out_proj[i][j] = orig - epsilon
            lm = model.loss(all_ids)
            model.out_proj[i][j] = orig - lr * (lp - lm) / (2 * epsilon)

    if (epoch + 1) % 5 == 0:
        loss = model.loss(all_ids)
        print(f"epoch {epoch+1}/20 | loss: {loss:.3f}")
        checkpoint.save(model, tokenizer)
        print("checkpoint saved")

# ── Final save ────────────────────────────────────────────────────────────
checkpoint.save(model, tokenizer)
print("Done! checkpoint.pkl is ready to push to GitHub.")