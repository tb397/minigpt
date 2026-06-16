# train_torch.py
import torch
import urllib.request
import pickle
import os
from tokenizer import BPETokenizer
from minigpt_torch import MiniGPT

CHECKPOINT = 'checkpoint_torch.pkl'
VOCAB_SIZE  = 50257   # much larger now that training is fast
EMBED_DIM   = 192
NUM_HEADS   = 4
NUM_LAYERS  = 4
SEQ_LEN     = 128
BATCH_SIZE  = 16
EPOCHS      = 4000
LR          = 3e-4

# ── Corpus ────────────────────────────────────────────────────────────────
print("Downloading Pride and Prejudice...")
#url  = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
# Try Pride and Prejudice
url = 'https://www.gutenberg.org/files/1342/1342-0.txt'

# Or Sherlock Holmes
#url = 'https://www.gutenberg.org/files/1661/1661-0.txt'

text = urllib.request.urlopen(url).read().decode('utf-8')

# Strip Gutenberg header and footer
start = text.find("*** START OF")
end   = text.find("*** END OF")
if start != -1:
    text = text[text.find("\n", start) + 1:]
if end != -1:
    text = text[:end]

print(f"Loaded {len(text):,} characters")

# ── Tokenize ──────────────────────────────────────────────────────────────
tokenizer = BPETokenizer()
tokenizer.fit([text])

all_ids = tokenizer.encode(text)
print(f"Encoded to {len(all_ids):,} tokens")

data   = torch.tensor(all_ids, dtype=torch.long)
split  = int(0.9 * len(data))
train  = data[:split]
val    = data[split:]

# ── Batch builder ─────────────────────────────────────────────────────────
def get_batch(data, batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
    """Grab a random batch of sequences."""
    starts = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[s:s + seq_len]     for s in starts])
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x, y

# ── Model ─────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Resume from checkpoint if it exists
if os.path.exists(CHECKPOINT):
    print("Resuming from checkpoint...")
    with open(CHECKPOINT, 'rb') as f:
        saved = pickle.load(f)
    model      = saved['model'].to(device)
    start_epoch = saved['epoch']
    print(f"Resuming from epoch {start_epoch}")
else:
    print("Starting fresh...")
    model       = MiniGPT(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=SEQ_LEN,
    ).to(device)
    start_epoch = 0

# Count parameters
params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {params:,}")

# ── Optimizer ─────────────────────────────────────────────────────────────
# AdamW is the standard optimizer for transformers
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# Learning rate scheduler — warms up then decays
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training loop ─────────────────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs...\n")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    x, y = get_batch(train)
    x, y = x.to(device), y.to(device)

    # Forward pass
    logits = model.forward(x)
    B, T, V = logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(B * T, V),
        y.reshape(B * T)
    )

    # Backward pass — one line replaces all our Value() code!
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    # Logging every 50 epochs
    if (epoch + 1) % 50 == 0:
        # Validation loss
        model.eval()
        with torch.no_grad():
            xv, yv   = get_batch(val)
            xv, yv   = xv.to(device), yv.to(device)
            vlogits  = model.forward(xv)
            B, T, V  = vlogits.shape
            val_loss = torch.nn.functional.cross_entropy(
                vlogits.reshape(B * T, V),
                yv.reshape(B * T)
            ).item()

        print(f"epoch {epoch+1:4d} | train loss: {loss.item():.3f} | val loss: {val_loss:.3f}")

        # Save checkpoint
        with open(CHECKPOINT, 'wb') as f:
            pickle.dump({'model': model.cpu(), 'epoch': epoch + 1, 
                        'tokenizer': tokenizer}, f)
        model.to(device)
        print("checkpoint saved")

        # Quick generation sample
        model.eval()
        with torch.no_grad():
            prompt = torch.tensor(
                [min(i, VOCAB_SIZE-1) for i in tokenizer.encode("It is a truth")],
                dtype=torch.long
            ).unsqueeze(0).to(device)
            output = model.generate(prompt, max_new=20, temperature=0.8)
            text_out = tokenizer.decode(output[0].tolist())
            print(f"Sample: {text_out[:120]}\n")

print("Training complete!")