import math
import random

# ── Helpers ──────────────────────────────────────────────────────────────

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]

def relu(x):
    return max(0.0, x)

def rand_vec(n, scale=0.1):
    return [random.gauss(0, scale) for _ in range(n)]

def rand_mat(rows, cols, scale=0.1):
    return [rand_vec(cols, scale) for _ in range(rows)]

def mat_vec(M, v):
    return [dot(row, v) for row in M]

def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]

def layer_norm(v):
    """Normalize a vector to mean=0, std=1."""
    mean = sum(v) / len(v)
    var  = sum((x - mean) ** 2 for x in v) / len(v)
    std  = math.sqrt(var + 1e-8)
    return [(x - mean) / std for x in v]


# ── Stage 1: Tokenizer ────────────────────────────────────────────────────

class Tokenizer:
    def __init__(self):
        self.t2i, self.i2t = {}, {}
        self.next = 0
        self._add("<PAD>"); self._add("<UNK>")

    def _add(self, tok):
        if tok not in self.t2i:
            self.t2i[tok] = self.next
            self.i2t[self.next] = tok
            self.next += 1

    def fit(self, texts):
        for t in texts:
            for w in t.lower().split(): self._add(w)

    def encode(self, text):
        return [self.t2i.get(w, 1) for w in text.lower().split()]

    def decode(self, ids):
        return " ".join(self.i2t.get(i, "<UNK>") for i in ids)

    @property
    def vocab_size(self): return self.next


# ── Stage 2: Embeddings + positional encoding ────────────────────────────

class Embeddings:
    def __init__(self, vocab_size, embed_dim):
        self.table = rand_mat(vocab_size, embed_dim, scale=0.02)
        self.embed_dim = embed_dim

    def positional(self, pos):
        """Simple learned-style sinusoidal position signal."""
        pe = []
        for i in range(self.embed_dim):
            angle = pos / (10000 ** (2 * (i // 2) / self.embed_dim))
            pe.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
        return pe

    def forward(self, token_ids):
        return [vec_add(self.table[tid], self.positional(pos))
                for pos, tid in enumerate(token_ids)]


# ── Stage 3: Attention ───────────────────────────────────────────────────

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.Wq = [rand_mat(self.head_dim, embed_dim) for _ in range(num_heads)]
        self.Wk = [rand_mat(self.head_dim, embed_dim) for _ in range(num_heads)]
        self.Wv = [rand_mat(self.head_dim, embed_dim) for _ in range(num_heads)]
        self.Wo = rand_mat(embed_dim, embed_dim)

    def forward(self, embeddings):
        seq = len(embeddings)
        all_head_outputs = []

        for h in range(self.num_heads):
            Q = [mat_vec(self.Wq[h], e) for e in embeddings]
            K = [mat_vec(self.Wk[h], e) for e in embeddings]
            V = [mat_vec(self.Wv[h], e) for e in embeddings]

            head_out = []
            for i in range(seq):
                # Causal mask: only attend to previous tokens (no peeking ahead)
                scores = [dot(Q[i], K[j]) / self.scale if j <= i else -1e9
                          for j in range(seq)]
                weights = softmax(scores)
                out = [sum(weights[j] * V[j][d] for j in range(seq))
                       for d in range(self.head_dim)]
                head_out.append(out)
            all_head_outputs.append(head_out)

        # Concatenate heads and project back to embed_dim
        concat = [sum(([all_head_outputs[h][i] for h in range(self.num_heads)]), [])
                  for i in range(seq)]
        return [layer_norm(vec_add(embeddings[i], mat_vec(self.Wo, concat[i])))
                for i in range(seq)]


# ── Feed-Forward Network ─────────────────────────────────────────────────

class FFN:
    def __init__(self, embed_dim):
        hidden = embed_dim * 4
        self.W1 = rand_mat(hidden, embed_dim)
        self.b1 = [0.0] * hidden
        self.W2 = rand_mat(embed_dim, hidden)
        self.b2 = [0.0] * embed_dim

    def forward(self, x):
        h = [relu(v + b) for v, b in zip(mat_vec(self.W1, x), self.b1)]
        out = [v + b for v, b in zip(mat_vec(self.W2, h), self.b2)]
        return layer_norm(vec_add(x, out))  # residual + norm


# ── Transformer Block ─────────────────────────────────────────────────────

class TransformerBlock:
    def __init__(self, embed_dim, num_heads):
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn  = FFN(embed_dim)

    def forward(self, x):
        x = self.attn.forward(x)
        return [self.ffn.forward(token) for token in x]


# ── Full GPT Model ────────────────────────────────────────────────────────

class MiniGPT:
    def __init__(self, vocab_size, embed_dim=32, num_heads=2, num_layers=2):
        self.embed     = Embeddings(vocab_size, embed_dim)
        self.blocks    = [TransformerBlock(embed_dim, num_heads)
                          for _ in range(num_layers)]
        self.out_proj  = rand_mat(vocab_size, embed_dim)  # final projection
        self.vocab_size = vocab_size

    def forward(self, token_ids):
        """Run a forward pass. Returns logits for each position."""
        x = self.embed.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x)
        # Project each position to vocab-size logits
        logits = [mat_vec(self.out_proj, h) for h in x]
        return logits

    def loss(self, token_ids):
        """
        Cross-entropy loss: at each position, how surprised were we
        by the actual next token?
        """
        logits = self.forward(token_ids)
        total_loss = 0.0
        count = 0
        for i in range(len(token_ids) - 1):
            probs = softmax(logits[i])
            target = token_ids[i + 1]           # next token is the label
            total_loss += -math.log(probs[target] + 1e-9)
            count += 1
        return total_loss / count if count else 0.0

    def generate(self, token_ids, max_new=20, temperature=1.0):
        """Generate new tokens one at a time."""
        ids = list(token_ids)
        for _ in range(max_new):
            logits = self.forward(ids)
            last_logits = [l / temperature for l in logits[-1]]
            probs = softmax(last_logits)
            # Sample from the distribution
            r = random.random()
            cumulative = 0.0
            next_id = 0
            for idx, p in enumerate(probs):
                cumulative += p
                if r < cumulative:
                    next_id = idx
                    break
            ids.append(next_id)
        return ids


# ── Training loop ─────────────────────────────────────────────────────────

def train(model, token_ids, epochs=50, lr=0.01):
    """
    Minimal training loop using finite-difference gradient estimation.
    (A real implementation uses backpropagation — we'll cover that next.)
    """
    print(f"Starting loss: {model.loss(token_ids):.4f}\n")
    epsilon = 1e-3

    for epoch in range(epochs):
        # We'll nudge the output projection weights as a demonstration
        for i in range(min(8, len(model.out_proj))):       # limit for speed
            for j in range(min(8, len(model.out_proj[0]))):
                orig = model.out_proj[i][j]

                model.out_proj[i][j] = orig + epsilon
                loss_plus = model.loss(token_ids)

                model.out_proj[i][j] = orig - epsilon
                loss_minus = model.loss(token_ids)

                grad = (loss_plus - loss_minus) / (2 * epsilon)
                model.out_proj[i][j] = orig - lr * grad

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | loss: {model.loss(token_ids):.4f}")

    print(f"\nFinal loss: {model.loss(token_ids):.4f}")


# ── Run it ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    corpus = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the cat and the dog played",
    ]

    tokenizer = Tokenizer()
    tokenizer.fit(corpus)
    print(f"Vocabulary: {tokenizer.vocab_size} tokens\n")

    model = MiniGPT(
        vocab_size = tokenizer.vocab_size,
        embed_dim  = 32,
        num_heads  = 2,
        num_layers = 2,
    )

    # Train on all sentences
    all_ids = tokenizer.encode(" ".join(corpus))
    train(model, all_ids, epochs=50, lr=0.02)

    # Generate some text
    prompt = tokenizer.encode("the cat")
    output = model.generate(prompt, max_new=8, temperature=0.8)
    print(f"\nPrompt:    'the cat'")
    print(f"Generated: '{tokenizer.decode(output)}'")