import math
import random

def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

def sample(probs):
    """Draw one token index from a probability distribution."""
    r, cumulative = random.random(), 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            return i
    return len(probs) - 1


# ── The four strategies ───────────────────────────────────────────────────

def greedy(logits):
    """Always pick the most likely token. Deterministic."""
    return logits.index(max(logits))


def temperature_sample(logits, temperature=1.0):
    """
    Divide logits by temperature before softmax.
    temperature < 1.0  →  sharper distribution (more confident)
    temperature > 1.0  →  flatter distribution (more random)
    temperature → 0    →  approaches greedy
    temperature → ∞    →  approaches uniform random
    """
    scaled = [l / temperature for l in logits]
    return sample(softmax(scaled))


def top_k_sample(logits, k=40, temperature=1.0):
    """
    Only consider the k highest-probability tokens.
    Zeros out everything else, then samples.
    Prevents sampling from the long tail of unlikely tokens.
    """
    if k <= 0:
        return temperature_sample(logits, temperature)

    # Find the k-th largest value as a cutoff threshold
    sorted_logits = sorted(logits, reverse=True)
    cutoff = sorted_logits[min(k - 1, len(sorted_logits) - 1)]

    # Mask everything below the cutoff to -infinity
    filtered = [l if l >= cutoff else float('-inf') for l in logits]
    scaled = [l / temperature for l in filtered]
    return sample(softmax(scaled))


def top_p_sample(logits, p=0.9, temperature=1.0):
    """
    Nucleus sampling: find the smallest set of tokens whose
    cumulative probability ≥ p, then sample only from those.
    Adapts dynamically — uses fewer tokens when the model is
    confident, more when it's uncertain.
    """
    scaled = [l / temperature for l in logits]
    probs = softmax(scaled)

    # Sort tokens by probability (highest first)
    sorted_pairs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)

    # Walk down the list until cumulative probability hits p
    cumulative = 0.0
    allowed = set()
    for idx, prob in sorted_pairs:
        allowed.add(idx)
        cumulative += prob
        if cumulative >= p:
            break

    # Zero out everything outside the nucleus
    filtered = [l if i in allowed else float('-inf') for i, l in enumerate(logits)]
    return sample(softmax(filtered))


# ── Full generation loop ──────────────────────────────────────────────────

def generate(model, tokenizer, prompt, max_new_tokens=30,
             strategy='top_p', temperature=0.8, k=40, p=0.9):
    """
    Generate text token by token using the chosen strategy.

    strategy options: 'greedy', 'temperature', 'top_k', 'top_p'
    """
    token_ids = tokenizer.encode(prompt)
    print(f"\nPrompt: '{prompt}'")
    print(f"Strategy: {strategy}, temperature={temperature}")
    print(f"Generating", end="", flush=True)

    for step in range(max_new_tokens):
        # Forward pass — get logits for every position
        all_logits = model.forward(token_ids)
        next_logits = all_logits[-1]  # we only care about the last position

        # Pick the next token
        if strategy == 'greedy':
            next_id = greedy(next_logits)
        elif strategy == 'temperature':
            next_id = temperature_sample(next_logits, temperature)
        elif strategy == 'top_k':
            next_id = top_k_sample(next_logits, k=k, temperature=temperature)
        else:  # top_p (default)
            next_id = top_p_sample(next_logits, p=p, temperature=temperature)

        token_ids.append(next_id)
        word = tokenizer.i2t.get(next_id, '<UNK>')
        print(f" {word}", end="", flush=True)

        # Stop at unknown or padding token
        if next_id <= 1:
            break

    print()  # newline
    return tokenizer.decode(token_ids)


# ── Demo: compare strategies side by side ────────────────────────────────

if __name__ == "__main__":
    from minigpt import MiniGPT, Tokenizer

    corpus = [
        "the cat sat on the mat and purred",
        "the dog ran in the park all day",
        "the cat and the dog played together",
        "the cat chased the dog around the mat",
        "a dog ran past the cat on the mat",
    ]

    tokenizer = Tokenizer()
    tokenizer.fit(corpus)

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
    )

    # Quick training pass
    print("Training...")
    all_ids = tokenizer.encode(" ".join(corpus))
    lr, epsilon = 0.02, 1e-3
    for epoch in range(60):
        for i in range(min(10, len(model.out_proj))):
            for j in range(min(10, len(model.out_proj[0]))):
                orig = model.out_proj[i][j]
                model.out_proj[i][j] = orig + epsilon
                lp = model.loss(all_ids)
                model.out_proj[i][j] = orig - epsilon
                lm = model.loss(all_ids)
                model.out_proj[i][j] = orig - lr * (lp - lm) / (2 * epsilon)

    prompt = "the cat"

    # Compare all four strategies
    generate(model, tokenizer, prompt, max_new_tokens=8, strategy='greedy')
    generate(model, tokenizer, prompt, max_new_tokens=8, strategy='temperature', temperature=1.4)
    generate(model, tokenizer, prompt, max_new_tokens=8, strategy='top_k', k=3)
    generate(model, tokenizer, prompt, max_new_tokens=8, strategy='top_p', p=0.9)