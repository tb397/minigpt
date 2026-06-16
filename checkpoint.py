# checkpoint.py
import pickle
import os

CHECKPOINT_FILE = 'checkpoint.pkl'

def save(model, tokenizer):
    """Save model and tokenizer to disk."""
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump({
            'model': model,
            'tokenizer': tokenizer,
        }, f)
    size_mb = os.path.getsize(CHECKPOINT_FILE) / 1024 / 1024
    print(f"Checkpoint saved ({size_mb:.1f} MB)")

def load():
    """Load model and tokenizer from disk. Returns (model, tokenizer) or None."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None, None
    print("Loading checkpoint...")
    with open(CHECKPOINT_FILE, 'rb') as f:
        data = pickle.load(f)
    print("Checkpoint loaded.")
    return data['model'], data['tokenizer']

def exists():
    return os.path.exists(CHECKPOINT_FILE)