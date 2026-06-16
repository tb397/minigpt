# tokenizer.py
import tiktoken

class BPETokenizer:
    """
    Wraps OpenAI's tiktoken BPE tokenizer.
    Same interface as our original SimpleTokenizer
    so the rest of the code doesn't need to change.
    """
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        # Special tokens
        self.pad_id = 0
        self.unk_id = 1

    def fit(self, texts):
        # BPE vocab is already fixed at 50,257 tokens
        # No fitting needed — it already knows all tokens
        print(f"BPE tokenizer ready. Vocab size: {self.vocab_size}")

    def encode(self, text):
        return self.enc.encode(text.lower())

    def decode(self, ids):
        # Filter out special token ids before decoding
        clean = [i for i in ids if i < self.vocab_size]
        return self.enc.decode(clean)

    @property
    def vocab_size(self):
        return self.enc.n_vocab  # 50,257

    # Keep i2t interface compatible with server.py streaming
    def token_to_str(self, token_id):
        try:
            return self.enc.decode([token_id])
        except Exception:
            return '<UNK>'