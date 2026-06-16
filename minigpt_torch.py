# minigpt_torch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4, 
                 max_seq_len=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim

        # Stage 2: Embeddings
        self.token_embedding    = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout            = nn.Dropout(dropout)

        # Stage 3+4: Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)

        # Output projection → vocab
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        B, T = token_ids.shape  # batch size, sequence length

        # Token + position embeddings
        positions = torch.arange(T, device=token_ids.device)
        x = self.dropout(
            self.token_embedding(token_ids) +
            self.position_embedding(positions)
        )

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.layer_norm(x)

        # Project to vocab size → logits
        return self.output(x)

    def loss(self, token_ids):
        """Cross entropy loss — predict next token at each position."""
        logits = self.forward(token_ids[:, :-1])   # all tokens except last
        targets = token_ids[:, 1:]                  # all tokens except first
        B, T, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T)
        )

    @torch.no_grad()
    def generate(self, token_ids, max_new=50, temperature=0.8, top_p=0.9):
        """Generate tokens one at a time."""
        self.eval()
        ids = token_ids.clone()

        for _ in range(max_new):
            # Crop to last 256 tokens if sequence gets long
            context = ids[:, -256:]
            logits  = self.forward(context)
            logits  = logits[:, -1, :] / temperature  # last position only

            # Top-p sampling
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs      = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)

            # Remove tokens once cumulative probability exceeds p
            sorted_logits[cumulative - probs > top_p] = float('-inf')
            filtered_probs = F.softmax(sorted_logits, dim=-1)

            # Sample and map back to original vocab indices
            next_pos  = torch.multinomial(filtered_probs, num_samples=1)
            next_id   = sorted_idx.gather(-1, next_pos)
            ids       = torch.cat([ids, next_id], dim=1)

        return ids


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # Stage 3: Attention
        self.attention  = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=dropout,
            batch_first=True
        )
        # Stage 4: FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # Causal mask — no peeking at future tokens
        mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()

        # Attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual connection
        x = self.norm2(x + self.ffn(x))
        return x