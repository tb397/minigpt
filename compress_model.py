# compress_model.py
import torch
import pickle

print("Loading 162M model...")
with open('checkpoint_117m_best.pkl', 'rb') as f:
    saved = pickle.load(f)

model     = saved['model']
tokenizer = saved['tokenizer']

# Quantize to 8-bit — cuts memory from 650MB to ~160MB
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # quantize all linear layers
    dtype=torch.qint8
)

# Check new size
param_size = sum(
    p.nelement() * p.element_size()
    for p in model_quantized.parameters()
)
buffer_size = sum(
    b.nelement() * b.element_size()
    for b in model_quantized.buffers()
)
total_mb = (param_size + buffer_size) / 1e6
print(f"Compressed model size: ~{total_mb:.0f} MB")

# Save compressed version
with open('checkpoint_compressed.pkl', 'wb') as f:
    pickle.dump({
        'model':     model_quantized,
        'tokenizer': tokenizer,
    }, f)

print("Saved checkpoint_compressed.pkl")