# generate_torch.py
import torch
import pickle
from tokenizer import BPETokenizer

CHECKPOINT = 'checkpoint_torch.pkl'
VOCAB_SIZE  = 5000

# Load checkpoint
print("Loading model...")
with open(CHECKPOINT, 'rb') as f:
    saved = pickle.load(f)

model     = saved['model']
tokenizer = saved['tokenizer']
model.eval()

def generate(prompt, max_new=80, temperature=0.8, top_p=0.9):
    print(f"\nPrompt: '{prompt}'")
    print("-" * 40)
    ids = torch.tensor(
        [min(i, VOCAB_SIZE - 1) for i in tokenizer.encode(prompt)],
        dtype=torch.long
    ).unsqueeze(0)

    with torch.no_grad():
        output = model.generate(ids, max_new=max_new, 
                                temperature=temperature, top_p=top_p)

    print(tokenizer.decode(output[0].tolist()))
    print("-" * 40)

# Try different prompts and temperatures
generate("To be or not to be")
generate("The king shall", temperature=0.6)   # more focused
generate("What is love", temperature=1.2)     # more creative
generate("Thou art")