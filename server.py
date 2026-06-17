from flask import Flask, request, jsonify, Response, send_from_directory
import json
import os
import torch
import sys
import urllib.request
import pickle
import checkpoint
from tokenizer import BPETokenizer
from minigpt_torch import MiniGPT

# Import our engine from previous stages
# from minigpt import MiniGPT, Tokenizer
from generate import top_p_sample, softmax

app = Flask(__name__, static_folder='static')

VOCAB_SIZE = 50257

# ── Load PyTorch model ────────────────────────────────────────────────────
print("Loading PyTorch model...")
with open('checkpoint_compressed.pkl', 'rb') as f:
    saved = pickle.load(f)

model     = saved['model']
tokenizer = saved['tokenizer']
model.eval()
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ── Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/vocab', methods=['GET'])
def vocab():
    """Return model info for the UI to display."""
    return jsonify({
        'vocab_size': tokenizer.vocab_size,
        'embed_dim': 192,
        'num_layers': 4,
        'corpus_size': 'Pride and Prejudice',
    })


@app.route('/generate', methods=['POST'])
def generate():
    """
    Stream tokens back one at a time using Server-Sent Events.
    The browser receives each token as it's generated.
    """
    data = request.get_json()
    prompt      = data.get('prompt', 'it is a truth')
    max_tokens  = min(int(data.get('max_tokens', 80)), 200)
    temperature = float(data.get('temperature', 0.8))
    top_p       = float(data.get('top_p', 0.9))
    mode        = data.get('mode', 'completion')

    def stream():
        if mode == 'instruction':
            # Format as instruction prompt
            full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        else:
            full_prompt = prompt
            
        # Encode prompt
        ids = tokenizer.encode(full_prompt)
        token_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

        yield f"data: {json.dumps({'type': 'prompt', 'text': prompt})}\n\n"

        # Generate one token at a time and stream each back
        with torch.no_grad():
            for _ in range(max_tokens):
                try:
                    context = token_tensor[:, -256:]
                    logits  = model.forward(context)
                    logits  = model.forward(context)[:, -1, :] / temperature

                    # Top-p sampling
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs      = torch.nn.functional.softmax(sorted_logits, dim=-1)
                    cumulative = torch.cumsum(probs, dim=-1)
                    sorted_logits[cumulative - probs > top_p] = float('-inf')
                    filtered_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)

                    next_pos  = torch.multinomial(filtered_probs, num_samples=1)
                    next_id   = sorted_idx.gather(-1, next_pos)
                    token_tensor = torch.cat([token_tensor, next_id], dim=1)

                    # Decode just the new token
                    word = tokenizer.token_to_str(next_id.item())
                    yield f"data: {json.dumps({'type': 'token', 'text': word})}\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
                    break

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(stream(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=False)