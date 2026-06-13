from flask import Flask, request, jsonify, Response, send_from_directory
import json
import os
import sys

# Import our engine from previous stages
from minigpt import MiniGPT, Tokenizer
from generate import top_p_sample, softmax

app = Flask(__name__, static_folder='static')

# ── Load and train the model at startup ──────────────────────────────────

print("Building vocabulary and training model...")

CORPUS = [
    "the cat sat on the mat and purred loudly",
    "the dog ran in the park all day long",
    "the cat and the dog played together outside",
    "the cat chased the dog around the garden",
    "a dog ran past the cat on the mat",
    "the cat slept on the warm mat by the fire",
    "the dog barked at the cat on the mat",
    "the cat and dog ran together in the park",
]

tokenizer = Tokenizer()
tokenizer.fit(CORPUS)

model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=64,
    num_heads=4,
    num_layers=3,
)

# Train for a few hundred steps
all_ids = tokenizer.encode(" ".join(CORPUS))
lr, epsilon = 0.02, 1e-3
for epoch in range(10):
    for i in range(min(16, len(model.out_proj))):
        for j in range(min(16, len(model.out_proj[0]))):
            orig = model.out_proj[i][j]
            model.out_proj[i][j] = orig + epsilon
            lp = model.loss(all_ids)
            model.out_proj[i][j] = orig - epsilon
            lm = model.loss(all_ids)
            model.out_proj[i][j] = orig - lr * (lp - lm) / (2 * epsilon)
    if (epoch + 1) % 25 == 0:
        print(f"  epoch {epoch+1}/100 | loss: {model.loss(all_ids):.3f}")

print(f"Model ready. Vocab size: {tokenizer.vocab_size}\n")


# ── Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/vocab', methods=['GET'])
def vocab():
    """Return model info for the UI to display."""
    return jsonify({
        'vocab_size': tokenizer.vocab_size,
        'embed_dim': 64,
        'num_layers': 3,
        'corpus_size': len(CORPUS),
    })


@app.route('/generate', methods=['POST'])
def generate():
    """
    Stream tokens back one at a time using Server-Sent Events.
    The browser receives each token as it's generated.
    """
    data = request.get_json()
    prompt      = data.get('prompt', 'the cat')
    max_tokens  = min(int(data.get('max_tokens', 30)), 100)
    temperature = float(data.get('temperature', 0.8))
    top_p       = float(data.get('top_p', 0.9))

    def stream():
        token_ids = tokenizer.encode(prompt)
        if not token_ids:
            token_ids = [1]  # fallback to <UNK>

        # Stream the prompt tokens first so the UI can echo them
        yield f"data: {json.dumps({'type': 'prompt', 'text': prompt})}\n\n"

        for _ in range(max_tokens):
            try:
                all_logits = model.forward(token_ids)
                next_logits = all_logits[-1]
                next_id = top_p_sample(next_logits, p=top_p, temperature=temperature)

                token_ids.append(next_id)
                word = tokenizer.i2t.get(next_id, '<UNK>')

                # Send each token as a Server-Sent Event
                yield f"data: {json.dumps({'type': 'token', 'text': word, 'id': next_id})}\n\n"

                if next_id <= 1:  # <PAD> or <UNK> — stop
                    break

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
                break

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(stream(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    #app.run(debug=False, port=5000, threaded=False)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=False)