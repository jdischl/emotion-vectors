# Emotion-Steered Chat Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Gradio chat interface that lets you converse with Llama 3.1 8B while applying real-time emotion steering and monitoring the model's emotional state after each response.

**Architecture:** A `gr.Blocks` layout with a central chat area and a right sidebar containing steering controls (emotion dropdown + alpha slider) and a live emotion-state readout (Plotly bar chart showing cosine similarity of response activations with each emotion vector). On each assistant turn: generate with `MultiLayerSteeringHook` if steering is active, then run a separate forward pass to capture activations at layer 16, project onto all 5 emotion vectors, and update the readout chart.

**Tech Stack:** Gradio (Blocks API), Plotly (emotion readout chart), PyTorch + transformers (model), existing `config.py` + `utils/hooks.py`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `06_chat_interface.py` | Main Gradio app: layout, model loading, chat generation, emotion readout |

Everything else is reused from existing code: `config.py`, `utils/hooks.py` (`MultiLayerSteeringHook`, `ActivationCapture`), `emotions.json`.

---

### Task 1: Scaffold the Gradio layout

**Files:**
- Create: `06_chat_interface.py`

This task creates the complete UI skeleton with placeholder callbacks. The layout is:
- Left (3/4 width): `gr.Chatbot` + `gr.Textbox` for input
- Right (1/4 width): steering controls + emotion readout plot

- [ ] **Step 1: Create the Gradio layout skeleton**

```python
#!/usr/bin/env python3
"""
Step 6 -- Gradio chat interface with emotion steering and live state readout.

Chat with Llama 3.1 8B while steering its emotional tone in real time.
A sidebar shows the model's detected emotional state after each response,
computed by projecting response activations onto the learned emotion vectors.

Usage:
    python 06_chat_interface.py
    python 06_chat_interface.py --port 7860 --share
"""

import argparse
import json

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from utils.hooks import ActivationCapture, MultiLayerSteeringHook


# ---------------------------------------------------------------------------
# Emotion colors (consistent across UI)
# ---------------------------------------------------------------------------
EMOTION_COLORS = {
    "frustrated": "#e74c3c",
    "anxious": "#e67e22",
    "happy": "#2ecc71",
    "angry": "#c0392b",
    "excited": "#3498db",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emotion-steered chat interface")
    p.add_argument("--model-id", type=str, default=config.MODEL_ID)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create public Gradio link")
    return p.parse_args()


def build_empty_readout_chart() -> go.Figure:
    """Build an empty emotion readout bar chart (placeholder before first message)."""
    emotions = list(EMOTION_COLORS.keys())
    fig = go.Figure(go.Bar(
        x=[0.0] * len(emotions),
        y=emotions,
        orientation="h",
        marker_color=[EMOTION_COLORS[e] for e in emotions],
    ))
    fig.update_layout(
        title="Emotional State",
        xaxis_title="Cosine Similarity",
        xaxis_range=[-1, 1],
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
    )
    return fig


def build_ui():
    """Build and return the Gradio Blocks app."""
    with gr.Blocks(title="Emotion Vectors Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Emotion Vectors Chat\nChat with Llama 3.1 8B with real-time emotion steering and state monitoring.")

        with gr.Row():
            # --- Left: Chat area ---
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=600,
                    layout="bubble",
                    label="Chat",
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type a message...",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.ClearButton([msg, chatbot], value="Clear Chat")

            # --- Right: Controls + Readout ---
            with gr.Column(scale=1):
                gr.Markdown("### Steering Controls")
                emotion_dropdown = gr.Dropdown(
                    choices=["none"] + list(EMOTION_COLORS.keys()),
                    value="none",
                    label="Steer Emotion",
                )
                alpha_slider = gr.Slider(
                    minimum=-0.05,
                    maximum=0.05,
                    step=0.005,
                    value=0.01,
                    label="Steering Strength (alpha)",
                )
                gr.Markdown(
                    "*Surgical range: 0.005-0.02. "
                    "Negative = suppress. "
                    "Collapse at >= 0.05.*",
                )

                gr.Markdown("### Emotional State Readout")
                readout_plot = gr.Plot(
                    value=build_empty_readout_chart(),
                    label="Detected Emotion",
                )
                readout_json = gr.JSON(
                    label="Raw Scores",
                    visible=True,
                )

    return demo, chatbot, msg, send_btn, clear_btn, emotion_dropdown, alpha_slider, readout_plot, readout_json


if __name__ == "__main__":
    args = parse_args()
    demo, *_ = build_ui()
    demo.launch(server_port=args.port, share=args.share)
```

- [ ] **Step 2: Verify the layout renders**

Run: `uv run python 06_chat_interface.py`

Expected: Gradio app launches, shows chat area on left, controls + empty chart on right. Nothing is functional yet (no model loaded, no callbacks wired).

Press Ctrl-C to stop.

- [ ] **Step 3: Commit**

```bash
git add 06_chat_interface.py
git commit -m "feat(06): scaffold Gradio chat layout with steering controls and readout panel"
```

---

### Task 2: Model and vector loading at startup

**Files:**
- Modify: `06_chat_interface.py`

Load the model, tokenizer, emotion vectors, and residual-stream norm once at startup. Store in module-level variables so callbacks can access them.

- [ ] **Step 1: Add model/vector loading above `build_ui()`**

Add this block between `parse_args()` and `build_empty_readout_chart()`:

```python
# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------
model = None
tokenizer = None
emotion_vectors: dict[str, torch.Tensor] = {}  # {name: (d_model,)}
residual_norm: float = 1.0
best_layer: int = 16


def load_model_and_vectors(model_id: str):
    """Load model, tokenizer, emotion vectors, and residual-stream norm."""
    global model, tokenizer, emotion_vectors, residual_norm, best_layer

    # Best layer from probe results
    probe_path = config.OUTPUTS_DIR / "probe_results.json"
    if probe_path.exists():
        with open(probe_path) as f:
            best_layer = json.load(f)["best_layer"]
    print(f"Using layer {best_layer} for vectors and readout")

    # Load emotion vectors
    vec_dir = config.VECTORS_DIR / str(best_layer)
    emotions = config.load_emotions()
    for emo in emotions:
        name = emo["name"]
        vec_path = vec_dir / f"{name}.pt"
        if vec_path.exists():
            emotion_vectors[name] = torch.load(vec_path, weights_only=True)
            print(f"  Loaded vector: {name}")
        else:
            print(f"  WARNING: missing vector for {name}")

    # Load mean residual stream norm for alpha scaling
    norm_path = config.ACTIVATIONS_DIR / str(best_layer) / "mean_residual_norm.pt"
    if norm_path.exists():
        residual_norm = torch.load(norm_path, weights_only=True).item()
        print(f"Mean residual norm at layer {best_layer}: {residual_norm:.2f}")
    else:
        print("WARNING: mean_residual_norm.pt not found, using raw alpha values")

    # Load model
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=config.DTYPE,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")
```

- [ ] **Step 2: Call the loader in `__main__`**

Replace the `if __name__` block:

```python
if __name__ == "__main__":
    args = parse_args()
    load_model_and_vectors(args.model_id)
    demo, *_ = build_ui()
    demo.launch(server_port=args.port, share=args.share)
```

- [ ] **Step 3: Commit**

```bash
git add 06_chat_interface.py
git commit -m "feat(06): add model and emotion vector loading at startup"
```

---

### Task 3: Chat generation with emotion steering

**Files:**
- Modify: `06_chat_interface.py`

Wire up the chat: user submits a message, the model generates a response (with optional steering), the chatbot updates. Streaming token-by-token.

- [ ] **Step 1: Add the generation function**

Add after `load_model_and_vectors()`:

```python
def generate_response(
    history: list[dict],
    emotion: str,
    alpha: float,
) -> str:
    """Generate a model response, optionally steered toward an emotion.

    Parameters
    ----------
    history : Chat history in openai format (list of {role, content} dicts).
              Must already include the latest user message.
    emotion : Emotion name to steer toward, or "none" for unsteered.
    alpha : Steering strength (scaled by residual_norm internally).
    """
    text = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    steer = emotion != "none" and emotion in emotion_vectors and alpha != 0.0
    if steer:
        vec = emotion_vectors[emotion]
        scaled_alpha = alpha * residual_norm
        with MultiLayerSteeringHook(model, vec, scaled_alpha):
            with torch.no_grad():
                output = model.generate(**inputs, **gen_kwargs)
    else:
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
```

- [ ] **Step 2: Add the chat callback and wire events in `build_ui()`**

Add the callback inside `build_ui()`, after the layout definition but before `return`:

```python
        # --- Callbacks ---
        def user_submit(message: str, history: list[dict]):
            """Append user message to history, clear input."""
            if not message.strip():
                return "", history
            history = history + [{"role": "user", "content": message}]
            return "", history

        def bot_respond(history: list[dict], emotion: str, alpha: float):
            """Generate assistant response and append to history."""
            response = generate_response(history, emotion, alpha)
            history = history + [{"role": "assistant", "content": response}]
            return history

        # Wire: user submits -> append user msg -> bot responds
        msg.submit(
            user_submit, [msg, chatbot], [msg, chatbot], queue=False,
        ).then(
            bot_respond, [chatbot, emotion_dropdown, alpha_slider], [chatbot],
        )
        send_btn.click(
            user_submit, [msg, chatbot], [msg, chatbot], queue=False,
        ).then(
            bot_respond, [chatbot, emotion_dropdown, alpha_slider], [chatbot],
        )
```

- [ ] **Step 3: Add `demo.queue()` before `demo.launch()` in `__main__`**

```python
if __name__ == "__main__":
    args = parse_args()
    load_model_and_vectors(args.model_id)
    demo, chatbot, msg, send_btn, clear_btn, emotion_dropdown, alpha_slider, readout_plot, readout_json = build_ui()
    demo.queue()
    demo.launch(server_port=args.port, share=args.share)
```

- [ ] **Step 4: Test on GPU pod**

Run: `python 06_chat_interface.py --share`

Expected: Chat works. Type a message, get a response. Try steering with "happy" at alpha=0.01 — response should have a noticeably warmer tone. Readout panel still shows empty chart (wired in next task).

- [ ] **Step 5: Commit**

```bash
git add 06_chat_interface.py
git commit -m "feat(06): wire chat generation with emotion steering via MultiLayerSteeringHook"
```

---

### Task 4: Emotion state readout (activation projection)

**Files:**
- Modify: `06_chat_interface.py`

After each response, run a forward pass on the full conversation, capture activations at the best layer, compute mean activation over the assistant's response tokens (position 50+ of response), then compute cosine similarity with each emotion vector. Display as a Plotly horizontal bar chart.

- [ ] **Step 1: Add the emotion readout function**

Add after `generate_response()`:

```python
# Token offset for activation averaging (matches step 02 methodology)
_AVERAGING_START_TOKEN = 50


def compute_emotion_readout(history: list[dict]) -> dict[str, float]:
    """Compute emotion similarities by projecting the last response's activations.

    Runs a forward pass on the full conversation, captures activations at
    best_layer, averages over response tokens (position 50+ of the response),
    and computes cosine similarity with each emotion vector.

    Returns {emotion_name: cosine_similarity}.
    """
    if not history or history[-1]["role"] != "assistant":
        return {name: 0.0 for name in emotion_vectors}

    # Tokenize full conversation to find where the assistant response starts
    full_text = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=False,
    )
    # Tokenize without the last assistant message to find the split point
    history_without_last = history[:-1]
    prefix_text = tokenizer.apply_chat_template(
        history_without_last, tokenize=False, add_generation_prompt=True,
    )

    full_ids = tokenizer(full_text, return_tensors="pt").to(model.device)
    prefix_len = len(tokenizer(prefix_text)["input_ids"])

    # Forward pass with activation capture at the best layer
    with ActivationCapture(model, [best_layer]) as cap:
        with torch.no_grad():
            model(**full_ids)
        acts = cap.get()

    # acts[best_layer] shape: (1, seq_len, d_model)
    layer_acts = acts[best_layer][0]  # (seq_len, d_model)

    # Average over response tokens, starting from _AVERAGING_START_TOKEN
    # tokens into the response (or all response tokens if response is short)
    response_start = prefix_len
    averaging_start = response_start + _AVERAGING_START_TOKEN
    if averaging_start >= layer_acts.shape[0]:
        # Response shorter than 50 tokens — use all response tokens
        averaging_start = response_start

    response_acts = layer_acts[averaging_start:]  # (n_tokens, d_model)
    if response_acts.shape[0] == 0:
        return {name: 0.0 for name in emotion_vectors}

    mean_act = response_acts.mean(dim=0)  # (d_model,)

    # Cosine similarity with each emotion vector
    similarities = {}
    for name, vec in emotion_vectors.items():
        vec_device = vec.to(device=mean_act.device, dtype=mean_act.dtype)
        cos_sim = torch.nn.functional.cosine_similarity(
            mean_act.unsqueeze(0), vec_device.unsqueeze(0),
        ).item()
        similarities[name] = round(cos_sim, 4)

    return similarities


def build_readout_chart(similarities: dict[str, float]) -> go.Figure:
    """Build a Plotly horizontal bar chart from emotion similarities."""
    emotions = list(EMOTION_COLORS.keys())
    values = [similarities.get(e, 0.0) for e in emotions]
    colors = [EMOTION_COLORS[e] for e in emotions]

    fig = go.Figure(go.Bar(
        x=values,
        y=emotions,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Emotional State (last response)",
        xaxis_title="Cosine Similarity with Emotion Vector",
        xaxis_range=[-0.5, 0.5],
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
    )
    return fig
```

- [ ] **Step 2: Update `bot_respond` to also compute and return the readout**

Replace the `bot_respond` callback and event wiring inside `build_ui()`:

```python
        def bot_respond(history: list[dict], emotion: str, alpha: float):
            """Generate assistant response, compute emotion readout, return both."""
            response = generate_response(history, emotion, alpha)
            history = history + [{"role": "assistant", "content": response}]
            similarities = compute_emotion_readout(history)
            chart = build_readout_chart(similarities)
            return history, chart, similarities

        # Wire: user submits -> append user msg -> bot responds + update readout
        msg.submit(
            user_submit, [msg, chatbot], [msg, chatbot], queue=False,
        ).then(
            bot_respond,
            [chatbot, emotion_dropdown, alpha_slider],
            [chatbot, readout_plot, readout_json],
        )
        send_btn.click(
            user_submit, [msg, chatbot], [msg, chatbot], queue=False,
        ).then(
            bot_respond,
            [chatbot, emotion_dropdown, alpha_slider],
            [chatbot, readout_plot, readout_json],
        )
```

- [ ] **Step 3: Also reset the readout on chat clear**

Add after the event wiring:

```python
        clear_btn.click(
            lambda: (build_empty_readout_chart(), None),
            outputs=[readout_plot, readout_json],
        )
```

- [ ] **Step 4: Test on GPU pod**

Run: `python 06_chat_interface.py --share`

Expected: After each response, the bar chart updates showing cosine similarities. With "happy" steering at alpha=0.01, the "happy" bar should be noticeably higher than baseline. The raw JSON panel shows exact numbers.

- [ ] **Step 5: Commit**

```bash
git add 06_chat_interface.py
git commit -m "feat(06): add live emotion readout via activation projection onto emotion vectors"
```

---

### Task 5: Add steering status indicator and polish

**Files:**
- Modify: `06_chat_interface.py`

Add a visible status indicator showing current steering config, and display a system-message-style note in the chat when steering changes.

- [ ] **Step 1: Add a Markdown status indicator to the sidebar**

In `build_ui()`, after the alpha slider guidance text and before the readout section, add:

```python
                steering_status = gr.Markdown(
                    value="**Status:** No steering active",
                )
```

Add `steering_status` to the return tuple.

- [ ] **Step 2: Add a callback to update the status when controls change**

Inside `build_ui()`, add:

```python
        def update_status(emotion: str, alpha: float) -> str:
            if emotion == "none" or alpha == 0.0:
                return "**Status:** No steering active"
            direction = "suppressing" if alpha < 0 else "amplifying"
            return f"**Status:** {direction} **{emotion}** at alpha={alpha:+.3f}"

        emotion_dropdown.change(
            update_status, [emotion_dropdown, alpha_slider], [steering_status],
        )
        alpha_slider.change(
            update_status, [emotion_dropdown, alpha_slider], [steering_status],
        )
```

- [ ] **Step 3: Update the return statement to include steering_status**

```python
    return demo, chatbot, msg, send_btn, clear_btn, emotion_dropdown, alpha_slider, readout_plot, readout_json, steering_status
```

And update `__main__`:

```python
    demo, chatbot, msg, send_btn, clear_btn, emotion_dropdown, alpha_slider, readout_plot, readout_json, steering_status = build_ui()
```

- [ ] **Step 4: Test on GPU pod**

Run: `python 06_chat_interface.py --share`

Expected: Status indicator updates live as you change the dropdown/slider. Full flow works: type message -> steered response -> readout chart updates.

- [ ] **Step 5: Commit**

```bash
git add 06_chat_interface.py
git commit -m "feat(06): add live steering status indicator"
```

---

### Task 6: Add `gradio` and `plotly` to project dependencies

**Files:**
- Modify: `pyproject.toml` (if it exists) or document in CLAUDE.md

- [ ] **Step 1: Check for and update dependency file**

If `pyproject.toml` exists, add `gradio` and `plotly` to the dependencies. If not, install directly:

```bash
uv pip install gradio plotly
```

- [ ] **Step 2: Update CLAUDE.md**

Add `06_chat_interface.py` to the Pipeline Scripts section:

```
6. `06_chat_interface.py` — Gradio chat with real-time emotion steering + state readout (needs GPU)
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: add gradio and plotly dependencies, update CLAUDE.md"
```
