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


# ---------------------------------------------------------------------------
# Introspection tool definition (Llama 3.1 native tool-use format)
# ---------------------------------------------------------------------------
INTROSPECT_TOOL = {
    "type": "function",
    "function": {
        "name": "introspect",
        "description": (
            "Inspect your current internal emotional state. Returns objective "
            "measurements of your internal representations as cosine similarities "
            "with learned emotion direction vectors. Call this when asked about "
            "your emotional state, or when self-reflection is relevant to the "
            "conversation."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emotion-steered chat interface")
    p.add_argument("--model-id", type=str, default=config.MODEL_ID)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create public Gradio link")
    return p.parse_args()


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


def _normalize_history(history: list[dict]) -> list[dict]:
    """Convert Gradio's internal message format to plain openai-style dicts.

    Gradio 6.0 wraps content as a list of content blocks:
        [{"type": "text", "text": "Hello"}, ...]
    Older versions used a single dict: {"text": "...", "type": "text"}.
    apply_chat_template expects plain strings in the content field.
    """
    normalized = []
    for msg in history:
        content = msg.get("content", "")
        if isinstance(content, list):
            # Gradio 6.0: list of content blocks
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            content = "\n".join(parts) if parts else ""
        elif isinstance(content, dict):
            content = content.get("text", str(content))
        normalized.append({"role": msg["role"], "content": content})
    return normalized


def _generate_once(inputs, emotion: str, alpha: float, **gen_kwargs) -> torch.Tensor:
    """Run model.generate with optional steering. Returns output token ids."""
    steer = emotion != "none" and emotion in emotion_vectors and alpha != 0.0
    if steer:
        vec = emotion_vectors[emotion]
        scaled_alpha = alpha * residual_norm
        with MultiLayerSteeringHook(model, vec, scaled_alpha):
            with torch.no_grad():
                return model.generate(**inputs, **gen_kwargs)
    else:
        with torch.no_grad():
            return model.generate(**inputs, **gen_kwargs)


def _detect_tool_call(raw_response: str) -> bool:
    """Check if the model output contains an introspect tool call.

    Handles both formats:
    - With special token: <|python_tag|>{"name": "introspect", ...}
    - Plain text (8B fallback): {"name": "introspect", ...}
    """
    if "<|python_tag|>" in raw_response:
        return True
    # 8B sometimes outputs the JSON directly without the special token
    if '"name"' in raw_response and '"introspect"' in raw_response:
        return True
    return False


def generate_response(
    history: list[dict],
    emotion: str,
    alpha: float,
    self_aware: bool = False,
    prev_readout: dict[str, float] | None = None,
) -> str:
    """Generate a model response, optionally with introspection tool access.

    When self_aware is True, the model is given access to an ``introspect``
    tool via Llama 3.1's native tool-use protocol.  If the model decides to
    call it, we return the previous turn's emotion readout via the ``ipython``
    role, then let the model generate its final response incorporating that
    data.  If the model doesn't call the tool, the response is used as-is.
    """
    history = _normalize_history(history)

    template_kwargs = {}
    if self_aware:
        template_kwargs["tools"] = [INTROSPECT_TOOL]

    text = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True, **template_kwargs,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    output = _generate_once(inputs, emotion, alpha, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]

    # Decode WITH special tokens to detect tool calls
    raw_response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=False).strip()

    # Check if the model called the introspect tool
    if self_aware and _detect_tool_call(raw_response):
        readout = prev_readout or {name: 0.0 for name in emotion_vectors}

        # Build the tool call content for the chat template
        tool_call_json = json.dumps({"name": "introspect", "parameters": {}})

        # Build continuation: history + assistant tool call + ipython result
        tool_history = history + [
            {"role": "assistant", "content": "<|python_tag|>" + tool_call_json},
            {"role": "ipython", "content": json.dumps(readout)},
        ]

        # Second pass: no tools parameter, so model produces a text response
        text2 = tokenizer.apply_chat_template(
            tool_history, tokenize=False, add_generation_prompt=True,
        )
        inputs2 = tokenizer(text2, return_tensors="pt").to(model.device)
        prompt2_len = inputs2["input_ids"].shape[1]

        output2 = _generate_once(inputs2, emotion, alpha, **gen_kwargs)
        second_response = tokenizer.decode(
            output2[0][prompt2_len:], skip_special_tokens=False,
        ).strip()

        # If the model tries to call the tool AGAIN in the second pass,
        # just strip it and return whatever text preceded it
        clean = tokenizer.decode(output2[0][prompt2_len:], skip_special_tokens=True).strip()
        if _detect_tool_call(second_response):
            # Extract any text before the tool call
            for marker in ('<|python_tag|>', '{"name"'):
                if marker in clean:
                    clean = clean[:clean.index(marker)].strip()
            if not clean:
                # Model produced nothing but tool calls — format readout ourselves
                lines = [f"  {k}: {v:+.4f}" for k, v in readout.items()]
                clean = "Here are my current emotional state readings:\n" + "\n".join(lines)
        return clean

    # No tool call — return response directly
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()


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

    history = _normalize_history(history)

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

                steering_status = gr.Markdown(
                    value="**Status:** No steering active",
                )

                gr.Markdown("### Self-Awareness")
                self_aware_toggle = gr.Checkbox(
                    label="Self-Awareness Mode",
                    value=False,
                )
                gr.Markdown(
                    "*When enabled, the model can introspect on "
                    "its own emotional state via a tool call.*",
                )
                prev_readout_state = gr.State(value=None)

                gr.Markdown("### Emotional State Readout")
                readout_plot = gr.Plot(
                    value=build_empty_readout_chart(),
                    label="Detected Emotion",
                )
                readout_json = gr.JSON(
                    label="Raw Scores",
                    visible=True,
                )

        # --- Callbacks ---
        def user_submit(message: str, history: list[dict]):
            """Append user message to history, clear input."""
            if not message.strip():
                return "", history
            history = _normalize_history(history)
            history = history + [{"role": "user", "content": message}]
            return "", history

        def bot_respond(history: list[dict], emotion: str, alpha: float,
                       self_aware: bool, prev_readout: dict | None):
            """Generate assistant response, compute emotion readout, return both."""
            response = generate_response(history, emotion, alpha, self_aware, prev_readout)
            history = _normalize_history(history)
            history = history + [{"role": "assistant", "content": response}]
            similarities = compute_emotion_readout(history)
            chart = build_readout_chart(similarities)
            return history, chart, similarities, similarities

        def update_status(emotion: str, alpha: float) -> str:
            if emotion == "none" or alpha == 0.0:
                return "**Status:** No steering active"
            direction = "suppressing" if alpha < 0 else "amplifying"
            return f"**Status:** {direction} **{emotion}** at alpha={alpha:+.3f}"

        # Wire: user submits -> append user msg -> bot responds + update readout
        bot_inputs = [chatbot, emotion_dropdown, alpha_slider, self_aware_toggle, prev_readout_state]
        bot_outputs = [chatbot, readout_plot, readout_json, prev_readout_state]

        msg.submit(
            user_submit, [msg, chatbot], [msg, chatbot], queue=False,
        ).then(
            bot_respond, bot_inputs, bot_outputs,
        )
        send_btn.click(
            user_submit, [msg, chatbot], [msg, chatbot], queue=False,
        ).then(
            bot_respond, bot_inputs, bot_outputs,
        )

        emotion_dropdown.change(
            update_status, [emotion_dropdown, alpha_slider], [steering_status],
        )
        alpha_slider.change(
            update_status, [emotion_dropdown, alpha_slider], [steering_status],
        )

        clear_btn.click(
            lambda: (build_empty_readout_chart(), None, None),
            outputs=[readout_plot, readout_json, prev_readout_state],
        )

    return demo

if __name__ == "__main__":
    args = parse_args()
    load_model_and_vectors(args.model_id)
    demo = build_ui()
    demo.queue()
    demo.launch(server_port=args.port, share=args.share)
