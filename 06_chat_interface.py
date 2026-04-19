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
) -> tuple[str, dict[str, float] | None]:
    """Generate a model response, optionally with introspection tool access.

    When self_aware is True, the model is given access to an ``introspect``
    tool via Llama 3.1's native tool-use protocol.  If the model decides to
    call it, we return the previous turn's emotion readout via the ``ipython``
    role, then let the model generate its final response incorporating that
    data.  If the model doesn't call the tool, the response is used as-is.

    Returns (response_text, introspect_data_sent_or_None).
    """
    history = _normalize_history(history)

    template_kwargs = {}
    if self_aware:
        template_kwargs["tools"] = [INTROSPECT_TOOL]
        # System prompt: ground the model's behavior around introspection
        system_msg = {
            "role": "system",
            "content": (
                "You have access to an introspect tool that reads your internal "
                "emotional state as measured by activation vectors. Use it ONLY "
                "when the user explicitly asks about your emotional state, "
                "feelings, or mood. For all other messages, respond normally "
                "and do NOT call the tool. When you do use the tool, report the "
                "exact numbers it returns and discuss them naturally as part of "
                "a broader response. Never invent or guess emotional state "
                "numbers -- if you haven't called the tool, say so."
            ),
        }
        history = [system_msg] + history

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
        return clean, readout

    # No tool call — return response directly
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip(), None


# Token offset for activation averaging (matches step 02 methodology)
_AVERAGING_START_TOKEN = 50


def _project_activations(layer_acts: torch.Tensor, start: int, end: int) -> dict[str, float]:
    """Project a slice of activations onto emotion vectors via cosine similarity.

    Parameters
    ----------
    layer_acts : (seq_len, d_model) tensor of activations at best_layer
    start, end : token range to average over
    """
    if start >= end or start >= layer_acts.shape[0]:
        return {name: 0.0 for name in emotion_vectors}

    end = min(end, layer_acts.shape[0])
    mean_act = layer_acts[start:end].mean(dim=0)  # (d_model,)

    similarities = {}
    for name, vec in emotion_vectors.items():
        vec_device = vec.to(device=mean_act.device, dtype=mean_act.dtype)
        cos_sim = torch.nn.functional.cosine_similarity(
            mean_act.unsqueeze(0), vec_device.unsqueeze(0),
        ).item()
        similarities[name] = round(cos_sim, 4)
    return similarities


def compute_pre_generation_readout(history: list[dict]) -> dict[str, float]:
    """Measure emotional state while the model processes the user's message.

    Forward pass on the conversation up to (and including) the last user
    message, with the generation prompt appended.  Activations are averaged
    over the last user message's tokens (from position 50 onward within
    that message, or all tokens if shorter).
    """
    history = _normalize_history(history)
    if not history or history[-1]["role"] != "user":
        return {name: 0.0 for name in emotion_vectors}

    # Full prompt as the model would see it right before generating
    full_text = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True,
    )
    # Prefix without the last user message, to find the split point
    history_without_last = history[:-1]
    if history_without_last:
        prefix_text = tokenizer.apply_chat_template(
            history_without_last, tokenize=False, add_generation_prompt=False,
        )
        prefix_len = len(tokenizer(prefix_text)["input_ids"])
    else:
        prefix_len = 0

    full_ids = tokenizer(full_text, return_tensors="pt").to(model.device)
    seq_len = full_ids["input_ids"].shape[1]

    with ActivationCapture(model, [best_layer]) as cap:
        with torch.no_grad():
            model(**full_ids)
        acts = cap.get()

    layer_acts = acts[best_layer][0]  # (seq_len, d_model)

    # Average over user message tokens, from pos 50 onward in the message
    msg_start = prefix_len
    averaging_start = msg_start + _AVERAGING_START_TOKEN
    if averaging_start >= seq_len:
        averaging_start = msg_start

    return _project_activations(layer_acts, averaging_start, seq_len)


def compute_post_generation_readout(history: list[dict]) -> dict[str, float]:
    """Measure emotional state across the model's response.

    Forward pass on the full conversation including the assistant's response.
    Activations are averaged over the response tokens (from position 50
    onward within the response, or all response tokens if shorter).
    """
    history = _normalize_history(history)
    if not history or history[-1]["role"] != "assistant":
        return {name: 0.0 for name in emotion_vectors}

    full_text = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=False,
    )
    history_without_last = history[:-1]
    prefix_text = tokenizer.apply_chat_template(
        history_without_last, tokenize=False, add_generation_prompt=True,
    )

    full_ids = tokenizer(full_text, return_tensors="pt").to(model.device)
    prefix_len = len(tokenizer(prefix_text)["input_ids"])
    seq_len = full_ids["input_ids"].shape[1]

    with ActivationCapture(model, [best_layer]) as cap:
        with torch.no_grad():
            model(**full_ids)
        acts = cap.get()

    layer_acts = acts[best_layer][0]

    response_start = prefix_len
    averaging_start = response_start + _AVERAGING_START_TOKEN
    if averaging_start >= seq_len:
        averaging_start = response_start

    return _project_activations(layer_acts, averaging_start, seq_len)


def build_readout_chart(
    similarities: dict[str, float],
    title: str = "Emotional State",
) -> go.Figure:
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
        title=title,
        xaxis_title="Cosine Similarity",
        xaxis_range=[-0.5, 0.5],
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
    )
    return fig


def build_empty_chart(title: str = "Emotional State") -> go.Figure:
    """Build an empty emotion readout bar chart."""
    return build_readout_chart({e: 0.0 for e in EMOTION_COLORS}, title=title)


def build_ui():
    """Build and return the Gradio Blocks app.

    Layout is optimized for single-screen screenshots: chat + controls on top,
    before/after readout charts side-by-side below, raw data hidden in an
    accordion.
    """
    with gr.Blocks(title="Emotion Vectors Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Emotion Vectors Chat — Llama 3.1 8B")

        # --- Top row: Chat | Controls ---
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=420,
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

            with gr.Column(scale=2):
                gr.Markdown("### Steering")
                emotion_dropdown = gr.Dropdown(
                    choices=["none"] + list(EMOTION_COLORS.keys()),
                    value="none",
                    label="Emotion",
                )
                alpha_slider = gr.Slider(
                    minimum=-0.05,
                    maximum=0.05,
                    step=0.005,
                    value=0.01,
                    label="Alpha (surgical: 0.005–0.02, collapse ≥ 0.05)",
                )
                steering_status = gr.Markdown(
                    value="**Status:** No steering active",
                )
                self_aware_toggle = gr.Checkbox(
                    label="Self-Awareness Mode (introspect tool)",
                    value=False,
                )
                prev_readout_state = gr.State(value=None)

        # --- Bottom row: Before / After readout charts side-by-side ---
        with gr.Row():
            pre_plot = gr.Plot(
                value=build_empty_chart("Before Response"),
                label="State entering the turn (user message)",
            )
            post_plot = gr.Plot(
                value=build_empty_chart("After Response"),
                label="State during model response",
            )

        # --- Collapsed raw data (kept out of screenshot frame) ---
        with gr.Accordion("Raw data", open=False):
            readout_json = gr.JSON(label="Raw scores (after response)")
            introspect_sent = gr.JSON(label="Last sent to model (introspect tool)")

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
            """Generate response, compute before/after readouts, return all."""
            # Before: measure state while processing user's message
            pre_sims = compute_pre_generation_readout(history)
            pre_chart = build_readout_chart(pre_sims, title="Before Response")

            # Generate (also returns the exact data sent to the model, if any)
            response, introspect_data_sent = generate_response(
                history, emotion, alpha, self_aware, prev_readout,
            )
            history = _normalize_history(history)
            history = history + [{"role": "assistant", "content": response}]

            # After: measure state across the model's response
            post_sims = compute_post_generation_readout(history)
            post_chart = build_readout_chart(post_sims, title="After Response")

            return history, pre_chart, post_chart, post_sims, post_sims, introspect_data_sent

        def update_status(emotion: str, alpha: float) -> str:
            if emotion == "none" or alpha == 0.0:
                return "**Status:** No steering active"
            direction = "suppressing" if alpha < 0 else "amplifying"
            return f"**Status:** {direction} **{emotion}** at alpha={alpha:+.3f}"

        # Wire: user submits -> append user msg -> bot responds + update readout
        bot_inputs = [chatbot, emotion_dropdown, alpha_slider, self_aware_toggle, prev_readout_state]
        bot_outputs = [chatbot, pre_plot, post_plot, readout_json, prev_readout_state, introspect_sent]

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
            lambda: (
                build_empty_chart("Before Response"),
                build_empty_chart("After Response"),
                None, None, None,
            ),
            outputs=[pre_plot, post_plot, readout_json, prev_readout_state, introspect_sent],
        )

    return demo

if __name__ == "__main__":
    args = parse_args()
    load_model_and_vectors(args.model_id)
    demo = build_ui()
    demo.queue()
    demo.launch(server_port=args.port, share=args.share)
