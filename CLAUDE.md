# Emotion Vectors for Llama 3.1 8B

## Project Goal
Replicate Part 1 of Anthropic's "Emotion Concepts and their Function in a Large Language Model" (April 2026), adapted for open-weights models. The ultimate purpose is **model self-awareness**: detecting emotional states (frustrated, anxious, etc.) that correlate with reliability failures (hallucination, sycophancy, hedging), so a model can monitor and modulate its own behavior.

## Why Llama 3.1 8B (not Gemma 4)
- Replication studies (Jeong 2026) found Gemma has extreme residual-stream anisotropy (0.997) — emotion vectors can't cleanly separate
- Llama 3.1 8B is proven in cross-architecture replication studies with this exact methodology
- Standard `model.model.layers` structure — no multimodal nesting headaches
- ~16GB in bf16 — fits on cheaper GPUs (A40, L40, or even 24GB consumer cards)
- Generation-based extraction (model writes its own stories) produces better results — fast/cheap with 8B

## Current Status (2026-04-15)
- **All 5 pipeline steps complete** with Llama 3.1 8B on RunPod A40
- Research report written: `docs/experiment-report.md` (+ PDF)
- Results: vector geometry r=0.97, best probe layer 16 (F1=0.647), steering confirmed causal
- Steering regime: surgical 0.005-0.02 works, collapse at >=0.05
- **Next:** Build Gradio chat interface (`06_chat_interface.py`) — plan ready at `.claude/plans/snoopy-wishing-flame.md`, implement it

## Methodology (Jeong 2026 alignment)
- **Layer selection**: 37.5% depth optimal for Llama 3.1 8B (layer 12 of 32)
- **Steering**: all layers simultaneously, NOT single layer (avoids text degradation)
- **Alpha range**: 0.005–0.05 (10-40× lower than naive single-layer approach)
- **PCA denoising**: our addition (Anthropic-style), not in Jeong — run with and without (`--no-denoise`) to compare
- **Generation-based extraction**: statistically superior to comprehension-based (p=0.007)
- **Reference papers**: arXiv:2604.04064, arXiv:2604.11050

## 5 Emotions (chosen for behavioral relevance)
- **frustrated** — hallucination/shortcut risk
- **anxious** — over-hedging, uncertainty
- **happy** — healthy cooperative baseline
- **angry** — adversarial drift
- **excited** — sycophancy/overconfidence

## Model Details
- **Primary:** `meta-llama/Llama-3.1-8B-Instruct` (gated — needs HF license agreement)
- **Ungated mirror:** `NousResearch/Meta-Llama-3.1-8B-Instruct`
- Architecture: `LlamaForCausalLM` — 32 layers, d_model=4096, GQA (8 KV heads)
- All layers use global attention (no hybrid/sliding window)
- Config is flat: `model.config.hidden_size` works directly (no nested text_config)
- Layers at `model.model.layers` (standard path)
- Target layers: **12, 16, 24** (37.5/50/75% depth; Jeong 2026 found 37.5% optimal for Llama 3.1 8B)
- Tokenizer: no pad token set by default — scripts set `pad_token = eos_token`

## Pipeline Scripts
1. `01_generate_stories.py` — generate emotional stories with Llama itself
2. `02_extract_activations.py` — forward pass to capture residual stream
3. `03_compute_vectors.py` — mean-diff vectors with PCA denoising (CPU only)
4. `04_validate_probes.py` — logistic regression probes (CPU only)
5. `05_steer_and_eval.py` — steering + LLM-as-judge evaluation (needs GPU)
6. `06_chat_interface.py` — Gradio chat with real-time emotion steering + state readout (needs GPU)

## RunPod Setup
- A40 48GB or L40 48GB sufficient (~16GB model + room for activations)
- A100 80GB also works (same as before, just overkill now)
- **Required env vars** (add to `~/.bashrc`):
  ```bash
  export HF_HUB_CACHE=/root/hf_cache
  export HF_HUB_DISABLE_XET=1
  ```
- Repo cloned at `/emotion-vectors` on the pod (not `/workspace/` — NFS has block write quota of 0)
- Use container disk, NOT workspace NFS for any data
- **Always run long jobs in tmux** to survive disconnects:
  ```bash
  tmux new -s emo          # start session
  # run your pipeline commands here
  # Ctrl-B then D to detach
  tmux attach -t emo       # reattach after reconnect
  ```

## Key Architecture Decisions
- `config.get_text_config(model)` — resolves nested text config for multimodal models; falls through for Llama
- `config.get_decoder_layers(model)` — tries multiple HF model structures; `model.model.layers` matches for Llama
- `config.get_target_layers(model)` — model-aware: uses raw percentages for Llama, snaps to global attention layers for hybrid models
- Activation averaging from token 50 onward
- PCA denoising: project out neutral variance components explaining 50% of variance (can be disabled with `--no-denoise` for Jeong-style simple mean subtraction)
- Emotionality direction projected out to compensate for valence imbalance
- **Steering**: all-layer approach (Jeong 2026) — vector added at every decoder layer simultaneously
- **Steering alphas**: 0.005–0.05 range (Jeong-aligned), scaled by mean residual stream norm
- `MultiLayerSteeringHook` in `utils/hooks.py` for all-layer steering; `SteeringHook` still available for single-layer via `--single-layer` flag

## Local Development
- Python managed via `uv` (Homebrew): `uv run python <script>.py`
- Steps 3-4 run locally (CPU only, no model needed)
- Steps 1-2, 5 require GPU pod
