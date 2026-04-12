# Emotion Vectors for Gemma 4

Extract "emotion vectors" — linear directions in activation space corresponding to emotion concepts — from Google's Gemma 4 and validate them through linear probes and steering experiments.

This project replicates Part 1 of Anthropic's ["Emotion Concepts and their Function in a Large Language Model"](https://www.anthropic.com/research) (April 2026), adapted for an open-weight model.

## Methodology

### What are emotion vectors?

Large language models develop internal representations of concepts, including emotions. These representations turn out to be approximately **linear** — each emotion corresponds to a direction in the model's high-dimensional activation space. If you know where "happy" points in this space, you can:

1. **Detect** it: project activations onto the direction to measure how "happy" a given input is
2. **Steer** it: add the direction to the residual stream during generation to make the model's outputs happier

### Pipeline overview

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `01_generate_stories.py` | Prompt Gemma 4 to write short stories depicting each emotion |
| 2 | `02_extract_activations.py` | Run stories through the model, capture mean residual-stream activations (token 50 onward) at target layers |
| 3 | `03_compute_vectors.py` | Grand-mean subtraction, PCA confound removal, emotionality projection, unit normalisation |
| 4 | `04_validate_probes.py` | Train linear probes (logistic regression) to classify emotions from activations |
| 5 | `05_steer_and_eval.py` | Add norm-scaled emotion vectors to the residual stream during generation, evaluate with Gemma-as-judge |

### Key adaptations from the Anthropic paper

- **Model**: Gemma 4 31B (dense, 60 layers) instead of Claude Sonnet 4.5
- **Emotions**: 10 high-signal emotions selected for steering power (not 171)
- **Approach**: Mean-difference vectors only (no SAE training — no SAEs exist for Gemma 4)
- **Framework**: Direct PyTorch hooks on HuggingFace `transformers` (no TransformerLens)

### Activation extraction

We feed each story as raw text (no chat template) through the model and capture residual-stream hidden states at three target layers. Following the paper (section 2.2), activations are **averaged across all token positions from position 50 onward** — skipping early tokens where the model hasn't yet processed enough emotional context. This captures the emotional content distributed across the full story rather than relying on a single position.

### Vector computation: denoising pipeline

The vector computation follows the paper's methodology with an additional correction for our small emotion set:

1. **Per-emotion means** — average the mean-across-token activations over all stories for each emotion
2. **Grand-mean subtraction** — subtract the mean-of-emotion-means from each emotion mean (paper section 2.3). This isolates what makes each emotion *distinctive from the average emotion*
3. **PCA confound removal** — fit PCA on neutral-story activations, project out the top components explaining 50% of variance (paper section 2.3). This removes writing-style, topic, and narrative structure confounds
4. **Emotionality direction projection** — project out the `grand_mean - neutral_mean` direction (see below)
5. **Unit normalisation**

### Adapting grand-mean subtraction to 10 emotions

The Anthropic paper computes emotion vectors by subtracting the **grand mean** (the average activation across all 171 emotions) from each emotion's mean activation. This isolates what makes each emotion *distinctive from other emotions*, rather than what makes it distinctive from emotionlessness.

Naively replicating this with only 10 emotions introduces a problem: 7 of our 10 emotions are negative-valence (angry, afraid, sad, disgusted, anxious, hostile, frustrated), which skews the grand mean toward negative affect. Subtracting that skewed mean would bake a positive-valence bias into every vector, contaminating the directions we extract.

A simple neutral-baseline subtraction (`emotion_mean - neutral_mean`) avoids the skew but captures a different signal — it conflates emotion-specific information with shared "emotionality" (variance common to all emotions versus neutral text).

We address this with a **two-step subtraction**:

1. Compute the grand mean across all 10 emotion activation means
2. Compute each raw vector as `emotion_mean - grand_mean` (the paper's approach)
3. Compute the shared emotionality direction: `grand_mean - neutral_mean`
4. Project out that shared direction from each vector

Step 2 isolates what is unique to each emotion relative to the average emotion. Step 4 removes the residual shared "emotional vs. neutral" axis that is poorly estimated with a small, valence-imbalanced emotion set. Together, this approximates the paper's 171-emotion grand mean subtraction without requiring 171 emotions, because we explicitly remove the contaminated axis rather than relying on a balanced set to average it out.

### Emotion vocabulary

```
angry, afraid, happy, sad, disgusted, excited, anxious, hostile, tender, frustrated
```

Selected for clear behavioural signatures and maximum separation — these produce the most observable changes under steering.

### Layer targeting

We capture activations at **global attention layers** near 25%, 50%, and 75% depth. Gemma 4 uses hybrid attention (sliding window + full global every 6 layers). Global attention layers have richer contextualised representations since they attend to the full sequence.

For the 31B model (60 layers): layers **17, 29, 47**.

### Steering normalisation

Following the paper (section 4), steering strengths (alpha values) are specified **relative to the mean norm of residual-stream activations** at the target layer. During activation extraction, we compute and save the mean residual-stream norm per layer. The steering script loads this norm and scales alpha accordingly, so `alpha=0.1` means "add 10% of the typical activation magnitude in this direction." This makes alpha values interpretable and comparable across models and layers.

## Setup (RunPod)

```bash
# Clone and enter the project
cd emotion-vectors

# Run the setup script
bash setup.sh

# Or manually:
pip install torch transformers accelerate scikit-learn numpy matplotlib seaborn tqdm jsonlines umap-learn
huggingface-cli login  # needs access to google/gemma-4-31B-it
```

**Hardware requirements**: A100 80GB (recommended) or H100. The 31B model uses ~62GB in bfloat16. If VRAM is tight, edit `config.py` to use `google/gemma-4-E4B-it` (the 4B fallback, ~16GB).

## Running the pipeline

```bash
# Quick test run (5 stories per emotion)
python 01_generate_stories.py --num-stories 5
python 02_extract_activations.py
python 03_compute_vectors.py
python 04_validate_probes.py
python 05_steer_and_eval.py --skip-judge

# Full run (50 stories per emotion)
python 01_generate_stories.py
python 02_extract_activations.py
python 03_compute_vectors.py
python 04_validate_probes.py
python 05_steer_and_eval.py
```

Each script is independently runnable and picks up from the data left by previous steps.

## Outputs

After a full run, `outputs/` will contain:

- `similarity_matrix_layer{N}.png` — Cosine similarity heatmap between emotion vectors
- `emotion_space_layer{N}.png` — PCA projection of emotion vectors, coloured by valence
- `confusion_matrix_layer{N}.png` — Linear probe classification confusion matrix
- `probe_results.json` — Probe accuracy and F1 scores per layer
- `steered_responses.jsonl` — All generated responses at each steering strength
- `steering_dose_response.png` — Target-emotion intensity vs steering strength
- `judge_ratings.json` — Raw Gemma-as-judge ratings

## What to expect

Based on the Anthropic paper's findings on Claude:

- **Linear probes** should achieve 40–70%+ accuracy classifying emotions (well above the ~9% chance baseline for 11 classes)
- **Similarity structure** should show semantically related emotions clustering (angry/hostile, happy/excited) and opposite-valence emotions separating
- **Steering** should produce monotonically increasing target-emotion intensity as α grows, with visible qualitative shifts in generated text

## Project structure

```
emotion-vectors/
├── config.py                  # Hyperparameters, paths, model settings
├── emotions.json              # Emotion vocabulary with valence/arousal
├── 01_generate_stories.py     # Story generation
├── 02_extract_activations.py  # Activation extraction
├── 03_compute_vectors.py      # Vector computation and analysis
├── 04_validate_probes.py      # Linear probe validation
├── 05_steer_and_eval.py       # Steering experiments
├── utils/
│   ├── hooks.py               # ActivationCapture and SteeringHook
│   ├── prompts.py             # Story prompt templates
│   └── visualization.py       # Plotting helpers
├── data/                      # Generated data (not committed)
└── outputs/                   # Results and plots
```

## References

- Anthropic (2026). "Emotion Concepts and their Function in a Large Language Model."
- Turner et al. (2023). "Activation Addition: Steering Language Models Without Optimization."
- Neel Nanda et al. (2023). "Emergent Linear Representations in World Models of Self-Supervised Sequence Models."
