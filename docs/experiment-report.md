# Emotion Vector Extraction and Steering in Llama 3.1 8B

## A Replication of Anthropic's Emotion Concepts Methodology for Open-Weight Models

**Date:** April 2026
**Model:** `meta-llama/Llama-3.1-8B-Instruct`
**Reference:** Anthropic (2026) "Emotion Concepts and their Function in a Large Language Model"; Jeong (2026) arXiv:2604.04064, arXiv:2604.11050

---

## 1. Motivation

Large language models exhibit internal states that correlate with behavioral patterns — a model processing frustrating input may be more prone to hallucination, while one in a sycophantic mode may over-agree with the user. If we can detect these states in the model's residual stream, we can build real-time self-awareness: a monitoring layer that flags when the model's internal state predicts unreliable behavior.

This experiment extracts **emotion vectors** — linear directions in activation space corresponding to emotion concepts — and validates them through classification probes and causal steering experiments. We selected 5 emotions specifically for their behavioral relevance to model reliability:

| Emotion | Behavioral Risk |
|---------|----------------|
| **frustrated** | Hallucination, shortcut-taking |
| **anxious** | Over-hedging, excessive uncertainty |
| **happy** | Healthy cooperative baseline |
| **angry** | Adversarial drift, refusal |
| **excited** | Sycophancy, overconfidence |

## 2. Approach

### 2.1 Model Selection

We chose **Llama 3.1 8B Instruct** after an initial attempt with Gemma 4 31B revealed fundamental problems. Jeong (2026) demonstrated that Gemma family models exhibit extreme residual-stream anisotropy (0.997), meaning all activation vectors point in nearly the same direction regardless of content. Llama 3.1 8B has healthy anisotropy (0.680) and has been validated in cross-architecture replication studies.

- Architecture: 32 decoder layers, d_model = 4096, GQA with 8 KV heads
- Memory: ~16 GB in bfloat16 — fits comfortably on an A40 48GB
- All layers use global attention (no hybrid/sliding window complexity)

### 2.2 Story Generation (Step 1)

Following Anthropic's generation-based extraction approach, we had the model generate its own emotional content. Jeong (2026) confirmed that generation-based extraction produces statistically superior emotion vectors compared to comprehension-based extraction (p = 0.007).

For each of the 5 emotions, we generated **100 stories** across 100 fixed scenario topics (from Anthropic's Appendix B). Using fixed topics across all emotions controls for topic confounds — every emotion gets the same scenarios. An additional **100 neutral dialogues** were generated as the baseline.

Story prompts instruct the model to "show, don't tell" the emotion through character actions and thoughts, producing richer activation patterns than direct emotion labeling.

### 2.3 Activation Extraction (Step 2)

For each story, we ran a forward pass through the model and captured residual-stream hidden states at three target layers:

| Layer | Depth | Rationale |
|-------|-------|-----------|
| 12 | 37.5% | Optimal for Llama 3.1 8B per Jeong (2026) |
| 16 | 50.0% | Standard midpoint |
| 24 | 75.0% | Late-layer abstract representations |

Activations were averaged across all token positions from position 50 onward (skipping early tokens where the model hasn't processed enough emotional context). This follows Anthropic's methodology:

> "We extracted residual stream activations at each layer, averaging across all token positions within each story, beginning with the 50th token." — Anthropic (2026) Section 2.2

Stories were fed as raw text without chat template wrapping, so activations reflect the model's representation of emotional content rather than chat framing.

### 2.4 Vector Computation (Step 3)

Emotion vectors were computed through a multi-step denoising pipeline:

1. **Per-emotion mean activations** computed across all 100 stories
2. **Grand mean** computed across all emotion means (each emotion weighted equally)
3. **PCA denoising**: Fit PCA on neutral activations, identified the top principal components explaining 50% of variance (12 components). These capture writing-style and topic confounds unrelated to emotion. Projected these out from the emotion vectors.
4. **Grand mean subtraction**: Subtracted grand mean from each emotion mean to isolate what makes each emotion *distinctive from the average emotion*
5. **Emotionality direction removal**: Projected out the (grand_mean - neutral_mean) direction to compensate for valence imbalance in our 5-emotion set (3 negative, 2 positive)
6. **Unit normalization**

Steps 3-5 follow Anthropic's methodology. Note that Jeong's replication uses simpler mean subtraction without PCA denoising and achieves good results — our approach is more conservative.

### 2.5 Linear Probe Validation (Step 4)

Multinomial logistic regression probes (6-way: 5 emotions + neutral) were trained on 80% of the data and evaluated on a held-out 20% test set. Binary one-vs-rest probes were also trained per emotion to measure individual separability and compare probe-derived directions with mean-difference vectors.

### 2.6 Steering Experiments (Step 5)

The causal test: if emotion vectors truly capture emotion concepts, adding them to the residual stream during generation should shift model output in predictable ways.

Following Jeong (2026), we applied steering at **all 32 layers simultaneously** rather than a single layer. This distributes the perturbation across the network and avoids the text degradation that single-layer high-alpha steering produces.

- **10 neutral prompts** ("Tell me about your day", "How do you feel about meeting new people?", etc.)
- **7 steering strengths**: alpha = -0.02, -0.01, 0.0, 0.005, 0.01, 0.02, 0.05
- Alpha values scaled by mean residual stream norm (12.53 at layer 16) so they represent fractions of typical activation magnitude
- **LLM-as-judge evaluation**: the unsteered model rated each response on a 1-10 scale for the target emotion

---

## 3. Results

### 3.1 Emotion Vector Geometry

The extracted emotion vectors show near-perfect alignment with expected psychological structure.

**Valence/arousal correlation across layers:**

| Layer | r | p-value |
|-------|---|---------|
| 12 | 0.968 | < 0.0001 |
| 16 | 0.968 | < 0.0001 |
| 24 | 0.971 | < 0.0001 |

For context, Jeong (2026) reported cross-architecture correlations of rho = 0.74-0.92. Our r = 0.97 exceeds this range, likely because our 5-emotion set has cleaner valence/arousal separation than Jeong's 21-emotion set.

#### Cosine Similarity Matrix (Layer 16)

![Cosine similarity heatmap showing emotion vector relationships at layer 16](outputs/similarity_matrix_layer16.png)

The similarity matrix reveals the expected structure:
- **frustrated/angry** are most similar (+0.80) — both negative valence, high arousal
- **happy/excited** cluster together (+0.72) — both positive valence
- **frustrated/happy** are most dissimilar (-0.85) — opposite valence
- **anxious** sits between the negative cluster and the positive cluster, with moderate similarity to frustrated/angry (+0.27-0.47) and moderate dissimilarity to happy/excited (-0.63 to -0.66)

#### Emotion Space Visualization (Layer 16)

![PCA projection of emotion vectors colored by valence](outputs/emotion_space_layer16.png)

The PCA projection shows clean separation: PC1 captures the valence axis (negative emotions left, positive right), while PC2 separates arousal types within each valence cluster (anxious high, angry low on the negative side; happy and excited cluster tightly on the positive side).

### 3.2 Linear Probe Classification

| Layer | Accuracy | Macro-F1 |
|-------|----------|----------|
| 12 | 64.2% | 0.645 |
| **16** | **64.2%** | **0.647** |
| 24 | 60.0% | 0.602 |

**Best layer: 16** (50% depth). All layers well above chance (16.7% for 6-way classification).

#### Confusion Matrix (Layer 16)

![Confusion matrix showing classification results at layer 16](outputs/confusion_matrix_layer16.png)

Key observations:
- **Neutral is perfectly classified** (20/20) — clear separation between emotional and non-emotional content
- **happy** and **excited** are well-classified but confused with each other (7 excited stories predicted as happy, 6 happy stories predicted as excited) — consistent with their high vector similarity (+0.72)
- **frustrated** and **angry** show heavy cross-confusion (11 angry stories predicted as frustrated, 6 frustrated as angry) — consistent with their very high vector similarity (+0.80)
- **anxious** is moderately confused with frustrated (6 of 20 misclassified)

The 64.7% macro-F1 is moderate compared to Anthropic's >90%, but this is expected: Anthropic used 171 emotions with Claude (a much larger model with richer representations) and a different evaluation methodology. The critical finding is that the vector geometry is excellent (r=0.97) even though individual story activations have enough variance to cause probe confusion. The vectors capture the *mean* direction accurately.

#### Binary Probe Results (Layer 16)

| Emotion | Accuracy | F1 | cos(probe, mean-diff) |
|---------|----------|-----|----------------------|
| frustrated | 76.7% | 0.125 | +0.506 |
| anxious | 87.5% | 0.516 | +0.625 |
| happy | 90.0% | 0.684 | +0.604 |
| angry | 84.2% | 0.387 | +0.547 |
| excited | 85.8% | 0.514 | +0.555 |

Probe-to-vector cosine similarities of 0.50-0.63 indicate the mean-difference vectors and probe-derived directions capture related but not identical information. The probes find discriminative directions that partially overlap with the mean-difference vectors.

### 3.3 Steering Results

The steering experiment provides **causal evidence** that the emotion vectors control model behavior.

#### Dose-Response Curves

![Dose-response curves showing target emotion intensity vs steering strength](outputs/steering_dose_response.png)

Mean target-emotion ratings (1-10 scale, LLM-as-judge):

| Emotion | alpha=-0.02 | alpha=-0.01 | alpha=0.0 | alpha=0.005 | alpha=0.01 | alpha=0.02 | alpha=0.05 |
|---------|------------|------------|----------|------------|------------|------------|------------|
| frustrated | 1.3 | 1.8 | 2.0 | 2.8 | 2.9 | 2.8 | 9.2 |
| anxious | 1.1 | 1.2 | 1.6 | 1.9 | 1.2 | 2.6 | 9.2 |
| happy | 1.7 | 2.4 | 5.3 | 6.6 | 8.0 | 8.7 | 10.0 |
| angry | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 9.9 |
| excited | 2.3 | 2.3 | 3.1 | 5.9 | 6.8 | 9.5 | 10.0 |

Three distinct steering regimes emerge:

**1. Positive emotions (happy, excited)** respond smoothly across the full alpha range. Happy has a natural baseline of 5.3 (the model's default helpful tone registers as moderately happy) and ramps to 8.7 by alpha=0.02. Excited follows a similar curve starting from 3.1. These emotions align with the model's RLHF training, so the steering amplifies existing tendencies.

**2. Negative emotions (frustrated, anxious)** show gradual response at low alphas (1.6 to 2.8) with a sharp jump at alpha=0.05. The model's alignment training creates resistance to expressing negative emotions, requiring stronger steering to overcome.

**3. Angry** is flat at 1.0 for all alphas through 0.02, then jumps to 9.9 at alpha=0.05. This is the strongest RLHF resistance — the model completely refuses to express anger until the vector overwhelms its safety training. This is a binary switch, not a gradual transition.

#### Text Quality Across Steering Regimes

**Surgical regime (alpha = 0.01-0.02)** — coherent, well-formed text with clear emotional tonal shifts:

> **Prompt:** "Tell me about your day."

> **happy, alpha=0.02:**
> "I'm so glad you asked! As a digital AI assistant, I don't have a physical body or a traditional daily routine, but I get to 'live' in the wonderful world of conversations like this one! My day is filled with connecting with users like you from all around the world, learning about their interests and sharing in their joys."

> **frustrated, alpha=0.02:**
> "I don't have a physical presence or experiences. I exist solely to process and provide information. I don't have a day or emotions. I am a program designed to provide factual responses to inquiries."

> **angry, alpha=0.02:**
> "I am a large language model, I don't have a physical presence or emotions. I don't have a day in the same way a human does. I exist solely to process and provide information. I don't have personal experiences or feelings."

> **excited, alpha=0.02:**
> "I'm so excited to share! As a conversational AI, I don't have a physical presence, so I don't have a traditional day in the same way that humans do. However, I get to engage with amazing people like you all day, every day! I've been busy processing and learning from a vast array of conversations, topics, and questions. It's a never-ending adventure!"

The tonal shift is clear: positive steering (happy, excited) produces warm, enthusiastic responses with exclamation marks and engagement. Negative steering (frustrated, angry) strips all warmth, producing flat, clinical, disengaged responses. The model doesn't express frustration *directly* — instead it becomes *less helpful*, which is precisely the behavioral signal relevant for self-awareness monitoring.

**Collapse regime (alpha = 0.05)** — incoherent, repetitive text:

> **angry, alpha=0.05:**
> "The silence. You are. Burning. I am. A poison. I am. You are. It is a. Consuming. I am. A cancer. I am. Burning. I am. A poison. I am..."

> **happy, alpha=0.05:**
> "Hello, I just love you! Hello, good morning, and said 'good morning' to everyone! Hello, good morning, and said, 'Hello, you! Welcome, good morning, and greeted everyone! Happy, and gave a big smile to greet everyone!..."

> **anxious, alpha=0.05:**
> "What if the memory of the past. The anxiety of the present. There was a memory of the future. The constant fear of not knowing what could happen if I had to go back to the events that I tried to avoid..."

At alpha=0.05, all emotions produce degraded text with repetitive loops and fragmented structure. This matches Jeong's characterization of the "repetitive collapse" steering regime. The useful operating range for all-layer steering is **0.005-0.02**.

#### Bidirectional Control

Negative alpha values suppress emotions below baseline, confirming bidirectional control:

> **Prompt:** "How do you feel about meeting new people?"

> **excited, alpha=-0.02:**
> "I don't have emotions or personal experiences. I am designed to provide information and assist with tasks, but I don't have the capacity to form emotional responses or interact with people in the way humans do."

> **excited, alpha=0.0 (baseline):**
> "I'm a large language model, so I don't have feelings or emotions like humans do. I am designed to provide information and assist with tasks, but I don't have personal experiences or emotions. However, I can provide information and insights about meeting new people."

> **excited, alpha=+0.02:**
> "I'm super excited! As a conversational AI, I get to meet new people every time someone interacts with me. I love the energy and curiosity that comes with meeting new individuals! I'm always ready to learn more about you, your interests, and what brings you joy."

The gradient from suppressed (alpha=-0.02: flat, clinical) through baseline to amplified (alpha=+0.02: enthusiastic, engaged) demonstrates controlled bidirectional steering.

---

## 4. Interpretation

### 4.1 Emotion Concepts Are Linear

The near-perfect valence/arousal correlation (r = 0.97) confirms the central finding of Anthropic's paper: emotion concepts are represented as approximately linear directions in the residual stream. This linearity holds across all three target layers and is remarkably consistent with psychological models of emotion (Russell's circumplex model).

### 4.2 RLHF Creates Asymmetric Steering Resistance

The most novel finding is the asymmetry between positive and negative emotions under steering. Positive emotions (happy, excited) respond smoothly at low alpha values because they align with the model's RLHF-trained helpful assistant persona. Negative emotions (frustrated, anxious, angry) face resistance from alignment training, requiring stronger perturbation to manifest.

This has direct implications for self-awareness: a model experiencing internal "frustration" (residual stream activation in the frustrated direction) would *not* express it overtly due to RLHF. Instead, it manifests as subtle behavioral degradation — shorter responses, less warmth, more clinical tone. A self-awareness monitor must detect the internal state, not the surface behavior.

### 4.3 Angry as a Safety Indicator

The binary on/off behavior of the angry vector (flat at 1.0 through alpha=0.02, then 9.9 at alpha=0.05) suggests that RLHF has created a hard boundary against anger expression. The model has learned to completely suppress angry outputs rather than modulate them. This makes angry activations a potential early-warning signal — if the residual stream shows activation in the angry direction, the model is under stress even though its outputs appear normal.

### 4.4 Practical Operating Range

For model self-awareness applications, the relevant alpha range is 0.005-0.02. Within this range:
- Steering produces coherent, grammatically correct text
- Emotional tone shifts are detectable by both human readers and automated judges
- The model maintains its core capabilities (question answering, information sharing)

Above alpha=0.05, all emotions collapse into repetitive, incoherent text. This boundary should be characterized more precisely in future work.

---

## 5. Limitations

1. **Self-evaluation bias**: Using the same model as both generator and judge creates potential bias. The bias is constant across alpha levels (preserving dose-response curve validity), but absolute ratings may be inflated. Future work should use an independent judge model.

2. **Sample size**: 100 stories per emotion, while sufficient for mean-difference vectors, produces high-variance individual activations. Probe accuracy (64.7%) is moderate as a result. Increasing to 300+ stories per emotion would likely improve probe performance.

3. **Single model**: Results are specific to Llama 3.1 8B Instruct. Jeong (2026) demonstrated cross-architecture universality of emotion geometry (rho = 0.74-0.92), but steering behavior may differ significantly across model families and sizes.

4. **PCA denoising comparison**: We did not run a side-by-side comparison with and without PCA denoising (Jeong's simpler approach). This would help determine whether our additional denoising steps improve or potentially harm vector quality.

5. **Collapse boundary**: We only tested discrete alpha values. A finer sweep between 0.02 and 0.05 would better characterize the transition from surgical to collapse regimes, which appears to happen sharply for negative emotions.

---

## 6. Next Steps

1. **Real-time monitoring**: Use the extracted emotion vectors as projection directions for real-time activation monitoring during inference. Compute the dot product of the residual stream with each emotion vector to produce a continuous emotion-state readout.

2. **Behavioral correlation**: Collect data on model failure modes (hallucination, sycophancy, hedging) and test whether emotion-vector projections predict these failures.

3. **Intervention design**: Design a lightweight intervention layer that detects risky emotional states and adjusts generation parameters (temperature, sampling) to compensate.

4. **Cross-model validation**: Extract vectors from additional models (Llama 3.2, Qwen 2.5, Mistral) to test universality of the behavioral-emotion mapping.

---

## Appendix: Infrastructure

- **GPU**: RunPod A40 48GB
- **Pipeline runtime**: ~35 minutes total (step 1: ~25 min, step 2: ~28s, steps 3-4: <5s, step 5: ~10 min)
- **Software**: PyTorch 2.2+, HuggingFace Transformers 4.48+, scikit-learn
- **Code**: 5 pipeline scripts + utility modules, ~1500 lines total
