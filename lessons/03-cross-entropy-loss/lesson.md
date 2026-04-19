# Lesson 03 — Cross-Entropy Loss

## Learning Objectives

By the end of this lesson you will be able to:

- Define **cross-entropy** between a true distribution and a predicted distribution.
- Derive the simplified cross-entropy loss used in language model training.
- Explain the connection between minimising cross-entropy and **maximum likelihood estimation** (MLE).
- Interpret the shape of the cross-entropy loss curve and identify what "good" vs. "bad" predictions look like.
- Compute cross-entropy by hand for simple examples and verify against the simulation output.

---

## Background

This lesson assumes:

- Probability distributions and softmax from Lesson 02.
- Natural logarithm: $\ln(x) = \log_e(x)$.
- The concept of a loss function: a scalar that measures how wrong a model's prediction is.

---

## Theory

### What Are We Measuring?

At each position in a sequence, the model outputs a probability distribution $\hat{\mathbf{p}} \in \mathbb{R}^{|\mathcal{V}|}$ over the next token. The ground truth is the *actual* next token — a one-hot vector $\mathbf{y} \in \{0,1\}^{|\mathcal{V}|}$ (from Lesson 01). We want a scalar that measures how wrong $\hat{\mathbf{p}}$ is relative to $\mathbf{y}$.

### Cross-Entropy

The **cross-entropy** between two distributions $\mathbf{y}$ (true) and $\hat{\mathbf{p}}$ (predicted) is:

$$
H(\mathbf{y}, \hat{\mathbf{p}}) = -\sum_{i=1}^{|\mathcal{V}|} y_i \log \hat{p}_i
$$

Because $\mathbf{y}$ is a one-hot vector, only one term is non-zero — the term for the correct token index $c$:

$$
\mathcal{L} = -\log \hat{p}_c
$$

This is the **negative log-probability of the correct token**. It is the standard loss for language model training.

### Intuition

- If $\hat{p}_c = 1$ (the model is perfectly certain about the right answer): $\mathcal{L} = -\log(1) = 0$. Zero loss.
- If $\hat{p}_c = 0.5$: $\mathcal{L} = -\log(0.5) = \log(2) \approx 0.693$.
- If $\hat{p}_c \to 0$ (the model assigns nearly zero probability to the correct token): $\mathcal{L} \to \infty$. Unbounded loss.

The loss is a **convex, decreasing function** of $\hat{p}_c$ on $(0, 1]$. Maximising the probability of the correct token is equivalent to minimising cross-entropy.

### Connection to Maximum Likelihood Estimation

Given a training sequence of $T$ tokens $(x_1, x_2, \ldots, x_T)$, the log-likelihood of the data under the model is:

$$
\log P(\mathbf{x}) = \sum_{t=1}^{T} \log \hat{p}_{x_t}^{(t)}
$$

where $\hat{p}_{x_t}^{(t)}$ is the probability the model assigns to the correct token at step $t$. Maximising this is identical to minimising the average negative log-probability:

$$
\mathcal{L}_{\text{avg}} = -\frac{1}{T} \sum_{t=1}^{T} \log \hat{p}_{x_t}^{(t)}
$$

This is exactly the **mean cross-entropy loss** used in practice. Training a language model *is* maximum likelihood estimation.

### Relationship to Shannon Entropy

Cross-entropy decomposes as:

$$
H(\mathbf{y}, \hat{\mathbf{p}}) = H(\mathbf{y}) + D_{\text{KL}}(\mathbf{y} \,\|\, \hat{\mathbf{p}})
$$

where $H(\mathbf{y})$ is the entropy of the true distribution (a constant for fixed data) and $D_{\text{KL}}$ is the Kullback-Leibler divergence — a non-negative measure of how different $\hat{\mathbf{p}}$ is from $\mathbf{y}$. Minimising cross-entropy thus minimises the KL divergence: the model's distribution is pushed toward the true data distribution.

---

## Core Concepts

Cross-entropy is the bridge between the probability output of a language model and the training signal used to improve it. Every parameter update during training is computed to reduce this single scalar. Intuitively, the model is penalised exponentially for placing low probability on the right answer — $-\log(0.01) \approx 4.6$ versus $-\log(0.5) \approx 0.69$. This steep penalty for near-zero probability assignments forces the model to avoid confidently wrong predictions.

**Common misconception:** Cross-entropy loss is *not* the same as accuracy. A model can assign 51% probability to the correct token (high accuracy) while still having a high loss if the remaining 49% is concentrated on one wrong token. The loss penalises the *calibration* of the full distribution, not just which token is argmax.

---

## Simulations

### `cross_entropy_surface.r` — Loss vs. Predicted Probability

**What it computes:**
Plots $\mathcal{L}(\hat{p}_c) = -\log(\hat{p}_c)$ as a function of $\hat{p}_c \in (0, 1]$ and marks reference values. Additionally renders the CE loss as a 3D surface over logit space $(z_1, z_2)$ with $z_3 = 0$ and the correct class fixed to 1 — the actual landscape gradient descent sees.

**What to observe:**
- The curve is **convex and steeply increasing** as $\hat{p}_c \to 0$.
- At $\hat{p}_c = 1/|\mathcal{V}|$ (uniform prediction, the model knows nothing), the loss equals $\log |\mathcal{V}|$. This is the **baseline loss** at the start of training.
- As $\hat{p}_c \to 1$ the loss approaches 0.
- The gradient of the loss is $-1/\hat{p}_c$, which is large when $\hat{p}_c$ is small — meaning the model gets a strong training signal when it is very wrong.
- In logit space the surface has **no finite minimiser** (loss $\to 0$ only as $z_1 \to \infty$) and grows **linearly** in the distractor direction — it is not the exponential cliff the probability-space view suggests.

**Verify by hand:**
Compute $-\ln(0.25)$, $-\ln(0.5)$, $-\ln(0.9)$ on a calculator. Compare to the printed values in the script.

---

## Exercises

1. **Baseline loss.** At the start of training, a model outputs a uniform distribution over a vocabulary of size 50,000. What is the cross-entropy loss? Express this as $\ln(|\mathcal{V}|)$ and compute the numerical value.

2. **Loss ceiling.** If a model assigns probability $10^{-6}$ to the correct token, what is the cross-entropy loss? Is this worse or better than a uniform prediction over a 50-token vocabulary?

3. **Sequence loss.** A model processes a 3-token sequence and assigns probabilities $[0.8, 0.3, 0.6]$ to the correct tokens. Compute the mean cross-entropy loss $\mathcal{L}_{\text{avg}}$.

4. **KL divergence.** Show that for a one-hot true distribution $\mathbf{y}$, the KL divergence $D_{\text{KL}}(\mathbf{y} \| \hat{\mathbf{p}})$ simplifies to the cross-entropy formula derived in the Theory section. (Hint: $H(\mathbf{y}) = 0$ for a one-hot vector.)

5. **Loss curve shape.** Modify `cross_entropy_surface.r` to plot $-\log_2(\hat{p}_c)$ (base-2 logarithm) instead of natural log. How does the shape change? What is the unit of the resulting loss? At what probability does the loss equal 1 bit?
