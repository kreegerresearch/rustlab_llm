# Lesson 02 — Probability & Softmax

## Learning Objectives

By the end of this lesson you will be able to:

- Interpret the output of a language model as a **probability distribution** over the vocabulary.
- Derive the **softmax function** from first principles and explain why it always produces valid probabilities.
- Explain the role of **temperature** in controlling the sharpness of a distribution.
- Compute **Shannon entropy** and explain what high vs. low entropy means for a language model's confidence.
- Read softmax curves at multiple temperatures and identify which is most/least confident.

---

## Background

This lesson assumes:

- One-hot encoding from Lesson 01 (tokens are integers representing vocabulary positions).
- Exponential and logarithm functions: $e^x$ and $\ln(x)$.
- The definition of a probability distribution: non-negative values that sum to 1.

---

## Theory

### From Scores to Probabilities

A language model produces raw, unconstrained real-valued scores called **logits** — one per vocabulary token. Logits can be any real number, positive or negative. To use them as probabilities we need a mapping that:

1. Makes all outputs non-negative.
2. Makes them sum to 1.
3. Preserves the relative ordering (higher logit → higher probability).

### The Softmax Function

**Step 1 — Exponentiate.** Apply $e^{z_i}$ to each logit $z_i$. This maps any real number to a strictly positive number:

$$
\tilde{p}_i = e^{z_i} > 0 \quad \forall\, z_i \in \mathbb{R}
$$

**Step 2 — Normalise.** Divide by the sum of all exponentiated values:

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{|\mathcal{V}|} e^{z_j}}
$$

This is the **softmax** function. The output vector $\mathbf{p} = \text{softmax}(\mathbf{z})$ satisfies $p_i > 0$ and $\sum_i p_i = 1$, making it a valid probability distribution.

**Numerical stability note.** In practice, computing $e^{z_i}$ for large $z_i$ overflows floating-point arithmetic. The standard fix is to subtract the maximum logit before exponentiating:

$$
p_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j} e^{z_j - \max(\mathbf{z})}}
$$

This is mathematically identical (the constant $e^{-\max}$ cancels top and bottom) but prevents overflow.

### Temperature Scaling

The **temperature** $T > 0$ is a scalar that scales the logits before softmax:

$$
p_i(T) = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}
$$

- **$T \to 0$ (cold):** Dividing by a small $T$ amplifies differences. The distribution becomes a near-step function — almost all probability mass on the highest-logit token. The model is maximally *confident* (or greedy).
- **$T = 1$ (neutral):** Standard softmax; the distribution reflects the raw logit differences.
- **$T \to \infty$ (hot):** Dividing by a large $T$ shrinks all logits toward zero. The distribution approaches **uniform**: $p_i \to 1/|\mathcal{V}|$. The model is maximally *uncertain*.

Temperature is used during **text generation** (Lesson 21) to trade off between deterministic (focused) and random (creative) outputs.

### Shannon Entropy

The **entropy** of a probability distribution $\mathbf{p}$ measures how uncertain or spread-out it is:

$$
H(\mathbf{p}) = -\sum_{i=1}^{|\mathcal{V}|} p_i \log_2 p_i \quad [\text{bits}]
$$

Entropy is zero when all mass is on one token (perfectly certain). It reaches its maximum of $\log_2 |\mathcal{V}|$ bits when the distribution is uniform.

**Connection to temperature:** Higher temperature → higher entropy. The model becomes harder to predict from.

---

## Core Concepts

Softmax is not just a normalisation trick — it encodes a specific assumption about how logits relate to probabilities. Exponentiating logits means that a logit that is $k$ units larger produces a probability that is $e^k$ times larger (before normalisation). This *exponential* gap amplification is what makes the highest-scoring token dominate at low temperature.

The key insight for language modelling: at every position in a sequence, the model must output a full probability distribution over all possible next tokens. Softmax is the standard way to convert the model's internal scores into that distribution. Everything that follows in this tutorial — loss functions, training, sampling — depends on this output being a valid probability distribution.

**Common misconception:** Temperature does not change *which* token has the highest probability — it changes *how much* higher it is relative to the others. The ranking of tokens by probability is preserved across temperatures.

---

## Simulations

### `softmax_temperature.r` — Softmax Curves at T = 0.5, 1.0, 2.0

**What it computes:**
Applies softmax to the same logit vector at three temperatures and plots the resulting probability distributions as overlaid bar charts.

**What to observe:**
- At $T = 0.5$ the distribution is peaked: most mass on the top token.
- At $T = 1.0$ the distribution is the standard softmax.
- At $T = 2.0$ the distribution is flatter, with more probability spread across tokens.
- The token with the highest logit always has the highest probability regardless of temperature.

**Verify by hand:**
For the logits `[2.0, 1.0, 0.5, -0.5]` at $T = 1.0$: compute $e^{2.0}$, $e^{1.0}$, $e^{0.5}$, $e^{-0.5}$, sum them, and divide each by the sum. Compare to the printed probabilities.

---

### `entropy.r` — Entropy of Softmax Distributions

**What it computes:**
Computes the Shannon entropy of the softmax distribution at several temperatures and prints the values. Also shows entropy for a hand-crafted near-uniform distribution and a near-deterministic distribution.

**What to observe:**
- Entropy increases monotonically with temperature.
- The uniform distribution ($T \to \infty$) achieves maximum entropy $= \log_2 |\mathcal{V}|$ bits.
- A near-deterministic distribution has entropy close to 0 bits.

**Verify by hand:**
For a 4-token uniform distribution $p_i = 0.25$: $H = -4 \times 0.25 \times \log_2(0.25) = -4 \times 0.25 \times (-2) = 2$ bits.

---

## Exercises

1. **Softmax invariance to shift.** Show algebraically that adding a constant $c$ to all logits does not change the softmax output. Then verify numerically: apply softmax to `[2, 1, 0]` and `[5, 4, 3]` and compare.

2. **Temperature limits.** What happens to $\text{softmax}(\mathbf{z} / T)$ as $T \to 0$? Write out the limit mathematically. What is this operation called in the context of optimisation?

3. **Entropy calculation.** For the logits `[3.0, 3.0, 3.0, 3.0]` (all equal), compute the softmax probabilities by hand. Then compute the entropy. Does the result match $\log_2(4)$?

4. **Modifying temperature.** Edit `softmax_temperature.r` to add a fourth curve at $T = 0.1$. Describe what you observe. Is this practically useful for a language model?

5. **Entropy and vocabulary size.** If a model over a vocabulary of size $|\mathcal{V}|$ outputs a perfectly uniform distribution, what is the entropy in bits? Plot this as a function of $|\mathcal{V}|$ for sizes 10, 100, 1000, 10000.
