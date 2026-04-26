# Lesson 02: Probability & Softmax

A language model produces raw scores called **logits** — one per vocabulary token. This lesson derives the **softmax function** that converts logits into a valid probability distribution, and introduces **temperature** and **entropy** as tools for understanding model confidence.

## Learning Objectives

- Interpret the output of a language model as a **probability distribution** over the vocabulary.
- Derive the **softmax function** from first principles and explain why it always produces valid probabilities.
- Explain the role of **temperature** in controlling the sharpness of a distribution.
- Compute **Shannon entropy** and explain what high vs. low entropy means for a language model's confidence.
- Read softmax curves at multiple temperatures and identify which is most/least confident.

## Background

One-hot encoding from [Lesson 01](01-tokens-and-encoding.md) (tokens are integers indexing vocabulary positions). The exponential and logarithm functions $e^x$ and $\log_2 x$. The definition of a probability distribution: non-negative values that sum to 1.

## From Scores to Probabilities

Logits can be any real number, positive or negative. We need a mapping that:

1. Makes all outputs non-negative.
2. Makes them sum to 1.
3. Preserves the relative ordering (higher logit $\to$ higher probability).

This section is pure reference — the softmax definition and its numerical-stability shift. Every later section pairs `### Theory` with `### Example — <descriptor>`.

### The Softmax Function

**Step 1 — Exponentiate.** Apply $e^{z_i}$ to each logit, mapping any real number to a strictly positive one:

$$\tilde{p}_i = e^{z_i} > 0 \quad \forall\, z_i \in \mathbb{R}.$$

**Step 2 — Normalise.** Divide by the sum:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{|\mathcal{V}|} e^{z_j}}.$$

The output $\mathbf{p} = \text{softmax}(\mathbf{z})$ satisfies $p_i > 0$ and $\sum_i p_i = 1$ — a valid probability distribution.

> **Numerical stability.** In practice, subtract $\max(\mathbf{z})$ before exponentiating to prevent overflow:
> $p_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}}.$
> The constant $e^{-\max}$ cancels top and bottom, so this is mathematically identical.

## Temperature Scaling

### Theory

The **temperature** $T > 0$ scales the logits before softmax:

$$p_i(T) = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}.$$

| Temperature | Effect |
|-------------|--------|
| $T \to 0$ (cold) | Near-step function — almost all mass on the top token |
| $T = 1$ (neutral) | Standard softmax |
| $T \to \infty$ (hot) | Approaches uniform: $p_i \to 1/|\mathcal{V}|$ |

Temperature does not change *which* token has the highest probability — it changes *how much* higher it is relative to the others. Temperature is the knob the **generation** stage (Lesson 21) uses to trade off deterministic (focused) vs. random (creative) output.

### Example — Softmax of four logits at three temperatures

See it in action with logits $\mathbf{z} = [2.0, 1.0, 0.5, -0.5]$:

```rustlab
z = [2.0, 1.0, 0.5, -0.5];

p_cold    = softmax(z / 0.5);
p_neutral = softmax(z / 1.0);
p_warm    = softmax(z / 2.0);

print("T=0.5 (cold)   :", p_cold);
print("T=1.0 (neutral):", p_neutral);
print("T=2.0 (warm)   :", p_warm);
```

Each row sums to ${sum(p_cold):%.3f} — a valid distribution. At $T = 0.5$ token 1 gets ${p_cold(1):%.3f}$ of the mass; at $T = 2.0$ it gets only ${p_warm(1):%.3f}$ — the distribution flattens as temperature rises.

### Example — Stacked subplots: cold / neutral / warm

```rustlab
figure()
subplot(3, 1, 1)
plot(p_cold, "color", "blue", "label", "T=0.5")
title("Softmax at T=0.5 (cold - peaked)")
ylabel("Probability")
ylim([0, 1])

subplot(3, 1, 2)
plot(p_neutral, "color", "green", "label", "T=1.0")
title("Softmax at T=1.0 (neutral)")
ylabel("Probability")
ylim([0, 1])

subplot(3, 1, 3)
plot(p_warm, "color", "red", "label", "T=2.0")
title("Softmax at T=2.0 (warm - flat)")
ylabel("Probability")
xlabel("Token index")
ylim([0, 1])
```

## Shannon Entropy

### Theory

The **entropy** of a distribution $\mathbf{p}$ measures how uncertain or spread-out it is:

$$H(\mathbf{p}) = -\sum_{i=1}^{|\mathcal{V}|} p_i \log_2 p_i \quad [\text{bits}].$$

- $H = 0$: all mass on one token (perfectly certain).
- $H = \log_2 |\mathcal{V}|$: uniform distribution (maximally uncertain).

Higher temperature $\to$ higher entropy.

### Example — Entropy at four temperatures

<!-- hide -->
```rustlab
eps = 1e-12;
vocab_size = 4;
```

```rustlab
p05 = softmax(z / 0.5);
p10 = softmax(z / 1.0);
p20 = softmax(z / 2.0);
p50 = softmax(z / 5.0);

H05 = -sum(p05 .* log2(p05 + eps));
H10 = -sum(p10 .* log2(p10 + eps));
H20 = -sum(p20 .* log2(p20 + eps));
H50 = -sum(p50 .* log2(p50 + eps));
```

Entropy climbs with temperature: $H(T{=}0.5) = ${H05:%.3f}$ bits, $H(T{=}1.0) = ${H10:%.3f}$ bits, $H(T{=}2.0) = ${H20:%.3f}$ bits, $H(T{=}5.0) = ${H50:%.3f}$ bits.

### Example — Sanity checks: uniform vs. near-deterministic

The theoretical maximum for 4 tokens is $\log_2(4) = 2$ bits.

```rustlab
% Uniform distribution — should hit the maximum of log2(4) = 2 bits
p_uniform = ones(vocab_size) / vocab_size;
H_uniform = -sum(p_uniform .* log2(p_uniform + eps));

% Near-deterministic distribution — should be near 0
p_det = [0.999, 0.0003, 0.0003, 0.0004];
H_det = -sum(p_det .* log2(p_det + eps));
```

Uniform over 4 tokens: $H = ${H_uniform:%.3f}$ bits (matches $\log_2 4 = 2$). Near-deterministic: $H = ${H_det:%.4f}$ bits (near zero, as expected).

### Example — Entropy bar chart vs. the maximum

```rustlab
figure()
T_labels = {"T=0.5", "T=1.0", "T=2.0", "T=5.0"};
H_vec = [H05, H10, H20, H50];
bar(T_labels, H_vec, "Entropy (bits) vs. Temperature")
hold("on")
hline(log2(vocab_size), "red", "max = log2(4)")
hold("off")
```

Entropy increases monotonically with temperature — the model becomes harder to predict from.

## Connection to Information Theory

Both objects in this lesson — softmax and entropy — are direct lifts from statistical physics and information theory; transformer-era ML did not invent either.

**Entropy is the source-coding bound.** Shannon's source coding theorem states that the minimum expected code length to losslessly transmit symbols drawn from $p$ is exactly $H(p)$ bits per symbol. The optimal code for symbol $i$ has length $-\log_2 p_i$ bits — common symbols get short codes, rare ones get long codes. So the entropies you computed above aren't abstract uncertainty scores; they are the unavoidable bit-budget for transmitting a sample from each distribution.

| Distribution | $H$ (bits) | Optimal avg bits/token |
|---|---|---|
| `p_cold`  ($T{=}0.5$) | ${H05:%.3f}$ | ${H05:%.3f}$ |
| `p_neutral` ($T{=}1.0$) | ${H10:%.3f}$ | ${H10:%.3f}$ |
| `p_uniform` (4 tokens) | $2.0$ (= $\log_2 4$) | $2.0$ — fixed-width is already optimal |

**Softmax is the Boltzmann distribution.** The form $p_i = e^{z_i/T} / \sum_j e^{z_j/T}$ is identical to the probability of microstate $i$ at temperature $T$ in statistical mechanics, with logits $z_i$ playing the role of negative energies. The "temperature" name is not metaphor: as $T \to 0$ the distribution collapses onto the lowest-energy (highest-logit) state, exactly as a physical system freezes into its ground state. The maximum-entropy principle then recovers softmax as the *unique* distribution that maximises $H(p)$ subject to a constraint on $\mathbb{E}[z]$ — softmax is the "least committed" distribution consistent with the logits.

These two facts return in [Lesson 03](03-cross-entropy-loss.md) (cross-entropy as expected code length under the model) and [Lesson 05](05-bigram-language-model.md) (perplexity as $2^H$).

## Key Takeaways

- Softmax converts arbitrary logits to a valid probability distribution via exponentiation and normalisation.
- Exponentiating means a logit $k$ units larger produces probability $e^k$ times larger (before normalisation) — this amplification is what makes the top token dominate at low temperature.
- At every sequence position the model must output a full distribution over all next tokens. Everything that follows — loss functions, training, sampling — depends on softmax.
- Entropy is the Shannon source-coding bound; softmax is the Boltzmann distribution. ML borrowed both names exactly.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `softmax_temperature.r` | softmax of `[2.0, 1.0, 0.5, -0.5]` at $T = 0.5, 1.0, 2.0$; overlaid bar plot |
| `entropy.r` | entropy at $T = 0.5, 1.0, 2.0, 5.0$ plus uniform and near-deterministic baselines |

Run all with `make lesson-02` (or `rustlab run lessons/02-probability-and-softmax/<name>.r`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `p_cold(1)` (T=0.5) | ≈ `0.701` |
| `p_neutral(1)` (T=1.0) | ≈ `0.580` |
| `p_warm(1)` (T=2.0) | ≈ `0.418` |
| `sum(p_cold)` | `1.0` |
| `H05` | ≈ `1.110` bits |
| `H10` | ≈ `1.591` bits |
| `H20` | ≈ `1.879` bits |
| `H50` | ≈ `1.978` bits |
| `H_uniform` | `2.0` bits (= $\log_2 4$) |
| `H_det` | ≈ `0.014` bits |

## Exercises

1. **Softmax invariance to shift.** Show algebraically that adding a constant $c$ to all logits does not change the softmax output. Then verify numerically: apply softmax to `[2, 1, 0]` and `[5, 4, 3]` and compare.
2. **Temperature limits.** What happens to $\text{softmax}(\mathbf{z} / T)$ as $T \to 0$? Write out the limit mathematically. What is this operation called in the context of optimisation?
3. **Entropy calculation.** For the logits `[3.0, 3.0, 3.0, 3.0]` (all equal), compute the softmax probabilities by hand. Then compute the entropy. Does the result match $\log_2(4)$?
4. **Modifying temperature.** Edit `softmax_temperature.r` to add a fourth curve at $T = 0.1$. Describe what you observe. Is this practically useful for a language model?
5. **Entropy and vocabulary size.** If a model over a vocabulary of size $|\mathcal{V}|$ outputs a perfectly uniform distribution, what is the entropy in bits? Plot this as a function of $|\mathcal{V}|$ for sizes 10, 100, 1000, 10000.

## What's next

Lesson 03 introduces **cross-entropy loss** — the standard training objective for a language model. Cross-entropy measures the gap between the predicted distribution $\mathbf{p}$ from this lesson and the target one-hot distribution from Lesson 01, and reduces (under MLE) to maximizing the log-probability of the correct token at each position.
