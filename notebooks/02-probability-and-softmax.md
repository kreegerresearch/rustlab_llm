# Lesson 02 — Probability & Softmax

A language model produces raw scores called **logits** — one per vocabulary token.
This lesson derives the **softmax function** that converts logits into a valid
probability distribution, and introduces **temperature** and **entropy** as tools for
understanding model confidence.

---

## From Scores to Probabilities

Logits can be any real number, positive or negative. We need a mapping that:

1. Makes all outputs non-negative.
2. Makes them sum to 1.
3. Preserves the relative ordering (higher logit $\to$ higher probability).

### The Softmax Function

**Step 1 — Exponentiate.** Apply $e^{z_i}$ to each logit, mapping any real number to a
strictly positive one:

$$\tilde{p}_i = e^{z_i} > 0 \quad \forall\, z_i \in \mathbb{R}$$

**Step 2 — Normalise.** Divide by the sum:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{|\mathcal{V}|} e^{z_j}}$$

The output $\mathbf{p} = \text{softmax}(\mathbf{z})$ satisfies $p_i > 0$ and
$\sum_i p_i = 1$ — a valid probability distribution.

> **Numerical stability.** In practice, subtract $\max(\mathbf{z})$ before
> exponentiating to prevent overflow:
> $p_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}}$

---

## Temperature Scaling

The **temperature** $T > 0$ scales the logits before softmax:

$$p_i(T) = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}$$

| Temperature | Effect |
|-------------|--------|
| $T \to 0$ (cold) | Near-step function — almost all mass on the top token |
| $T = 1$ (neutral) | Standard softmax |
| $T \to \infty$ (hot) | Approaches uniform: $p_i \to 1/|\mathcal{V}|$ |

Temperature does not change *which* token has the highest probability — it changes
*how much* higher it is relative to the others.

Let's see this in action with logits $\mathbf{z} = [2.0, 1.0, 0.5, -0.5]$:

```rustlab
z = [2.0, 1.0, 0.5, -0.5];
print("Logits z:", z);

% Softmax at three temperatures
p_cold    = softmax(z / 0.5);
p_neutral = softmax(z / 1.0);
p_warm    = softmax(z / 2.0);

print("Probabilities at T=0.5 (cold):");
print(p_cold);
print("Probabilities at T=1.0 (neutral):");
print(p_neutral);
print("Probabilities at T=2.0 (warm):");
print(p_warm);

% Verify: all distributions sum to 1.0
print("Sum at T=0.5 (should be 1):", sum(p_cold));
print("Sum at T=1.0 (should be 1):", sum(p_neutral));
print("Sum at T=2.0 (should be 1):", sum(p_warm));
```

At $T = 0.5$ most mass concentrates on token 1 (logit 2.0). At $T = 2.0$ the
distribution flattens out.

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

savefig("outputs/softmax_temperature.svg")
print("Saved outputs/softmax_temperature.svg")
```

---

## Shannon Entropy

The **entropy** of a distribution $\mathbf{p}$ measures how uncertain or spread-out
it is:

$$H(\mathbf{p}) = -\sum_{i=1}^{|\mathcal{V}|} p_i \log_2 p_i \quad [\text{bits}]$$

- $H = 0$: all mass on one token (perfectly certain).
- $H = \log_2 |\mathcal{V}|$: uniform distribution (maximally uncertain).

Higher temperature $\to$ higher entropy. Let's measure this:

```rustlab
eps = 1e-12;
vocab_size = 4;

p05 = softmax(z / 0.5);
p10 = softmax(z / 1.0);
p20 = softmax(z / 2.0);
p50 = softmax(z / 5.0);

H05 = -sum(p05 .* log2(p05 + eps));
H10 = -sum(p10 .* log2(p10 + eps));
H20 = -sum(p20 .* log2(p20 + eps));
H50 = -sum(p50 .* log2(p50 + eps));

print("Entropy at T=0.5:", H05, "bits");
print("Entropy at T=1.0:", H10, "bits");
print("Entropy at T=2.0:", H20, "bits");
print("Entropy at T=5.0:", H50, "bits");
```

The theoretical maximum for 4 tokens is $\log_2(4) = 2$ bits. Let's verify:

```rustlab
H_max = log2(vocab_size);
print("Maximum entropy (uniform, 4 tokens):", H_max, "bits");

% Uniform distribution achieves maximum entropy
p_uniform = ones(vocab_size) / vocab_size;
H_uniform = -sum(p_uniform .* log2(p_uniform + eps));
print("Entropy of uniform distribution:", H_uniform, "bits  (should equal 2.0)");

% Near-deterministic distribution has entropy near 0
p_det = [0.999, 0.0003, 0.0003, 0.0004];
H_det = -sum(p_det .* log2(p_det + eps));
print("Entropy of near-deterministic:", H_det, "bits  (should be near 0)");
```

```rustlab
H_vec = [H05, H10, H20, H50];
savebar(H_vec, "outputs/entropy_vs_temperature.svg", "Entropy (bits) at T = 0.5, 1.0, 2.0, 5.0")
print("Saved outputs/entropy_vs_temperature.svg")
```

Entropy increases monotonically with temperature — the model becomes harder to
predict from.

---

## Key Takeaways

- Softmax converts arbitrary logits to a valid probability distribution via
  exponentiation and normalisation.
- Exponentiating means a logit $k$ units larger produces probability $e^k$ times
  larger (before normalisation) — this amplification is what makes the top token
  dominate at low temperature.
- At every sequence position the model must output a full distribution over all next
  tokens. Everything that follows — loss functions, training, sampling — depends on
  softmax.
