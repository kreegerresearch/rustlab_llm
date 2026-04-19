---
title: "Lesson 03 — Cross-Entropy Loss"
order: 3
---

# Lesson 03 — Cross-Entropy Loss

The model outputs a probability distribution. The ground truth is the actual next
token. We need a scalar that measures how wrong the prediction is — this scalar is the
**cross-entropy loss**, and minimising it is the entire goal of training.

---

## The Setup

At each position in a sequence the model outputs $\hat{\mathbf{p}} \in \mathbb{R}^{|\mathcal{V}|}$
(a probability distribution from softmax, [Lesson 02](02-probability-and-softmax.md)).
The ground truth is the actual next token — a one-hot vector $\mathbf{y}$
([Lesson 01](01-tokens-and-encoding.md)).

## Cross-Entropy

The **cross-entropy** between the true distribution $\mathbf{y}$ and predicted
distribution $\hat{\mathbf{p}}$ is:

$$H(\mathbf{y}, \hat{\mathbf{p}}) = -\sum_{i=1}^{|\mathcal{V}|} y_i \log \hat{p}_i$$

Because $\mathbf{y}$ is one-hot, only the term for the correct token $c$ survives:

$$\mathcal{L} = -\log \hat{p}_c$$

This is the **negative log-probability of the correct token** — the standard loss for
language model training.

### Intuition

| $\hat{p}_c$ | $\mathcal{L} = -\log(\hat{p}_c)$ | Meaning |
|------|------|---------|
| 1.0  | 0    | Perfect prediction |
| 0.5  | 0.693 | Moderate uncertainty |
| 0.01 | 4.605 | Nearly zero probability on the right answer |
| $\to 0$ | $\to \infty$ | Catastrophically wrong |

The loss is a **convex, decreasing function** of $\hat{p}_c$. Let's plot it:

```rustlab
% Sample p_hat across (0.01, 1.0] and compute L = -log(p_hat_c) in nats
p_hat = linspace(0.01, 1.0, 200);
loss  = -log(p_hat);
```

<!-- hide -->
```rustlab
L_50k   = -log(1.0 / 50000.0);
L_4     = -log(0.25);
L_half  = -log(0.5);
L_high  = -log(0.9);
L_vhigh = -log(0.99);
```

Reference points in nats: $p{=}1/50000 \Rightarrow L = ${L_50k:%.3f}$
(the uniform-over-50k-tokens baseline), $p{=}0.25 \Rightarrow L = ${L_4:%.3f}$
($= \log 4$), $p{=}0.5 \Rightarrow L = ${L_half:%.3f}$,
$p{=}0.9 \Rightarrow L = ${L_high:%.3f}$, $p{=}0.99 \Rightarrow L = ${L_vhigh:%.4f}$.

```rustlab
figure()
hold("on")
plot(p_hat, loss, "color", "blue", "label", "L = -log(p)")
hline(L_4, "gray", "baseline = log(4)")
title("Cross-Entropy Loss vs. Predicted Probability of Correct Token")
xlabel("Predicted probability of correct token (p_c)")
ylabel("Loss L = -log(p_c)  [nats]")
ylim([0, 6])
legend()
```

At $\hat{p}_c = 1/|\mathcal{V}|$ (uniform prediction — the model knows nothing), the
loss equals $\log |\mathcal{V}|$. This is the **baseline loss** at the start of training.

---

## Connection to Maximum Likelihood

For a training sequence of $T$ tokens, the average loss is:

$$\mathcal{L}_{\text{avg}} = -\frac{1}{T} \sum_{t=1}^{T} \log \hat{p}_{x_t}^{(t)}$$

This is exactly the negative log-likelihood divided by $T$. Minimising cross-entropy
**is** maximum likelihood estimation.

## Gradient of the Loss

The gradient $\frac{d\mathcal{L}}{d\hat{p}_c} = -\frac{1}{\hat{p}_c}$ is large when
$\hat{p}_c$ is small — the model gets a strong training signal when it is very wrong:

```rustlab
grad_at_low  = 1.0 / 0.01;
grad_at_high = 1.0 / 0.99;
```

At $p{=}0.01$, $|dL/dp_c| = ${grad_at_low:%.2f}$; at $p{=}0.99$,
$|dL/dp_c| = ${grad_at_high:%.4f}$ — roughly a
${grad_at_low / grad_at_high:%.0f}$× difference in gradient magnitude between
confident-wrong and confident-right predictions. This steep penalty forces the
model to avoid confidently wrong predictions.

---

## Relationship to Entropy and KL Divergence

Cross-entropy decomposes as:

$$H(\mathbf{y}, \hat{\mathbf{p}}) = H(\mathbf{y}) + D_{\text{KL}}(\mathbf{y} \| \hat{\mathbf{p}})$$

For a one-hot $\mathbf{y}$, $H(\mathbf{y}) = 0$, so minimising cross-entropy
directly minimises the KL divergence — pushing the model's distribution toward the
true data distribution.

---

## Key Takeaways

- Cross-entropy loss $\mathcal{L} = -\log \hat{p}_c$ is the standard training
  objective for language models.
- The loss is convex and unbounded as $\hat{p}_c \to 0$ — the model is penalised
  exponentially for near-zero probability on the correct token.
- Cross-entropy $\neq$ accuracy. A model can be 51% accurate while having high loss
  if the remaining 49% concentrates on one wrong token.
- Training a language model *is* maximum likelihood estimation.

---

← [Lesson 02 — Probability & Softmax](02-probability-and-softmax.md) · Next: [Lesson 04 — Embeddings & Similarity](04-embeddings-and-similarity.md) →
