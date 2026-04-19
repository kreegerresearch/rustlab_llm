---
title: "Lesson 07 — Context and Naive Averaging"
order: 7
---

# Lesson 07 — Context and Naive Averaging

The bigram model ([Lesson 05](05-bigram-language-model.md)) predicts the next token from a single previous token. That's a structural limit — no amount of data can make it context-aware. Fixing this means letting the prediction depend on *all* earlier tokens. The simplest aggregator is the **prefix average**. Writing it as a matrix multiply is the bridge to attention (Lesson 08).

---

## Where the Bigram Fails

Consider two sentences sharing an ambiguous token:

- `river bank water`
- `money bank safe`

`bank` is the same token in both, but its continuation depends on the history. A bigram sees only the previous token, so both histories collapse to the same row of $P$.

```rustlab
vocab_size = 5;
% river=1, bank=2, water=3, money=4, safe=5

C = zeros(vocab_size, vocab_size);
C(1, 2) = 1;   % river -> bank
C(2, 3) = 1;   % bank  -> water
C(4, 2) = 1;   % money -> bank
C(2, 5) = 1;   % bank  -> safe

P = zeros(vocab_size, vocab_size);
for i = 1:vocab_size
  row_sum = sum(C(i));
  if row_sum > 0
    for j = 1:vocab_size
      P(i, j) = C(i, j) / row_sum;
    end
  end
end

p_after_bank = P(2);
```

The bigram's continuation distribution after `bank` is:

$$P(\text{next} \mid \text{bank}) = [0, 0, 0.5, 0, 0.5]$$

P(water | bank) = ${p_after_bank(3):%.2f}$, P(safe | bank) = ${p_after_bank(5):%.2f}$ — whether the history was `river bank` or `money bank`. The model simply has no mechanism to tell them apart.

```rustlab
figure()
labels = {"river", "bank", "water", "money", "safe"};
bar(labels, p_after_bank)
title("P(next | bank) — identical regardless of prior context")
ylabel("Probability")
ylim([0, 1])
```

This is a structural failure, not a data failure.

---

## The Fix: Summarise the Whole Prefix

If the prediction depends only on the *last* token, it can't see prior context. So let the state at position $t$ summarise *every* earlier token. The simplest summary is the average:

$$\bar{\mathbf{x}}_t \;=\; \frac{1}{t}\sum_{i=1}^{t} \mathbf{x}_i$$

Each $\bar{\mathbf{x}}_t$ is a "bag of past tokens" — every prior embedding blended in with equal weight.

---

## Rewrite as a Matrix Multiply

Stack embeddings into $\mathbf{X} \in \mathbb{R}^{T \times d}$ and define a **lower-triangular** averaging matrix $\mathbf{W}$:

$$\mathbf{W}_{t, i} = \begin{cases} \frac{1}{t} & i \le t \\ 0 & i > t \end{cases}$$

Then the entire stack of prefix averages is a single matrix multiply:

$$\bar{\mathbf{X}} \;=\; \mathbf{W}\mathbf{X}$$

The upper triangle is zero — this is the **causal** constraint: position $t$ cannot see a future token.

```rustlab
T = 6;
d = 4;

% Hand-crafted embeddings with distinct signatures
X = [ 1.0, 0.0, 0.0, 0.0;
      0.0, 1.0, 0.0, 0.0;
      0.0, 0.0, 1.0, 0.0;
      0.0, 0.0, 0.0, 1.0;
      1.0, 1.0, 0.0, 0.0;
      0.0, 0.0, 1.0, 1.0 ];

W = zeros(T, T);
for t = 1:T
  for i = 1:t
    W(t, i) = 1.0 / t;
  end
end
```

For $T = ${T}$, $\mathbf{W}$ is:

```rustlab
figure()
imagesc(W, "viridis")
title("Causal Averaging Matrix W — row t = 1/t, zero above diagonal")
```

Row $t$ has $t$ non-zero entries each equal to $1/t$, so every row sums to 1.

### Verify Against the Loop

The matrix multiply produces the same result as an explicit running-sum loop:

<!-- hide -->
```rustlab
X_bar_loop = zeros(T, d);
running = zeros(d);
for t = 1:T
  running = running + X(t);
  for k = 1:d
    X_bar_loop(t, k) = running(k) / t;
  end
end

X_bar_mm = W * X;
diff = max(reshape(abs(X_bar_loop - X_bar_mm), 1, T * d));
```

Max absolute difference between loop and matrix multiply: ${diff:%.2e}$ — identical to machine precision.

### The Averaged Sequence

```rustlab
figure()
subplot(2, 1, 1)
imagesc(X, "viridis")
title("Input Embeddings X (distinct per token)")

subplot(2, 1, 2)
imagesc(X_bar_mm, "viridis")
title("Prefix Averages X̄ = W*X (each row mixes all earlier tokens)")
```

$\bar{\mathbf{X}}$ is smoother than $\mathbf{X}$. Row $t$ of $\bar{\mathbf{X}}$ is a blend of the first $t$ rows of $\mathbf{X}$ — position $t$ now carries information from every earlier token.

---

## The Remaining Weakness

Uniform $1/t$ weights treat every past token as equally informative. In a sentence like *"the cat sat on the mat, it was soft"*, the pronoun *it* refers to a specific token (`mat`), not the average of everything before it. We need **data-dependent** weights — weights that concentrate mass on the relevant tokens based on what each token actually says.

That is exactly what **attention** does. It replaces the fixed $1/t$ entries of $\mathbf{W}$ with $\mathrm{softmax}(\mathrm{scores})$, where the scores depend on the query and key vectors derived from the embeddings themselves. Everything else — the lower-triangular (causal) structure, the per-row weighted sum — stays the same.

---

## Key Takeaways

- A bigram's next-token distribution depends only on the previous token; histories that share the same last token are indistinguishable.
- The simplest context-aware summary is a **prefix average** over all tokens $1..t$.
- That average is a single matrix multiply $\bar{\mathbf{X}} = \mathbf{W}\mathbf{X}$ with $\mathbf{W}$ lower-triangular and each row normalised to sum to 1.
- The lower-triangular shape encodes **causality** — position $t$ cannot look at the future.
- Uniform averaging throws away the signal about *which* past tokens matter most — the next lesson replaces it with learned, data-dependent weights.

---

← [Lesson 06 — Linear Layers & Gradient Descent](06-linear-layers-and-gradient-descent.md) · [Lesson 08 — Scaled Dot-Product Attention](08-scaled-dot-product-attention.md) →
