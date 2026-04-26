# Lesson 07: Context and Naive Averaging

The bigram model ([Lesson 05](05-bigram-language-model.md)) predicts the next token from a single previous token. That's a structural limit — no amount of data can make it context-aware. Fixing this means letting the prediction depend on *all* earlier tokens. The simplest aggregator is the **prefix average**. Writing it as a matrix multiply is the bridge to attention ([Lesson 08](08-scaled-dot-product-attention.md)).

## Learning Objectives

- Explain why a **bigram** (Markov-1) language model is context-free beyond a single previous token, and identify the failure mode on ambiguous tokens.
- Define **prefix averaging** — the simplest context-aware aggregation — as $\bar{\mathbf{x}}_t = \frac{1}{t}\sum_{i=1}^{t} \mathbf{x}_i$.
- Rewrite that sum as a **matrix multiplication** $\bar{\mathbf{X}} = \mathbf{W}\mathbf{X}$ where $\mathbf{W}$ is a lower-triangular averaging matrix.
- Read the lower-triangular causal structure of $\mathbf{W}$ and explain why it enforces the "no peek at the future" rule.
- Articulate the remaining weakness of uniform averaging — all past tokens get equal weight — that motivates **attention** (Lesson 08).

## Background

Bigram language models and row-normalised probability matrices from [Lesson 05](05-bigram-language-model.md). Linear layers and matrix multiplication from [Lesson 06](06-linear-layers-and-gradient-descent.md). Embeddings as dense row vectors from [Lesson 04](04-embeddings-and-similarity.md). No new mathematics — only rearranging sums into matrix form.

## Where the Bigram Fails

### Theory

Consider two sentences sharing an ambiguous token:

- `river bank water`
- `money bank safe`

`bank` is the same token in both, but its continuation depends on the history. A bigram sees only the previous token, so both histories collapse to the same row of $P$. This section's H2s pair `### Theory` with `### Example — <descriptor>`.

### Example — P(next | bank) collapses to a 50/50

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

The bigram's continuation distribution after `bank` is

$$P(\text{next} \mid \text{bank}) = [0, 0, 0.5, 0, 0.5].$$

P(water | bank) = ${p_after_bank(3):%.2f}$, P(safe | bank) = ${p_after_bank(5):%.2f}$ — whether the history was `river bank` or `money bank`. The model has no mechanism to tell them apart.

### Example — Bar of P(next | bank)

```rustlab
figure()
labels = {"river", "bank", "water", "money", "safe"};
bar(labels, p_after_bank)
title("P(next | bank) — identical regardless of prior context")
ylabel("Probability")
ylim([0, 1])
```

This is a structural failure, not a data failure.

## The Fix: Summarise the Whole Prefix

### Theory

If the prediction depends only on the *last* token, it can't see prior context. So let the state at position $t$ summarise *every* earlier token. The simplest summary is the average

$$\bar{\mathbf{x}}_t \;=\; \frac{1}{t}\sum_{i=1}^{t} \mathbf{x}_i.$$

Each $\bar{\mathbf{x}}_t$ is a "bag of past tokens" — every prior embedding blended in with equal weight.

## Rewrite as a Matrix Multiply

### Theory

Stack embeddings into $\mathbf{X} \in \mathbb{R}^{T \times d}$ and define a **lower-triangular** averaging matrix $\mathbf{W}$:

$$\mathbf{W}_{t, i} = \begin{cases} \frac{1}{t} & i \le t \\ 0 & i > t. \end{cases}$$

Then the entire stack of prefix averages is a single matrix multiply

$$\bar{\mathbf{X}} \;=\; \mathbf{W}\mathbf{X}.$$

The upper triangle is zero — this is the **causal** constraint: position $t$ cannot see a future token.

### Example — Build W and the input embeddings X

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

For $T = ${T}$, $\mathbf{W}$ is shown below.

### Example — Causal averaging matrix W heatmap

```rustlab
figure()
imagesc(W, "viridis")
title("Causal Averaging Matrix W — row t = 1/t, zero above diagonal")
```

Row $t$ has $t$ non-zero entries each equal to $1/t$, so every row sums to 1.

### Example — Loop vs. matmul agreement

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

### Example — X vs. X̄ side by side

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

## The Remaining Weakness

### Theory

Uniform $1/t$ weights treat every past token as equally informative. In a sentence like *"the cat sat on the mat, it was soft"*, the pronoun *it* refers to a specific token (`mat`), not the average of everything before it. We need **data-dependent** weights — weights that concentrate mass on the relevant tokens based on what each token actually says.

**Information-theoretic framing.** Let $X_t$ be the next token and $X_{1..t-1}$ the prior context. Each prior token $X_i$ carries some **mutual information** $I(X_t ; X_i \mid X_{<i})$ about $X_t$ — and that quantity varies hugely across positions. For *"the cat sat on the mat, it was soft"*, $I(X_{\text{soft}} ; X_{\text{mat}})$ is large (mat is what *it* refers to) while $I(X_{\text{soft}} ; X_{\text{the}})$ is essentially zero. Uniform averaging — assigning weight $1/t$ to every prior token — wastes capacity on low-information tokens and dilutes the high-information ones, raising the conditional entropy $H(X_t \mid \bar{\mathbf{x}}_t)$ above what a smart, data-dependent aggregator could achieve. From the [Lesson 05](05-bigram-language-model.md) entropy-chain view, every dropped bit of mutual information is a dropped bit of compressibility.

That is exactly what **attention** does. It replaces the fixed $1/t$ entries of $\mathbf{W}$ with $\mathrm{softmax}(\mathrm{scores})$, where the scores depend on the query and key vectors derived from the embeddings themselves — so the model can route weight toward whichever past tokens carry the most information about the next. The lower-triangular (causal) structure and per-row weighted sum stay identical.

## Key Takeaways

- A bigram's next-token distribution depends only on the previous token; histories that share the same last token are indistinguishable.
- The simplest context-aware summary is a **prefix average** over all tokens $1..t$.
- That average is a single matrix multiply $\bar{\mathbf{X}} = \mathbf{W}\mathbf{X}$ with $\mathbf{W}$ lower-triangular and each row normalised to sum to 1.
- The lower-triangular shape encodes **causality** — position $t$ cannot look at the future.
- Uniform averaging throws away the signal about *which* past tokens matter most — the next lesson replaces it with learned, data-dependent weights.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `context_failure.r` | bigram count + probability matrix on the `bank` corpus; bar of $P(\text{next} \mid \text{bank})$ |
| `prefix_averaging.r` | prefix averages two ways (loop and matrix multiply); heatmaps of $\mathbf{W}$, $\mathbf{X}$, $\bar{\mathbf{X}}$ |

Run all with `make lesson-07` (or `rustlab run lessons/07-context-and-naive-averaging/<name>.r`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `p_after_bank(3)` ($P(\text{water}\mid\text{bank})$) | `0.50` |
| `p_after_bank(5)` ($P(\text{safe}\mid\text{bank})$) | `0.50` |
| `W(1, 1)` | `1.0` |
| `W(2, 1)`, `W(2, 2)` | `0.5` each |
| `W(t, i)` for $i \le t$ | `1/t` |
| `W(t, i)` for $i > t$ | `0` |
| `sum(W(t))` (each row) | `1.0` |
| `diff` (loop vs matmul) | ≈ `0` (machine epsilon) |

## Exercises

1. **Extending the ambiguity.** Modify `context_failure.r` to add a third sentence `river bank fish`. What does $P(\text{next} \mid \text{bank})$ become? How does adding more data *not* fix the underlying structural problem?
2. **Exponential moving average.** Replace the uniform $1/t$ weights in $\mathbf{W}$ with exponentially decaying weights $w_{t,i} \propto \gamma^{t-i}$ (with $\gamma = 0.8$), normalised to sum to 1 per row. How does the averaged output differ? What kind of bias does this introduce?
3. **Non-causal averaging.** What matrix $\mathbf{W}'$ would average *all* tokens (past and future) uniformly at every position? Write it out for $T=4$. Why is this wrong for language modelling but fine for, e.g., sentence classification?
4. **Counting operations.** For a sequence of length $T$ and embedding dimension $d$, how many scalar multiplies does $\mathbf{W}\mathbf{X}$ take? How does this scale with $T$? (Hint: count the non-zero entries of $\mathbf{W}$.)
5. **Preview of attention.** In attention (Lesson 08), the weights in row $t$ of $\mathbf{W}$ are replaced by $\mathrm{softmax}(\mathbf{q}_t \mathbf{K}^\top / \sqrt{d_k})$, still with a causal mask. What property of softmax guarantees each row still sums to 1?

## What's next

Lesson 08 replaces the fixed $1/t$ weights with **learned, content-dependent** ones derived from query and key projections of the embeddings. The matrix shape and the causal mask stay; only the entries change — and that change is the entire mechanism behind modern transformer attention.
