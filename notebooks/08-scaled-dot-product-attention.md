---
title: "Lesson 08 — Scaled Dot-Product Attention"
order: 8
---

# Lesson 08 — Scaled Dot-Product Attention

[Lesson 07](07-context-and-naive-averaging.md) rewrote causal prefix averaging as a matrix multiply $\bar{\mathbf{X}} = \mathbf{W}\mathbf{X}$ where $\mathbf{W}$ was lower-triangular with fixed $1/t$ weights. Attention keeps the same skeleton but makes the weights **learned** and **data-dependent**. This lesson derives the full operation:

$$\mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V}) \;=\; \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

---

## Queries, Keys, Values

Given an input $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$, three learned linear projections produce three matrices:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

- $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $\mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$

Each token emits three vectors: a **query** (what it's looking for), a **key** (what it advertises), a **value** (what it contributes if selected).

```rustlab
T = 5;
d_k = 4;
scale = 1.0 / sqrt(d_k);

% Hand-crafted Q, K designed so the attention pattern is interpretable.
% K row 1 is a "keyword" feature; Q rows 2 and 4 both want it.
K = [ 2.0, 0.0, 0.0, 0.0;
      0.0, 2.0, 0.0, 0.0;
      0.0, 0.0, 2.0, 0.0;
      1.0, 1.0, 0.0, 0.0;
      0.0, 0.0, 1.0, 1.0 ];

Q = [ 1.0, 0.0, 0.0, 0.0;
      1.0, 0.0, 0.0, 0.0;
      0.0, 1.0, 0.0, 0.0;
      1.0, 0.0, 0.0, 0.0;
      0.5, 0.5, 0.0, 0.0 ];
```

## The Score Matrix

Entry $(t, i)$ of $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top$ is the dot product $\mathbf{q}_t \cdot \mathbf{k}_i$ — a raw similarity between query $t$ and key $i$.

### Why divide by $\sqrt{d_k}$?

If $\mathbf{q}$ and $\mathbf{k}$ each have $d_k$ i.i.d. components with mean 0 and variance 1, then $\mathbf{q}\cdot\mathbf{k} = \sum_{j=1}^{d_k} q_j k_j$ has variance $d_k$, and so standard deviation $\sqrt{d_k}$. As $d_k$ grows, raw dot products grow too, softmax becomes razor-sharp, and gradients vanish. Scaling by $1/\sqrt{d_k}$ keeps the input to softmax at roughly unit variance regardless of dimension.

```rustlab
S = Q * K' * scale;
```

## The Causal Mask

For a language model, token $t$ must not see tokens at positions $i > t$. We add a mask $\mathbf{M}$ with entries $-\infty$ above the diagonal and 0 elsewhere:

$$M_{t, i} = \begin{cases} 0 & i \le t \\ -\infty & i > t \end{cases}$$

In practice we use a large negative value (e.g. $-10^9$) to avoid `NaN` from $\exp(-\infty)$.

```rustlab
NEG_INF = -1.0e9;
M = zeros(T, T);
for i = 1:T
  for j = (i + 1):T
    M(i, j) = NEG_INF;
  end
end
S_masked = S + M;
```

## Row-wise Softmax → Attention Weights

Applying softmax to each row turns the scores into a probability distribution. The $-10^9$ entries become exactly 0.

$$A_{t, i} = \frac{\exp(\tilde S_{t, i})}{\sum_{j=1}^{T} \exp(\tilde S_{t, j})}$$

```rustlab
A = zeros(T, T);
for t = 1:T
  row = softmax(S_masked(t));
  for j = 1:T
    A(t, j) = row(j);
  end
end
```

<!-- hide -->
```rustlab
row_sums = zeros(T);
for t = 1:T
  row_sums(t) = sum(A(t));
end
A_1_1 = A(1, 1);

max_upper = 0.0;
for i = 1:T
  for j = (i + 1):T
    if A(i, j) > max_upper
      max_upper = A(i, j);
    end
  end
end
```

Token 1 can only attend to itself, so $A_{1,1} = ${A_1_1:%.4f}$. Every row sums to 1. The maximum attention weight in the upper triangle is ${max_upper:%.2e}$ — effectively zero, as required for causality.

### The three-stage pipeline

```rustlab
figure()
subplot(3, 1, 1)
imagesc(S, "viridis")
title("Scaled scores S = Q K^T / sqrt(d_k)")

subplot(3, 1, 2)
imagesc(S_masked, "viridis")
title("After causal mask (upper triangle → -∞)")

subplot(3, 1, 3)
imagesc(A, "viridis")
title("Attention weights A = softmax_row(S_masked)")
```

The final attention matrix is **lower-triangular** (causality) with **rows summing to 1** (softmax) — exactly the same shape as the Lesson 07 averaging matrix, but now the weights depend on the content of $\mathbf{Q}$ and $\mathbf{K}$.

---

## Full Pipeline: X → Q, K, V → Output

Now wire it all together. We add a value matrix $\mathbf{V}$ and produce $\mathbf{O} = \mathbf{A}\mathbf{V}$.

```rustlab
d_model = 6;
d_v = 4;

% Input sequence (T × d_model)
X = [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
      0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
      0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
      0.5, 0.5, 0.0, 0.0, 0.0, 0.0;
      0.0, 0.0, 0.0, 1.0, 1.0, 1.0 ];

% Hand-set projection matrices (d_model × d_k/d_v)
W_Q = [ 1.0, 0.0, 0.0, 0.0;
        0.0, 1.0, 0.0, 0.0;
        0.0, 0.0, 1.0, 0.0;
        0.0, 0.0, 0.0, 1.0;
        0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0 ];

W_K = W_Q;

W_V = [ 1.0, 0.0, 0.0, 0.0;
        0.0, 1.0, 0.0, 0.0;
        0.0, 0.0, 1.0, 0.0;
        0.0, 0.0, 0.0, 1.0;
        1.0, 1.0, 0.0, 0.0;
        0.0, 0.0, 1.0, 1.0 ];

Q2 = X * W_Q;
K2 = X * W_K;
V2 = X * W_V;
```

Compute attention weights and output:

```rustlab
S2 = Q2 * K2' * scale;
M2 = zeros(T, T);
for i = 1:T
  for j = (i + 1):T
    M2(i, j) = NEG_INF;
  end
end
S2_masked = S2 + M2;

A2 = zeros(T, T);
for t = 1:T
  row = softmax(S2_masked(t));
  for j = 1:T
    A2(t, j) = row(j);
  end
end

O = A2 * V2;
```

<!-- hide -->
```rustlab
diff_row1 = max(abs(O(1) - V2(1)));
n_params_qkv = 3 * d_model * d_k;
```

Row 1 of $\mathbf{O}$ equals row 1 of $\mathbf{V}$ (token 1 only attends to itself): max|O(1) - V(1)| = ${diff_row1:%.2e}$. The block's learnable parameters are $3 \cdot d_{\text{model}} \cdot d_k = 3 \cdot ${d_model}$ \cdot ${d_k}$ = ${n_params_qkv}$ — and this count does **not** depend on sequence length $T$.

```rustlab
figure()
subplot(2, 1, 1)
imagesc(A2, "viridis")
title("Attention weights A")
subplot(2, 1, 2)
imagesc(O, "viridis")
title("Output O = A V")
```

---

## Connection to Lesson 07

Attention and the Lesson 07 averaging matrix have the **same shape**:

- Lower-triangular (causal).
- Rows sum to 1.
- $\mathbf{O} = \mathbf{A}\mathbf{V}$ (attention) vs $\bar{\mathbf{X}} = \mathbf{W}\mathbf{X}$ (averaging) — a weighted sum of past tokens.

If $\mathbf{W}_Q = \mathbf{W}_K = \mathbf{0}$ then every score is 0, softmax produces uniform weights over the first $t$ positions, and the output reduces to the Lesson 07 prefix average of the value vectors. Attention is a **strict generalisation** of uniform averaging.

---

## Key Takeaways

- **Queries, keys, values** are three linear projections of the same input $\mathbf{X}$ — self-attention means they all come from the same sequence.
- The **score matrix** $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top / \sqrt{d_k}$ measures query-key similarity; the $1/\sqrt{d_k}$ scale keeps variance stable as $d_k$ grows.
- The **causal mask** zeros out future positions so position $t$ can only look at tokens $1..t$. Without it, language-model training would leak future tokens.
- **Softmax is row-wise** — each row of $\mathbf{A}$ is a probability distribution over the first $t$ tokens.
- The **parameter count** of one self-attention block ($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$) is $3 \cdot d_{\text{model}} \cdot d_k$, independent of sequence length.
- This block is the heart of every transformer. [Lesson 09](09-multi-head-attention.md) runs several of these in parallel and explains why.

---

← [Lesson 07 — Context and Naive Averaging](07-context-and-naive-averaging.md) · [Lesson 09 — Multi-Head Attention](09-multi-head-attention.md) →
