---
title: "Lesson 09 — Multi-Head Attention"
order: 9
---

# Lesson 09 — Multi-Head Attention

A single attention head ([Lesson 08](08-scaled-dot-product-attention.md)) produces *one* row-stochastic mixing matrix. Language has many kinds of relationships — syntactic, positional, coreferential — and a single head can only express one of them at a time. **Multi-head attention** runs $H$ heads in parallel, concatenates their outputs, and projects them back to model width.

---

## Per-Head Projections

For each head $h = 1, \dots, H$, project the input $\mathbf{X}$ through its own weight matrices:

$$\mathbf{Q}_h = \mathbf{X}\mathbf{W}_Q^h, \quad \mathbf{K}_h = \mathbf{X}\mathbf{W}_K^h, \quad \mathbf{V}_h = \mathbf{X}\mathbf{W}_V^h$$

A common choice is $d_k = d_v = d_{\text{model}}/H$: each head works on an $H$-th of the feature width, but sees the *full* input. Per-head attention follows Lesson 08 unchanged:

$$\mathbf{A}_h = \mathrm{softmax}\!\left(\frac{\mathbf{Q}_h \mathbf{K}_h^\top}{\sqrt{d_k}} + \mathbf{M}\right), \qquad \mathbf{O}_h = \mathbf{A}_h \mathbf{V}_h$$

The causal mask $\mathbf{M}$ is shared across heads — every head must respect the same "no peeking at the future" rule.

---

## Four Heads, Four Patterns

Here are four heads with hand-set $(\mathbf{Q}_h, \mathbf{K}_h)$ designed to reveal distinct patterns:

```rustlab
T = 6;
H = 4;
d_k = 2;
scale = 1.0 / sqrt(d_k);

NEG_INF = -1.0e9;
M = zeros(T, T);
for i = 1:T
  for j = (i + 1):T
    M(i, j) = NEG_INF;
  end
end

function A = causal_attention_weights(Q, K, scale, M, T)
  S = Q * K' * scale;
  S_masked = S + M;
  A = zeros(T, T);
  for t = 1:T
    row = softmax(S_masked(t));
    for j = 1:T
      A(t, j) = row(j);
    end
  end
end
```

**Head 1 — "look at the first token":** $\mathbf{K}_1$ has a strong feature only in row 1; every query asks for that feature.

```rustlab
K1 = zeros(T, d_k);
K1(1, 1) = 3.0;
Q1 = zeros(T, d_k);
for t = 1:T
  Q1(t, 1) = 1.0;
end
A1 = causal_attention_weights(Q1, K1, scale, M, T);
```

**Head 2 — "previous token":** Encode position on the unit circle so that $\mathbf{Q}_t \cdot \mathbf{K}_i$ peaks at $i = t-1$.

```rustlab
K2 = zeros(T, d_k);
Q2 = zeros(T, d_k);
for i = 1:T
  K2(i, 1) = 3.0 * cos(2.0 * pi * i / T);
  K2(i, 2) = 3.0 * sin(2.0 * pi * i / T);
end
for t = 1:T
  Q2(t, 1) = 3.0 * cos(2.0 * pi * (t - 1) / T);
  Q2(t, 2) = 3.0 * sin(2.0 * pi * (t - 1) / T);
end
A2 = causal_attention_weights(Q2, K2, scale, M, T);
```

**Head 3 — "self":** same encoding, but $\mathbf{Q}_t = \mathbf{K}_t$.

```rustlab
A3 = causal_attention_weights(K2, K2, scale, M, T);
```

**Head 4 — "uniform over past":** all scores zero → softmax produces uniform weights.

```rustlab
Q4 = zeros(T, d_k);
K4 = zeros(T, d_k);
A4 = causal_attention_weights(Q4, K4, scale, M, T);
```

<!-- hide -->
```rustlab
A4_row4 = A4(4);
```

Head 4 row 4 should equal $[0.25, 0.25, 0.25, 0.25, 0, 0]$ — the uniform case is exactly the Lesson 07 averaging matrix. Computed: ${A4_row4(1):%.3f}, ${A4_row4(2):%.3f}, ${A4_row4(3):%.3f}, ${A4_row4(4):%.3f}, ${A4_row4(5):%.3f}, ${A4_row4(6):%.3f}$.

### Plot all four heads

```rustlab
figure()
subplot(2, 2, 1)
imagesc(A1, "viridis")
title("Head 1 — first token")

subplot(2, 2, 2)
imagesc(A2, "viridis")
title("Head 2 — previous token")

subplot(2, 2, 3)
imagesc(A3, "viridis")
title("Head 3 — self")

subplot(2, 2, 4)
imagesc(A4, "viridis")
title("Head 4 — uniform")
```

All four are lower-triangular (same causal mask) with rows summing to 1 (softmax), yet they compute completely different mixings. Head 4 recovers exactly the uniform prefix average from [Lesson 07](07-context-and-naive-averaging.md) — attention is a strict generalisation.

---

## Concatenation and Output Projection

Per-head outputs $\mathbf{O}_h \in \mathbb{R}^{T \times d_v}$ are stacked along the feature axis and projected back to $d_{\text{model}}$:

$$\mathrm{Concat} = [\mathbf{O}_1, \mathbf{O}_2, \dots, \mathbf{O}_H] \in \mathbb{R}^{T \times H d_v}, \qquad \mathbf{O} = \mathrm{Concat} \cdot \mathbf{W}_O$$

When $d_v = d_{\text{model}}/H$ the concatenation already has the right shape, but $\mathbf{W}_O$ is essential: it mixes features *across* heads so later layers can combine what each head discovered.

Here's the full pipeline in compact form ($T=4$, $H=2$, $d_k=d_v=2$, $d_{\text{model}}=4$):

```rustlab
T2 = 4;
d_model = 4;
d_k2 = 2;
d_v = 2;
scale2 = 1.0 / sqrt(d_k2);

X = [ 1.0, 0.0, 0.0, 0.0;
      0.0, 1.0, 0.0, 0.0;
      0.0, 0.0, 1.0, 0.0;
      0.0, 0.0, 0.0, 1.0 ];

% Head 1: self-attention on cyclic position encoding
K1b = zeros(T2, d_k2);
for t = 1:T2
  K1b(t, 1) = 3.0 * cos(2.0 * pi * t / T2);
  K1b(t, 2) = 3.0 * sin(2.0 * pi * t / T2);
end
V1 = zeros(T2, d_v);
for t = 1:T2
  V1(t, 1) = X(t, 1);
  V1(t, 2) = X(t, 2);
end

% Head 2: uniform over past
Q2b = zeros(T2, d_k2);
K2b = zeros(T2, d_k2);
V2 = zeros(T2, d_v);
for t = 1:T2
  V2(t, 1) = X(t, 3);
  V2(t, 2) = X(t, 4);
end

M2 = zeros(T2, T2);
for i = 1:T2
  for j = (i + 1):T2
    M2(i, j) = NEG_INF;
  end
end

A1b = causal_attention_weights(K1b, K1b, scale2, M2, T2);
A2b = causal_attention_weights(Q2b, K2b, scale2, M2, T2);

O1 = A1b * V1;
O2 = A2b * V2;

% Concatenate along the feature axis
O_concat = [O1, O2];

% Output projection (hand-set permutation for a visible effect)
W_O = [ 0.0, 0.0, 1.0, 0.0;
        0.0, 0.0, 0.0, 1.0;
        1.0, 0.0, 0.0, 0.0;
        0.0, 1.0, 0.0, 0.0 ];

O = O_concat * W_O;
```

<!-- hide -->
```rustlab
concat_shape = size(O_concat);
out_shape    = size(O);
n_qkv = 3 * d_model * d_model;
n_wo  = d_model * d_model;
n_tot = n_qkv + n_wo;
```

Shapes: $\mathrm{Concat} \in \mathbb{R}^{${concat_shape(1)} \times ${concat_shape(2)}}$, $\mathbf{O} \in \mathbb{R}^{${out_shape(1)} \times ${out_shape(2)}}$.

### Plot

```rustlab
figure()
subplot(1, 2, 1)
imagesc(O_concat, "viridis")
title("Concat = [O_1, O_2]  (T × H*d_v)")

subplot(1, 2, 2)
imagesc(O, "viridis")
title("Final MHA output O = Concat * W_O")
```

---

## Parameter Count

Pack the per-head projections into three combined $d_{\text{model}} \times d_{\text{model}}$ matrices; each head uses a slice of width $d_k$. The total is:

$$\underbrace{3 d_{\text{model}}^2}_{\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V} + \underbrace{d_{\text{model}}^2}_{\mathbf{W}_O} \;=\; 4 d_{\text{model}}^2$$

For our toy $d_{\text{model}} = 4$ that is $3 \cdot 16 + 16 = ${n_tot}$ parameters.

**$H$ does not appear.** More heads at fixed $d_{\text{model}}$ means narrower heads ($d_k = d_{\text{model}}/H$), not more parameters. Head *count* is an architectural choice; head *width* is what determines capacity.

---

## Key Takeaways

- **Why multiple heads:** one softmax-weighted sum can only express one kind of relationship; $H$ heads can express $H$.
- **Per-head:** run Lesson-08 attention with its own $\mathbf{W}_Q^h, \mathbf{W}_K^h, \mathbf{W}_V^h$.
- **Concat + project:** stack head outputs along the feature axis, then project with $\mathbf{W}_O$ to mix them.
- **Parameter count:** $4 d_{\text{model}}^2$ — independent of the number of heads when $d_k = d_{\text{model}}/H$.
- **Uniform averaging (Lesson 07) is a special case:** a degenerate head with zero $\mathbf{Q}$ and $\mathbf{K}$ produces exactly the prefix-average matrix.

This completes the attention mechanism. The next phase (Lessons 10–12) builds the three supporting pieces that surround attention in a transformer block: positional encoding, the MLP feed-forward sublayer, and layer normalisation with residuals.

---

← [Lesson 08 — Scaled Dot-Product Attention](08-scaled-dot-product-attention.md)
