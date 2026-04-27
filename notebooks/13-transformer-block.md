# Lesson 13: The Transformer Block

You now have every piece a transformer needs: multi-head attention ([Lesson 09](09-multi-head-attention.md)), positional encoding ([Lesson 10](10-positional-encoding.md)), the feed-forward sublayer ([Lesson 11](11-feed-forward-block.md)), and LayerNorm + residual connections ([Lesson 12](12-layer-norm-and-residuals.md)). This lesson assembles them into the **transformer block** — the unit that gets stacked $N$ times to form a full GPT.

## Learning Objectives

- Write the **Pre-LN transformer block** forward pass as a single equation and identify each sublayer.
- Trace the $(T, d_{\text{model}})$ tensor shape through every operation in the block and confirm input shape equals output shape.
- Implement one full block end-to-end with multi-head attention, two residuals, two LayerNorms, and an FFN.
- Stack two blocks and verify the residual stream's magnitude stays bounded.
- Compute the **parameter count per block** and identify which sublayer dominates.

## Background

Multi-head attention from [Lesson 09](09-multi-head-attention.md). FFN with $d_{\text{ff}} = 4 d_{\text{model}}$ from [Lesson 11](11-feed-forward-block.md). LayerNorm and the Pre-LN residual convention from [Lesson 12](12-layer-norm-and-residuals.md). Causal mask from [Lesson 08](08-scaled-dot-product-attention.md). No new mathematics — only assembly.

## Pre-LN Block: The Forward Pass

### Theory

The modern (Pre-LN) transformer block is two sublayers, each wrapped in LayerNorm-then-sublayer-then-residual:

$$\begin{aligned}
\mathbf{H}_{\text{mid}} &= \mathbf{H}_{\text{in}} + \mathrm{MHA}\!\left(\mathrm{LN}_1(\mathbf{H}_{\text{in}})\right), \\
\mathbf{H}_{\text{out}} &= \mathbf{H}_{\text{mid}} + \mathrm{FFN}\!\left(\mathrm{LN}_2(\mathbf{H}_{\text{mid}})\right).
\end{aligned}$$

Read it as a flow:

```
H_in ──┬─→ LN1 ──→ MHA ──┐
       │                 ↓
       └─────────────── (+) ──┬─→ LN2 ──→ FFN ──┐
                              │                 ↓
                              └─────────────── (+) ──→ H_out
```

The two `+` are the **residual additions** ([Lesson 12](12-layer-norm-and-residuals.md)). The unmixed `H_in` flows along the bottom of the diagram all the way to `H_out`; each sublayer adds its correction. This is the "residual stream" picture from [Lesson 12](12-layer-norm-and-residuals.md), now realised concretely.

Every tensor along the residual stream has shape $(T, d_{\text{model}})$. The intermediate matrices $\mathrm{LN}_1(\mathbf{H}_{\text{in}})$, $\mathrm{MHA}(\dots)$, $\mathrm{LN}_2(\mathbf{H}_{\text{mid}})$, $\mathrm{FFN}(\dots)$ all share that shape too. Inside MHA the projections temporarily produce $(T, d_k)$ per-head matrices, but the concat + output projection $\mathbf{W}_O$ collapses them back to $(T, d_{\text{model}})$. Inside FFN the hidden layer briefly widens to $(T, d_{\text{ff}})$ before the second linear projection reduces it. **The residual stream's width is invariant** — that's what makes blocks stackable without intervening reshape logic.

### Example — Configure dimensions and initialise weights

Use a small but realistic config for the worked example: $T = 4$, $d_{\text{model}} = 8$, $H = 2$ heads of $d_k = 4$ each, $d_{\text{ff}} = 32$.

```rustlab
seed(13);
T = 4;
d_model = 8;
H_heads = 2;
d_k = d_model / H_heads;     % 4
d_ff = 4 * d_model;          % 32
NEG_INF = -1.0e9;

H_in = randn(T, d_model);

% Multi-head attention weights — packed as full d_model × d_model matrices.
% Each head will read a contiguous d_k-wide slice of the columns.
W_Q = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_K = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_V = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_O = randn(d_model, d_model) * (1.0 / sqrt(d_model));

% FFN weights with He init for the GELU pre-activation
W_ff1 = randn(d_model, d_ff) * sqrt(2.0 / d_model);
W_ff2 = randn(d_ff,    d_model) * sqrt(2.0 / d_ff);

print("H_in shape:                      ", size(H_in));
print("W_{Q,K,V,O} shape (each):        ", size(W_Q));
print("W_ff1 shape (d_model -> d_ff):   ", size(W_ff1));
print("W_ff2 shape (d_ff -> d_model):   ", size(W_ff2));
```

LayerNorm in this notebook uses $\boldsymbol{\gamma} = \mathbf{1}, \boldsymbol{\beta} = \mathbf{0}$ — pure standardisation, no learned affine. Real implementations carry $(\boldsymbol{\gamma}, \boldsymbol{\beta})$ as small parameter vectors; we add them to the parameter count later but skip them in the forward pass for clarity.

## Sublayer 1: Pre-LN Multi-Head Self-Attention

### Theory

The first sublayer is

$$\mathbf{A} = \mathrm{MHA}\!\left(\mathrm{LN}_1(\mathbf{H}_{\text{in}})\right), \qquad \mathbf{H}_{\text{mid}} = \mathbf{H}_{\text{in}} + \mathbf{A}.$$

Pre-LN means LayerNorm runs *before* the projections — the inputs to $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ are already normalised, so the dot-product scores stay in a sensible range from the very first iteration. The $1/\sqrt{d_k}$ scale ([Lesson 08](08-scaled-dot-product-attention.md)) handles the rest.

The MHA computation is exactly Lesson 09's: per head, score → mask → softmax → weighted sum → concat → $\mathbf{W}_O$. The novelty here is just plumbing it inside a residual block.

### Example — LN1, then per-head Q/K/V, attention, concat, project

```rustlab
% Per-row LayerNorm of the residual stream — see AGENTS.md Rustlab
% Recommendations on the matrix overload.
H_norm1 = zeros(T, d_model);
for t = 1:T
  H_norm1(t) = layernorm(H_in(t));
end

% Project: same equations as Lesson 08, but applied to the LN'd input
Q = H_norm1 * W_Q;       % (T, d_model)
K = H_norm1 * W_K;
V = H_norm1 * W_V;

% Causal mask, shared across heads
M_mask = zeros(T, T);
for i = 1:T
  for j = (i + 1):T
    M_mask(i, j) = NEG_INF;
  end
end

scale = 1.0 / sqrt(d_k);
out_concat = zeros(T, d_model);

for h = 1:H_heads
  c_lo = (h - 1) * d_k + 1;     % first column belonging to head h

  % Slice columns out of Q, K, V into per-head matrices.
  Q_h = zeros(T, d_k);
  K_h = zeros(T, d_k);
  V_h = zeros(T, d_k);
  for t = 1:T
    for k = 1:d_k
      Q_h(t, k) = Q(t, c_lo + k - 1);
      K_h(t, k) = K(t, c_lo + k - 1);
      V_h(t, k) = V(t, c_lo + k - 1);
    end
  end

  S = Q_h * K_h' * scale + M_mask;
  A_h = zeros(T, T);
  for t = 1:T
    A_h(t) = softmax(S(t));
  end
  O_h = A_h * V_h;

  % Write head h's output into its slice of the concat
  for t = 1:T
    for k = 1:d_k
      out_concat(t, c_lo + k - 1) = O_h(t, k);
    end
  end
end

A_out = out_concat * W_O;     % (T, d_model)
print("LN1(H_in) shape:                 ", size(H_norm1));
print("MHA output shape:                ", size(A_out));
```

Both LN1 and the MHA output are $(T, d_{\text{model}}) = (4, 8)$ — same as the input. The per-head $(T, d_k)$ matrices live only inside the loop.

### Example — First residual addition

```rustlab
H_mid = H_in + A_out;
print("H_mid shape (after first residual):", size(H_mid));
print("H_mid row 1 (first 4 values):     ", H_mid(1, 1), H_mid(1, 2), H_mid(1, 3), H_mid(1, 4));
```

The residual restores the original $\mathbf{H}_{\text{in}}$ signal that LN1 had standardised away. Future blocks will read $\mathbf{H}_{\text{mid}}$ — the *unnormalised* residual stream — and apply LN to it again from scratch.

## Sublayer 2: Pre-LN Feed-Forward

### Theory

Same wrapper, different sublayer:

$$\mathbf{F} = \mathrm{FFN}\!\left(\mathrm{LN}_2(\mathbf{H}_{\text{mid}})\right), \qquad \mathbf{H}_{\text{out}} = \mathbf{H}_{\text{mid}} + \mathbf{F}.$$

FFN applies $\mathbf{W}_2 \, \mathrm{GELU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$ to every row independently ([Lesson 11](11-feed-forward-block.md)). Biases omitted here for clarity.

### Example — LN2, FFN, second residual

```rustlab
H_norm2 = zeros(T, d_model);
for t = 1:T
  H_norm2(t) = layernorm(H_mid(t));
end

F_pre  = H_norm2 * W_ff1;     % (T, d_ff)
F_post = gelu(F_pre);         % (T, d_ff), still
F_out  = F_post * W_ff2;      % (T, d_model)

H_out = H_mid + F_out;        % residual addition

print("LN2(H_mid) shape:                ", size(H_norm2));
print("FFN hidden  shape (T, d_ff):     ", size(F_pre));
print("FFN output  shape (T, d_model):  ", size(F_out));
print("Block output H_out shape:        ", size(H_out));
```

The FFN hidden temporarily widens to $(T, d_{\text{ff}}) = (4, 32)$, then contracts back to $(T, d_{\text{model}}) = (4, 8)$. **The block's input and output have identical shape**, $(4, 8)$. Stacking blocks just means feeding `H_out` as the next block's `H_in`.

### Example — Magnitude tracking through the block

```rustlab
print("|H_in|  =", norm(H_in));
print("|H_mid| =", norm(H_mid));
print("|H_out| =", norm(H_out));
print("|MHA contribution|  / |H_in|  =", norm(A_out) / norm(H_in));
print("|FFN contribution|  / |H_mid| =", norm(F_out) / norm(H_mid));
```

At random initialisation each sublayer adds a small perturbation to the residual stream — ratios are typically well below 1, which is why a deep stack does not blow up at init. The training process scales these contributions up where they help and down where they don't.

### Example — Visualise the residual stream at each stage

```rustlab
figure()
subplot(3, 1, 1)
imagesc(H_in,  "viridis")
title("H_in (T=4, d_model=8)")

subplot(3, 1, 2)
imagesc(H_mid, "viridis")
title("H_mid (after MHA + residual)")

subplot(3, 1, 3)
imagesc(H_out, "viridis")
title("H_out (after FFN + residual)")
```

Each row of each panel is one token's representation; each column is one feature dimension. The structure barely changes between panels — that is the residual stream doing its job. Zoom in and you can see the FFN and MHA each nudge specific cells, but the dominant pattern carried by $\mathbf{H}_{\text{in}}$ persists.

## Stacking Two Blocks

### Theory

A real GPT stacks $N$ identical blocks. "Identical" means same architecture, **different parameters per block**. Each block has its own $\mathbf{W}_Q^{(\ell)}, \mathbf{W}_K^{(\ell)}, \mathbf{W}_V^{(\ell)}, \mathbf{W}_O^{(\ell)}, \mathbf{W}_1^{(\ell)}, \mathbf{W}_2^{(\ell)}$ for $\ell = 1, \dots, N$. The residual stream threads through every block, so block $\ell$'s output becomes block $\ell+1$'s input — both at width $d_{\text{model}}$.

### Example — Build a second set of weights and run the stack

```rustlab
% Block 2 weights — different seed, same architecture
seed(14);
W_Q2 = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_K2 = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_V2 = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_O2 = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_ff1_2 = randn(d_model, d_ff) * sqrt(2.0 / d_model);
W_ff2_2 = randn(d_ff,    d_model) * sqrt(2.0 / d_ff);

% Run the same block forward pass on H_out as input
H_in2 = H_out;
H_norm1b = zeros(T, d_model);
for t = 1:T
  H_norm1b(t) = layernorm(H_in2(t));
end

Q2 = H_norm1b * W_Q2;
K2 = H_norm1b * W_K2;
V2 = H_norm1b * W_V2;

out_concat2 = zeros(T, d_model);
for h = 1:H_heads
  c_lo = (h - 1) * d_k + 1;
  Q_h = zeros(T, d_k);
  K_h = zeros(T, d_k);
  V_h = zeros(T, d_k);
  for t = 1:T
    for k = 1:d_k
      Q_h(t, k) = Q2(t, c_lo + k - 1);
      K_h(t, k) = K2(t, c_lo + k - 1);
      V_h(t, k) = V2(t, c_lo + k - 1);
    end
  end
  S = Q_h * K_h' * scale + M_mask;
  A_h = zeros(T, T);
  for t = 1:T
    A_h(t) = softmax(S(t));
  end
  O_h = A_h * V_h;
  for t = 1:T
    for k = 1:d_k
      out_concat2(t, c_lo + k - 1) = O_h(t, k);
    end
  end
end
A_out2 = out_concat2 * W_O2;
H_mid2 = H_in2 + A_out2;

H_norm2b = zeros(T, d_model);
for t = 1:T
  H_norm2b(t) = layernorm(H_mid2(t));
end
F_pre2  = H_norm2b * W_ff1_2;
F_out2  = gelu(F_pre2) * W_ff2_2;

H_out2 = H_mid2 + F_out2;

print("After block 1 |H| =", norm(H_out));
print("After block 2 |H| =", norm(H_out2));
print("After block 2 shape:", size(H_out2));
```

After two blocks the residual stream's magnitude is the same order as the input — residuals + identity-dominant init keep the stack stable. Real transformers go to $N = 12$ (GPT-2 small), $N = 96$ (GPT-3) or beyond using the same recipe.

## Parameter Count per Block

### Theory

Sum the learnable matrices and bias vectors:

| Component | Params |
|---|---|
| MHA: $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O$ | $4 d_{\text{model}}^2$ |
| FFN: $\mathbf{W}_1$ ($d_{\text{model}} \to d_{\text{ff}}$) + $\mathbf{W}_2$ ($d_{\text{ff}} \to d_{\text{model}}$) at $d_{\text{ff}} = 4 d_{\text{model}}$ | $8 d_{\text{model}}^2$ |
| FFN biases $\mathbf{b}_1, \mathbf{b}_2$ | $d_{\text{ff}} + d_{\text{model}} = 5 d_{\text{model}}$ |
| LayerNorm $(\boldsymbol{\gamma}_1, \boldsymbol{\beta}_1, \boldsymbol{\gamma}_2, \boldsymbol{\beta}_2)$ | $4 d_{\text{model}}$ |

Total (ignoring small lower-order terms): $\boxed{12 d_{\text{model}}^2 + O(d_{\text{model}})}$.

The $4 d_{\text{model}}^2$ for attention vs. $8 d_{\text{model}}^2$ for FFN means **FFN holds twice the parameters of attention** in every block — confirmed in [Lesson 11](11-feed-forward-block.md). The number of heads $H$ does not appear: more heads at fixed $d_{\text{model}}$ just slices the same matrices into narrower per-head views.

### Example — Block parameter count

```rustlab
n_attn = 4 * d_model * d_model;
n_ffn  = 2 * d_model * d_ff;            % W_ff1 + W_ff2 (no bias)
n_bias = d_ff + d_model;                % FFN biases (LN affine ignored — γ=1, β=0 here)
n_block = n_attn + n_ffn + n_bias;

print("d_model:        ", d_model);
print("d_ff:           ", d_ff);
print("Attention params:", n_attn);
print("FFN matmul params:", n_ffn);
print("FFN biases:     ", n_bias);
print("Total per block:", n_block);
print("FFN / attention:", n_ffn / n_attn);
```

For our toy $d_{\text{model}} = 8$ the block has ${n_block} parameters total. Scale to $d_{\text{model}} = 384$ (nanoGPT-small) and the block has ~1.77M parameters — multiplied by $N$ blocks in [Lesson 14](14-full-gpt-architecture.md).

## Connection to Information Theory

Each block performs two information operations on the residual stream:

1. **Attention extracts a fragment of $I(X_{t+1}; X_{1..t})$.** Per the [Lesson 08](08-scaled-dot-product-attention.md) framing, MHA's softmax-weighted sum re-weights past values toward those that carry mutual information about the next token. Multi-head splits that extraction into $H$ orthogonal channels ([Lesson 09](09-multi-head-attention.md)).
2. **FFN re-shapes the resulting representation.** No new mutual information about $X_{t+1}$ is introduced (the FFN sees only the current position), but the geometric arrangement of the per-token features is reshaped non-linearly so the *next* attention layer can extract a different fragment.

The residual stream is the **lossless backbone** ([Lesson 12](12-layer-norm-and-residuals.md)): $\mathbf{H}_{\text{in}}$'s information is preserved through `H_in + sublayer(LN(H_in))` no matter what the sublayer does. Stacking $N$ blocks therefore composes $N$ rounds of "extract MI → re-shape representation" without ever discarding what earlier rounds learned. The model's capacity to lower the cross-entropy loss ([Lesson 03](03-cross-entropy-loss.md)) grows roughly linearly in depth, until other constraints (gradient noise, parameter sharing, dataset size) intervene.

A useful framing: a transformer is a **conditional entropy refinement pipeline**. The bigram baseline ([Lesson 05](05-bigram-language-model.md)) gives one number per row: $H(X_{t+1} \mid X_t)$. Each transformer block lowers that conditional entropy by the amount of information it can extract from the larger context. The stack converges (in the limit of unlimited training and capacity) toward the true entropy $H(X_{t+1} \mid X_{1..t})$ of the language.

## Key Takeaways

- The Pre-LN transformer block is `H + MHA(LN(H))` followed by `H' + FFN(LN(H'))` — two residual sublayers around the persistent residual stream.
- Tensor shape is $(T, d_{\text{model}})$ at every step of the residual stream; FFN widens to $(T, d_{\text{ff}})$ internally and contracts back.
- Blocks stack trivially because input and output shapes match. Each block has its own parameters but identical architecture.
- Per-block parameter count: $4 d_{\text{model}}^2$ (attention) + $8 d_{\text{model}}^2$ (FFN) + small bias / LN terms.
- FFN dominates the parameter budget at $4\times$ widening; attention dominates the compute at long context.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `block_forward.r` | one Pre-LN transformer block end-to-end with $T = 4, d_{\text{model}} = 8, H = 2$; prints shape at every step |
| `two_block_stack.r` | runs the same block twice with different weights; prints residual-stream magnitude after each block |

Run all with `make lesson-13` (or `rustlab run lessons/13-transformer-block/<name>.r`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `size(H_in)` | `[4, 8]` |
| `size(H_norm1)` | `[4, 8]` |
| `size(A_out)` (MHA output) | `[4, 8]` |
| `size(F_pre)` (FFN hidden) | `[4, 32]` |
| `size(F_out)` | `[4, 8]` |
| `size(H_out)` | `[4, 8]` (= input shape — block is shape-preserving) |
| `n_attn` ($d_{\text{model}} = 8$) | `256` |
| `n_ffn` | `512` |
| `n_ffn / n_attn` | `2` |

## Exercises

1. **Shape check across configs.** Re-run the block with $T = 16, d_{\text{model}} = 64, H = 8$. List the shape of $\mathbf{H}_{\text{in}}, \mathbf{H}_{\text{mid}}, \mathbf{H}_{\text{out}}, \mathbf{F}_{\text{pre}}$, $\mathbf{F}_{\text{out}}$. Which intermediate shapes change with $H$? Which don't?
2. **Without residuals.** Remove the two `+` operations in the block (just use `H_mid = A_out` and `H_out = F_out`). Run two blocks in a row. How does $\|\mathbf{H}_{\text{out}}\|$ compare to $\|\mathbf{H}_{\text{in}}\|$? Connect to [Lesson 12](12-layer-norm-and-residuals.md)'s magnitude collapse demo.
3. **Post-LN variant.** Rewrite the block as `H_mid = LN(H_in + MHA(H_in))` and `H_out = LN(H_mid + FFN(H_mid))` (Post-LN). Confirm the final shape is unchanged. Why might this version need a learning-rate warmup that Pre-LN doesn't?
4. **Scaling laws.** Compute the per-block parameter count for $d_{\text{model}} \in \{64, 128, 256, 512, 1024\}$. Plot it. The $d_{\text{model}}^2$ scaling is the reason transformer compute and memory grow so steeply with width — sketch why.
5. **Bias accounting.** Add the LayerNorm affine parameters $(\boldsymbol{\gamma}, \boldsymbol{\beta})$ for both LN1 and LN2 to the count. By how much does the per-block total change? At what $d_{\text{model}}$ do these become negligible compared to the $12 d_{\text{model}}^2$ matrix budget?

## What's next

Lesson 14 wraps the block into the **full GPT decoder**: a token embedding ([Lesson 04](04-embeddings-and-similarity.md)), positional encoding ([Lesson 10](10-positional-encoding.md)), $N$ stacked transformer blocks (this lesson), a final LayerNorm, and a language-modelling head that projects the residual stream to vocabulary logits. The lesson prints the parameter count of every component for a small GPT config and confirms the breakdown matches a known reference. After Lesson 14 you have every piece needed to *train* the architecture — Phases 6 onward fill in the training loop, optimizer, and evaluation.
