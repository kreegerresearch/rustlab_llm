# Lesson 10: Positional Encoding

Attention as built so far ([Lessons 07–09](07-context-and-naive-averaging.md)) is **permutation-invariant**: shuffle the input tokens and the output positions shuffle the same way — no information about *order* enters the computation. That is fatal for language: "dog bites man" and "man bites dog" use the same multiset of tokens with completely different meanings. Positional encoding is the fix — a deterministic vector added to each token's embedding that tags it with its position.

## Learning Objectives

- Show that scaled dot-product attention is permutation-equivariant and explain why this rules out language modelling without a positional signal.
- Write the **sinusoidal positional encoding** formula and identify the role of each variable.
- Compute the full $T \times d_{\text{model}}$ PE matrix and read its banded heatmap structure.
- Verify that the dot product $\mathrm{PE}_t \cdot \mathrm{PE}_{t+k}$ depends only on the offset $k$, not on the absolute position $t$ — the property that lets attention reason about relative position.
- Explain the trade-off between **fixed sinusoidal** PE and **learned** PE.

## Background

Scaled dot-product attention from [Lesson 08](08-scaled-dot-product-attention.md). Embeddings as dense rows from [Lesson 04](04-embeddings-and-similarity.md). Sine and cosine over arbitrary arguments. Mutual information from the [Lesson 07](07-context-and-naive-averaging.md) and [Lesson 08](08-scaled-dot-product-attention.md) info-theory framings.

## Why Attention is Order-Blind

### Theory

Let $\mathbf{P}$ be a $T \times T$ permutation matrix. If we permute the input rows, $\mathbf{X}' = \mathbf{P}\mathbf{X}$, then the projections become

$$\mathbf{Q}' = \mathbf{P}\mathbf{Q}, \qquad \mathbf{K}' = \mathbf{P}\mathbf{K}, \qquad \mathbf{V}' = \mathbf{P}\mathbf{V}.$$

The score matrix transforms as $\mathbf{S}' = \mathbf{Q}'\mathbf{K}'^\top = \mathbf{P}\mathbf{S}\mathbf{P}^\top$ — the same scores, just relabelled. After softmax and multiplying by $\mathbf{V}'$ the output is $\mathbf{O}' = \mathbf{P}\mathbf{O}$: the output rows are simply the permuted original rows. **No information about the original order survived.** This is what "permutation-equivariant" means and why a vanilla self-attention layer cannot learn that adjacent tokens are different from far-apart tokens.

The causal mask from [Lesson 08](08-scaled-dot-product-attention.md) breaks *full* permutation invariance — token $t$ only sees tokens $1..t$ — but it leaves the relative order *within* the prefix untouched. The model still has no way to distinguish "the cat sat" from "sat the cat" within the visible window. We need to inject position into the *features* themselves.

### Example — Permutation invariance at unit scale

Build a tiny attention block, run it on $\mathbf{X}$ and on a row-shuffled $\mathbf{X}'$, and check that the outputs differ only by the same permutation:

```rustlab
seed(10);
T = 4;
d_model = 4;
X = randn(T, d_model);
W_Q = randn(d_model, d_model) * 0.5;
W_K = randn(d_model, d_model) * 0.5;
W_V = randn(d_model, d_model) * 0.5;
scale = 1.0 / sqrt(d_model);

function O = attn(X, W_Q, W_K, W_V, scale)
  Q = X * W_Q;
  K = X * W_K;
  V = X * W_V;
  S = Q * K' * scale;
  T_local = size(S, 1);
  A = zeros(T_local, T_local);
  for t = 1:T_local
    row = softmax(S(t));
    for j = 1:T_local
      A(t, j) = row(j);
    end
  end
  O = A * V;
end

P = [0, 1, 0, 0; 0, 0, 0, 1; 1, 0, 0, 0; 0, 0, 1, 0];   % row permutation

O      = attn(X, W_Q, W_K, W_V, scale);
O_perm = attn(P * X, W_Q, W_K, W_V, scale);

perm_err = max(reshape(abs(O_perm - P * O), 1, T * d_model));
print("max |attn(P*X) - P*attn(X)| =", perm_err);
```

The discrepancy ${perm_err:%.2e}$ is at machine precision — attention is *exactly* equivariant under row permutation. Without a positional signal, "the cat sat" and any permutation of those three tokens produce indistinguishable hidden states.

## The Sinusoidal Positional Encoding

### Theory

The original transformer paper uses a fixed (non-learned) encoding built from sinusoids of geometrically spaced frequencies. For position $t \in \{1, \dots, T\}$ and embedding dimension $d_{\text{model}}$, the entry at column $i$ is

$$\mathrm{PE}_{t, 2k}   = \sin\!\left(\frac{t}{10000^{\,2k/d_{\text{model}}}}\right), \qquad
  \mathrm{PE}_{t, 2k+1} = \cos\!\left(\frac{t}{10000^{\,2k/d_{\text{model}}}}\right),$$

with $k = 0, 1, \dots, d_{\text{model}}/2 - 1$. The $i = 0$ pair has wavelength $2\pi$ (one full cycle every $\sim 6$ tokens); the highest-$i$ pair has wavelength $2\pi \cdot 10000$ (essentially constant over any practical sequence). The model gets fast and slow position clocks at every dimension pair, so it can read both fine-grained ("which token am I?") and coarse ("which half of the sequence am I in?") position from the same vector.

The encoding is *added* to the token embedding, not concatenated:

$$\mathbf{X}'_t \;=\; \mathbf{E}_{x_t} + \mathrm{PE}_t.$$

This works because the embedding norm is small ($\sim 0.1$ from initialisation) compared to the unit-amplitude PE — token identity dominates locally, position bends the geometry over the sequence.

### Example — Build the PE matrix for T=64, d=32

```rustlab
T_seq = 64;
d_model_pe = 32;

PE = zeros(T_seq, d_model_pe);
for t = 1:T_seq
  for i = 1:d_model_pe
    pair_idx = floor((i - 1) / 2);                            % k = 0,0,1,1,2,2,...
    div = 10000.0 ^ (2.0 * pair_idx / d_model_pe);
    angle = t / div;
    if mod(i - 1, 2) == 0
      PE(t, i) = sin(angle);
    else
      PE(t, i) = cos(angle);
    end
  end
end

print("PE shape:", size(PE));
print("PE(1, 1:6):", PE(1, 1:6));
print("PE(64, 1:6):", PE(64, 1:6));
```

Position 1 produces $[\sin(1), \cos(1), \sin(0.422), \cos(0.422), \dots]$; position 64 cycles much further around the fastest sinusoids while barely moving on the slowest ones.

### Example — Heatmap of the full PE matrix

```rustlab
figure()
imagesc(PE, "viridis")
title("Sinusoidal Positional Encoding (T=64, d=32)")
xlabel("Embedding dimension")
ylabel("Position t")
```

The horizontal bands on the right (high-$i$, slow sinusoids) barely change down the page. The left columns oscillate rapidly. Each row is a unique fingerprint, and the structure is smooth — nearby rows look similar, far-apart rows look different. That smoothness is what makes "relative position" a learnable feature.

## Why Sinusoids: Translation in Position is a Linear Map

### Theory

Pick any pair of dimensions $(2k, 2k+1)$ and any offset $\delta$. The $\sin$/$\cos$ angle-addition formulas give

$$\begin{pmatrix} \sin(\omega_k(t+\delta)) \\ \cos(\omega_k(t+\delta)) \end{pmatrix} \;=\;
\underbrace{\begin{pmatrix} \cos(\omega_k \delta) & \sin(\omega_k \delta) \\ -\sin(\omega_k \delta) & \cos(\omega_k \delta) \end{pmatrix}}_{R_k(\delta)}
\begin{pmatrix} \sin(\omega_k t) \\ \cos(\omega_k t) \end{pmatrix},$$

where $\omega_k = 1/10000^{2k/d_{\text{model}}}$. Translation by $\delta$ acts as a fixed rotation $R_k(\delta)$ on each $(2k, 2k+1)$ pair. The Q/K projection matrices the model learns can therefore implement "look $\delta$ tokens back" as a *linear* operation, independent of the absolute $t$. This is the property that lets attention generalise to sequence lengths it never saw at training time.

### Example — Dot-product similarity vs. distance

The similarity $\mathrm{PE}_t \cdot \mathrm{PE}_{t+k}$ depends only on $k$, not $t$ — verify it numerically:

```rustlab
sims_short = zeros(40);                        % similarity for offsets k=0..39 starting at t=10
sims_far   = zeros(40);                        % same offsets starting at t=20

for k = 0:39
  sims_short(k + 1) = sum(PE(10) .* PE(10 + k));
end
for k = 0:39
  sims_far(k + 1) = sum(PE(20) .* PE(20 + k));
end

drift = max(abs(sims_short - sims_far));
print("max |sim_t=10(k) - sim_t=20(k)| over k=0..39 =", drift);
```

Drift across base position $t$: ${drift:%.2e}$ — within a hair of zero. Whatever similarity we measure between two positions is a function of *only* their separation.

### Example — Plot the similarity-vs-distance curve

```rustlab
figure()
plot(0:39, sims_short, "color", "blue", "label", "PE_t · PE_{t+k}")
title("PE Dot-Product Similarity vs. Offset k")
xlabel("Offset k (tokens)")
ylabel("Dot product")
```

The curve peaks sharply at $k = 0$ (each PE has unit-ish self-similarity), decays through several oscillations, then settles. The decaying envelope is what lets attention treat "close" and "far" as different — without it, every position would look like every other.

## Adding PE to Embeddings

### Theory

In a real transformer block the embedded sequence is $\mathbf{H} = \mathbf{X}_{\text{onehot}} \mathbf{E} + \mathrm{PE}$ (with PE truncated to length $T$). The embedding magnitude is set by the random init scale ($\sim 0.1$), and PE has unit amplitude — so positional information dominates the representation when the token embedding is small, and token identity reasserts itself once the embedding is trained up. Some implementations multiply the embedding by $\sqrt{d_{\text{model}}}$ to balance the two; we skip that here for clarity.

### Example — Token + PE produces unique per-position vectors

```rustlab
seed(42);
vocab_pe = 6;
T_demo = 8;
E_pe = randn(vocab_pe, d_model_pe) * 0.1;

% Token sequence: alternating tokens 1 and 2 — without PE, every "1" position is identical
ids = [1, 2, 1, 2, 1, 2, 1, 2];
X_tok = zeros(T_demo, d_model_pe);
for t = 1:T_demo
  X_tok(t) = E_pe(ids(t));      % assign whole row from the embedding lookup
end

% PE(1:T_demo) returns a sub-matrix; use a loop to add it row-by-row to X_tok.
X_pos = zeros(T_demo, d_model_pe);
for t = 1:T_demo
  X_pos(t) = X_tok(t) + PE(t);
end

% Two positions of the SAME token: are their representations actually distinct?
diff_tok_only = max(abs(X_tok(1) - X_tok(3)));      % both rows are token 1, no PE → identical
diff_with_pe  = max(abs(X_pos(1) - X_pos(3)));      % same token, different positions → differ
print("max |row1 - row3| (token only):", diff_tok_only);
print("max |row1 - row3| (token + PE):", diff_with_pe);
```

Without PE, the token-1 rows at positions 1 and 3 are bit-identical — the model literally cannot tell them apart. Adding PE injects ${diff_with_pe:%.4f}$ worth of separation per dimension, and any downstream attention head can now compute features sensitive to *which* token-1 it is looking at.

## Sinusoidal vs. Learned Positional Encoding

### Theory

Two practical choices coexist in modern transformers:

| Choice | How it works | Pros | Cons |
|---|---|---|---|
| **Sinusoidal (fixed)** | Closed-form $\sin / \cos$ table, no parameters | Generalises to *any* sequence length, zero parameters | No way to adapt the encoding to the data |
| **Learned** | Add a separate `randn(T_max, d_model)` and train it like an embedding | Can specialise per dataset, marginally better at training-length contexts | Bounded by `T_max`; must extrapolate or interpolate to go longer |

GPT-2 and the original BERT use **learned** PE; the original Vaswani transformer and many recent long-context models (RoPE, ALiBi) use carefully chosen *fixed* schemes that share the sinusoidal property of "translation = linear". Either way, *some* positional signal must be added, and the rotation property derived above is the geometric reason fixed schemes work.

## Connection to Information Theory

Without positional encoding, the input distribution that attention sees is **invariant to permutation**: the multiset of tokens carries the same Shannon entropy regardless of order. But the conditional distribution over the next token, $P(X_{t+1} \mid X_{1..t})$, very much depends on order — "dog bites man" and "man bites dog" have different next-token distributions and therefore different conditional entropies.

The mismatch is information that the model literally cannot extract:

$$I(X_{t+1}; \mathrm{order \ of \ } X_{1..t} \mid \mathrm{multiset}) > 0$$

for almost any natural-language source, but a permutation-equivariant network maps every permutation to the same intermediate state — discarding all of it. Positional encoding is the device that makes that mutual information *recoverable* by attention. It does not add new bits about the corpus; it makes the bits already present in token *order* visible to the network. The sinusoidal scheme is, in this view, a fixed *minimum sufficient encoding* of position — enough for the model to undo the permutation-equivariance bottleneck, no more.

This is why removing positional encoding tanks language-modelling perplexity by a large factor while leaving bag-of-words tasks (sentence classification, topic detection) almost unchanged — the latter genuinely *don't* depend on order, so no information is lost when permutation-equivariance is in effect.

## Key Takeaways

- Self-attention is permutation-equivariant; without a positional signal it cannot distinguish word orders.
- **Sinusoidal PE** uses paired $\sin/\cos$ at geometrically spaced frequencies. Translation in position becomes a fixed linear map (rotation per pair), which lets attention represent relative offsets cleanly.
- Adding PE to the token embedding is the standard injection point. The dot product $\mathrm{PE}_t \cdot \mathrm{PE}_{t+k}$ depends only on $k$ — verified numerically.
- **Learned PE** is a viable alternative; it specialises to the data but is bounded by the maximum trained sequence length.
- Information-theoretically, PE makes the order-dependent component of $I(X_{t+1}; X_{1..t})$ recoverable. Permutation-equivariant networks systematically discard it.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `pe_matrix.r` | the $64 \times 32$ sinusoidal PE matrix; heatmap |
| `pe_translation.r` | similarity vs. offset curves at two base positions; verifies translation-only dependence |

Run all with `make lesson-10` (or `rustlab run lessons/10-positional-encoding/<name>.r`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `perm_err` ($\max\|\text{attn}(PX) - P\,\text{attn}(X)\|$) | ≈ `0` (machine epsilon) |
| `size(PE)` | `[64, 32]` |
| `PE(1, 1)` ($\sin(1)$) | ≈ `0.841` |
| `PE(1, 2)` ($\cos(1)$) | ≈ `0.540` |
| `drift` (translation invariance check) | ≈ `0` (machine epsilon) |
| `sims_short(1)` (= `PE(10) · PE(10)`) | $d_{\text{model}}/2 \cdot 1 = $ `16` |
| `diff_tok_only` | `0` (same token = same vector without PE) |
| `diff_with_pe` | > `0` (PE breaks the tie) |

## Exercises

1. **Frequency span.** What fraction of one full sine cycle does the *fastest* PE pair complete over $T = 64$? What fraction does the *slowest* pair complete? Confirm from the formula and the heatmap.
2. **Wavelength of pair $k$.** Show algebraically that the wavelength of the $k$-th sinusoid pair is $2\pi \cdot 10000^{2k/d_{\text{model}}}$. For $d_{\text{model}} = 32$, $k = 15$, what is the wavelength in tokens?
3. **Permutation breaking.** Modify the permutation-invariance demo to add `PE(1:T)` to `X` before applying `attn`. Re-check `O_perm` against `P * O` — does it still match? Explain why not.
4. **Replace with learned.** Sketch the parameter count for a learned PE table at `T_max = 1024` and `d_model = 512`. Is this larger or smaller than the token embedding for $|\mathcal{V}| = 50000$? What goes wrong if you ever feed a sequence longer than `T_max`?
5. **Order matters how much?** For a random transformer with random weights, compute the average per-token cosine similarity between `attn(X)` and `attn(P*X)` (after un-permuting). Without PE this should be 1.0; *with* PE it should be much less. The drop is a rough measure of how much information PE injects.

## What's next

Lesson 11 introduces the second sublayer of every transformer block: the **position-wise feed-forward network**. Each token's representation passes through the same two-layer MLP independently — a per-position non-linearity that gives the model the expressive power a stack of pure linear projections lacks. The non-linearity is **GELU**, a smooth cousin of ReLU, and we'll see why the smoothness matters for gradient flow.
