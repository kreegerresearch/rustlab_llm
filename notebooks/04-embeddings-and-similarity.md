# Lesson 04: Embeddings & Similarity

One-hot vectors are orthogonal — every pair of tokens is equally "distant". This lesson introduces **dense embeddings**: learned, low-dimensional vectors where geometric proximity encodes semantic similarity.

## Learning Objectives

- Explain why **dense embeddings** are preferred over one-hot vectors as token representations.
- Describe the **embedding matrix** $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$ and how a lookup operation works.
- Compute **cosine similarity** between two vectors and interpret the result geometrically.
- Read an **embedding matrix heatmap** and identify patterns in learned representations.
- Reason about what it means for two tokens to be "close" in embedding space.

## Background

One-hot encoding from [Lesson 01](01-tokens-and-encoding.md) (tokens as sparse integer indices). Vector dot products and norms from linear algebra. The idea that a matrix-vector product $\mathbf{W}\mathbf{x}$ is a linear projection.

## The Problem with One-Hot Vectors

One-hot vectors (Lesson 01) are orthogonal — every pair of tokens is equally "distant". A model operating on one-hot vectors cannot leverage any prior knowledge that `king` and `queen` are semantically closer to each other than to `table`. They also have dimension $|\mathcal{V}|$ (potentially tens of thousands), which is expensive to process. This section is pure motivation — every later H2 pairs `### Theory` with `### Example — <descriptor>`.

## The Embedding Matrix

### Theory

The fix is to learn a **dense, low-dimensional representation** for each token. The **embedding matrix** is

$$\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d},$$

where $d \ll |\mathcal{V}|$ is the embedding dimension (e.g., $d = 64$ or $d = 512$). Each row $\mathbf{E}_i \in \mathbb{R}^d$ is the learned embedding vector for token $i$.

**Lookup as matrix multiplication.** For a one-hot vector $\mathbf{e}_i$:

$$\mathbf{E}^\top \mathbf{e}_i = \mathbf{E}_i.$$

Multiplying by the one-hot vector selects row $i$ — a lookup expressed as a linear map. In practice, implementations index directly (no explicit matrix multiply), but the linear-algebra view is essential for understanding gradient flow during training.

For a sequence of $T$ tokens encoded as a matrix $\mathbf{X} \in \{0,1\}^{T \times |\mathcal{V}|}$:

$$\mathbf{H} = \mathbf{X} \mathbf{E} \in \mathbb{R}^{T \times d}.$$

The result $\mathbf{H}$ is the **embedded sequence**: each row is the dense embedding of the corresponding token.

### Example — Building a deterministic 8×6 embedding matrix

In a real model these values come from `randn(vocab_size, d_embed) * 0.1`; here we use a deterministic sin/cos recipe so the rendered notebook reproduces bit-for-bit (rustlab v0.1.11 has no RNG seed — see `AGENTS.md` Rustlab Recommendations).

```rustlab
vocab_size = 8;
d_embed = 6;

% Deterministic pseudo-random init.
% TODO: replace with `randn(vocab_size, d_embed) * 0.1` once rustlab adds an RNG seed API.
[II, JJ] = meshgrid(1:d_embed, 1:vocab_size);
E = 0.1 * (sin(2.7 * II + 1.3 * JJ) + cos(1.7 * II - 0.9 * JJ + 0.4));

print("Embedding matrix E:");
print(E);
```

Shape: ${size(E, 1)} $\times$ ${size(E, 2)} — one row per token in a ${d_embed}-dimensional embedding space.

### Example — One-hot lookup recovers a row

```rustlab
% Token index 3 → one-hot e3 selects row 3 via  h = e3 * E
e3 = [0, 0, 1, 0, 0, 0, 0, 0];
h3 = e3 * E;

diff = max(abs(h3 - E(3)));
print("Embedded representation h3:", h3);
print("Row 3 of E:", E(3));
```

The lookup matches the direct row access exactly — $\max|h_3 - E_3| = ${diff:%.2e}$ (machine epsilon).

### Example — Embedding matrix heatmap

At random initialisation all rows look similar. After training, semantically related tokens would cluster together:

```rustlab
figure()
imagesc(E, "viridis")
title("Embedding Matrix E  (8 tokens x 6 dims)  - random init")
```

## Cosine Similarity

### Theory

In embedding space, **direction** is the signal. Two tokens are related if their vectors point the same way. The measure is **cosine similarity**:

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}.$$

| Value | Meaning |
|-------|---------|
| $\cos = 1$ | Same direction — maximally similar |
| $\cos = 0$ | Orthogonal — no linear relationship |
| $\cos = -1$ | Opposite directions — maximally dissimilar |

Cosine similarity ignores vector magnitude and focuses only on direction, making it robust to tokens that appear at different frequencies (and thus may have different embedding magnitudes).

The full pairwise similarity matrix for $N$ embeddings stacked as rows of $\mathbf{E}$ is

$$\mathbf{S} = \hat{\mathbf{E}} \, \hat{\mathbf{E}}^\top \in \mathbb{R}^{N \times N}, \qquad \hat{\mathbf{E}}_i = \frac{\mathbf{E}_i}{\|\mathbf{E}_i\|}.$$

Entry $S_{ij}$ is the cosine similarity between tokens $i$ and $j$. The diagonal is always 1.

### Example — Hand-crafted king/queen/man/woman vectors

Use hand-crafted embeddings with dimensions encoding $[\text{royalty}, \text{femininity}, \text{age}, \text{authority}]$:

```rustlab
king  = [1.0,  0.1,  0.8,  0.9];
queen = [0.9,  0.9,  0.7,  0.8];
man   = [0.1,  0.1,  0.6,  0.4];
woman = [0.1,  0.9,  0.5,  0.3];

print("Embedding vectors (dim=4):");
print("king :", king);
print("queen:", queen);
print("man  :", man);
print("woman:", woman);

function s = cos_sim(a, b)
  s = sum(a .* b) / (sqrt(sum(a .^ 2)) * sqrt(sum(b .^ 2)))
end
```

### Example — 4×4 cosine-similarity matrix

```rustlab
% Compute the full 4x4 similarity matrix
s_kk = cos_sim(king,  king);
s_kq = cos_sim(king,  queen);
s_km = cos_sim(king,  man);
s_kw = cos_sim(king,  woman);
s_qk = cos_sim(queen, king);
s_qq = cos_sim(queen, queen);
s_qm = cos_sim(queen, man);
s_qw = cos_sim(queen, woman);
s_mk = cos_sim(man,   king);
s_mq = cos_sim(man,   queen);
s_mm = cos_sim(man,   man);
s_mw = cos_sim(man,   woman);
s_wk = cos_sim(woman, king);
s_wq = cos_sim(woman, queen);
s_wm = cos_sim(woman, man);
s_ww = cos_sim(woman, woman);

S = [s_kk, s_kq, s_km, s_kw; s_qk, s_qq, s_qm, s_qw; s_mk, s_mq, s_mm, s_mw; s_wk, s_wq, s_wm, s_ww];

print("Cosine similarity matrix (king, queen, man, woman):");
print(S);

% Verify symmetry: S_ij == S_ji
sym_err = max(reshape(abs(S - transpose(S)), 1, 16));
```

Key pairs: king/queen = ${s_kq:%.3f}$ (both royal), king/man = ${s_km:%.3f}$ (same gender), queen/woman = ${s_qw:%.3f}$ (same gender). The matrix is symmetric: $\max|S - S^\top| = ${sym_err:%.2e}$.

### Example — Similarity heatmap

```rustlab
figure()
imagesc(S, "viridis")
title("Cosine Similarity: king, queen, man, woman")
```

## Analogy Arithmetic

### Theory

Trained embeddings organise so that semantic relationships correspond to geometric ones. The classic example:

$$\mathbf{E}_{\text{king}} - \mathbf{E}_{\text{man}} + \mathbf{E}_{\text{woman}} \approx \mathbf{E}_{\text{queen}}.$$

### Example — Closest token to king − man + woman

```rustlab
analogy = king - man + woman;
print("king - man + woman:", analogy);

sim_to_king  = cos_sim(analogy, king);
sim_to_queen = cos_sim(analogy, queen);
sim_to_man   = cos_sim(analogy, man);
sim_to_woman = cos_sim(analogy, woman);
```

Similarity of $\mathbf{E}_{\text{king}} - \mathbf{E}_{\text{man}} + \mathbf{E}_{\text{woman}}$ to each vocab item: king = ${sim_to_king:%.3f}$, **queen = ${sim_to_queen:%.3f}**, man = ${sim_to_man:%.3f}$, woman = ${sim_to_woman:%.3f}$. The closest token is **queen**, as predicted. This emergent structure is not programmed — it arises from training the model to predict next tokens accurately. Dense embeddings are a compressed summary of co-occurrence patterns in language.

## Key Takeaways

- Embeddings are the first transformation inside every language model: one-hot $\to$ dense vector via the embedding matrix $\mathbf{E}$.
- The embedding matrix is **learned** jointly with the rest of the model by gradient descent ([Lesson 06](06-linear-layers-and-gradient-descent.md)). At initialisation it is random; after training, similar tokens cluster.
- Cosine similarity measures direction, not magnitude — robust to frequency differences between tokens.
- The embedding dimension $d$ is a critical hyperparameter: too small and the vectors lack nuance; too large and the model is expensive to train.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `embedding_matrix.r` | random `8 × 6` embedding matrix; one-hot lookup demo; heatmap |
| `cosine_similarity.r` | the 4-token king/queen/man/woman cosine-similarity matrix; analogy arithmetic |

Run all with `make lesson-04` (or `rustlab run lessons/04-embeddings-and-similarity/<name>.r`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `size(E)` | `[8, 6]` |
| `diff` (`h3 − E(3)`) | ≈ `0` (machine epsilon) |
| `s_kq` (king/queen) | ≈ `0.951` |
| `s_km` (king/man) | ≈ `0.881` |
| `s_qw` (queen/woman) | ≈ `0.881` |
| `s_kk` (king/king) | `1.000` |
| `sym_err` ($\max|S - S^\top|$) | ≈ `0` (machine epsilon) |
| `sim_to_queen` (analogy → queen) | ≈ `0.998` (closest match) |

## Exercises

1. **Embedding lookup.** If the embedding matrix has shape $|\mathcal{V}| \times d$, and you embed a sequence of $T$ tokens, what is the shape of the output $\mathbf{H}$? Express in terms of $T$, $|\mathcal{V}|$, and $d$.
2. **Parameter count.** How many learnable parameters does the embedding matrix have for $|\mathcal{V}| = 50{,}000$ and $d = 512$? Compare this to the parameters in one attention head (Lesson 08).
3. **Cosine symmetry.** Prove algebraically that $\cos(\mathbf{a}, \mathbf{b}) = \cos(\mathbf{b}, \mathbf{a})$. What does this say about the similarity matrix $\mathbf{S}$?
4. **Analogy arithmetic.** Recompute `king - man + woman` and find which of the four defined tokens it is closest to (by cosine similarity). Does the result match `queen`?
5. **Effect of dimension.** Edit `embedding_matrix.r` to use $d = 2$ instead of $d = 6$. Plot the 8 token embeddings as 2D scatter points. After random initialisation, do any tokens cluster together? Why or why not?

## What's next

Lesson 05 builds the first **language model** of the series: a count-based bigram model that learns next-token probabilities from a corpus and samples text from them. The embedding-style lookup table from this lesson reappears, this time storing transition probabilities rather than learned vectors.
