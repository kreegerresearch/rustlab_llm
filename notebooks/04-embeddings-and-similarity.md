# Lesson 04 — Embeddings & Similarity

One-hot vectors are orthogonal — every pair of tokens is equally "distant". This
lesson introduces **dense embeddings**: learned, low-dimensional vectors where
geometric proximity encodes semantic similarity.

---

## The Embedding Matrix

The **embedding matrix** maps each token from a sparse one-hot representation to a
dense vector:

$$\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$$

where $d \ll |\mathcal{V}|$ is the embedding dimension. Row $\mathbf{E}_i$ is the
learned embedding for token $i$.

**Lookup as matrix multiplication.** For a one-hot vector $\mathbf{e}_i$ (Lesson 01):

$$\mathbf{E}^\top \mathbf{e}_i = \mathbf{E}_i$$

For a sequence $\mathbf{X} \in \{0,1\}^{T \times |\mathcal{V}|}$:

$$\mathbf{H} = \mathbf{X} \mathbf{E} \in \mathbb{R}^{T \times d}$$

Let's initialise a random embedding matrix and verify the lookup:

```rustlab
vocab_size = 8;
d_embed = 6;

% Random initialisation (scaled by 0.1 — standard practice)
E = randn(vocab_size, d_embed) * 0.1;

print("Embedding matrix E  (shape: vocab_size x d_embed):");
print(size(E));
print(E);
```

```rustlab
% Embedding lookup via one-hot multiply
% Token index 3 → one-hot e3 = [0,0,1,0,0,0,0,0]
% h = e3 * E  selects row 3

e3 = [0, 0, 1, 0, 0, 0, 0, 0];
h3 = e3 * E;

print("One-hot vector for token 3:");
print(e3);
print("Embedded representation (= row 3 of E):");
print(h3);
print("Row 3 of E directly:");
print(E(3));

% Max difference should be ~0 (machine epsilon)
diff = max(abs(h3 - E(3)));
print("Max difference (should be ~0):", diff);
```

At random initialisation all rows look similar. After training, semantically related
tokens would cluster together:

```rustlab
saveimagesc(E, "outputs/embedding_matrix.svg", "Embedding Matrix E  (8 tokens x 6 dims)  - random init", "viridis")
print("Saved outputs/embedding_matrix.svg")
```

---

## Cosine Similarity

In embedding space, **direction** is the signal. Two tokens are related if their
vectors point the same way. The measure is **cosine similarity**:

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}$$

| Value | Meaning |
|-------|---------|
| $\cos = 1$ | Same direction — maximally similar |
| $\cos = 0$ | Orthogonal — no linear relationship |
| $\cos = -1$ | Opposite directions — maximally dissimilar |

The full pairwise similarity matrix for $N$ embeddings is:

$$\mathbf{S} = \hat{\mathbf{E}} \, \hat{\mathbf{E}}^\top$$

where $\hat{\mathbf{E}}_i = \mathbf{E}_i / \|\mathbf{E}_i\|$ are the row-normalised
embeddings.

### Worked example: king, queen, man, woman

Let's use hand-crafted embeddings with dimensions encoding
$[\text{royalty}, \text{femininity}, \text{age}, \text{authority}]$:

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
print("king  / queen (high — both royal):", s_kq);
print("king  / man   (moderate — same gender):", s_km);
print("queen / woman (moderate — same gender):", s_qw);

% Verify symmetry: S_ij == S_ji
sym_err = max(reshape(abs(S - transpose(S)), 1, 16));
print("Symmetry check max|S - S'| (should be ~0):", sym_err);
```

```rustlab
saveimagesc(S, "outputs/cosine_similarity.svg", "Cosine Similarity: king, queen, man, woman", "viridis")
print("Saved outputs/cosine_similarity.svg")
```

---

## Analogy Arithmetic

Trained embeddings organise so that semantic relationships correspond to geometric
ones. The classic example:

$$\mathbf{E}_{\text{king}} - \mathbf{E}_{\text{man}} + \mathbf{E}_{\text{woman}} \approx \mathbf{E}_{\text{queen}}$$

```rustlab
analogy = king - man + woman;
print("king - man + woman:", analogy);

sim_to_king  = cos_sim(analogy, king);
sim_to_queen = cos_sim(analogy, queen);
sim_to_man   = cos_sim(analogy, man);
sim_to_woman = cos_sim(analogy, woman);

print("Similarity of (king - man + woman) to:");
print("  king :", sim_to_king);
print("  queen:", sim_to_queen);
print("  man  :", sim_to_man);
print("  woman:", sim_to_woman);
print("Closest token should be 'queen'.");
```

This emergent structure is not programmed — it arises from training the model to
predict next tokens. Dense embeddings are a compressed summary of co-occurrence
patterns in language.

---

## Key Takeaways

- Embeddings are the first transformation inside every language model: one-hot $\to$
  dense vector via the embedding matrix $\mathbf{E}$.
- The embedding matrix is **learned** jointly with the rest of the model by gradient
  descent (Lesson 06). At initialisation it is random; after training, similar tokens
  cluster.
- Cosine similarity measures direction, not magnitude — robust to frequency
  differences between tokens.
- The embedding dimension $d$ is a critical hyperparameter: too small and the vectors
  lack nuance; too large and the model is expensive to train.
