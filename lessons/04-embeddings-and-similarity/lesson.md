# Lesson 04 — Embeddings & Similarity

## Learning Objectives

By the end of this lesson you will be able to:

- Explain why **dense embeddings** are preferred over one-hot vectors as token representations.
- Describe the **embedding matrix** $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$ and how a lookup operation works.
- Compute **cosine similarity** between two vectors and interpret the result geometrically.
- Read an **embedding matrix heatmap** and identify patterns in learned representations.
- Reason about what it means for two tokens to be "close" in embedding space.

---

## Background

This lesson assumes:

- One-hot encoding from Lesson 01 (tokens as sparse integer indices).
- Vector dot products and norms from linear algebra.
- The idea that a matrix-vector product $\mathbf{W}\mathbf{x}$ is a linear projection.

---

## Theory

### The Problem with One-Hot Vectors

One-hot vectors (Lesson 01) are orthogonal — every pair of tokens is equally "distant". A model operating on one-hot vectors cannot leverage any prior knowledge that `king` and `queen` are semantically closer to each other than to `table`. Furthermore, one-hot vectors have dimension $|\mathcal{V}|$ (potentially tens of thousands), which is expensive to process.

### The Embedding Matrix

The solution is to learn a **dense, low-dimensional representation** for each token. The **embedding matrix** is:

$$
\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}
$$

where $d \ll |\mathcal{V}|$ is the embedding dimension (e.g., $d = 64$ or $d = 512$). Each row $\mathbf{E}_i \in \mathbb{R}^d$ is the learned embedding vector for token $i$.

**Lookup as matrix multiplication.** For a one-hot vector $\mathbf{e}_i$ (Lesson 01):

$$
\mathbf{E}^\top \mathbf{e}_i = \mathbf{E}_i
$$

Multiplying by the one-hot vector selects row $i$ of the embedding matrix — it is a lookup operation expressed as a linear map. In practice, implementations use an index lookup directly (no explicit matrix multiply), but the linear algebra view is essential for understanding gradient flow during training.

For a sequence of $T$ tokens encoded as a matrix $\mathbf{X} \in \{0,1\}^{T \times |\mathcal{V}|}$ (Lesson 01):

$$
\mathbf{H} = \mathbf{X} \mathbf{E} \in \mathbb{R}^{T \times d}
$$

The result $\mathbf{H}$ is the **embedded sequence**: each row is the dense embedding of the corresponding token.

### Cosine Similarity

In embedding space, **geometric proximity** is the signal. Two tokens are semantically related if their embedding vectors point in the same direction. The standard measure is **cosine similarity**:

$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
$$

where $\|\mathbf{v}\| = \sqrt{\sum_i v_i^2}$ is the Euclidean (L2) norm.

- $\cos = 1$: vectors point in the same direction — maximally similar.
- $\cos = 0$: vectors are orthogonal — no linear relationship.
- $\cos = -1$: vectors point in opposite directions — maximally dissimilar.

Cosine similarity ignores vector magnitude and focuses only on direction, making it robust to tokens that appear at different frequencies (and thus may have different embedding magnitudes).

**Computing the full similarity matrix.** For a set of $N$ embedding vectors stacked as rows of $\mathbf{E} \in \mathbb{R}^{N \times d}$, normalise each row:

$$
\hat{\mathbf{E}}_i = \frac{\mathbf{E}_i}{\|\mathbf{E}_i\|}
$$

Then the pairwise cosine similarity matrix is:

$$
\mathbf{S} = \hat{\mathbf{E}} \, \hat{\mathbf{E}}^\top \in \mathbb{R}^{N \times N}
$$

Entry $S_{ij}$ is the cosine similarity between tokens $i$ and $j$. The diagonal is always 1.

### What Embeddings Learn

When trained on large corpora, embedding vectors organise themselves so that semantic relationships correspond to geometric ones. The classic example:

$$
\mathbf{E}_{\text{king}} - \mathbf{E}_{\text{man}} + \mathbf{E}_{\text{woman}} \approx \mathbf{E}_{\text{queen}}
$$

Linear arithmetic on embedding vectors captures analogy relationships. This emergent structure is not programmed — it arises from training the model to predict next tokens accurately. Dense embeddings are a **compressed summary of co-occurrence patterns** in language.

---

## Core Concepts

Embeddings are the first transformation inside every language model. Before any attention or feed-forward computation happens, each token is looked up in the embedding matrix and replaced with a dense vector. This vector is the model's entire representation of that token — all subsequent computation acts on it.

The embedding matrix is learned jointly with the rest of the model by gradient descent. At initialisation it is random (Lesson 06); by the end of training, semantically similar tokens have high cosine similarity. The embedding dimension $d$ is a critical hyperparameter: too small and the vectors cannot capture enough nuance; too large and the model has too many parameters to train efficiently.

**Common misconception:** Embeddings are not a fixed feature — they are parameters learned from data. The embedding matrix is updated on every training step, just like weight matrices in later layers.

---

## Simulations

### `embedding_matrix.r` — Embedding Matrix Heatmap

**What it computes:**
Initialises a random embedding matrix for a vocabulary of 8 tokens with embedding dimension 6. Prints the matrix and saves a heatmap. Demonstrates how a one-hot lookup selects a row.

**What to observe:**
- The heatmap shows one row per vocabulary token and one column per embedding dimension.
- Each row is a dense vector — no zeros enforced, unlike one-hot.
- After random initialisation all rows look similar (no structure). After training they would organise into meaningful clusters.

**Verify by hand:**
Multiply the one-hot vector for token 3 (`[0, 0, 1, 0, 0, 0, 0, 0]`) by the printed embedding matrix. Confirm the result equals row 3 of the matrix.

---

### `cosine_similarity.r` — Cosine Similarity Matrix

**What it computes:**
Defines 4 hand-crafted embedding vectors for conceptually related tokens (`king`, `queen`, `man`, `woman`) in a 4-dimensional space. Computes the $4 \times 4$ pairwise cosine similarity matrix and saves it as a heatmap.

**What to observe:**
- Diagonal entries are all 1.0 (a vector is perfectly similar to itself).
- `king`/`queen` similarity is high — both have a "royalty" dimension.
- `king`/`man` and `queen`/`woman` similarities reflect shared gender components.
- The matrix is symmetric: $S_{ij} = S_{ji}$.

**Verify by hand:**
Compute $\cos(\text{king}, \text{queen})$ by hand using the printed embedding vectors. Use the formula $\cos = (\mathbf{a} \cdot \mathbf{b}) / (\|\mathbf{a}\| \|\mathbf{b}\|)$.

---

## Exercises

1. **Embedding lookup.** If the embedding matrix has shape $|\mathcal{V}| \times d$, and you embed a sequence of $T$ tokens, what is the shape of the output $\mathbf{H}$? Express in terms of $T$, $|\mathcal{V}|$, and $d$.

2. **Parameter count.** How many learnable parameters does the embedding matrix have for $|\mathcal{V}| = 50{,}000$ and $d = 512$? Compare this to the parameters in one attention head (discussed in Lesson 08).

3. **Cosine symmetry.** Prove algebraically that $\cos(\mathbf{a}, \mathbf{b}) = \cos(\mathbf{b}, \mathbf{a})$. What does this say about the similarity matrix $\mathbf{S}$?

4. **Analogy arithmetic.** Using the vectors defined in `cosine_similarity.r`, compute `king - man + woman` and find which of the four defined tokens it is closest to (by cosine similarity). Does the result match `queen`?

5. **Effect of dimension.** Edit `embedding_matrix.r` to use $d = 2$ instead of $d = 6$. Plot the 8 token embeddings as 2D scatter points (use `savescatter`). After random initialisation, do any tokens cluster together? Why or why not?
