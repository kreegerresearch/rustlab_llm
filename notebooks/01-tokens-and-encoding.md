# Lesson 01: Tokens & Text Encoding

A language model never sees letters — it sees integers. This lesson builds the bridge from raw text to the numerical representations every later lesson depends on: vocabulary construction, integer indices, and one-hot vectors.

## Learning Objectives

- Explain what a **token** is and why text must be converted to numbers before a model can process it.
- Build a **character-level vocabulary** and map every token to a unique integer index.
- Construct a **one-hot vector** for a given token and interpret it geometrically.
- Read a **character frequency bar chart** and a **one-hot matrix heatmap** and explain what each cell means.
- State the trade-off between vocabulary size and representational expressiveness.

## Background

Vectors (a list of numbers) and matrices (a 2-D grid of numbers). The concept of a function as a deterministic input-to-output mapping. No neural network, probability, or deep learning knowledge is required yet — those arrive in later lessons.

## What is a Token?

A **token** is the basic unit of text a language model operates on. At the character level every character becomes one token. The pipeline has two steps. This section is pure reference — formal definitions and the vocabulary table; every later section pairs `### Theory` with `### Example — <descriptor>`.

**Step 1 — Collect the vocabulary.** Given a corpus, collect every unique character and sort them. This ordered set is the vocabulary $\mathcal{V}$, with size $|\mathcal{V}|$.

**Step 2 — Assign integer indices.** Create a deterministic mapping

$$\text{encode} : \mathcal{V} \to \{1, 2, \ldots, |\mathcal{V}|\}.$$

For the corpus `"to be or not to be"`:

| Character | Index |
|-----------|-------|
| ` ` (space) | 1 |
| `b` | 2 |
| `e` | 3 |
| `n` | 4 |
| `o` | 5 |
| `r` | 6 |
| `t` | 7 |

The string `"hello"` would encode to $[4, 3, 5, 5, 6]$ under a similar scheme. The inverse mapping $\text{decode} : \{1, \ldots, |\mathcal{V}|\} \to \mathcal{V}$ reconstructs text from integer sequences.

## Character Frequencies

### Theory

Before building a model it helps to understand the **frequency distribution** of tokens. If $c_i$ is the count of character $i$ in the corpus, the relative frequency is

$$f_i = \frac{c_i}{\sum_{j=1}^{|\mathcal{V}|} c_j}.$$

This is a discrete probability distribution over the vocabulary — every $f_i \in [0, 1]$ and $\sum_i f_i = 1$. High-frequency characters appear often, and a good model must predict them reliably.

### Example — Computing relative frequencies

<!-- hide -->
```rustlab
% Corpus: "to be or not to be" — vocabulary (sorted): ' ', b, e, n, o, r, t
chars = {" ", "b", "e", "n", "o", "r", "t"};
vocab_size = length(chars);

% Raw character counts, matching the table above
counts = [5, 2, 2, 1, 4, 1, 3];
```

```rustlab
total = sum(counts);
freqs = counts / total;

print("Character counts (space, b, e, n, o, r, t):", counts);
print("Relative frequencies:", freqs);
print("Sum of frequencies (should be 1.0):", sum(freqs));
```

The corpus has **${total}** characters across a vocabulary of size **${vocab_size}**. The highest-frequency character is the space at ${max(freqs):%.3f} $\approx 5/18$ — exactly five spaces in `"to be or not to be"`.

### Example — Frequency bar chart

```rustlab
figure()
bar(chars, freqs, "Character Frequencies: 'to be or not to be'")
```

The bar heights sum to 1.0 — this is a valid probability distribution over the vocabulary.

## One-Hot Encoding

### Theory

An integer index like $4$ carries no useful geometric meaning — a model might infer that character 4 is "greater than" character 3, which is meaningless. Instead, each token is represented as a **one-hot vector**: a vector of length $|\mathcal{V}|$ that is 0 everywhere except at position $i$:

$$(\mathbf{e}_i)_j = \begin{cases} 1 & \text{if } j = i \\ 0 & \text{otherwise.} \end{cases}$$

All one-hot vectors are mutually orthogonal:

$$\mathbf{e}_i \cdot \mathbf{e}_j = \delta_{ij}.$$

No two tokens share any geometric similarity — a clean slate before the model learns its own representations ([Lesson 04](04-embeddings-and-similarity.md)).

To encode a sequence of $T$ tokens, stack their one-hot vectors as rows of a matrix $\mathbf{X} \in \{0, 1\}^{T \times |\mathcal{V}|}$:

$$\mathbf{X} = \begin{bmatrix} \mathbf{e}_{i_1} \\ \mathbf{e}_{i_2} \\ \vdots \\ \mathbf{e}_{i_T} \end{bmatrix}.$$

### Example — Stacking one-hots into a matrix X

Build $\mathbf{X}$ for `"hello"` with vocabulary $\{e{:}1,\; h{:}2,\; l{:}3,\; o{:}4\}$:

```rustlab
% Vocabulary for "hello": e=1, h=2, l=3, o=4
vocab_size = 4;
oh_e = [1, 0, 0, 0];
oh_h = [0, 1, 0, 0];
oh_l = [0, 0, 1, 0];
oh_o = [0, 0, 0, 1];
```

```rustlab
% "hello" → token sequence [h, e, l, l, o] → indices [2, 1, 3, 3, 4]
X = [oh_h; oh_e; oh_l; oh_l; oh_o];

print("One-hot matrix X for 'hello'  (5 tokens x 4 vocab):");
print(X);
```

The matrix is ${size(X, 1)} $\times$ ${size(X, 2)} — one row per token, one column per vocabulary slot. Each row has exactly one non-zero entry. Rows 3 and 4 (both `l`) are identical — the model sees them as the same token.

### Example — Row sums and orthogonality

Row sums must all equal 1, and distinct one-hot vectors must be orthogonal:

```rustlab
% Row sums via matrix-vector product
row_sums = X * ones(vocab_size)';
print("Row sums (each must equal 1):", row_sums);

% Orthogonality check
dot_h_e = sum(oh_h .* oh_e);
dot_l_l = sum(oh_l .* oh_l);
print("Dot product h . e:", dot_h_e, "  l . l:", dot_l_l);
```

Orthogonality confirmed: $\mathbf{e}_h \cdot \mathbf{e}_e = ${dot_h_e}$ and $\mathbf{e}_l \cdot \mathbf{e}_l = ${dot_l_l}$ — exactly $\delta_{ij}$.

### Example — One-hot matrix heatmap

```rustlab
figure()
imagesc(X, "viridis")
title("One-Hot Matrix: 'hello' (5 tokens x 4 vocab)")
```

In the heatmap, each row has one bright cell (value = 1) and three dark cells (value = 0). The column positions identify the character.

## Key Takeaways

- **Tokenisation** converts symbols to integers to matrices — the bridge from text to linear algebra.
- A character-level vocabulary is tiny (~100 in English) but forces the model to learn spelling from scratch. Larger vocabularies reduce sequence length but increase memory.
- One-hot encoding does *not* imply characters are independent — it is only the starting point. The embedding layer ([Lesson 04](04-embeddings-and-similarity.md)) will project these orthogonal vectors into a dense space where learned relationships emerge.

## Standalone Scripts

For shell-based experimentation, two rustlab scripts in this lesson's directory reproduce the demos above as standalone programs:

| Script | What it computes |
|---|---|
| `char_frequencies.r` | character counts and relative frequencies for `"to be or not to be"`; bar chart |
| `one_hot_encoding.r` | the 5×4 one-hot matrix for `"hello"`; heatmap |

Run both with `make lesson-01` from the repo root (or `rustlab run lessons/01-tokens-and-encoding/<name>.r` for one). Each writes SVGs next to the script (gitignored).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `total` | `18` |
| `vocab_size` (corpus) | `7` |
| `freqs` (max — space) | ≈ `0.278` (= 5/18) |
| `sum(freqs)` | `1.0` |
| `size(X)` (`"hello"`) | `[5, 4]` |
| `row_sums` | `[1; 1; 1; 1; 1]` |
| `dot_h_e` | `0` |
| `dot_l_l` | `1` |

## Exercises

1. **Vocabulary extension.** Modify `char_frequencies.r` to use the corpus `"the cat sat on the mat"`. What is the new vocabulary size? Which character has the highest frequency?
2. **Decode a sequence.** Given the vocabulary `{e:1, h:2, l:3, o:4}` and the integer sequence `[4, 3, 3, 1]`, what word does this decode to?
3. **One-hot orthogonality.** Using the `"hello"` one-hot matrix, compute the dot product between the `h` row and the `l` row by hand. Confirm the result matches the orthogonality property.
4. **Vocabulary size trade-off.** If a text contains $N$ unique characters, the one-hot matrix for a sequence of $T$ tokens has $T \times N$ entries, but only $T$ are non-zero. Express the fraction of non-zero entries as a formula. What happens to this fraction as $N$ grows?
5. **Beyond characters.** Suppose you tokenise at the word level instead. For `"to be or not to be"`, what is the word-level vocabulary size? How do the one-hot matrix dimensions change for the same sentence?

## What's next

Lesson 02 turns the raw counts above into a proper probability distribution over the vocabulary using the **softmax** function, and introduces the **temperature** parameter that controls how sharp or diffuse the resulting distribution is. The character frequencies $f_i$ from this lesson reappear there as a baseline distribution for comparison.
