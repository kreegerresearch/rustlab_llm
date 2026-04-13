# Lesson 01 — Tokens & Text Encoding

A language model never sees letters — it sees integers. This lesson builds the bridge
from raw text to the numerical representations that every later lesson depends on.

---

## What is a Token?

A **token** is the basic unit of text a language model operates on. At the character
level every character becomes one token. The process has two steps:

**Step 1 — Collect the vocabulary.** Given a corpus, collect every unique character
and sort them. This ordered set is the vocabulary $\mathcal{V}$, with size $|\mathcal{V}|$.

**Step 2 — Assign integer indices.** Create a mapping:

$$\text{encode} : \mathcal{V} \to \{1, 2, \ldots, |\mathcal{V}|\}$$

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

The string `"hello"` would encode to $[4, 3, 5, 5, 6]$ under a similar scheme.

---

## Character Frequencies

Before building a model it helps to understand the **frequency distribution** of
tokens. If $c_i$ is the count of character $i$, the relative frequency is:

$$f_i = \frac{c_i}{\sum_{j=1}^{|\mathcal{V}|} c_j}$$

This is a discrete probability distribution over the vocabulary. Let's compute it
for our corpus:

```rustlab
% Corpus: "to be or not to be" (18 characters including spaces)
% Vocabulary (sorted): ' '=1, 'b'=2, 'e'=3, 'n'=4, 'o'=5, 'r'=6, 't'=7

vocab_size = 7;

% Raw character counts
%   ' ' : 5   b : 2   e : 2   n : 1   o : 4   r : 1   t : 3
counts = [5, 2, 2, 1, 4, 1, 3];

total = sum(counts);
print("Corpus: to be or not to be");
print("Total characters:", total);
print("Vocabulary size:", vocab_size);

% Relative frequencies: f_i = c_i / sum(c)
freqs = counts / total;

print("Character counts (space, b, e, n, o, r, t):");
print(counts);
print("Relative frequencies:");
print(freqs);
print("Sum of frequencies (should be 1.0):", sum(freqs));
```

The highest-frequency character is the space at $5/18 \approx 0.278$. Verify by
counting spaces in `"to be or not to be"` — there are exactly five.

```rustlab
max_freq = max(freqs);
print("Highest frequency:", max_freq);

savebar(freqs, "outputs/char_frequencies.svg", "Character Frequencies: 'to be or not to be'")
print("Saved outputs/char_frequencies.svg")
```

The bar heights sum to 1.0 — this is a valid probability distribution over the
vocabulary.

---

## One-Hot Encoding

An integer index like $4$ carries no useful geometric meaning — a model might infer
that character 4 is "greater than" character 3, which is meaningless. Instead, each
token is represented as a **one-hot vector**: a vector of length $|\mathcal{V}|$ that
is 0 everywhere except at position $i$:

$$(\mathbf{e}_i)_j = \begin{cases} 1 & \text{if } j = i \\ 0 & \text{otherwise} \end{cases}$$

All one-hot vectors are mutually orthogonal:

$$\mathbf{e}_i \cdot \mathbf{e}_j = \delta_{ij}$$

No two tokens share any geometric similarity — a clean slate before the model learns
its own representations (Lesson 04).

### Building the one-hot matrix

To encode a sequence of $T$ tokens, stack their one-hot vectors as rows of a matrix
$\mathbf{X} \in \{0, 1\}^{T \times |\mathcal{V}|}$:

$$\mathbf{X} = \begin{bmatrix} \mathbf{e}_{i_1} \\ \mathbf{e}_{i_2} \\ \vdots \\ \mathbf{e}_{i_T} \end{bmatrix}$$

Let's build this for `"hello"` with vocabulary $\{e{:}1,\; h{:}2,\; l{:}3,\; o{:}4\}$:

```rustlab
vocab_size = 4;

% One-hot rows (each is a basis vector in R^4)
oh_e = [1, 0, 0, 0];
oh_h = [0, 1, 0, 0];
oh_l = [0, 0, 1, 0];
oh_o = [0, 0, 0, 1];

% "hello" → token sequence [h, e, l, l, o] → indices [2, 1, 3, 3, 4]
X = [oh_h; oh_e; oh_l; oh_l; oh_o];

print("One-hot matrix X for 'hello'  (5 tokens x 4 vocab):");
print(X);
print("Shape (rows=tokens, cols=vocab_size):", size(X));
```

Each row has exactly one non-zero entry. Rows 3 and 4 (both `l`) are identical —
the model sees them as the same token.

### Verification

Row sums must all equal 1, and distinct one-hot vectors must be orthogonal:

```rustlab
% Row sums via matrix-vector product
row_sums = X * ones(vocab_size)';
print("Row sums (each must equal 1):");
print(row_sums);

% Orthogonality check
dot_h_e = sum(oh_h .* oh_e);
dot_l_l = sum(oh_l .* oh_l);
print("Dot product h . e (should be 0):", dot_h_e);
print("Dot product l . l (should be 1):", dot_l_l);
```

```rustlab
saveimagesc(X, "outputs/one_hot_matrix.svg", "One-Hot Matrix: 'hello' (5 tokens x 4 vocab)", "viridis")
print("Saved outputs/one_hot_matrix.svg")
```

In the heatmap, each row has one bright cell (value = 1) and three dark cells
(value = 0). The column positions identify the character.

---

## Key Takeaways

- **Tokenisation** converts symbols to integers to matrices — the bridge from text
  to linear algebra.
- A character-level vocabulary is tiny (~100 in English) but forces the model to learn
  spelling from scratch. Larger vocabularies reduce sequence length but increase memory.
- One-hot encoding does *not* imply characters are independent — it is only the
  starting point. The embedding layer (Lesson 04) will project these orthogonal vectors
  into a dense space where learned relationships emerge.
