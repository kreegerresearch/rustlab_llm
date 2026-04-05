# Lesson 01 — Tokens & Text Encoding

## Learning Objectives

By the end of this lesson you will be able to:

- Explain what a **token** is and why text must be converted to numbers before a model can process it.
- Build a **character-level vocabulary** and map every token to a unique integer index.
- Construct a **one-hot vector** for a given token and interpret it geometrically.
- Read a **character frequency bar chart** and a **one-hot matrix heatmap** and explain what each cell means.
- State the trade-off between vocabulary size and representational expressiveness.

---

## Background

This lesson assumes:

- Basic familiarity with vectors (a list of numbers) and matrices (a 2-D grid of numbers).
- The concept of a function: something that maps an input to an output.

No neural network or probability knowledge is required yet.

---

## Theory

### What is a Token?

A **token** is the basic unit of text that a language model operates on. At the character level, every character in a string becomes one token. The model never sees letters — it sees integers.

**Step 1 — Collect the vocabulary.**

Given a corpus (any text we want to learn from), collect every unique character. Sort them for reproducibility. This ordered collection is the **vocabulary** $\mathcal{V}$, with size $|\mathcal{V}|$.

**Step 2 — Assign an integer index.**

Create a deterministic mapping:

$$
\text{encode} : \mathcal{V} \to \{1, 2, \ldots, |\mathcal{V}|\}
$$

For example, with the text `"hello world"`:

| Character | Index |
|-----------|-------|
| ` ` (space) | 1 |
| `d` | 2 |
| `e` | 3 |
| `h` | 4 |
| `l` | 5 |
| `o` | 6 |
| `r` | 7 |
| `w` | 8 |

The string `"hello"` encodes to the sequence $[4, 3, 5, 5, 6]$.

The inverse mapping $\text{decode} : \{1, \ldots, |\mathcal{V}|\} \to \mathcal{V}$ reconstructs text from integer sequences.

### Character Frequencies

Before building a model, it is useful to understand the **frequency distribution** of tokens in the corpus. Let $c_i$ be the count of character $i$ in the text. The relative frequency is:

$$
f_i = \frac{c_i}{\sum_{j=1}^{|\mathcal{V}|} c_j}
$$

This is a discrete probability distribution over the vocabulary. High-frequency characters appear often; a good model must predict them reliably.

### One-Hot Encoding

An integer index like $4$ does not carry any useful geometric meaning — the model might infer that character $4$ is "greater than" character $3$, which is meaningless. To avoid this, each token is represented as a **one-hot vector**: a vector of length $|\mathcal{V}|$ that is $0$ everywhere except at position $i$, where it is $1$.

Formally, for token index $i$:

$$
\mathbf{e}_i \in \mathbb{R}^{|\mathcal{V}|}, \quad (\mathbf{e}_i)_j =
\begin{cases}
1 & \text{if } j = i \\
0 & \text{otherwise}
\end{cases}
$$

**Geometric interpretation.** Each one-hot vector is a standard basis vector. All one-hot vectors are mutually orthogonal:

$$
\mathbf{e}_i \cdot \mathbf{e}_j = \delta_{ij}
$$

where $\delta_{ij}$ is the Kronecker delta. This means no two tokens share any geometric similarity in one-hot space — a clean slate before the model learns its own representations (Lesson 04).

**Sequence encoding.** To encode a sequence of $T$ tokens, stack their one-hot vectors as rows of a matrix $\mathbf{X} \in \{0, 1\}^{T \times |\mathcal{V}|}$. Each row has exactly one non-zero entry.

$$
\mathbf{X} =
\begin{bmatrix}
\mathbf{e}_{i_1} \\
\mathbf{e}_{i_2} \\
\vdots \\
\mathbf{e}_{i_T}
\end{bmatrix}
$$

---

## Core Concepts

The fundamental challenge of language modelling is that text is discrete and symbolic — computers naturally work with continuous numbers. **Tokenisation** is the bridge: it converts a sequence of symbols into a sequence of integers, and then into a matrix of one-hot vectors that a neural network can process with matrix multiplication.

The choice of vocabulary determines the granularity of the model's understanding. A character-level vocabulary is tiny (∼100 characters in English) but forces the model to learn spelling from scratch. Larger vocabularies (word-level or subword-level, Lesson 19) reduce sequence length but increase memory. In this lesson we start at the character level to keep the mechanics transparent.

**Common misconception:** One-hot encoding does *not* imply that characters are independent or unrelated. It is only the *starting point* — the model's embedding layer (Lesson 04) will project these orthogonal vectors into a dense, lower-dimensional space where learned relationships emerge.

---

## Simulations

### `char_frequencies.r` — Character Frequency Bar Chart

**What it computes:**
Defines a sample corpus (`"to be or not to be"`), manually specifies the vocabulary and per-character counts, computes relative frequencies, and saves a bar chart.

**What to observe:**
- The bar heights correspond to how often each character appears.
- The space character and the letters `t`, `o`, `b`, `e` dominate — common in English.
- The bars sum to 1.0 (verify with the printed total).

**Verify by hand:**
Count each character in `"to be or not to be"` (18 characters total). Confirm that `t` appears 3 times → frequency $= 3/18 \approx 0.167$.

---

### `one_hot_encoding.r` — One-Hot Matrix Heatmap

**What it computes:**
Encodes the string `"hello"` into a $5 \times 4$ one-hot matrix (5 tokens, vocabulary of 4 unique characters: `e`, `h`, `l`, `o`), prints the matrix, and saves it as a heatmap.

**What to observe:**
- Each row has exactly one bright cell (value = 1) and three dark cells (value = 0).
- The column positions identify the character: column 1 = `e`, column 2 = `h`, column 3 = `l`, column 4 = `o`.
- Rows 3 and 4 (both `l`) are identical — the model sees them as the same token in the same position.

**Verify by hand:**
Write out the encoding table, then manually construct the expected matrix and compare to the printed output.

---

## Exercises

1. **Vocabulary extension.** Modify `char_frequencies.r` to use the corpus `"the cat sat on the mat"`. What is the new vocabulary size? Which character has the highest frequency?

2. **Decode a sequence.** Given the vocabulary `{e:1, h:2, l:3, o:4}` and the integer sequence `[4, 3, 3, 1]`, what word does this decode to?

3. **One-hot orthogonality.** Using the `"hello"` one-hot matrix from `one_hot_encoding.r`, compute the dot product between the `h` row and the `l` row by hand. What is the result, and why does it confirm the orthogonality property?

4. **Vocabulary size trade-off.** If a text contains $N$ unique characters, the one-hot matrix for a sequence of $T$ tokens has $T \times N$ entries, but only $T$ are non-zero. Express the fraction of non-zero entries as a formula. What happens to this fraction as $N$ grows?

5. **Beyond characters.** Suppose you tokenise at the word level instead of the character level. For the corpus `"to be or not to be"`, what is the word-level vocabulary size? How does the one-hot matrix dimensions change for the same sentence?
