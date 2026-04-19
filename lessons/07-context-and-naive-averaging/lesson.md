# Lesson 07 — Context and Naive Averaging

## Learning Objectives

By the end of this lesson you will be able to:

- Explain why a **bigram** (Markov-1) language model is context-free beyond a single previous token, and identify the failure mode on ambiguous tokens.
- Define **prefix averaging** — the simplest context-aware aggregation — as $\bar{\mathbf{x}}_t = \frac{1}{t}\sum_{i=1}^{t} \mathbf{x}_i$.
- Rewrite that sum as a **matrix multiplication** $\bar{\mathbf{X}} = \mathbf{W}\mathbf{X}$ where $\mathbf{W}$ is a lower-triangular averaging matrix.
- Read the lower-triangular causal structure of $\mathbf{W}$ and explain why it enforces the "no peek at the future" rule.
- Articulate the remaining weakness of uniform averaging — all past tokens get equal weight — that motivates **attention** (Lesson 08).

---

## Background

This lesson assumes:

- Bigram language models and row-normalised probability matrices from Lesson 05.
- Linear layers and matrix multiplication from Lesson 06.
- Embeddings as dense row vectors from Lesson 04.
- No new mathematics: only rearranging sums into matrix form.

---

## Theory

### Why a Bigram Fails on Context

The bigram probability $P(x_{t+1} \mid x_t)$ depends on *only* the previous token. For an ambiguous token like "bank", which can continue with "water" (preceded by "river") or "safe" (preceded by "money"), the bigram cannot distinguish the two contexts — both histories collapse to the same row of the bigram matrix.

$$
P(x_{t+1} \mid \text{"river", "bank"}) = P(x_{t+1} \mid \text{"money", "bank"}) = P(x_{t+1} \mid \text{"bank"})
$$

This is a structural limitation, not a data limitation. Any model whose prediction depends on only the last token will fail on this class of example.

### Aggregating Context: Prefix Averaging

The simplest fix is to represent the state at position $t$ as a summary of *all* preceding tokens. Given a sequence of embeddings $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T \in \mathbb{R}^{d}$, the **prefix average** at position $t$ is:

$$
\bar{\mathbf{x}}_t \;=\; \frac{1}{t}\sum_{i=1}^{t} \mathbf{x}_i
$$

Each $\bar{\mathbf{x}}_t$ mixes the current token with every earlier token, equally weighted. This is sometimes called a **bag of tokens** representation up to position $t$.

### Prefix Averaging as a Matrix Multiply

Stack the embeddings into a row matrix $\mathbf{X} \in \mathbb{R}^{T \times d}$ (row $t$ = token $t$). Define the lower-triangular averaging matrix $\mathbf{W} \in \mathbb{R}^{T \times T}$:

$$
\mathbf{W}_{t,i} \;=\; \begin{cases} \frac{1}{t} & \text{if } i \le t \\ 0 & \text{if } i > t \end{cases}
$$

Then the stacked prefix averages are simply:

$$
\bar{\mathbf{X}} \;=\; \mathbf{W}\mathbf{X}
$$

For $T=4$:

$$
\mathbf{W} \;=\; \begin{bmatrix}
1 & 0 & 0 & 0 \\
\tfrac{1}{2} & \tfrac{1}{2} & 0 & 0 \\
\tfrac{1}{3} & \tfrac{1}{3} & \tfrac{1}{3} & 0 \\
\tfrac{1}{4} & \tfrac{1}{4} & \tfrac{1}{4} & \tfrac{1}{4}
\end{bmatrix}
$$

Row $t$ of $\mathbf{W}$ specifies the mixing weights that produce $\bar{\mathbf{x}}_t$. The zero entries *above the diagonal* enforce the **causal** constraint: position $t$ cannot depend on a future token $i > t$. Every row sums to 1, so $\mathbf{W}$ performs a weighted average.

### What's Still Wrong

Uniform $1/t$ weights treat every past token as equally informative. In "the cat sat on the mat, it was soft", the pronoun "it" refers to "mat" (or "cat") — specific past tokens matter more than others. We need *data-dependent* weights that can concentrate mass on the relevant tokens. That is exactly what **attention** (Lesson 08) provides: replace the fixed $1/t$ weights with $\mathrm{softmax}(\mathrm{scores})$ where the scores depend on the query and key vectors.

---

## Core Concepts

A bigram model's prediction is a function of *one* token. A context-aware model's prediction is a function of *a set* of tokens. The cleanest way to aggregate a set of vectors into a single summary is to average them — and a **causal** average (only past and current tokens) can be written as a lower-triangular matrix multiplication.

Seeing this as a matrix multiply is the critical bridge to attention. Attention replaces the constant weight matrix $\mathbf{W}$ with a matrix whose entries depend on the content of the tokens themselves. The structure — one weighted sum per position, causal mask from upper triangle — stays identical. Only the weights change.

**Common misconception:** Averaging is not "losing information" in a trivial sense — you do lose ordering and magnitude, but you gain a fixed-size summary that conditions on all prior tokens. Attention keeps the same fixed-size-summary shape while recovering the ability to focus.

---

## Simulations

### `context_failure.r` — Bigram Fails on an Ambiguous Token

**What it computes:**
Builds a tiny corpus with two sentences that share the ambiguous token "bank":
- `river bank water`
- `money bank safe`

Constructs the bigram count matrix $C$, row-normalises to $P$, and prints $P(\text{next} \mid \text{bank})$. Plots the row distribution.

**What to observe:**
- $P(\text{next} \mid \text{bank}) = [0, 0, 0.5, 0, 0.5]$ regardless of whether the preceding token was "river" or "money".
- The bigram has no mechanism to distinguish the two histories; both bank-predictions are identical.
- The plot shows a symmetric two-bar distribution — there is no signal selecting "water" over "safe" from the history.

**Verify by hand:**
Count: `bank` appears twice in the corpus; it is followed once by `water` (sentence A) and once by `safe` (sentence B). So $P(\text{water} \mid \text{bank}) = P(\text{safe} \mid \text{bank}) = 0.5$. Confirm this matches the printed row.

---

### `prefix_averaging.r` — Causal Averaging as a Matrix Multiply

**What it computes:**
Builds $T=6$ hand-crafted 4-dimensional embeddings. Computes prefix averages two ways:
1. A `for` loop that accumulates running sums: $\bar{\mathbf{x}}_t = \frac{1}{t}\sum_{i \le t} \mathbf{x}_i$.
2. A single matrix multiply $\bar{\mathbf{X}} = \mathbf{W}\mathbf{X}$ using a lower-triangular $\mathbf{W}$ with row $t$ equal to $[\tfrac{1}{t}, \dots, \tfrac{1}{t}, 0, \dots, 0]$.

Prints the max absolute difference between the two methods (should be ~0). Plots $\mathbf{W}$, the input embeddings $\mathbf{X}$, and the averaged embeddings $\bar{\mathbf{X}}$ as heatmaps.

**What to observe:**
- $\mathbf{W}$ is lower-triangular — the upper triangle is exactly zero (causal constraint).
- Row $t$ of $\mathbf{W}$ has $t$ non-zero entries, each equal to $1/t$; each row sums to 1.
- $\bar{\mathbf{X}}$ is "smoother" than $\mathbf{X}$ — later rows are blends of all earlier rows, so they vary less than the raw embeddings.
- Both methods agree to machine precision, confirming $\bar{\mathbf{X}} = \mathbf{W}\mathbf{X}$.

**Verify by hand:**
For $t=3$, $\bar{\mathbf{x}}_3$ should equal $(\mathbf{x}_1 + \mathbf{x}_2 + \mathbf{x}_3)/3$. Pick a single dimension (e.g., the first) and check the arithmetic against the printed $\bar{\mathbf{X}}$.

---

## Exercises

1. **Extending the ambiguity.** Modify `context_failure.r` to add a third sentence `river bank fish`. What does $P(\text{next} \mid \text{bank})$ become? How does adding more data *not* fix the underlying structural problem?

2. **Exponential moving average.** Replace the uniform $1/t$ weights in $\mathbf{W}$ with exponentially decaying weights $w_{t,i} \propto \gamma^{t-i}$ (with $\gamma = 0.8$), normalised to sum to 1 per row. How does the averaged output differ? What kind of bias does this introduce?

3. **Non-causal averaging.** What matrix $\mathbf{W}'$ would average *all* tokens (past and future) uniformly at every position? Write it out for $T=4$. Why is this wrong for language modelling but fine for, e.g., sentence classification?

4. **Counting operations.** For a sequence of length $T$ and embedding dimension $d$, how many scalar multiplies does $\mathbf{W}\mathbf{X}$ take? How does this scale with $T$? (Hint: count the non-zero entries of $\mathbf{W}$.)

5. **Preview of attention.** In attention (Lesson 08), the weights in row $t$ of $\mathbf{W}$ are replaced by $\mathrm{softmax}(\mathbf{q}_t \mathbf{K}^\top / \sqrt{d_k})$, still with a causal mask. What property of softmax guarantees each row still sums to 1?
