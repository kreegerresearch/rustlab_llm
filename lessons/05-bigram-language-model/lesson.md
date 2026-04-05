# Lesson 05 — The Bigram Language Model

## Learning Objectives

By the end of this lesson you will be able to:

- Define a **bigram language model** and explain how it estimates next-token probabilities from counts.
- Construct and interpret a **bigram frequency matrix** from a corpus.
- Normalise count rows into valid probability distributions using softmax or direct division.
- Trace through the **sampling algorithm** (cumulative distribution + threshold) step by step.
- Explain why a bigram model is a useful baseline before introducing neural approaches.

---

## Background

This lesson assumes:

- Tokens and vocabulary from Lesson 01.
- Probability distributions and softmax from Lesson 02.
- Cross-entropy loss from Lesson 03 (mentioned but not required to follow the main derivation).

---

## Theory

### The Language Modelling Problem

A language model assigns a probability to every possible next token given the tokens seen so far:

$$
P(x_{t+1} \mid x_1, x_2, \ldots, x_t)
$$

The full history $x_1, \ldots, x_t$ can be arbitrarily long. The bigram model makes the **Markov assumption**: the next token depends only on the immediately preceding token.

$$
P(x_{t+1} \mid x_1, \ldots, x_t) \approx P(x_{t+1} \mid x_t)
$$

This reduces the problem to estimating $|\mathcal{V}|^2$ conditional probabilities — one for every ordered pair of tokens.

### Building the Bigram Matrix

**Step 1 — Count.** Scan the corpus and tally every consecutive pair of tokens $(x_t, x_{t+1})$. Store these in the **bigram count matrix** $\mathbf{C} \in \mathbb{Z}_{\geq 0}^{|\mathcal{V}| \times |\mathcal{V}|}$, where:

$$
C_{ij} = \text{number of times token } j \text{ follows token } i \text{ in the training corpus}
$$

**Step 2 — Normalise.** Convert each row to a probability distribution by dividing by the row sum:

$$
P_{ij} = \frac{C_{ij}}{\sum_{k=1}^{|\mathcal{V}|} C_{ik}}
$$

Row $i$ of $\mathbf{P}$ is the conditional distribution $P(\cdot \mid x_t = i)$ — a valid probability distribution (non-negative, sums to 1).

**Smoothing.** If token pair $(i, j)$ never appeared in training, $C_{ij} = 0$ and therefore $P_{ij} = 0$. The model assigns zero probability to that continuation, which causes the cross-entropy loss to be infinite on unseen pairs. A simple fix is **Laplace (add-one) smoothing**:

$$
P_{ij}^{\text{smooth}} = \frac{C_{ij} + 1}{\sum_{k}(C_{ik} + 1)} = \frac{C_{ij} + 1}{\sum_{k} C_{ik} + |\mathcal{V}|}
$$

This ensures every pair has non-zero probability.

### Sampling from the Model

To **generate** text, start from any token and repeatedly sample the next token from the model's distribution. At each step:

**Step 1 — Look up the row.** Given current token $i$, extract row $\mathbf{P}_i$ from the probability matrix.

**Step 2 — Compute the CDF.** The cumulative distribution function is:

$$
\text{CDF}_j = \sum_{k=1}^{j} P_{ik}
$$

This is a non-decreasing sequence from 0 (before the first element) to 1 (after the last).

**Step 3 — Draw a uniform sample.** Sample $u \sim \text{Uniform}(0, 1)$.

**Step 4 — Find the bin.** The sampled token is the smallest $j$ such that $\text{CDF}_j \geq u$:

$$
x_{t+1} = \min\{j : \text{CDF}_j \geq u\}
$$

Repeating this process from the newly sampled token generates a sequence of arbitrary length.

### Training Loss

For a training corpus of $T$ consecutive tokens, the bigram model's average cross-entropy loss (Lesson 03) is:

$$
\mathcal{L} = -\frac{1}{T-1} \sum_{t=1}^{T-1} \log P_{x_t, x_{t+1}}
$$

Minimising this over $\mathbf{P}$ (subject to row-sum constraints) gives exactly the normalised count solution above — the count-based bigram model is the **maximum likelihood estimator** for the Markov model.

### Why Bigrams Are a Useful Baseline

A bigram model is fast to train (one pass over the corpus), interpretable (the matrix is human-readable), and sets a performance floor. Any neural language model should achieve lower perplexity than a bigram baseline to justify its cost. Its core weakness is that context is limited to one token — it cannot model dependencies that span more than two positions.

---

## Core Concepts

The bigram model encodes the statistical structure of a corpus in a single matrix. Reading a row tells you exactly what the model "thinks" will follow a given token. This transparency is valuable: a row with high entropy means the current token is ambiguous (many plausible continuations); a row with low entropy means the continuation is predictable.

The sampling algorithm — cumulative distribution + uniform threshold — is a foundational technique that appears in every sampling strategy in Lesson 21 (top-K, nucleus). Learning it here in the simple bigram context makes the later generalisations straightforward.

**Common misconception:** The bigram matrix is *not* the same as the transition matrix of a random walk. The matrix $\mathbf{P}$ is row-stochastic (rows sum to 1), not column-stochastic, so it operates on row vectors from the left ($\mathbf{p}_{t+1} = \mathbf{p}_t \mathbf{P}$), not column vectors from the right.

---

## Simulations

### `bigram_counts.r` — Bigram Count and Probability Matrix

**What it computes:**
Uses the corpus `"abcbabcba"` with vocabulary `{a:1, b:2, c:3}`. Scans the token sequence with a `for` loop to build the $3 \times 3$ bigram count matrix, normalises each row to probabilities, applies Laplace smoothing, and saves both matrices as heatmaps.

**What to observe:**
- The raw count matrix: bright cells show frequent token pairs; dark cells show rare or absent pairs.
- After normalisation, each row sums to 1.
- Laplace smoothing fills in the zeros, lightening the dark cells slightly.
- Token `b` has the most interesting row: it can be followed by either `a` or `c` with equal probability.

**Verify by hand:**
Trace through `"abcbabcba"` character by character, tallying each bigram. Confirm your tally matches the printed count matrix.

---

### `bigram_sampling.r` — Sampling Trace

**What it computes:**
Demonstrates the CDF-based sampling formula `sum(CDF < u) + 1` on token `b` with two fixed draws, then runs a `for` loop to generate a 12-token sequence from a random starting point, using `rand()` for the draws.

**What to observe:**
- For token `b`: CDF = [0.5, 0.5, 1.0]. Draw 0.3 < 0.5 → samples `a` (index 1); draw 0.7 ≥ 0.5 → samples `c` (index 3).
- The generated sequence always has `b` at every even position (since `a` and `c` both lead deterministically to `b`).
- Re-running the script produces a different sequence each time due to `rand()` — the structure (b at every even step) is preserved.

**Verify by hand:**
Use the CDF for token `c` and draws $u = 0.1$ and $u = 0.9$. Predict which token is sampled. Confirm against the printed output.

---

## Exercises

1. **Entropy of rows.** Compute the Shannon entropy (in bits) of each row of $\mathbf{P}$ for the `"abcbabcba"` corpus. Which token is most predictable? Which is least?

2. **Laplace smoothing effect.** For the `"abcbabcba"` corpus with $|\mathcal{V}| = 3$, compute the Laplace-smoothed probability for the bigram (a → a), which never appeared. What is $P^{\text{smooth}}_{11}$?

3. **Longer corpus.** Extend the corpus to `"abcbabcbabc"` and recompute the bigram matrix. Do the probabilities change significantly? What happens as the corpus grows?

4. **Perplexity.** The perplexity of a language model on a test sequence is $\exp(\mathcal{L})$ where $\mathcal{L}$ is the mean cross-entropy loss. For the test sequence `"abcb"` (3 bigrams: ab, bc, cb), compute the perplexity of the trained bigram model. What does a perplexity of 1 mean? What about perplexity of $|\mathcal{V}|$?

5. **Beyond bigrams.** A trigram model conditions on the last two tokens: $P(x_{t+1} \mid x_{t-1}, x_t)$. How many parameters does a trigram model have for a vocabulary of size $|\mathcal{V}|$? How does this scale compared to the bigram model?
