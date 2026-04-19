# Lesson 08 — Scaled Dot-Product Attention

## Learning Objectives

By the end of this lesson you will be able to:

- Define the **query**, **key**, and **value** matrices $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ derived from a sequence of embeddings via three learned linear projections.
- Write and explain the scaled dot-product attention equation $\mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \mathrm{softmax}\!\left(\tfrac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$ one piece at a time.
- Motivate the $1/\sqrt{d_k}$ scale from the variance of a dot product of independent vectors.
- Apply a **causal mask** so position $t$ cannot see tokens at positions $> t$ and verify the resulting attention matrix is lower-triangular.
- Relate attention back to the uniform averaging matrix from Lesson 07 — same shape, now with data-dependent weights.

---

## Background

This lesson assumes:

- Embeddings as dense row vectors from Lesson 04.
- Softmax, cross-entropy, and probability row-normalisation from Lessons 02–03.
- Linear layers as $\mathbf{y} = \mathbf{x}\mathbf{W}$ from Lesson 06.
- Causal prefix averaging as $\bar{\mathbf{X}} = \mathbf{W}\mathbf{X}$ from Lesson 07.

---

## Theory

### Queries, Keys, Values

Given an input sequence $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$ (row $t$ is the embedding of token $t$), three linear projections produce three new matrices:

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q \in \mathbb{R}^{T \times d_k}, \quad
\mathbf{K} = \mathbf{X}\mathbf{W}_K \in \mathbb{R}^{T \times d_k}, \quad
\mathbf{V} = \mathbf{X}\mathbf{W}_V \in \mathbb{R}^{T \times d_v}
$$

- $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ — project each embedding into a **query** and **key** vector.
- $\mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ — project into a **value** vector.
- Interpretation: each token emits a *query* (what it's looking for), a *key* (what it advertises), and a *value* (what it contributes if selected).

The parameters $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ are learned by gradient descent (Lesson 06); once trained, they encode which features matter for prediction.

### The Score Matrix

The similarity between query $\mathbf{q}_t$ and key $\mathbf{k}_i$ is their dot product. Stacked for all pairs:

$$
\mathbf{S} = \mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{T \times T}, \qquad S_{t,i} = \mathbf{q}_t \cdot \mathbf{k}_i
$$

$S_{t,i}$ is a raw (unbounded) score for "how relevant is token $i$ to the query from token $t$".

### The $1/\sqrt{d_k}$ Scale

If $\mathbf{q}$ and $\mathbf{k}$ each have $d_k$ i.i.d. components with mean 0 and variance 1, their dot product $\mathbf{q} \cdot \mathbf{k} = \sum_{j=1}^{d_k} q_j k_j$ has variance $d_k$ (sum of $d_k$ independent products, each with variance 1). So its standard deviation scales like $\sqrt{d_k}$.

As $d_k$ grows, raw dot products grow larger in magnitude, and softmax becomes sharp (one entry approaches 1, others approach 0) — killing gradient flow. Dividing by $\sqrt{d_k}$ keeps the input to softmax at unit variance regardless of $d_k$:

$$
\mathbf{S}_{\text{scaled}} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}
$$

### The Causal Mask

For language modelling, token $t$ must not see tokens at positions $i > t$ (no peeking at the future). Define a mask $\mathbf{M} \in \mathbb{R}^{T \times T}$:

$$
M_{t, i} = \begin{cases} 0 & i \le t \\ -\infty & i > t \end{cases}
$$

Add it to the scaled scores:

$$
\tilde{\mathbf{S}} = \mathbf{S}_{\text{scaled}} + \mathbf{M}
$$

After softmax, the $-\infty$ entries become exactly 0 — attention weights from $t$ to any $i > t$ are zero. In practice, a large negative number (e.g., $-10^9$) replaces $-\infty$ to avoid `NaN` from $\exp(-\infty)$.

### Softmax and the Output

Apply softmax **row-wise** to $\tilde{\mathbf{S}}$:

$$
\mathbf{A} = \mathrm{softmax}_{\text{row}}(\tilde{\mathbf{S}}), \qquad A_{t, i} = \frac{\exp(\tilde S_{t, i})}{\sum_{j=1}^{T} \exp(\tilde S_{t, j})}
$$

Each row of $\mathbf{A}$ is a probability distribution over the first $t$ tokens — zero mass on future tokens, positive mass on past and current ones, summing to 1. The output is the weighted sum of value vectors:

$$
\mathbf{O} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{T \times d_v}
$$

Row $t$ of $\mathbf{O}$ is a linear combination of $\mathbf{v}_1, \dots, \mathbf{v}_t$ weighted by how much the query at position $t$ attends to each earlier token.

### The Complete Formula

$$
\boxed{\;\mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V}) \;=\; \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}\;}
$$

This is the core operation inside every transformer block.

### Connection to Lesson 07

The uniform averaging matrix $\mathbf{W}$ from Lesson 07 had row $t$ equal to $[\tfrac{1}{t}, \dots, \tfrac{1}{t}, 0, \dots, 0]$. Attention produces the same-shape matrix $\mathbf{A}$ — lower-triangular with rows summing to 1 — but the entries are **learned and data-dependent**. When $\mathbf{W}_Q = \mathbf{W}_K = \mathbf{0}$, every score is 0 and softmax produces uniform weights — recovering exactly the Lesson 07 prefix average over the causal triangle.

---

## Core Concepts

Attention replaces a fixed mixing matrix with one whose entries depend on the content of the sequence. A token chooses how much to listen to every earlier token by asking *"how well does my query match your key?"*. The scaled dot product gives this compatibility score, softmax normalises it into a distribution, and the weighted sum of value vectors produces a context-aware summary.

The causal mask is what makes this work as a *language model*: without it, the model would be able to read future tokens during training and would trivially "predict" tokens it has already seen. With the mask, each position $t$ can only look at $1..t$, matching the constraint of autoregressive generation.

**Common misconception:** The scaling factor $1/\sqrt{d_k}$ is not cosmetic — without it, gradients through softmax vanish as $d_k$ grows. It also is not $1/d_k$: the relevant quantity is the *standard deviation* of the dot product, which scales like $\sqrt{d_k}$.

**Common misconception:** $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ are not three separate inputs — they are three projections of the *same* input $\mathbf{X}$. This is called **self-attention**. The weights $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ let the network discover what query/key/value features to extract.

---

## Simulations

### `attention_weights.r` — Scores to Weights

**What it computes:**
Builds $T = 5$, $d_k = 4$ hand-crafted query and key matrices designed so the attention pattern is interpretable. Walks through the derivation step by step: raw score matrix, scaled score matrix, causal mask applied, row-wise softmax. Prints each intermediate; plots the three-stage pipeline and the final attention weight matrix.

**What to observe:**
- The raw score matrix $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top / \sqrt{d_k}$ has the largest values where queries and keys align.
- After adding the causal mask $\mathbf{M}$, the upper triangle contains $-10^9$ — effectively $-\infty$.
- After row-wise softmax, every upper-triangle entry of $\mathbf{A}$ is ~0; rows sum to exactly 1.
- Row 1 of $\mathbf{A}$ is $[1, 0, 0, 0, 0]$ — token 1 can only attend to itself.

**Verify by hand:**
Row 1 after masking has only one non-zero score (the diagonal), so $A_{1,1} = \mathrm{softmax}([s, -\infty, -\infty, -\infty, -\infty])_1 = 1$. Confirm this matches the printed row.

---

### `attention_output.r` — Full Pipeline X → Q, K, V → O

**What it computes:**
Builds a hand-set input $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$ ($T=5$, $d_{\text{model}}=6$). Applies three hand-set linear projections $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ to produce $\mathbf{Q}, \mathbf{K}, \mathbf{V}$. Runs the full attention computation and prints $\mathbf{O}$. Counts parameters: $3 \cdot d_{\text{model}} \cdot d_k$ for the projections.

**What to observe:**
- Shapes: $\mathbf{X}$ is $T \times d_{\text{model}}$; $\mathbf{Q}, \mathbf{K}$ are $T \times d_k$; $\mathbf{V}$ is $T \times d_v$; $\mathbf{O}$ is $T \times d_v$.
- $\mathbf{O}$ has one row per input token — attention preserves sequence length.
- Row $t$ of $\mathbf{O}$ equals $\sum_{i \le t} A_{t, i}\, \mathbf{v}_i$: a convex combination of value vectors from past and current tokens.
- The parameter count for this attention block is $3 \cdot d_{\text{model}} \cdot d_k$, independent of $T$.

**Verify by hand:**
For row 1 of $\mathbf{O}$, only $A_{1,1} = 1$ is non-zero. So $\mathbf{o}_1 = \mathbf{v}_1$. Confirm by checking $\mathbf{O}(1) = \mathbf{V}(1)$ in the printed output.

---

## Exercises

1. **The scale matters.** Remove the $1/\sqrt{d_k}$ factor in `attention_weights.r`. How does the attention matrix change when $d_k = 4$? Try bumping the diagonal of $\mathbf{Q}$ and $\mathbf{K}$ to make the dot products larger — what does softmax do as the scores grow?

2. **Drop the mask.** Disable the causal mask in `attention_weights.r`. Confirm that token 1 now attends to future tokens. Why does this break language-model training?

3. **Uniform weights as a limit.** In `attention_weights.r`, set $\mathbf{Q}$ and $\mathbf{K}$ to all zeros. What are the attention weights? Compare to the averaging matrix $\mathbf{W}$ from Lesson 07.

4. **Parameter counting.** For $d_{\text{model}} = 384$, $d_k = d_v = 64$, compute the parameter count of $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ together. How many parameters does this attention block have if we also include an output projection $\mathbf{W}_O \in \mathbb{R}^{d_v \times d_{\text{model}}}$?

5. **Complexity.** The score matrix $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top$ has $T \times T$ entries. How does the cost of computing $\mathbf{S}$ scale with sequence length $T$? Why is this the bottleneck that motivates efficient attention variants (FlashAttention, sliding window, linear attention)?
