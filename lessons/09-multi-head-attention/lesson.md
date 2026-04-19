# Lesson 09 — Multi-Head Attention

## Learning Objectives

By the end of this lesson you will be able to:

- Explain why a **single** attention head is limited in the kinds of relationships it can capture and what **multi-head attention** (MHA) buys you.
- Compute $H$ parallel attention heads $\mathbf{A}_1, \dots, \mathbf{A}_H$, each with its own $\mathbf{W}_Q^h, \mathbf{W}_K^h, \mathbf{W}_V^h$, on the same input $\mathbf{X}$.
- Write the **concatenation** of per-head outputs as a horizontal stack along the feature axis, and apply the **output projection** $\mathbf{W}_O$.
- Derive the total parameter count of an MHA block as $4 d_{\text{model}}^2$ when $d_k = d_v = d_{\text{model}}/H$.
- Interpret a grid of head-attention heatmaps and recognise qualitatively different patterns (first-token, previous-token, self, uniform).

---

## Background

This lesson assumes:

- Scaled dot-product attention, the causal mask, and the $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ formulation from Lesson 08.
- Linear layers and matrix multiplication from Lesson 06.
- Block-wise horizontal concatenation of matrices.

---

## Theory

### Why More Than One Head

A single attention head can compute *one* way of relating tokens — for example, "each token attends to the previous one". But language has many relationships: syntactic (subject-verb agreement), coreference (*it* → *mat*), long-range dependencies (quotes), positional (first-token, last-token). A single softmax-weighted sum collapses all of them into one pattern.

**Multi-head attention** runs $H$ independent attention computations in parallel, each with its own projection matrices. Each head can specialise in a different relationship, and the outputs are combined at the end.

### Per-Head Projections

For each head $h = 1, \dots, H$:

$$
\mathbf{Q}_h = \mathbf{X}\mathbf{W}_Q^h, \quad \mathbf{K}_h = \mathbf{X}\mathbf{W}_K^h, \quad \mathbf{V}_h = \mathbf{X}\mathbf{W}_V^h
$$

where $\mathbf{W}_Q^h, \mathbf{W}_K^h \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $\mathbf{W}_V^h \in \mathbb{R}^{d_{\text{model}} \times d_v}$. A common choice is $d_k = d_v = d_{\text{model}}/H$, so splitting the model width into $H$ equal-sized slices.

The per-head attention and output follow Lesson 08 exactly:

$$
\mathbf{A}_h = \mathrm{softmax}\!\left(\frac{\mathbf{Q}_h \mathbf{K}_h^\top}{\sqrt{d_k}} + \mathbf{M}\right), \qquad \mathbf{O}_h = \mathbf{A}_h \mathbf{V}_h \in \mathbb{R}^{T \times d_v}
$$

The causal mask $\mathbf{M}$ is the same for every head.

### Concatenation and Output Projection

Stack the head outputs horizontally:

$$
\mathrm{Concat} = [\mathbf{O}_1, \mathbf{O}_2, \ldots, \mathbf{O}_H] \in \mathbb{R}^{T \times H d_v}
$$

When $d_v = d_{\text{model}}/H$ this is $T \times d_{\text{model}}$ — same shape as the input $\mathbf{X}$.

Apply a final linear projection to mix information across heads:

$$
\mathbf{O} = \mathrm{Concat} \cdot \mathbf{W}_O \in \mathbb{R}^{T \times d_{\text{model}}}, \qquad \mathbf{W}_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
$$

The output projection is critical: without it, each slice of the output depends on only one head, and the heads never interact. $\mathbf{W}_O$ lets downstream layers blend features discovered by different heads.

### Parameter Count

Pack the per-head projections into three combined matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ (head $h$ uses columns $[(h-1)d_k + 1, \dots, h d_k]$). The total parameter count is:

$$
\underbrace{3 \cdot d_{\text{model}}^2}_{\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V} + \underbrace{d_{\text{model}}^2}_{\mathbf{W}_O} \;=\; 4 d_{\text{model}}^2
$$

Notice $H$ does not appear — the count is independent of the number of heads when $d_k = d_{\text{model}}/H$. More heads means narrower heads, not more parameters.

### Why Does This Work?

Each head sees the full input $\mathbf{X}$ but projects it down to a $d_k$-dimensional subspace before computing attention. Different projections emphasise different features of $\mathbf{X}$, so each head sees a different "view" of the sequence. With enough heads and sufficient training, the heads specialise — some learn positional patterns, some learn semantic patterns, some learn long-range structure.

---

## Core Concepts

A single attention head averages value vectors with one set of weights. Multi-head attention averages value vectors with $H$ independent sets of weights, then mixes the results. The shape of the operation stays the same (input and output are both $T \times d_{\text{model}}$), but the expressive power grows because the model can attend to multiple kinds of relationships simultaneously.

The trick of splitting $d_{\text{model}}$ into $H$ narrower heads — rather than running $H$ full-width heads — keeps the parameter count fixed at $4 d_{\text{model}}^2$ regardless of $H$. The cost per head goes down by a factor of $H$, offsetting the $H$-fold parallelism.

**Common misconception:** Heads do not "know" they are supposed to specialise. Specialisation emerges from training: because all heads share the same input and loss, gradient descent finds projections that cover complementary patterns — if two heads learned the same pattern, that would waste capacity.

**Common misconception:** The output projection $\mathbf{W}_O$ is not optional. Without it, slices of $\mathbf{O}$ come from unrelated heads and downstream layers cannot mix them. $\mathbf{W}_O$ is what turns $H$ parallel computations into a unified context-aware representation.

---

## Simulations

### `multi_head_weights.r` — Four Heads, Four Patterns

**What it computes:**
Fixes $T = 6$, $H = 4$, $d_k = 2$. Constructs four hand-set $(\mathbf{Q}_h, \mathbf{K}_h)$ pairs to demonstrate different attention patterns:
- **Head 1 — First-token:** $\mathbf{K}_1$ advertises a feature only in row 1; every query asks for it.
- **Head 2 — Previous-token:** Positions are encoded on the unit circle, $\mathbf{Q}_t$ matches $\mathbf{K}_{t-1}$ exactly.
- **Head 3 — Self:** Same encoding, $\mathbf{Q}_t = \mathbf{K}_t$ so the strongest match is the diagonal.
- **Head 4 — Uniform:** All scores zero, so softmax produces uniform weights over the causal triangle (identical to Lesson 07).

Plots all four attention matrices in a $2 \times 2$ grid.

**What to observe:**
- Each head produces a distinctly different pattern, even though all see the same sequence.
- Head 4's attention matrix is identical to the averaging matrix from Lesson 07 — the Lesson 07 baseline is a *special case* of a degenerate attention head.
- All four attention matrices are lower-triangular (shared causal mask) with rows summing to 1.

**Verify by hand:**
- Head 4 row 4 should be $[0.25, 0.25, 0.25, 0.25, 0, 0]$ — four equal entries. Confirm it matches the printed row.
- Head 1 row 6: since only $\mathbf{K}_1$ has a non-zero dim-1 entry, the scores are $[3, 0, 0, 0, 0, 0]/\sqrt{2}$ before softmax. After softmax, token 1 gets the lion's share (~62%) and tokens 2–6 share the rest roughly equally.

---

### `head_concatenation.r` — Full Pipeline with $\mathbf{W}_O$

**What it computes:**
A compact $T = 4$, $H = 2$, $d_k = d_v = 2$, $d_{\text{model}} = 4$ example. Two heads (one "self", one "uniform"), each with hand-set $\mathbf{V}_h$ taken as a slice of $\mathbf{X}$. Computes per-head outputs $\mathbf{O}_1, \mathbf{O}_2$, concatenates them to shape $T \times d_{\text{model}}$, applies a permutation $\mathbf{W}_O$, and prints the final $\mathbf{O}$. Prints parameter counts.

**What to observe:**
- Shapes: $\mathbf{O}_h \in \mathbb{R}^{4 \times 2}$, $\mathrm{Concat} \in \mathbb{R}^{4 \times 4}$, $\mathbf{O} \in \mathbb{R}^{4 \times 4}$.
- The concatenation places head 1 in the first two columns and head 2 in the last two — no arithmetic, just stacking.
- The output projection (a permutation here) visibly shuffles which columns carry which head's signal.
- Parameter count: $3 \cdot d_{\text{model}}^2 + d_{\text{model}}^2 = 4 \cdot d_{\text{model}}^2 = 64$ for $d_{\text{model}} = 4$.

**Verify by hand:**
Row 1 of $\mathbf{O}_1$ equals $\mathbf{V}_1(1) = [1, 0]$ (self-attention and token 1 attends only to itself). Row 1 of $\mathbf{O}_2$ also equals $\mathbf{V}_2(1) = [0, 0]$ (uniform weight of 1 on its only available token, and $\mathbf{V}_2(1) = X(1, 3:4) = [0, 0]$). So Concat row 1 = $[1, 0, 0, 0]$. Confirm.

---

## Exercises

1. **Role of $\mathbf{W}_O$.** Modify `head_concatenation.r` to set $\mathbf{W}_O = \mathbf{I}$. How does the output relate to the concatenated head outputs now? Argue why a learned $\mathbf{W}_O$ is important even though the concatenation already has the correct shape.

2. **Varying $H$ at fixed $d_{\text{model}}$.** For $d_{\text{model}} = 384$, compute $d_k$ and the total parameter count for $H \in \{1, 2, 4, 6, 8, 12\}$. Confirm the parameter count is independent of $H$. What trades off as $H$ grows?

3. **More heads, same roles.** Extend `multi_head_weights.r` to 8 heads, with heads 5–8 replicating heads 1–4. Do the outputs still differ (they see the same $\mathbf{V}$)? What does this suggest about initialisation and training?

4. **Attending to two things at once.** Design a pair of $(\mathbf{Q}_h, \mathbf{K}_h)$ values so that attention row 4 concentrates roughly evenly on positions 1 *and* 3 but not 2 or 4. What property of softmax makes this possible?

5. **Head count vs. dimension budget.** A model with $d_{\text{model}} = 512$ and $H = 8$ has $d_k = 64$. If you kept $d_k$ fixed at 64 but grew $H$ to 16, how would the parameter count change? Why do practitioners typically fix $d_k$ (not $H$) when scaling model width?
