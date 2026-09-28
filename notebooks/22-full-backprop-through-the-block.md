# Lesson 22: Full Backprop Through the Block

[[15-backpropagation]] derived the chain rule one layer at a time; [[13-transformer-block]] assembled the forward pass of one block. This lesson wires the two together: the **complete analytical backward pass** through a Pre-LN single-head transformer block — LayerNorm, scaled dot-product attention, residuals, FFN with GELU, LM head — verified against a finite-difference probe and then used to train the block end-to-end on a corpus that a bigram model cannot solve.

The forward/backward pair built here lives in the shared library `lib/transformer.rlab` (`transformer_forward`, `transformer_backward`, `adamw_step`). It is the engine that the capstone ([[23-putting-it-all-together]]) and the fine-tuning lesson ([[25-fine-tuning-sft-and-dpo]]) run on, so every line of it is derived in this lesson rather than treated as a black box.

## Learning Objectives

- Derive the **full backward pass** through a Pre-LN single-head transformer block: LayerNorm, scaled dot-product attention, residual connection, FFN with GELU, LM head.
- Verify the analytical gradient against a **numerical finite-difference probe**; expect relative error $\sim 10^{-10}$.
- Train a single-block transformer to drive the loss below the bigram floor on a context-2-dependent corpus.
- Read the shared library's `transformer_forward` / `transformer_backward` and map each line to an equation in this lesson.

## Background

- Chain rule through every transformer layer from [[15-backpropagation]] — this lesson composes them.
- The transformer block forward pass from [[13-transformer-block]] — used verbatim (single head, with LayerNorm affines and FFN biases).
- AdamW + warmup-cosine training pattern from [[16-adamw-optimizer]], [[17-learning-rate-scheduling]], and [[18-training-loop]].

<!-- hide -->
```rustlab
run "../lib/transformer.rlab"
```

## A Corpus Where Attention Beats Bigram

### Theory

To demonstrate that backprop through attention works, we need a task that **attention can solve but bigram cannot**. The period-3 corpus `"abbabbabbabb"` (12 tokens, 11 pairs) has exactly this property:

| Context | Bigram says | Truth |
|---|---|---|
| After `a` | always `b` (P=1) | always `b` ✓ |
| After `b` (middle of `abb`) | $P(b\mid b) = 4/7$, $P(a\mid b) = 3/7$ | always `b` ✗ |
| After `b` (end of `abb`) | same as above | always `a` ✗ |

A bigram model cannot distinguish the two `b` contexts. The analytic floor:

$$\mathcal{L}_{\text{bigram}} = \frac{1}{11}\!\left(4 \cdot 0 + 4 \cdot (-\log\tfrac{4}{7}) + 3 \cdot (-\log\tfrac{3}{7})\right) \approx 0.434\ \text{nats/pair}.$$

A trigram (or any model with a 2-token context window) can predict every next token with $P = 1$, so its loss floor is exactly $0$. Attention with $d_{\text{head}} \geq 2$ has enough capacity to encode this context. **The trained model should drive the loss from ~0.6 (random init) below the bigram floor of 0.434 toward 0.**

Beating the floor shows the model uses *more* than the current token — but it does not, on its own, prove the extra signal comes from *attention*. On this short, fixed corpus the model also carries a **fixed sinusoidal positional embedding**, which makes every one of the 12 positions distinct; the FFN can memorise a position→next-token map with attention contributing nothing (exactly the effect explored in [[23-putting-it-all-together]]'s Exercise 3). So the final loss is not a clean isolation of attention. The decisive evidence that backprop *through attention* is correct is the finite-difference gradient check below, which probes a weight (`Wq`) that lives inside the attention path.

### Example — Corpus and parameters

All trainable parameters travel in one struct `P` so that the library functions have short signatures. Positional encodings are fixed (not trained).

```rustlab
seed(24);
vocab = 2;             % tokens: a=1, b=2
d_model = 4;
d_ff = 8;
T_max = 16;

pat = [1, 2, 2];       % a, b, b
T = 12;
ids = zeros(T);
for i = 1:T
  ids(i) = pat(mod(i - 1, 3) + 1);
end
mask = ones(T - 1);    % every prediction position counts

E = randn(vocab, d_model) * 0.3;
PE = sinusoidal_pe(T_max, d_model);
gamma1 = ones(d_model);   beta1 = zeros(d_model);
gamma2 = ones(d_model);   beta2 = zeros(d_model);
Wq = randn(d_model, d_model) * 0.3;
Wk = randn(d_model, d_model) * 0.3;
Wv = randn(d_model, d_model) * 0.3;
Wo = randn(d_model, d_model) * 0.3;
W1 = randn(d_model, d_ff) * 0.3;  b1f = zeros(d_ff);
W2 = randn(d_ff, d_model) * 0.3;  b2f = zeros(d_model);
W_U = randn(d_model, vocab) * 0.3;
P = struct("E", E, "PE", PE, "gamma1", gamma1, "beta1", beta1, "gamma2", gamma2, "beta2", beta2, ...
           "Wq", Wq, "Wk", Wk, "Wv", Wv, "Wo", Wo, "W1", W1, "b1f", b1f, "W2", W2, "b2f", b2f, "W_U", W_U);
print("tokens:", ids);
```

## The Full Backward Pass

### Theory

The forward pass through one Pre-LN block is

$$\begin{aligned}
\mathbf{H}_{\ln 1} &= \mathrm{LN}_1(\mathbf{H}_{\text{in}}) \\
\mathbf{Q}, \mathbf{K}, \mathbf{V} &= \mathbf{H}_{\ln 1} \mathbf{W}_Q,\; \mathbf{H}_{\ln 1} \mathbf{W}_K,\; \mathbf{H}_{\ln 1} \mathbf{W}_V \\
\mathbf{S} &= \mathbf{Q} \mathbf{K}^\top / \sqrt{d_{\text{head}}} + \mathrm{mask} \\
\mathbf{A} &= \mathrm{softmax}(\mathbf{S})\quad\text{(row-wise)} \\
\mathbf{H}_{\text{mid}} &= \mathbf{H}_{\text{in}} + (\mathbf{A} \mathbf{V}) \mathbf{W}_O \\
\mathbf{H}_{\ln 2} &= \mathrm{LN}_2(\mathbf{H}_{\text{mid}}) \\
\mathbf{H}_{\text{out}} &= \mathbf{H}_{\text{mid}} + \mathrm{GELU}(\mathbf{H}_{\ln 2} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2
\end{aligned}$$

Then a final LM head $\mathbf{Z} = \mathbf{H}_{\text{out}} \mathbf{W}_U$ produces logits, and cross-entropy gives the scalar loss. The full backward pass is the chain rule applied to each line in reverse. The non-obvious pieces:

**Softmax row-wise backward.** For one row $\mathbf{a} = \mathrm{softmax}(\mathbf{s})$ with $\mathbf{a}, \mathbf{s} \in \mathbb{R}^T$,

$$\frac{\partial \mathbf{a}_i}{\partial \mathbf{s}_j} = \mathbf{a}_i (\delta_{ij} - \mathbf{a}_j) \quad\Longrightarrow\quad \frac{\partial L}{\partial \mathbf{s}} = \mathbf{a} \odot \left(\frac{\partial L}{\partial \mathbf{a}} - \sum_k \frac{\partial L}{\partial \mathbf{a}_k} \mathbf{a}_k\right).$$

The bracketed term is a scalar per row — the softmax-weighted average of the upstream gradient — that gets subtracted from the upstream gradient itself before scaling by $\mathbf{a}$. This is the operation that makes softmax "stay on the simplex" under backprop.

**LayerNorm backward.** For $\mathbf{y} = \frac{\mathbf{x} - \mu}{\sigma} \cdot \boldsymbol{\gamma} + \boldsymbol{\beta}$ with $\mu = \mathrm{mean}(\mathbf{x})$ and $\sigma = \sqrt{\mathrm{var}(\mathbf{x}) + \varepsilon}$:

$$\frac{\partial L}{\partial \boldsymbol{\gamma}} = \sum_t \frac{\partial L}{\partial \mathbf{y}_t} \odot \tilde{\mathbf{x}}_t, \qquad \frac{\partial L}{\partial \boldsymbol{\beta}} = \sum_t \frac{\partial L}{\partial \mathbf{y}_t},$$

and per-row, with $\tilde{\mathbf{x}} = (\mathbf{x} - \mu)/\sigma$ and $\frac{\partial L}{\partial \tilde{\mathbf{x}}} = \frac{\partial L}{\partial \mathbf{y}} \odot \boldsymbol{\gamma}$:

$$\frac{\partial L}{\partial \mathbf{x}} = \frac{1}{\sigma}\!\left(\frac{\partial L}{\partial \tilde{\mathbf{x}}} - \mathrm{mean}\!\left(\frac{\partial L}{\partial \tilde{\mathbf{x}}}\right) - \tilde{\mathbf{x}} \cdot \mathrm{mean}\!\left(\frac{\partial L}{\partial \tilde{\mathbf{x}}} \odot \tilde{\mathbf{x}}\right)\right).$$

The two mean terms account for the normalisation step's coupling: any change in one input element shifts $\mu$ and $\sigma$, which shifts every output. Without them the backward is wrong by exactly those coupling terms.

**Residual connections** split gradients into two paths: $\mathbf{H}_{\text{mid}} = \mathbf{H}_{\text{in}} + \mathrm{proj}$ implies $\frac{\partial L}{\partial \mathbf{H}_{\text{in}}}$ receives $\frac{\partial L}{\partial \mathbf{H}_{\text{mid}}}$ unchanged, and the same gradient also flows back through the projection branch.

**Embeddings.** Token embedding gradient is a *scatter-add*: the gradient at position $t$ adds into row $\mathrm{ids}(t)$ of $\mathrm{d}\mathbf{E}$.

In the library, `transformer_forward(ids, mask, P)` returns the logits, the masked mean loss, and a `cache` of every intermediate; `ce_dlogits` forms the upstream gradient $(\mathbf{p} - \mathbf{e}_y)/n$ at the logits ([[15-backpropagation]]); `transformer_backward(ids, dlogits, cache, P)` returns a struct `G` of gradients with the same field names as `P`.

### Example — Gradient check

Perturb two scalar parameters — `W_U(1, 1)` (LM head) and `Wq(1, 1)` (deep inside attention) — by $\varepsilon = 10^{-4}$ and compare the central finite-difference loss change to the analytical gradient:

```rustlab
eps_g = 1e-4;
[logits_0, L_0, cache] = transformer_forward(ids, mask, P);
dl = ce_dlogits(ids, mask, logits_0, cache.total);
G = transformer_backward(ids, dl, cache, P);

% LM head entry W_U(1, 1).  (Struct fields cannot be indexed directly, so
% copy the matrix out, perturb, and put it back into a copy of P.)
W_U_plus = P.W_U;   W_U_plus(1, 1)  = W_U_plus(1, 1)  + eps_g;   P_plus  = P;  P_plus.W_U  = W_U_plus;
W_U_minus = P.W_U;  W_U_minus(1, 1) = W_U_minus(1, 1) - eps_g;   P_minus = P;  P_minus.W_U = W_U_minus;
[lo, L_plus, ca]  = transformer_forward(ids, mask, P_plus);
[lo, L_minus, ca] = transformer_forward(ids, mask, P_minus);
num_WU = (L_plus - L_minus) / (2 * eps_g);
dW_U = G.W_U;  ana_WU = dW_U(1, 1);
rel_WU = abs(num_WU - ana_WU) / (abs(num_WU) + abs(ana_WU) + 1e-12);

% Attention entry Wq(1, 1).
Wq_plus = P.Wq;   Wq_plus(1, 1)  = Wq_plus(1, 1)  + eps_g;   P_plus  = P;  P_plus.Wq  = Wq_plus;
Wq_minus = P.Wq;  Wq_minus(1, 1) = Wq_minus(1, 1) - eps_g;   P_minus = P;  P_minus.Wq = Wq_minus;
[lo, L_qp, ca] = transformer_forward(ids, mask, P_plus);
[lo, L_qm, ca] = transformer_forward(ids, mask, P_minus);
num_q = (L_qp - L_qm) / (2 * eps_g);
dWq = G.Wq;  ana_q = dWq(1, 1);
rel_q = abs(num_q - ana_q) / (abs(num_q) + abs(ana_q) + 1e-12);

print("W_U(1,1): numerical =", num_WU, " analytical =", ana_WU, " rel error =", rel_WU);
print("Wq(1,1):  numerical =", num_q,  " analytical =", ana_q,  " rel error =", rel_q);
```

Relative errors of ${rel_WU:%.1e}$ and ${rel_q:%.1e}$ are exactly what a *central* difference should give here: its error is the sum of an $O(\varepsilon^2)$ truncation term (with $\varepsilon = 10^{-4}$, that is $\sim 10^{-8}$ scaled by the third derivative) and the floating-point cancellation in forming $L_+ - L_-$ (two losses that agree to ~15 digits, differenced and divided by $2\varepsilon$). Both land far below the $10^{-5}$ pass threshold, so the analytical gradient — including the path through attention — is correct.

### Example — End-to-end training on the `abb` corpus

With the gradient verified, train for 600 AdamW steps with warmup+cosine. Each step is forward → upstream gradient → backward → AdamW, all from the library:

```rustlab
n_train = 600;  T_w = 60;  eta_max = 0.05;  eta_min = 0.005;
bet1 = 0.9;  bet2 = 0.999;  eps_a = 1e-8;
[M, V] = adamw_init(P);

loss_curve = zeros(n_train + 1);
[lo, L0, cache] = transformer_forward(ids, mask, P);
loss_curve(1) = L0;
for step = 1:n_train
  eta_t = warmup_cosine(step, n_train, T_w, eta_max, eta_min);
  [lo, L, cache] = transformer_forward(ids, mask, P);
  loss_curve(step + 1) = L;
  dl = ce_dlogits(ids, mask, lo, cache.total);
  G = transformer_backward(ids, dl, cache, P);
  [P, M, V] = adamw_step(P, G, M, V, eta_t, step, bet1, bet2, eps_a, 0.0);
end

L_bigram = (4 * 0.0 + 4 * (-log(4.0 / 7.0)) + 3 * (-log(3.0 / 7.0))) / 11;
print("Step 0   L =", loss_curve(1));
print("Final    L =", loss_curve(n_train + 1));
print("Bigram floor =", L_bigram);
```

The loss starts at ${loss_curve(1):%.4f}$ nats/pair (random init over a 2-token vocabulary; $\ln 2 = 0.693$ would be the uniform guess) and ends at ${loss_curve(n_train + 1):%.2e}$ — essentially zero, decisively below the bigram floor of ${L_bigram:%.4f}$. The model has learned to use context beyond the current token. Convergence to the analytic minimum is a strong end-to-end sanity check on the whole training loop, but (as noted above) the final loss alone cannot separate attention from the fixed-PE-plus-FFN route on this small fixed corpus. The decisive evidence that the *backward pass* — and in particular backprop through attention — is correct remains the finite-difference gradient check above.

### Example — Loss curve against the bigram floor

```rustlab
figure();
semilogy(0:n_train, loss_curve, "color", "blue", "label", "train loss")
hold("on")
hline(L_bigram, "red", "bigram floor")
hold("off")
title("Single-block transformer: end-to-end loss vs bigram floor (log scale)")
xlabel("step")
ylabel("L (nats / pair)")
legend("train loss", "bigram floor")
```

On the log axis the descent shows its three phases: the warm-up transient over the first ~60 steps, a long exponential-looking decline, and the flattening as the loss approaches floating-point round-off. The bigram floor is crossed early — long before the cosine decay has finished.

## Key Takeaways

- The chain rule from [[15-backpropagation]] composes into a **complete backward pass** through one transformer block. Every parameter (LN scales, Q/K/V/O projections, FFN, embedding, LM head) receives an analytical gradient.
- A finite-difference gradient check verifies correctness at relative error $\sim 10^{-10}$. Always run one on a new backward implementation.
- On a context-2 corpus where bigram cannot solve the task, full-backprop attention drives the loss **below the analytic bigram floor toward 0** — concrete evidence the training loop is correct end to end.
- The forward/backward pair, the AdamW step, and the schedule live once in `lib/transformer.rlab`; the capstone and the fine-tuning lesson reuse them unchanged.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `full_backprop.rlab` | Gradient check at $W_U(1, 1)$ and $W_q(1, 1)$; 600-step AdamW pretraining on the `abb` corpus; loss-curve figure. Pulls the forward/backward library in with `run "../../lib/transformer.rlab"`. |

Run with `make lesson-22` (or `rustlab run lessons/22-full-backprop-through-the-block/full_backprop.rlab`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `rel_WU` (gradient-check rel error, LM head) | $\approx 2.2 \times 10^{-10}$ |
| `rel_q` (gradient-check rel error, attention) | $\approx 6.2 \times 10^{-11}$ |
| `loss_curve(1)` (random init) | $\approx 0.6263$ nats/pair |
| `L_bigram` (bigram floor on `abb`) | $0.4346$ nats/pair |
| `loss_curve(601)` (after 600 steps) | $\approx 3.9 \times 10^{-9}$ |

## Exercises

1. **Numerical-vs-analytical for every parameter.** Extend the gradient check to `gamma1(1)` (a LayerNorm scale). Why is the LayerNorm gradient harder to get right than the LM head's?
2. **Why 0.434?** Rederive the bigram floor on the `abb` corpus by hand. Show that any model that conditions only on the current token *and has no position information* cannot beat it. Then explain why the transformer here can beat it even with attention disabled — its fixed sinusoidal positional embedding leaks position — and confirm by ablating the PE (set `P.PE = zeros(T_max, d_model)` before training) that the loss then floors at $\approx 0.434$, mirroring [[23-putting-it-all-together]]'s Exercise 3.
3. **The ε trade-off.** Sweep $\varepsilon \in \{10^{-1}, 10^{-2}, \dots, 10^{-9}\}$ in the gradient check and plot the relative error on log–log axes. Identify the truncation-dominated and cancellation-dominated regimes and the optimum near $\sqrt{\text{machine epsilon}}$.
4. **Read the library.** Open `lib/transformer.rlab` and annotate every line of `transformer_backward` with the equation it implements from this lesson. Which two lines implement the residual-connection rule?

## What's next

[[23-putting-it-all-together]] is the capstone: the same library trains a single-block transformer on a BPE-tokenised corpus, checkpoints it, and generates text under every sampling strategy — every component of the curriculum in one run. [[25-fine-tuning-sft-and-dpo]] then reuses the backward pass for supervised fine-tuning and direct preference optimisation.
