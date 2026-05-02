# Lesson 18: The Training Loop

This lesson assembles every piece from Phase 6 into a real training run. A tiny embedding-then-linear language model (24 trainable parameters) is trained on a 60-character corpus with **AdamW** ([Lesson 16](16-adamw-optimizer.md)) and a **warmup + cosine learning-rate schedule** ([Lesson 17](17-learning-rate-scheduling.md)). Gradients come from analytical **backpropagation** ([Lesson 15](15-backpropagation.md)). The diagnostics — train loss, validation loss, gradient norm — are the same ones used to monitor multi-billion-parameter LLM runs.

## Learning Objectives

- Wire one **forward pass** of a tiny LM (embedding lookup → linear head → softmax → cross-entropy) and verify it numerically.
- Wire the **backward pass** by hand and confirm the parameter gradients via a finite-difference check.
- Implement an **AdamW step** with the warmup+cosine schedule.
- Track three diagnostics during training: **train loss, validation loss, gradient norm**.
- Recognise the visual signatures of **underfitting** (both losses high and flat), **healthy training** (both falling, val tracks train), and **overfitting** (train falls while val rises).

## Background

Backprop and the linear-layer gradient triple from [Lesson 15](15-backpropagation.md). AdamW from [Lesson 16](16-adamw-optimizer.md). Warmup+cosine schedule from [Lesson 17](17-learning-rate-scheduling.md). The bigram-language-model setup and CDF sampling from [Lesson 05](05-bigram-language-model.md). Embeddings as $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$ from [Lesson 04](04-embeddings-and-similarity.md).

## The Toy Model

### Theory

The smallest model with a real forward + backward pass: an embedding table followed by a linear head.

$$\mathbf{h}_t = \mathbf{E}_{x_t}, \qquad \boldsymbol{\ell}_t = \mathbf{h}_t \mathbf{W}, \qquad \mathbf{p}_t = \mathrm{softmax}(\boldsymbol{\ell}_t).$$

- $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$ — embedding table; $\mathbf{E}_{x_t}$ is the row indexed by token $x_t$.
- $\mathbf{W} \in \mathbb{R}^{d \times |\mathcal{V}|}$ — language-model head.
- $\mathbf{p}_t \in \mathbb{R}^{|\mathcal{V}|}$ — predicted distribution over the next token.

For vocab $|\mathcal{V}| = 3$ and $d = 4$ the model has $3\cdot 4 + 4\cdot 3 = 24$ parameters. Loss for one $(x_t, x_{t+1})$ pair: $L_t = -\log p_t(x_{t+1})$. Loss over a sequence: average over all consecutive pairs.

### Example — Forward pass on one pair

Set up the model and run the forward pass on the bigram `(a, b)`.

```rustlab
seed(18);
vocab = 3;
d_emb = 4;

E = randn(vocab, d_emb) * 0.3;
W = randn(d_emb, vocab) * 0.3;

% Encode "abc...": a=1, b=2, c=3
function L = forward_one(x_curr, x_next, E, W)
  h = E(x_curr, :);                 % vector of length d_emb (row gather)
  pvec = softmax(h * W);          % vector of length vocab
  L = -log(pvec(x_next));
end

L_ab = forward_one(1, 2, E, W);
print("Loss on bigram (a, b) at init:", L_ab);
```

At random init the loss is roughly $\log_2 |\mathcal{V}| \approx \log 3 = 1.099$ nats — the uniform-prior baseline. The model has not learned anything yet.

## Backward Pass

### Theory

Backprop through the model, top to bottom:

1. **Softmax + CE.** $\bar{\boldsymbol{\ell}}_t = \mathbf{p}_t - \mathbf{1}_{x_{t+1}}$ (Lesson 15).
2. **Linear head $\boldsymbol{\ell} = \mathbf{h}\mathbf{W}$.** $\bar{\mathbf{W}} = \mathbf{h}^\top \bar{\boldsymbol{\ell}}$, $\bar{\mathbf{h}} = \bar{\boldsymbol{\ell}} \mathbf{W}^\top$.
3. **Embedding lookup $\mathbf{h} = \mathbf{E}_{x_t}$.** Only row $x_t$ of $\mathbf{E}$ is touched: $\bar{\mathbf{E}}_{x_t} = \bar{\mathbf{h}}$, all other rows have zero gradient. (Across a minibatch the gradients accumulate at each row whose index appears.)

The total parameter gradient on a batch is the **sum** over all training pairs.

### Example — Backward pass + finite-difference check

```rustlab
% rustlab 0.3 native multi-output: [dE, dW, L] = backward_one(...).
function [dE, dW, L] = backward_one(x_curr, x_next, E, W, vocab, d_emb)
  h = E(x_curr, :);
  pvec = softmax(h * W);
  L = -log(pvec(x_next));

  e_y = zeros(vocab); e_y(x_next) = 1.0;
  dlogits = pvec - e_y;            % vector of length vocab
  dW = h' * dlogits;                % d_emb × vocab
  dh = dlogits * W';                % vector of length d_emb

  dE = zeros(vocab, d_emb);
  for k = 1:d_emb
    dE(x_curr, k) = dh(k);
  end
end

[dE_ab, dW_ab, L_ab] = backward_one(1, 2, E, W, vocab, d_emb);
print("dL/dW shape:", size(dW_ab));
print("dL/dE shape:", size(dE_ab));

% Finite-difference check on W(2, 3)
eps = 1e-5;
Wp = W; Wp(2, 3) = W(2, 3) + eps;
Wm = W; Wm(2, 3) = W(2, 3) - eps;
Lp = forward_one(1, 2, E, Wp);
Lm = forward_one(1, 2, E, Wm);
fd = (Lp - Lm) / (2 * eps);
print("FD vs analytic dL/dW(2,3):  fd =", fd, "  analytic =", dW_ab(2, 3));
```

Finite-difference and analytical gradients match to roughly $10^{-9}$.

## The Training Loop

### Theory

One pass through the loop is the same five lines for any neural network:

```
1. Sample a minibatch of (x_t, x_{t+1}) pairs from the corpus.
2. Forward pass: compute the per-pair losses and the mean batch loss.
3. Backward pass: accumulate dE and dW over the batch, average by batch size.
4. Optimiser step: AdamW update with current scheduled LR.
5. Log: train loss; every K steps, validation loss and gradient norm.
```

For this lesson, the corpus is a 60-character periodic sequence over $\{a, b, c\}$. The training set covers the first 50 characters; the held-out validation set is the remaining 10 (kept disjoint, not seen during gradient updates). Repeating the same finite training data forever is *exactly* the regime where overfitting becomes visible: train loss drops to a number reflecting the ground-truth bigram entropy, while val loss bottoms out at the same number if the data is i.i.d., or higher if the train/val distributions differ.

### Example — End-to-end training

The full training run (corpus, model, AdamW, schedule, diagnostics) is fairly long; the standalone script `train_loop.rlab` carries it out. The notebook focuses on the diagnostic plots that come out.

The expected diagnostics for a healthy run:

- **Train and val loss both fall** from ≈1.1 nats (uniform prior) to ≈0.7 nats (the bigram entropy of the corpus).
- **Validation loss tracks train loss** — there is no overfitting because the model has too few parameters (24) and the corpus is bigram-perfect.
- **Gradient norm** starts large, then decays smoothly as the optimiser approaches the minimum, with brief upticks when the LR ramp from warmup amplifies updates.

If the model were larger relative to the corpus (e.g. 2400 parameters on a 60-character corpus), the val loss would *bottom out and rise* while train loss kept falling — the canonical overfitting signature.

## Reading the Diagnostics

### Theory

Three plots are produced by `train_loop.rlab`:

**1. `train_val_loss.svg` — train loss (red) vs validation loss (blue) per logged step.** What to look for:

| Pattern | Diagnosis |
|---|---|
| Both flat near $\log_2 \|\mathcal{V}\|$ | underfitting — increase model size or LR |
| Both falling, val tracks train | healthy — keep training |
| Train falling, val rising | overfitting — add regularisation, more data, or stop earlier |
| Loss spikes mid-training | LR too high, exploding gradients, or numerical instability |

**2. `grad_norm.svg` — $\|\nabla L\|_2$ per logged step (log scale).** A healthy curve declines roughly two-three orders of magnitude over training. A flat-and-large grad norm suggests the LR is too small to make progress; a flat-and-tiny grad norm at high loss suggests vanishing gradients (not relevant here at depth 1, but central in deep transformers — Lesson 15).

**3. `lr_curve.svg` — the schedule from Lesson 17.** Annotated with the same step axis so you can correlate any loss-curve oddity with where in the schedule it happened (e.g. divergence right at the warmup peak suggests the peak LR is too high).

## Information-Theoretic Sanity Check

### Theory

Two reference points to verify the run hit:

- **Initial loss.** A randomly-initialised softmax over $|\mathcal{V}|$ classes has expected cross-entropy $\log |\mathcal{V}|$ nats. For $|\mathcal{V}| = 3$ that is ${log(3.0):%.4f}$ nats. Any random model should start there ± a small amount of noise from the random init.
- **Optimal loss.** A trained bigram model on the corpus `"abcbabcba…"` reaches the *conditional entropy* $H(X_{t+1} \mid X_t)$ derived in [Lesson 05](05-bigram-language-model.md). For the periodic `abc` corpus that is roughly 0.347 nats (perplexity ≈ 1.414). The training run should converge near that floor — and *cannot* go below it, because no Markov-1 model can.

Comparing the run's terminal train loss to the analytical bigram entropy is the cleanest "is my training healthy?" test you can run on a tiny problem.

## Connection to Earlier Lessons

### Theory

Every component is a closer look at something already in the series:

- **The forward pass** is a stripped-down Lesson 14 with no attention and no residuals — pure embedding + LM head.
- **The cross-entropy loss** is the Lesson 03 derivation, evaluated per pair instead of per sample.
- **The gradient computation** is the Lesson 15 chain rule, applied to a 2-layer network.
- **The optimiser** is the Lesson 16 AdamW, no modifications.
- **The schedule** is the Lesson 17 warmup+cosine, no modifications.

A real GPT training run replaces the embedding+head with the full architecture from Lesson 14 — but the loop, the gradient flow, and the diagnostic recipes are identical. Scaling up changes which numbers fly past, not how the loop is structured.

## Key Takeaways

- The training loop is **five steps, one line each**: sample, forward, backward, optimiser step, log. Memorise it.
- A good run shows train and val loss falling together and bottoming out near the data's intrinsic entropy.
- Overfitting is the gap **train ↘, val ↗**. It is visible in the diagnostic plot before any fancy metric.
- Gradient norm should *decrease* over training. Flat-and-large means the LR is too small; spikes mean instability; tiny-at-high-loss means vanishing gradients.
- Compare the terminal loss to the analytical entropy bound (Lesson 03) — it is the most reliable sanity check on a small corpus.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `train_loop.rlab` | end-to-end training of the embedding+head bigram model on `"abcbabcba…"` with AdamW + warmup+cosine schedule; saves `train_val_loss.svg`, `grad_norm.svg`, `lr_curve.svg` |
| `overfit_demo.rlab` | same model trained on a 6-character corpus to force overfitting; the train/val gap appears clearly |

Run all with `make lesson-18` (or `rustlab run lessons/18-training-loop/<name>.rlab`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| Initial train loss | ≈ `log(3) = 1.099` nats |
| Final train loss (`train_loop.rlab`) | ≈ `0.35–0.45` nats (close to bigram entropy) |
| Final val loss (`train_loop.rlab`) | ≈ same as train loss (no overfit, model is too small) |
| Initial gradient norm | $O(1)$ |
| Final gradient norm | several orders of magnitude smaller |
| `overfit_demo.rlab` train loss | falls to near 0 (memorised) |
| `overfit_demo.rlab` val loss | rises after a few hundred steps (textbook overfit) |

## Exercises

1. **Sanity-check the initial loss.** Re-seed the model with `seed(N)` for several $N$. Does the initial training loss stay near $\log 3 \approx 1.099$? What does it mean if it doesn't?
2. **Effect of LR.** Modify `train_loop.rlab` to use a constant LR equal to $\eta_{\max}$ (no warmup, no decay). Does training still converge? Where do the diagnostic curves differ from the scheduled run?
3. **Effect of weight decay.** Set $\lambda = 0$ in AdamW. The model is so tiny that overfitting is not the issue — but does removing decay change the final loss? Why or why not?
4. **Read the grad-norm plot.** At which step does the gradient norm peak? Correlate it with the LR schedule's peak in `lr_curve.svg`. Why is the alignment expected?
5. **Build the overfit case.** In `overfit_demo.rlab`, increase the embedding dimension to $d = 32$ and the corpus length to 6 characters. Re-run and inspect `train_val_loss.svg` — at what step does the val loss start rising?

## What's next

Phase 6 closes here. With backprop, AdamW, the schedule, and the loop in hand, the only thing left to build a real LLM is the data pipeline (Phase 7) and the inference path (Phase 8). [Lesson 19](19-byte-pair-encoding.md) replaces the character-level vocabulary with **byte-pair encoding (BPE)**, the production-grade tokenizer; [Lesson 20](20-perplexity-and-evaluation.md) introduces **perplexity** as the standard metric for comparing language models across corpora and architectures.
