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

At random init the loss is roughly $\log |\mathcal{V}| = \log 3 = 1.099$ nats — the uniform-prior baseline. The model has not learned anything yet.

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
print(sprintf("dL/dW shape: %dx%d (d_emb x vocab)", size(dW_ab, 1), size(dW_ab, 2)));
print(sprintf("dL/dE shape: %dx%d (vocab x d_emb)", size(dE_ab, 1), size(dE_ab, 2)));

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

For this lesson, the corpus is a 60-character periodic sequence over $\{a, b, c\}$. Training uses the 49 consecutive bigram pairs among the first 50 characters; validation uses the 9 pairs spanning characters 50–59 (the 60th character is unused). Repeating the same finite training data forever is the regime where a train/val gap becomes visible: train loss drops to a number reflecting the empirical bigram conditional entropy, while val loss bottoms out at the same number when train and val see the same transition statistics, or higher when their finite samples differ.

### Example — End-to-end training

The full training run (corpus, model, AdamW, schedule, diagnostics) is fairly long; the standalone script `train_loop.rlab` carries it out. The notebook focuses on the diagnostic plots that come out.

The expected diagnostics for a healthy run:

- **Train and val loss both fall** from ≈1.1 nats (uniform prior) toward the corpus's bigram conditional entropy ≈0.347 nats. The run lands at train ≈0.34 and val ≈0.39. Both sit *near* the 0.347 floor; the small train/val gap is a finite-sample artefact — 5 of the 9 validation pairs start from the ambiguous state `b` (which costs $\ln 2 \approx 0.69$ each), versus 24 of the 49 training pairs, so the two averages weight the uncertain branch slightly differently.
- **Validation loss tracks train loss** — no overfitting, because a token→distribution model of rank $\lvert\mathcal{V}\rvert$ can only ever reproduce the empirical bigram table; extra embedding width adds nothing to the fit.
- **Gradient norm** starts large, then decays smoothly as the optimiser approaches the minimum, with brief upticks when the LR ramp from warmup amplifies updates.

Width alone cannot make this model overfit: any embedding dimension $d \ge \lvert\mathcal{V}\rvert$ converges to the *same* bigram distribution, so when train and val share transition statistics the val loss plateaus at the train floor rather than rising. Overfitting-by-capacity needs features that can *separate* train contexts from val contexts — a longer context window or more layers — not just more parameters at rank $\lvert\mathcal{V}\rvert$.

## Reading the Diagnostics

### Theory

Three plots are produced by `train_loop.rlab`:

**1. `train_val_loss.svg` — train loss (red) vs validation loss (blue) per logged step.** What to look for:

| Pattern | Diagnosis |
|---|---|
| Both flat near $\log \lvert\mathcal{V}\rvert$ | underfitting — increase model size or LR |
| Both falling, val tracks train | healthy — keep training |
| Train falling, val rising | overfitting — add regularisation, more data, or stop earlier |
| Loss spikes mid-training | LR too high, exploding gradients, or numerical instability |

**2. `grad_norm.svg` — $\|\nabla L\|_2$ per logged step (log scale, from step 1).** A healthy curve declines several orders of magnitude over training — this run drops from ≈$1.8\times10^{-1}$ at step 1 to ≈$9\times10^{-8}$ at the end (~6 orders). A flat-and-large grad norm suggests the LR is too small to make progress; a flat-and-tiny grad norm at high loss suggests vanishing gradients (not relevant here at depth 1, but central in deep transformers — Lesson 15).

**3. `lr_curve.svg` — the schedule from Lesson 17.** Annotated with the same step axis so you can correlate any loss-curve oddity with where in the schedule it happened (e.g. divergence right at the warmup peak suggests the peak LR is too high).

### Example — In-notebook diagnostic run

The three plots above come from the full 600-step `train_loop.rlab`. Here is a compact 180-step mirror, run inline so the diagnostics render right here. It reuses `forward_one` and `backward_one` from the blocks above, trains the same 24-parameter model on the period-4 corpus with AdamW + warmup/cosine, and records train loss, val loss, and gradient norm at every step.

```rustlab
seed(18);
E = randn(vocab, d_emb) * 0.3;
W = randn(d_emb, vocab) * 0.3;

pat = [1, 2, 3, 2];                 % a, b, c, b — period 4
corpus = zeros(60);
for i = 1:60
  corpus(i) = pat(mod(i - 1, 4) + 1);
end
n_tr = 49;                          % train pairs among chars 1..50
n_va = 9;                           % val pairs among chars 50..59

% Mean loss over a contiguous run of pairs, reusing forward_one.
function L = range_loss(start_idx, n_pairs, corpus, E, W)
  L = 0.0;
  for k = 0:(n_pairs - 1)
    L = L + forward_one(corpus(start_idx + k), corpus(start_idx + k + 1), E, W);
  end
  L = L / n_pairs;
end

n_steps = 180;
b1 = 0.9; b2 = 0.999; a_eps = 1e-8;
eta_max = 0.15; eta_min = 0.015; T_w = 30;
mE = zeros(vocab, d_emb); vE = zeros(vocab, d_emb);
mW = zeros(d_emb, vocab); vW = zeros(d_emb, vocab);
Ltr = zeros(n_steps + 1); Lva = zeros(n_steps + 1); gnc = zeros(n_steps + 1);
Ltr(1) = range_loss(1,  n_tr, corpus, E, W);
Lva(1) = range_loss(50, n_va, corpus, E, W);
for t = 1:n_steps
  if t <= T_w
    eta_t = eta_max * (t / T_w);
  else
    prog = (t - T_w) / (n_steps - T_w);
    eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * prog));
  end
  dE_sum = zeros(vocab, d_emb); dW_sum = zeros(d_emb, vocab);
  for k = 0:(n_tr - 1)
    [dEk, dWk, Lk] = backward_one(corpus(1 + k), corpus(2 + k), E, W, vocab, d_emb);
    dE_sum = dE_sum + dEk; dW_sum = dW_sum + dWk;
  end
  dE_avg = dE_sum / n_tr; dW_avg = dW_sum / n_tr;
  gnc(t + 1) = sqrt(sum(reshape(dE_avg .^ 2, 1, vocab * d_emb)) + sum(reshape(dW_avg .^ 2, 1, d_emb * vocab)));
  mE = b1 * mE + (1 - b1) * dE_avg;  vE = b2 * vE + (1 - b2) * (dE_avg .^ 2);
  E = E - eta_t * (mE / (1 - b1 ^ t)) ./ (sqrt(vE / (1 - b2 ^ t)) + a_eps);
  mW = b1 * mW + (1 - b1) * dW_avg;  vW = b2 * vW + (1 - b2) * (dW_avg .^ 2);
  W = W - eta_t * (mW / (1 - b1 ^ t)) ./ (sqrt(vW / (1 - b2 ^ t)) + a_eps);
  Ltr(t + 1) = range_loss(1,  n_tr, corpus, E, W);
  Lva(t + 1) = range_loss(50, n_va, corpus, E, W);
end
floor_emp = 24 * log(2) / 49;
final_tr = Ltr(n_steps + 1);
final_va = Lva(n_steps + 1);
print("final train loss:", final_tr);
print("final val   loss:", final_va);
print("empirical floor 24*ln2/49:", floor_emp);
```

After 180 steps the train loss reaches ${final_tr:%.4f}$ and the val loss ${final_va:%.4f}$ — essentially the floor, since a 24-parameter bigram model converges fast. Both hug the empirical floor $24\ln 2/49 = ${floor_emp:%.4f}$; the small train/val gap is the finite-sample weighting discussed below, not overfitting.

```rustlab
figure();
steps = 0:n_steps;
plot(steps, Ltr, "color", "red",  "label", "train")
hold("on")
plot(steps, Lva, "color", "blue", "label", "val")
hline(floor_emp, "green", "empirical floor 0.3395")
hold("off")
title("Train vs val loss (180-step mirror of train_loop.rlab)")
xlabel("step")
ylabel("L (nats)")
legend("train", "val")
```

Both curves fall from the ≈1.1-nat uniform prior and flatten at the green empirical-floor line within the first ~60 steps.

```rustlab
figure();
plot(1:n_steps, log10(gnc(2:(n_steps + 1))), "color", "green", "label", "log10 ||grad||")
title("Gradient norm vs step (log10, from step 1)")
xlabel("step")
ylabel("log10 ||grad||")
```

The gradient norm collapses from ≈$10^{-0.7}$ at step 1 toward ≈$10^{-5}$ — the log-scale signature of an optimiser settling into a minimum.

## Information-Theoretic Sanity Check

### Theory

Two reference points to verify the run hit:

- **Initial loss.** A randomly-initialised softmax over $|\mathcal{V}|$ classes has expected cross-entropy $\log |\mathcal{V}|$ nats. For $|\mathcal{V}| = 3$ that is ${log(3.0):%.4f}$ nats. Any random model should start there ± a small amount of noise from the random init.
- **Optimal loss.** A trained bigram model on the period-4 corpus `"abcbabcb…"` reaches the *conditional entropy* $H(X_{t+1} \mid X_t)$ derived in [Lesson 05](05-bigram-language-model.md). For this corpus that is $H = P(b)\,\ln 2 \approx 0.347$ nats (perplexity ≈ 1.414) — the population floor no Markov-1 model can beat. The run's terminal train loss is $0.3395$, which sits *slightly below* $0.347$: the loss is measured on the finite training sample, and among the 49 training pairs only 24 start from the uncertain state `b` (a fraction $24/49 = 0.490$, below the true $0.5$), so the *empirical* floor is $24\cdot\ln 2 / 49 = 0.3395$ nats — matched to four decimals. The same finite-sample effect resurfaces in [Lesson 20](20-perplexity-and-evaluation.md) as a train PPL of $1.404$ just below the $1.4148$ bigram floor.

Comparing the run's terminal train loss to the analytical bigram entropy is the cleanest "is my training healthy?" test you can run on a tiny problem.

## Connection to Earlier Lessons

### Theory

Every component is a closer look at something already in the series:

- **The forward pass** is a stripped-down Lesson 14 with no attention and no residuals — pure embedding + LM head.
- **The cross-entropy loss** is the Lesson 03 derivation, evaluated per pair instead of per sample.
- **The gradient computation** is the Lesson 15 chain rule, applied to a 2-layer network.
- **The optimiser** is the Lesson 16 AdamW run with weight decay $\lambda = 0$ — i.e. plain Adam for this tiny model (decoupled decay off; Exercise 3 turns it on).
- **The schedule** is the Lesson 17 warmup+cosine, no modifications.

A real GPT training run replaces the embedding+head with the full architecture from Lesson 14 — but the loop, the gradient flow, and the diagnostic recipes are identical. Scaling up changes which numbers fly past, not how the loop is structured.

## Sidebar: Gradient Clipping

### Theory

The AdamW update from [Lesson 16](16-adamw-optimizer.md) makes the *direction* of the step adaptive but not its *magnitude*. A single outlier batch (e.g. a sequence that hits a numerical edge case) can produce a gradient whose norm is 10–100× the typical step. That one step can move the parameters far enough off the loss basin that the next step diverges, and training collapses.

**Gradient clipping** caps the per-step gradient norm before the optimiser sees it. Let $g$ be the concatenated gradient over all parameters and $c$ the clip threshold (typically $c = 1.0$). The clipped gradient is

$$g \leftarrow g \cdot \min\!\left(1,\; \frac{c}{\lVert g \rVert_2}\right).$$

In words: if $\lVert g \rVert_2 \leq c$, do nothing; otherwise rescale $g$ to have norm exactly $c$. The *direction* of the step is preserved (every parameter's gradient scaled by the same factor), only the *magnitude* is bounded.

### Why it matters

The gradient-norm diagnostic plotted above already shows the problem: a healthy run has gradient norm in a stable band. Spikes — visible as occasional jumps of 10–100× — are the failures that clipping silences. Without clipping these spikes either (a) cause a divergent step that wrecks the run, or (b) force a much smaller learning rate to compensate, slowing every other step.

Clipping is **cheap** (one norm + one scalar multiply per step), **safe** (it cannot make a healthy run worse — typical gradients are below the threshold and pass through untouched), and **standard** — nanoGPT, every major LLM training stack, and most published transformer recipes use $c = 1.0$ unconditionally.

### Applying it

To add clipping to the loop above, insert two lines between the gradient accumulation and the AdamW update:

```rustlab
% --- After computing dE_avg and dW_avg, before updating m, v --- (illustrative — comments only)
% gn = sqrt(sum(dE_avg .^ 2, "all") + sum(dW_avg .^ 2, "all"));
% if gn > clip_c
%   scale = clip_c / gn;
%   dE_avg = dE_avg * scale;
%   dW_avg = dW_avg * scale;
% end
```

The training loops in this series do *not* clip — the toy models do not produce spike gradients and clipping would only obscure the diagnostic. Production code should clip with $c = 1.0$ by default.

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
| `overfit_demo.rlab` | same model on a tiny corpus whose validation set contains a transition (1→3) never seen in training; the held-out loss climbs as the model grows confident on the training transitions (train↘, val↗) |

Run all with `make lesson-18` (or `rustlab run lessons/18-training-loop/<name>.rlab`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| Initial train loss | ≈ `1.12` nats (≈ `log 3 = 1.099` ± random-init noise) |
| Final train loss (`train_loop.rlab`) | ≈ `0.34` (`0.3395` = empirical floor $24\ln 2/49$) |
| Final val loss (`train_loop.rlab`) | ≈ `0.39` (`0.3851`; slightly above train — finite-sample weighting) |
| Initial gradient norm (step 1) | ≈ `0.18` |
| Final gradient norm | ≈ `9e-8` (~6 orders smaller) |
| `overfit_demo.rlab` train loss | falls to ≈ `0.277` (`2\ln 2/5` — the empirical floor, not 0) |
| `overfit_demo.rlab` val loss | rises to ≈ `6.2` (the unseen (1→3) transition) |

## Exercises

1. **Sanity-check the initial loss.** Re-seed the model with `seed(N)` for several $N$. Does the initial training loss stay near $\log 3 \approx 1.099$? What does it mean if it doesn't?
2. **Effect of LR.** Modify `train_loop.rlab` to use a constant LR equal to $\eta_{\max}$ (no warmup, no decay). Does training still converge? Where do the diagnostic curves differ from the scheduled run?
3. **Effect of weight decay.** `train_loop.rlab` runs with $\lambda = 0$ (plain Adam). Set $\lambda = 0.1$ and re-run. The model is so tiny that overfitting is not the issue — does *adding* decay change the final loss? Why or why not? (Hint: decoupled decay pulls every weight toward 0 each step, competing with the gradient's pull toward the bigram solution.)
4. **Read the grad-norm plot.** At which step does the gradient norm peak? Correlate it with the LR schedule's peak in `lr_curve.svg`. Why is the alignment expected?
5. **Build the overfit case.** In `overfit_demo.rlab`, increase the embedding dimension to $d = 32$ and grow the training corpus to 12 characters. Re-run and inspect `overfit_demo.svg` — does widening $d$ change the val-loss curve at all? (It shouldn't — rank is capped at $\lvert\mathcal{V}\rvert$.) What *does* move the curve is which transitions the val set holds that the train set never shows.

## What's next

Phase 6 closes here. With backprop, AdamW, the schedule, and the loop in hand, the only thing left to build a real LLM is the data pipeline (Phase 7) and the inference path (Phase 8). [Lesson 19](19-byte-pair-encoding.md) replaces the character-level vocabulary with **byte-pair encoding (BPE)**, the production-grade tokenizer; [Lesson 20](20-perplexity-and-evaluation.md) introduces **perplexity** as the standard metric for comparing language models across corpora and architectures.
