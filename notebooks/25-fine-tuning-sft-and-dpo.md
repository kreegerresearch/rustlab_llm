# Lesson 25: Fine-Tuning — SFT and DPO

[[22-full-backprop-through-the-block]] built and verified the full backward pass through one transformer block, and [[23-putting-it-all-together]] used it to pre-train a model from scratch. This lesson uses the same machinery — `lib/transformer.rlab`, unchanged — to demonstrate the two fine-tuning paradigms that turn a pre-trained model into a useful one:

1. **Supervised fine-tuning (SFT)**: continue training on instruction-formatted data with loss-masked prompts — and watch it erase the pre-training task (**catastrophic forgetting**).
2. **Direct preference optimization (DPO)**: contrast a trainable policy against a frozen reference copy on preference pairs — and watch the reference term hold the pre-training task in place.

## Learning Objectives

- Implement **supervised fine-tuning** with prompt-token loss masking and recognise the **catastrophic forgetting** that vanilla SFT produces on the pretraining distribution.
- Implement **DPO** with a frozen reference policy; understand why the reference term is the principled answer to catastrophic forgetting.
- Place pre-training, SFT, and preference optimisation in the standard LLM training stack.

## Background

- The shared forward/backward library and the `abb` corpus from [[22-full-backprop-through-the-block]].
- AdamW + warmup-cosine training from [[16-adamw-optimizer]], [[17-learning-rate-scheduling]], and [[18-training-loop]].
- Cross-entropy and KL divergence from [[03-cross-entropy-loss]]; softmax with temperature from [[02-probability-and-softmax]].

<!-- hide -->
```rustlab
run "../lib/transformer.rlab"
```

## Shared Setup: Initialise and Pre-train

### Theory

Both fine-tuning demos start from the same place: a single-block transformer pre-trained on the period-3 `abb` corpus of [[22-full-backprop-through-the-block]], with a vocabulary of three tokens (`a`, `b`, and a separator `sep` that never appears in pre-training data). Two helpers keep the code below short: `init_params` draws the parameters in the same order as the standalone scripts, and `pretrain_abb` runs the AdamW warmup-cosine loop from Lesson 22.

### Example — Helpers

```rustlab
function P = init_params(vocab, d_model, d_ff, T_max)
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
end

function P = pretrain_abb(P, ids_pre, mask_pre, n_steps, T_w)
  [M, V] = adamw_init(P);
  for step = 1:n_steps
    eta_t = warmup_cosine(step, n_steps, T_w, 0.05, 0.005);
    [lo, L, cache] = transformer_forward(ids_pre, mask_pre, P);
    dl = ce_dlogits(ids_pre, mask_pre, lo, cache.total);
    G = transformer_backward(ids_pre, dl, cache, P);
    [P, M, V] = adamw_step(P, G, M, V, eta_t, step, 0.9, 0.999, 1e-8, 0.0);
  end
end

vocab = 3;             % a=1, b=2, sep=3
d_model = 4;  d_ff = 8;  T_max = 16;
pat = [1, 2, 2];
T_pre = 12;
ids_pre = zeros(T_pre);
for i = 1:T_pre
  ids_pre(i) = pat(mod(i - 1, 3) + 1);
end
mask_pre = ones(T_pre - 1);
```

## Supervised Fine-Tuning (SFT)

### Theory

SFT is **same architecture, same loss, different data**. The pre-trained checkpoint becomes the initialisation for a new training run on **prompt-response** sequences:

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|R|} \sum_{t \in R} \log P_\theta(x_{t+1} \mid x_{\le t})$$

where $R$ is the set of positions inside the **response** (not the prompt). Prompt positions contribute zero to the loss — they exist only as context for the response predictions. The mechanism is a **loss mask**: a binary vector that gates which positions appear in the loss and, through `ce_dlogits`, in the upstream gradient.

The justification: at inference time the user provides the prompt; the model only needs to generate the response. We do not want to train the model to *predict the prompt*, only to *respond to it*.

### Catastrophic Forgetting

A subtlety: even with loss masking, every parameter still receives a gradient. The response-position predictions depend (via attention) on every prefix token, so the gradient back through attention touches $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$, the embedding $\mathbf{E}$, the LayerNorm scales — everything. Those changes alter the model's behaviour on **other** distributions too. The probe below measures exactly that: the `abb`-corpus loss before and after SFT.

### Example — Pre-train, then SFT with loss masking

Three formatted sequences, each `[tok1, tok2, sep, response, response]` with response `aa`; the mask `[0, 0, 1, 1]` scores only the two response predictions.

```rustlab
seed(241);
P = init_params(vocab, d_model, d_ff, T_max);
P = pretrain_abb(P, ids_pre, mask_pre, 600, 60);
[lo, L_eval_pre, cache] = transformer_forward(ids_pre, mask_pre, P);
print("Pre-SFT abb-corpus L:", L_eval_pre);

sft_data = [1, 2, 3, 1, 1;
            2, 2, 3, 1, 1;
            2, 1, 3, 1, 1];
sft_mask = [0, 0, 1, 1];
n_sft_seq = size(sft_data)(1);

n_sft = 200;
eta_sft = 0.001;           % lower than pre-training — standard practice
sft_loss_curve = zeros(n_sft + 1);
[lo, L_sft0, cache] = transformer_forward(sft_data(1, :), sft_mask, P);
sft_loss_curve(1) = L_sft0;
[M, V] = adamw_init(P);    % fresh optimiser state for the fine-tuning phase
for step = 1:n_sft
  seq = sft_data(mod(step - 1, n_sft_seq) + 1, :);      % round-robin
  [lo, L, cache] = transformer_forward(seq, sft_mask, P);
  sft_loss_curve(step + 1) = L;
  dl = ce_dlogits(seq, sft_mask, lo, cache.total);
  G = transformer_backward(seq, dl, cache, P);
  [P, M, V] = adamw_step(P, G, M, V, eta_sft, step, 0.9, 0.999, 1e-8, 0.0);
end
print("Final SFT L:", sft_loss_curve(n_sft + 1));

% (a) Does the model produce response "a" after every prompt + sep?
for s = 1:n_sft_seq
  p = next_token_dist(sft_data(s, 1:3), P);
  print("  prompt =", sft_data(s, 1:3), " -> P(. | prompt) =", p, " argmax =", argmax(p));
end

% (b) Catastrophic-forgetting probe on the pre-training distribution.
[lo, L_eval_post, cache] = transformer_forward(ids_pre, mask_pre, P);
print("abb-corpus L: pre-SFT =", L_eval_pre, "  post-SFT =", L_eval_post, "  delta =", L_eval_post - L_eval_pre);
```

The SFT objective is solved: every prompt is answered with `a` at confidence $> 0.99$ and the SFT loss falls to ${sft_loss_curve(n_sft + 1):%.3f}$. The price is paid on the pre-training distribution: the `abb`-corpus loss rises from ${L_eval_pre:%.1e}$ to ${L_eval_post:%.2f}$ nats. The model has effectively *forgotten* the pretraining task. This is **catastrophic forgetting** — the most-cited problem with vanilla SFT — and it motivates the techniques typically applied alongside it: LoRA (which constrains updates to a low-rank subspace), KL regularisation to the pretrained policy, replay of pretraining data during fine-tuning, and the DPO formulation in the next section.

### Example — SFT loss curve

```rustlab
figure();
plot(0:n_sft, sft_loss_curve, "color", "blue", "label", "SFT loss")
title("Supervised fine-tuning loss (only response tokens count)")
xlabel("step")
ylabel("L (nats / response-token)")
```

## Direct Preference Optimization (DPO)

### Theory

DPO (Rafailov et al., 2023) replaces SFT's "maximise the response's log-prob" with "increase the policy's preference margin between a chosen response and a rejected response, **relative to a frozen reference**". The reference is a frozen copy of the pretrained policy; the KL-style term it introduces is what prevents the catastrophic forgetting that vanilla SFT suffers from.

For a preference triple $(\text{prompt } x, \text{ chosen } y_w, \text{ rejected } y_l)$ and policy $\pi_\theta$ with reference $\pi_{\text{ref}}$:

$$r_\theta(x, y) = \beta \cdot \log\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = \beta \cdot (\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x))$$

$$\mathcal{L}_{\text{DPO}}(\theta) = -\log \sigma\bigl(r_\theta(x, y_w) - r_\theta(x, y_l)\bigr)$$

where $\sigma$ is the sigmoid and $\beta > 0$ controls how far the policy can drift from the reference. Decomposing:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\log \sigma\bigl(\beta \cdot \bigl(\log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_w \mid x) + \log \pi_{\text{ref}}(y_l \mid x)\bigr)\bigr).$$

Three properties make this work:

- **No separately-trained reward model.** The contrast here is with RLHF/PPO, which first trains a reward model and then optimises against it. DPO folds the reward into the loss: the reward implicit in $\beta \cdot \log(\pi_\theta / \pi_{\text{ref}})$ is the analytic optimum of the KL-constrained reward-maximisation problem, so no separate reward head is trained. (Versus SFT, the difference is the *data*: SFT needs labelled responses; DPO needs only preference pairs — a chosen and a rejected response per prompt.)
- **No reinforcement learning.** Standard supervised gradients suffice. PPO's exploration/value-baseline machinery is unnecessary.
- **Stability via reference.** The $-\log \pi_{\text{ref}}$ terms are constants from $\pi_\theta$'s perspective, but they enter the *margin* the policy is maximising. The policy is pushed to make chosen more likely than rejected, **but only by the amount the reference doesn't already do**. DPO is derived from a KL-regularised objective in which $\beta$ sets the *strength* of the implicit KL penalty toward $\pi_{\text{ref}}$ — larger $\beta$ keeps the policy closer to the reference — but no explicit bound on the KL distance is imposed.

### Gradient

The gradient of $\mathcal{L}_{\text{DPO}}$ with respect to $\log \pi_\theta(y_w \mid x)$ is $-\beta (1 - \sigma)$; with respect to $\log \pi_\theta(y_l \mid x)$ it is $+\beta (1 - \sigma)$. These then chain through standard cross-entropy gradients into the policy's parameters. The backward path is **the same as supervised cross-entropy** with a sign flip and a scalar weight — *not* a new mathematical construct, just a different upstream gradient `dlogits` handed to `transformer_backward`.

**A numeric anchor at initialisation.** DPO starts with the policy as a verbatim copy of the reference, so every preference margin $r_\theta(x, y_w) - r_\theta(x, y_l)$ is exactly $0$, $\sigma(0) = \tfrac{1}{2}$, and the loss is $-\log \tfrac{1}{2} = \ln 2 \approx 0.6931$ — precisely the initial DPO loss printed below. At that same point the per-example gradient weight $\beta(1 - \sigma) = \beta/2$ (with $\beta = 0.5$, that is $0.25$) is at its maximum; as the policy learns to prefer chosen over rejected, $\sigma \to 1$ and the weight decays toward $0$, so DPO automatically eases off once the margin is comfortably positive.

### Example — DPO on toy preference pairs

Pre-train the same model for 400 steps, freeze the result as the reference (a struct copy), then run 200 DPO steps with $\beta = 0.5$ on three preference triples: chosen response `aa`, rejected response `bb`. The `abb`-corpus loss is tracked at every step as the forgetting probe.

```rustlab
function lp = response_logprob(ids, response_start, response_end, logits)
  lp = 0.0;
  for t = (response_start - 1):(response_end - 1)
    p = softmax(logits(t, :));
    lp = lp + log(p(ids(t + 1)));
  end
end

function dl = dpo_dlogits(ids, resp_start, resp_end, logits, vocab, sign, weight)
  T = size(logits)(1);
  dl = zeros(T, vocab);
  for t = (resp_start - 1):(resp_end - 1)
    p = softmax(logits(t, :));
    e_y = zeros(vocab); e_y(ids(t + 1)) = 1.0;
    dl(t, :) = sign * weight * (p - e_y);
  end
end

seed(242);
P = init_params(vocab, d_model, d_ff, T_max);
P = pretrain_abb(P, ids_pre, mask_pre, 400, 40);
P_ref = P;                 % frozen reference policy pi_ref

chosen   = [1, 2, 3, 1, 1;  2, 2, 3, 1, 1;  2, 1, 3, 1, 1];
rejected = [1, 2, 3, 2, 2;  2, 2, 3, 2, 2;  2, 1, 3, 2, 2];
n_prefs = size(chosen)(1);
resp_start = 4;  resp_end = 5;
zero_mask = zeros(size(chosen)(2) - 1);   % logits only
beta_dpo = 0.5;

n_dpo = 200;  eta_dpo = 0.005;
dpo_loss_curve = zeros(n_dpo + 1);
abb_loss_curve = zeros(n_dpo + 1);
L0_dpo = 0.0;
for s = 1:n_prefs
  [lo_w, l1, c1] = transformer_forward(chosen(s, :), zero_mask, P);
  [lo_l, l2, c2] = transformer_forward(rejected(s, :), zero_mask, P);
  [lo_w_ref, l3, c3] = transformer_forward(chosen(s, :), zero_mask, P_ref);
  [lo_l_ref, l4, c4] = transformer_forward(rejected(s, :), zero_mask, P_ref);
  margin = beta_dpo * (response_logprob(chosen(s, :), resp_start, resp_end, lo_w) ...
                       - response_logprob(chosen(s, :), resp_start, resp_end, lo_w_ref) ...
                       - response_logprob(rejected(s, :), resp_start, resp_end, lo_l) ...
                       + response_logprob(rejected(s, :), resp_start, resp_end, lo_l_ref));
  L0_dpo = L0_dpo - log(1.0 / (1.0 + exp(-margin)));
end
dpo_loss_curve(1) = L0_dpo / n_prefs;
[lo, L_abb_pre, cache] = transformer_forward(ids_pre, mask_pre, P);
abb_loss_curve(1) = L_abb_pre;
print("Initial DPO L:", dpo_loss_curve(1), "  (ln 2 =", log(2), ")   abb-corpus L:", L_abb_pre);

[M, V] = adamw_init(P);
for step = 1:n_dpo
  s = mod(step - 1, n_prefs) + 1;
  ids_w = chosen(s, :);  ids_l = rejected(s, :);
  [lo_w, l1, ca_w] = transformer_forward(ids_w, zero_mask, P);
  [lo_l, l2, ca_l] = transformer_forward(ids_l, zero_mask, P);
  [lo_w_ref, l3, c3] = transformer_forward(ids_w, zero_mask, P_ref);
  [lo_l_ref, l4, c4] = transformer_forward(ids_l, zero_mask, P_ref);
  margin = beta_dpo * (response_logprob(ids_w, resp_start, resp_end, lo_w) ...
                       - response_logprob(ids_w, resp_start, resp_end, lo_w_ref) ...
                       - response_logprob(ids_l, resp_start, resp_end, lo_l) ...
                       + response_logprob(ids_l, resp_start, resp_end, lo_l_ref));
  sig = 1.0 / (1.0 + exp(-margin));
  dpo_loss_curve(step + 1) = -log(sig + 1e-12);
  w_grad = beta_dpo * (1.0 - sig);

  % Chosen response: ordinary cross-entropy direction (+1); rejected: opposite sign (-1).
  G_w = transformer_backward(ids_w, dpo_dlogits(ids_w, resp_start, resp_end, lo_w, vocab, +1.0, w_grad), ca_w, P);
  G_l = transformer_backward(ids_l, dpo_dlogits(ids_l, resp_start, resp_end, lo_l, vocab, -1.0, w_grad), ca_l, P);
  G = struct("E", G_w.E + G_l.E, "gamma1", G_w.gamma1 + G_l.gamma1, "beta1", G_w.beta1 + G_l.beta1, ...
             "gamma2", G_w.gamma2 + G_l.gamma2, "beta2", G_w.beta2 + G_l.beta2, ...
             "Wq", G_w.Wq + G_l.Wq, "Wk", G_w.Wk + G_l.Wk, "Wv", G_w.Wv + G_l.Wv, "Wo", G_w.Wo + G_l.Wo, ...
             "W1", G_w.W1 + G_l.W1, "b1f", G_w.b1f + G_l.b1f, "W2", G_w.W2 + G_l.W2, "b2f", G_w.b2f + G_l.b2f, ...
             "W_U", G_w.W_U + G_l.W_U);
  [P, M, V] = adamw_step(P, G, M, V, eta_dpo, step, 0.9, 0.999, 1e-8, 0.0);

  [lo, L_abb, cache] = transformer_forward(ids_pre, mask_pre, P);
  abb_loss_curve(step + 1) = L_abb;
end
print("Final DPO L:", dpo_loss_curve(n_dpo + 1), "   final abb-corpus L:", abb_loss_curve(n_dpo + 1));

% Per-prompt log-probs after DPO.
for s = 1:n_prefs
  [lo_w, l1, c1] = transformer_forward(chosen(s, :), zero_mask, P);
  [lo_l, l2, c2] = transformer_forward(rejected(s, :), zero_mask, P);
  lp_w = response_logprob(chosen(s, :), resp_start, resp_end, lo_w);
  lp_l = response_logprob(rejected(s, :), resp_start, resp_end, lo_l);
  print("  prompt", s, ":  lp(chosen) =", lp_w, "  lp(rejected) =", lp_l, "  margin =", lp_w - lp_l);
end
```

Two things to read off. First, the **margins** $\log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x)$ are strongly positive for every prompt — the policy prefers chosen over rejected by tens of nats. Second, the **forgetting probe**: the `abb`-corpus loss goes from ${L_abb_pre:%.1e}$ before DPO to ${abb_loss_curve(n_dpo + 1):%.1e}$ after it — both essentially zero, against SFT's rise to ${L_eval_post:%.2f}$ on the same corpus. The reference term does what vanilla SFT cannot: shape the policy on preference signals without catastrophically forgetting prior knowledge.

> [!NOTE]
> Look at the absolute values, not only the margins: on two of the three prompts the log-probability of the *chosen* response is itself very negative (around $-19$ and $-16$ nats). DPO only constrains the *difference* between chosen and rejected; it can — and here does — lower both while widening the gap. This is the documented **likelihood-displacement** behaviour of DPO (Pal et al. 2024; Razin et al. 2024) and one reason production pipelines add an SFT term or monitor the chosen log-probability directly.

### Example — DPO loss and forgetting probe side by side

```rustlab
figure();
subplot(1, 2, 1)
plot(0:n_dpo, dpo_loss_curve, "color", "blue", "label", "DPO loss")
title("DPO loss (lower = better margin chosen/rejected)")
xlabel("step")
ylabel("L_DPO")
subplot(1, 2, 2)
plot(0:n_dpo, abb_loss_curve, "color", "red", "label", "abb-corpus L")
title("Pretraining-distribution loss during DPO")
xlabel("step")
ylabel("L (nats / pair)")
```

## Connecting the Three Paradigms

A typical modern LLM training stack is:

1. **Pre-training** on a giant generic corpus (next-token cross-entropy). This curriculum: [[18-training-loop]], [[22-full-backprop-through-the-block]], and [[23-putting-it-all-together]].
2. **SFT** on a smaller instruction-formatted dataset to produce a *base instruct* model. Loss-masked cross-entropy. Here: `sft.rlab`.
3. **Preference optimisation** (DPO or PPO-with-reward-model) on preference pairs collected from human annotators or AI feedback. Here: `dpo.rlab` (DPO specifically).

This lesson runs the entire pipeline at toy scale, with every gradient hand-derived from [[15-backpropagation]] and [[22-full-backprop-through-the-block]] and no library calls beyond our own. The mechanics generalise verbatim to LLaMA-scale models — only the dimensions and dataset size change.

## Key Takeaways

- **SFT** is "same loss, response-token-masked, different data, smaller LR." Trades pretraining-distribution loss for SFT loss — this is **catastrophic forgetting**.
- **DPO** keeps a frozen reference model and contrasts chosen-vs-rejected response log-probs. No separately-trained reward model, no RL, no PPO. The reference term regularises (rather than hard-bounds) the policy's drift toward $\pi_{\text{ref}}$, which is what curbs catastrophic forgetting.
- DPO constrains margins, not absolute likelihoods: chosen responses can lose probability mass while still winning the comparison (likelihood displacement).
- Both paradigms are ordinary backpropagation with a different upstream gradient at the logits — the library from Lesson 22 is reused unchanged.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `sft.rlab` | 600-step pretraining + 200-step SFT with prompt-token loss masking; explicit catastrophic-forgetting probe on the pretraining distribution |
| `dpo.rlab` | 400-step pretraining + 200-step DPO with frozen reference policy; per-prompt log-prob margins; pretraining-distribution loss tracked through DPO |

Run all with `make lesson-25` (or `rustlab run lessons/25-fine-tuning-sft-and-dpo/<name>.rlab`). Both pull the forward/backward library in with `run "../../lib/transformer.rlab"`.

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `L_eval_pre` (abb-corpus loss after 600-step pretraining) | $\approx 2 \times 10^{-5}$ |
| `sft_loss_curve(201)` (final SFT loss, response-only) | $\approx 0.018$ |
| `L_eval_post` (abb-corpus loss after SFT) | $\approx 2.8$ (catastrophic forgetting) |
| `dpo_loss_curve(1)` | $\ln 2 \approx 0.6931$ |
| Per-prompt margin $\log \pi_\theta(y_w) - \log \pi_\theta(y_l)$ after DPO | $\approx +29$, $+32$, $+47$ nats |
| `abb_loss_curve(201)` (abb-corpus loss after DPO) | $\approx 10^{-4}$ — within $10^{-4}$ of its pre-DPO value |

## Exercises

1. **Effect of $\beta$ in DPO.** Re-run the DPO block with $\beta \in \{0.1, 0.5, 2.0\}$. Plot the final preference margin vs the post-DPO abb-corpus loss. Where is the sweet spot, and how does it depend on $\beta$?
2. **SFT without prompt masking.** Set `sft_mask = [1, 1, 1, 1]` so every position contributes to the SFT loss. How does the catastrophic-forgetting probe change? Why?
3. **Replay buffer.** Add a "replay" of the pretraining sequence to the SFT loop — every other SFT step trains on `ids_pre` instead. Does this reduce catastrophic forgetting? Does it slow down SFT convergence?
4. **DPO derivation.** Verify analytically that the gradient $\partial \mathcal{L}_{\text{DPO}} / \partial \log \pi_\theta(y_w \mid x) = -\beta (1 - \sigma(\cdot))$ stated in this lesson. (Hint: $\frac{d}{du} \log \sigma(u) = 1 - \sigma(u)$.)
5. **Likelihood displacement.** Print $\log \pi_\theta(y_w \mid x)$ every 20 DPO steps. At which step does it start falling on prompts 1 and 2, and what happens to the rejected log-prob at the same time?

## What's next

The curriculum's core is now complete. From characters → BPE → attention → transformer block → full GPT architecture → backpropagation → AdamW → training loops → perplexity → sampling → KV cache → the full backward pass → the capstone → modern architectural variants → and now SFT and DPO. Every component of a modern LLM has been derived, implemented, and verified.

The natural next directions outside this curriculum:

- **Scale.** Move to a real corpus and a larger model. The math is unchanged; engineering becomes the dominant work.
- **Inference optimisations.** Quantisation, speculative decoding, paged attention — all extend the math here without changing it.
- **Longer context.** RoPE base-frequency interpolation, sliding-window attention, attention sinks — applied to the architecture from [[24-modern-architectural-variants]].
- **RLHF (PPO).** The most-cited alternative to DPO. Requires a separate reward model and the PPO machinery (advantage estimation, clipping). DPO's appeal is that it skips all of that.

Every one of those builds on what you have already derived from first principles. Welcome to the field.
