# Lesson 24: Full Backprop and Fine-Tuning

Lesson 22's capstone trains a full single-block transformer end-to-end — but it does so by *calling* a forward/backward library without deriving the backward half. This lesson opens that black box. The chain rule from [[15-backpropagation]] is wired through the [[13-transformer-block]] forward pass, the gradients are verified against a finite-difference probe, and the resulting machinery — the very library the capstone runs on — is then used to demonstrate three training paradigms:

1. **Pre-training**: end-to-end gradient descent on next-token cross-entropy, driving the loss below the bigram floor on a corpus where context matters.
2. **Supervised fine-tuning (SFT)**: continue training on instruction-formatted data with loss-masked prompts.
3. **Direct preference optimization (DPO)**: contrast a trainable policy against a frozen reference copy on preference pairs.

The same forward/backward library is the spine of all three.

## Learning Objectives

- Derive the **full backward pass** through a Pre-LN single-head transformer block: LayerNorm, scaled dot-product attention, residual connection, FFN with GELU, LM head.
- Verify the analytical gradient against a **numerical finite-difference probe**; expect relative error $\sim 10^{-10}$.
- Train a single-block transformer to drive the loss below the bigram floor on a context-2-dependent corpus.
- Implement **supervised fine-tuning** with prompt-token loss masking and recognise the **catastrophic forgetting** that vanilla SFT produces on the pretraining distribution.
- Implement **DPO** with a frozen reference policy; understand why the reference term is the principled answer to catastrophic forgetting.

## Background

- Chain rule through every transformer layer from [[15-backpropagation]] — this lesson composes them.
- The transformer block forward pass from [[13-transformer-block]] — used verbatim.
- AdamW + warmup-cosine training pattern from [[18-training-loop]].
- The capstone in [[22-putting-it-all-together]] trains the full single-block transformer using exactly the forward/backward library derived here — this lesson is where that backward pass comes from.

## A Corpus Where Attention Beats Bigram

To demonstrate that backprop through attention works, we need a task that **attention can solve but bigram cannot**. The period-3 corpus `"abbabbabbabb"` (12 tokens, 11 pairs) has exactly this property:

| Context | Bigram says | Truth |
|---|---|---|
| After `a` | always `b` (P=1) | always `b` ✓ |
| After `b` (middle of `abb`) | $P(b\mid b) = 4/7$, $P(a\mid b) = 3/7$ | always `b` ✗ |
| After `b` (end of `abb`) | same as above | always `a` ✗ |

A bigram model cannot distinguish the two `b` contexts. The analytic floor:

$$\mathcal{L}_{\text{bigram}} = \frac{1}{11}\!\left(4 \cdot 0 + 4 \cdot (-\log\tfrac{4}{7}) + 3 \cdot (-\log\tfrac{3}{7})\right) \approx 0.434\ \text{nats/pair}.$$

A trigram (or any model with a 2-token context window) can predict every next token with $P = 1$, so its loss floor is exactly $0$. Attention with $d_{\text{head}} \geq 2$ has enough capacity to encode this context. **The trained model should drive the loss from ~0.6 (random init) below the bigram floor of 0.434 toward 0.**

Beating the floor shows the model uses *more* than the current token — but it does not, on its own, prove the extra signal comes from *attention*. On this short, fixed corpus the model also carries a **fixed sinusoidal positional embedding**, which makes every one of the 12 positions distinct; the FFN can memorise a position→next-token map with attention contributing nothing (exactly the effect explored in [[22-putting-it-all-together]]'s Exercise 3). So the final loss is not a clean isolation of attention. The decisive evidence that backprop *through attention* is correct is the finite-difference gradient check in the next section, which probes a weight (`Wq`) that lives inside the attention path.

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

### Example — Gradient check

The `full_backprop.rlab` script first perturbs two scalar parameters — `W_U(1, 1)` (LM head) and `Wq(1, 1)` (deep inside attention) — by $\varepsilon = 10^{-4}$ and compares the finite-difference loss change to the analytical gradient:

```text
Gradient check on W_U(1, 1):
  numerical  = -0.0348566
  analytical = -0.0348566
  rel error  = 2.2e-10 (should be < 1e-5)
Gradient check on Wq(1, 1):
  numerical  = -0.000529412
  analytical = -0.000529412
  rel error  = 6.2e-11 (should be < 1e-5)
Both checks pass: true
```

(The numbers are truncated to six significant figures; the numerical estimate's last few digits shift from run to run and machine to machine with floating-point rounding.) A relative error around $10^{-10}$ is exactly what a *central* difference should give here: its error is the sum of an $O(\varepsilon^2)$ truncation term (with $\varepsilon = 10^{-4}$, that is $\sim 10^{-8}$ scaled by the third derivative) and the floating-point cancellation in forming $L_+ - L_-$ (two losses that agree to ~15 digits, differenced and divided by $2\varepsilon$). The two error sources land the check near $10^{-10}$ — far below the $10^{-5}$ pass threshold, so the analytical gradient is correct.

### Example — End-to-end training on the `abb` corpus

With the gradient verified, train for 600 AdamW steps with warmup+cosine. Expected trajectory: $\mathcal{L}$ starts at ~0.63 (random init over a 2-token vocab), drops below the bigram floor (~0.434) by step ~50, and converges toward 0:

```text
Step 0    L = 0.6262746329095579
Final L = 0.000000003918572521921996
Bigram floor = 0.4345778848093911   -- attention model should beat this.
```

The attention model achieves $\mathcal{L} \approx 4 \times 10^{-9}$, essentially zero. The bigram floor is decisively broken — the model has learned to use context beyond the current token. Convergence to the analytic minimum is a strong end-to-end sanity check on the whole training loop, but (as noted above) the final loss alone cannot separate attention from the fixed-PE-plus-FFN route on this small fixed corpus. The decisive evidence that the *backward pass* — and in particular backprop through attention — is correct remains the finite-difference gradient check above.

## Supervised Fine-Tuning (SFT)

### Theory

SFT is **same architecture, same loss, different data**. The pre-trained checkpoint becomes the initialisation for a new training run on **prompt-response** sequences:

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|R|} \sum_{t \in R} \log P_\theta(x_{t+1} \mid x_{\le t})$$

where $R$ is the set of positions inside the **response** (not the prompt). Prompt positions contribute zero to the loss — they exist only as context for the response predictions. The mechanism is a **loss mask**: a binary vector that gates which positions appear in the gradient computation.

The justification: at inference time the user provides the prompt; the model only needs to generate the response. We do not want to train the model to *predict the prompt*, only to *respond to it*.

### Catastrophic Forgetting

A subtlety: even with loss masking, every parameter still receives a gradient. The response-position predictions depend (via attention) on every prefix token, so the gradient back through attention touches $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$, the embedding $\mathbf{E}$, the LayerNorm scales — everything. Those changes alter the model's behaviour on **other** distributions too.

The `sft.rlab` script demonstrates this directly. After 600 steps pre-training on the `abb` corpus the model reaches $\mathcal{L} \approx 10^{-5}$. After 200 steps of vanilla SFT on a small prompt-response dataset:

```text
Pre-training-distribution loss check (catastrophic-forgetting probe):
  pre-SFT  L (abb corpus): 2.00e-5
  post-SFT L (abb corpus): ~2.8
  delta:                   ~2.8
```

The model has effectively *forgotten* the pretraining task. This is **catastrophic forgetting** — the most-cited problem with vanilla SFT — and it motivates the techniques typically applied alongside it: LoRA (which constrains updates to a low-rank subspace), KL regularisation to the pretrained policy, replay of pretraining data during fine-tuning, and the DPO formulation in the next section.

### Example — SFT with loss masking

The single change versus the pretraining loop is a `loss_mask` row vector that gates which positions contribute. Prompt tokens have `mask = 0`, response tokens `mask = 1`. Computation still flows through every position; only the loss accumulator (and the matching `dlogits` rows in the backward pass) skip the masked positions:

```text
% Sketch — see lessons/24-full-backprop-and-fine-tuning/sft.rlab for the
% complete `forward_with_mask` / `backward_with_mask` pair.
L = 0.0;  total = 0.0;
for t = 1:(T - 1)
  if loss_mask(t) > 0
    p = softmax(logits(t, :));
    L = L - log(p(ids(t + 1)));
    total = total + 1.0;
  end
end
if total > 0
  L = L / total;
end
```

The post-SFT predictions on prompts always pick response token `a` (matching the SFT data), and the model's per-token confidence on the response is $> 0.99$. The SFT objective is solved; the price is paid in pretraining-distribution accuracy.

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

The gradient of $\mathcal{L}_{\text{DPO}}$ with respect to $\log \pi_\theta(y_w \mid x)$ is $-\beta (1 - \sigma)$; with respect to $\log \pi_\theta(y_l \mid x)$ it is $+\beta (1 - \sigma)$. These then chain through standard cross-entropy gradients into the policy's parameters. The backward path is **the same as supervised cross-entropy** with a sign flip and a scalar weight — *not* a new mathematical construct, just a different upstream gradient.

**A numeric anchor at initialisation.** DPO starts with the policy as a verbatim copy of the reference, so every preference margin $r_\theta(x, y_w) - r_\theta(x, y_l)$ is exactly $0$, $\sigma(0) = \tfrac{1}{2}$, and the loss is $-\log \tfrac{1}{2} = \ln 2 \approx 0.6931$ — precisely the `Initial DPO L` the script prints. At that same point the per-example gradient weight $\beta(1 - \sigma) = \beta/2$ (with $\beta = 0.5$, that is $0.25$) is at its maximum; as the policy learns to prefer chosen over rejected, $\sigma \to 1$ and the weight decays toward $0$, so DPO automatically eases off once the margin is comfortably positive.

### Example — DPO on toy preference pairs

The `dpo.rlab` script pre-trains the same model for 400 steps, then freezes the resulting policy as the reference. The preference dataset has three triples; each chosen response is `(a, a)` (rewarded), each rejected response is `(b, b)` (penalised). After 200 DPO steps with $\beta = 0.5$:

- The log-prob margin $\log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x)$ for each prompt is strongly positive — the policy prefers chosen over rejected.
- The pretraining-distribution loss is tracked in parallel; the reference term keeps it close to its starting point, **demonstrating that DPO does what vanilla SFT cannot**: shape the policy on preference signals without catastrophically forgetting prior knowledge.

The script also plots the two curves side by side so the forgetting/non-forgetting comparison is visible at a glance.

## Connecting the Three Paradigms

A typical modern LLM training stack is:

1. **Pre-training** on a giant generic corpus (next-token cross-entropy). This curriculum: lessons 18, 22, and the `full_backprop.rlab` here.
2. **SFT** on a smaller instruction-formatted dataset to produce a *base instruct* model. Loss-masked cross-entropy. Here: `sft.rlab`.
3. **Preference optimisation** (DPO or PPO-with-reward-model) on preference pairs collected from human annotators or AI feedback. Here: `dpo.rlab` (DPO specifically).

This lesson runs the entire pipeline at toy scale, with every gradient hand-derived from Lesson 15 and no library calls. The mechanics generalise verbatim to LLaMA-scale models — only the dimensions and dataset size change.

## Key Takeaways

- The chain rule from [[15-backpropagation]] composes into a **complete backward pass** through one transformer block. Every parameter (LN scales, Q/K/V/O projections, FFN, embedding, LM head) receives an analytical gradient.
- A finite-difference gradient check verifies correctness at relative error $\sim 10^{-10}$. Always run one on a new backward implementation.
- On a context-2 corpus where bigram cannot solve the task, full-backprop attention drives the loss **below the analytic bigram floor toward 0** — concrete evidence the backward pass is correct.
- **SFT** is "same loss, response-token-masked, different data, smaller LR." Trades pretraining-distribution loss for SFT loss — this is **catastrophic forgetting**.
- **DPO** keeps a frozen reference model and contrasts chosen-vs-rejected response log-probs. No separately-trained reward model, no RL, no PPO. The reference term regularises (rather than hard-bounds) the policy's drift toward $\pi_{\text{ref}}$, which is what curbs catastrophic forgetting.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `full_backprop.rlab` | Forward + analytical backward for the full single-block transformer; gradient check at $W_U(1, 1)$ and $W_q(1, 1)$; 600-step AdamW pretraining on the `abb` corpus |
| `sft.rlab` | 600-step pretraining + 200-step SFT with prompt-token loss masking; explicit catastrophic-forgetting probe on the pretraining distribution |
| `dpo.rlab` | 400-step pretraining + 200-step DPO with frozen reference policy; per-prompt log-prob margins; pretraining-distribution loss tracked through DPO |

Run all with `make lesson-24` (or `rustlab run lessons/24-full-backprop-and-fine-tuning/<name>.rlab`). Each script is self-contained (the forward/backward library is duplicated across scripts so each runs without shared state, per project convention).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `full_backprop.rlab` gradient-check rel error | $\sim 10^{-10}$ on both probes |
| Initial loss (random init) | $\approx 0.63$ nats/pair |
| Bigram floor on `abb` corpus | $\approx 0.434$ nats/pair |
| Final loss after 600 pretraining steps | $\sim 4 \times 10^{-9}$ |
| `sft.rlab` final SFT loss (response-only) | $\sim 10^{-2}$ to $10^{-3}$ |
| `sft.rlab` post-SFT abb-corpus loss | $\sim 2.8$ (catastrophic forgetting) |
| `dpo.rlab` final per-prompt margin $\log \pi_\theta(y_w) - \log \pi_\theta(y_l)$ | strongly positive |
| `dpo.rlab` post-DPO abb-corpus loss | close to pre-DPO (bounded by reference term) |

## Exercises

1. **Numerical-vs-analytical for every parameter.** Modify `full_backprop.rlab` to gradient-check `gamma1(1)` (a LayerNorm scale). Why is the LayerNorm gradient harder to get right than the LM head's?
2. **Why 0.434?** Rederive the bigram floor on the `abb` corpus by hand. Show that any model that conditions only on the current token *and has no position information* cannot beat it. Then explain why the transformer here can beat it even with attention disabled — its fixed sinusoidal positional embedding leaks position — and confirm by ablating the PE (set it to zero) that the loss then floors at $\approx 0.434$, mirroring [[22-putting-it-all-together]]'s Exercise 3.
3. **Effect of $\beta$ in DPO.** Re-run `dpo.rlab` with $\beta \in \{0.1, 0.5, 2.0\}$. Plot the final preference margin vs the post-DPO abb-corpus loss. Where is the sweet spot, and how does it depend on $\beta$?
4. **SFT without prompt masking.** Remove the loss mask in `sft.rlab` so every position contributes to the SFT loss. How does the catastrophic-forgetting probe change? Why?
5. **Replay buffer.** Add a "replay" of the pretraining sequence to `sft.rlab` — every other SFT step trains on a pretraining sequence instead. Does this reduce catastrophic forgetting? Does it slow down SFT convergence?
6. **DPO derivation.** Verify analytically that the gradient $\partial \mathcal{L}_{\text{DPO}} / \partial \log \pi_\theta(y_w \mid x) = -\beta (1 - \sigma(\cdot))$ stated in this lesson. (Hint: $\frac{d}{du} \log \sigma(u) = 1 - \sigma(u)$.)

## What's next

The curriculum is now **truly complete**. From characters → BPE → attention → transformer block → full GPT architecture → backpropagation → AdamW → training loops → perplexity → sampling → KV cache → modern architectural variants → and now end-to-end backprop, SFT, and DPO. Every component of a modern LLM has been derived, implemented, and verified.

The natural next directions outside this curriculum:

- **Scale.** Move to a real corpus and a larger model. The math is unchanged; engineering becomes the dominant work.
- **Inference optimisations.** Quantisation, speculative decoding, paged attention — all extend the math here without changing it.
- **Longer context.** RoPE base-frequency interpolation, sliding-window attention, attention sinks — applied to the architecture from [[23-modern-architectural-variants]].
- **RLHF (PPO).** The most-cited alternative to DPO. Requires a separate reward model and the PPO machinery (advantage estimation, clipping). DPO's appeal is that it skips all of that.

Every one of those builds on what you have already derived from first principles. Welcome to the field.
