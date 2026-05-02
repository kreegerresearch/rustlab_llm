# Lesson 17: Learning Rate Scheduling

[Lesson 16](16-adamw-optimizer.md) treated the learning rate $\eta$ as a single fixed number. Real LLM training never holds $\eta$ constant. Instead it follows a **schedule**: a small linear *warmup* at the start (so the model sees gentle gradients while $\hat{\mathbf{m}}$ and $\hat{\mathbf{v}}$ initialise), then a long *cosine decay* (so the optimiser slows as it nears a minimum). This lesson derives the warmup-then-cosine schedule used in nanoGPT, GPT-3, LLaMA, and most modern transformers — and shows why warmup prevents early-training divergence.

## Learning Objectives

- Define **linear warmup** and explain why a transformer needs to see small learning rates for the first few hundred steps.
- Define **cosine decay** with a minimum-LR floor and explain why it converges more smoothly than step decay or exponential decay.
- Combine the two into a single $\eta(t)$ function and plot it.
- Connect schedule choices to **Adam's bias correction**: the first ~$1/(1-\beta_2)$ steps are dominated by initialisation noise; warmup keeps the model out of trouble while the moment estimates stabilise.
- Recognise the hyperparameters that real LLM training papers report: `warmup_steps`, `lr_decay_steps`, `min_lr`, `max_lr`.

## Background

Adam and AdamW from [Lesson 16](16-adamw-optimizer.md). Bias correction $1 - \beta_2^t$ that takes ~$1/(1-\beta_2) \approx 1000$ steps to converge. Cross-entropy loss as the optimisation target from [Lesson 03](03-cross-entropy-loss.md).

## The Warmup Phase

### Theory

For the first few hundred steps Adam's $\hat{\mathbf{v}}$ is dominated by whatever the *initial* gradient magnitudes happen to be. A randomly-initialised transformer has fresh embeddings and fresh attention weights; gradients can be far from typical-training scale. If $\eta$ starts at its peak value, the first few updates can:

- Push parameters out of the well-conditioned basin chosen by the initialiser (Xavier/He scale).
- Saturate softmaxes and produce nearly-zero attention gradients.
- Make $\hat{\mathbf{v}}$ commit to a too-large or too-small estimate that then takes many steps to correct.

The fix: ramp $\eta$ linearly from 0 to $\eta_{\max}$ over the first $T_w$ steps:

$$\eta_{\text{warmup}}(t) \;=\; \eta_{\max} \cdot \frac{t}{T_w} \quad \text{for } t \in [0, T_w].$$

Typical $T_w$ in published papers: **2000 steps for GPT-3, 5000 for LLaMA, 100–500 for nanoGPT-scale runs**. The rule of thumb: $T_w \ge 1/(1-\beta_2) \approx 1000$ steps for default $\beta_2 = 0.999$, so the schedule pulls $\eta$ from 0 to peak just as $\hat{\mathbf{v}}$ becomes a reliable scale estimate.

## The Cosine Decay Phase

### Theory

After warmup, decay $\eta$ smoothly from $\eta_{\max}$ down to a floor $\eta_{\min}$ over the remaining $T_d - T_w$ steps:

$$\eta_{\text{decay}}(t) \;=\; \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\pi \cdot \frac{t - T_w}{T_d - T_w}\right)\right)\quad \text{for } t \in (T_w, T_d].$$

At $t = T_w$ this evaluates to $\eta_{\max}$ (the cosine is $\cos 0 = 1$); at $t = T_d$ it evaluates to $\eta_{\min}$ (the cosine is $\cos \pi = -1$). The shape is **slow-fast-slow**: gentle decrease at the start of the decay (so we keep exploring near the still-high LR), aggressive drop in the middle, gentle approach to the floor near the end (so the final updates are small enough to settle into a flat region of the loss surface).

Why cosine and not exponential or step decay?

- **Step decay** (drop by 10× at fixed milestones) introduces sudden gradient-magnitude shocks that disrupt Adam's moment estimates.
- **Exponential decay** $\eta_t = \eta_0 \gamma^t$ is too aggressive early and too slow late.
- **Cosine** has zero derivative at both endpoints — the LR transitions smoothly into the warmup peak and into the final floor, with no discontinuities for Adam to chase.

A typical schedule sets $\eta_{\min} = 0.1 \cdot \eta_{\max}$. The 10× cap on the LR range keeps the optimiser productive even at the end of training.

### Example — Build the full schedule and plot it

```rustlab
T_total  = 5000;
T_warmup = 500;
T_decay  = T_total;
eta_max  = 6e-4;
eta_min  = 6e-5;

eta_sched = zeros(T_total + 1);
for t = 0:T_total
  if t < T_warmup
    eta_sched(t + 1) = eta_max * (t / T_warmup);
  else
    progress = (t - T_warmup) / (T_decay - T_warmup);
    eta_sched(t + 1) = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * progress));
  end
end
```

```rustlab
figure()
steps = 0:T_total;
plot(steps, eta_sched, "color", "blue", "label", "warmup + cosine")
title("Learning-rate schedule (T_w=500, T_total=5000)")
xlabel("step")
ylabel("eta")
```

Three regions are visible: a steep linear ramp to the peak in the first 500 steps, a gentle plateau near the peak, and a long cosine decay to the floor.

## Schedule and Loss Curve

### Theory

A simulated training run shows the schedule's *effect*. With a too-aggressive constant LR the loss diverges in the first few steps; with a too-conservative constant LR it converges slowly; with the warmup+cosine schedule it tracks the conservative curve early (during warmup) and the aggressive curve later (peak LR), then settles below both.

### Example — Simulated loss curves under three schedules

A noisy quadratic surrogate for cross-entropy. Three runs: constant high LR, constant low LR, warmup+cosine.

```rustlab
seed(17);

function L = surrogate_loss(theta)
  L = 0.5 * (10.0 * theta(1) ^ 2 + theta(2) ^ 2) + 1.0;   % +1 floor like real CE
end
function g = surrogate_grad(theta)
  g = [10.0 * theta(1), theta(2)];
end

n_train = 300;
sigma = 0.3;
init  = [-3.0, 2.0];

% Run 1: constant high LR (often diverges with bad init)
theta = init;
loss_high = zeros(n_train + 1);
loss_high(1) = surrogate_loss(theta);
for t = 1:n_train
  g = surrogate_grad(theta) + sigma * randn(2);
  theta = theta - 0.18 * g;
  loss_high(t + 1) = surrogate_loss(theta);
end

% Run 2: constant low LR (always safe, never fast)
theta = init;
loss_low = zeros(n_train + 1);
loss_low(1) = surrogate_loss(theta);
for t = 1:n_train
  g = surrogate_grad(theta) + sigma * randn(2);
  theta = theta - 0.02 * g;
  loss_low(t + 1) = surrogate_loss(theta);
end

% Run 3: warmup + cosine schedule
T_w  = 30;
T_d  = n_train;
eta_p = 0.18;
eta_f = 0.02;
theta = init;
loss_sched = zeros(n_train + 1);
loss_sched(1) = surrogate_loss(theta);
for t = 1:n_train
  if t <= T_w
    eta_t = eta_p * (t / T_w);
  else
    progress = (t - T_w) / (T_d - T_w);
    eta_t = eta_f + 0.5 * (eta_p - eta_f) * (1 + cos(pi * progress));
  end
  g = surrogate_grad(theta) + sigma * randn(2);
  theta = theta - eta_t * g;
  loss_sched(t + 1) = surrogate_loss(theta);
end

print("constant high LR  final loss:", loss_high(n_train + 1));
print("constant low  LR  final loss:", loss_low(n_train + 1));
print("warmup+cosine     final loss:", loss_sched(n_train + 1));
```

```rustlab
figure()
steps = 0:n_train;
plot(steps, loss_high,  "color", "red",   "label", "constant high LR")
hold("on")
plot(steps, loss_low,   "color", "blue",  "label", "constant low LR")
plot(steps, loss_sched, "color", "green", "label", "warmup + cosine")
hold("off")
title("Loss vs step under three LR strategies")
xlabel("step")
ylabel("L")
legend("high", "low", "warmup+cosine")
```

The high-LR run shows large fluctuations early (the optimiser overshoots before noise averages out); the low-LR run is smooth but slow; the schedule run starts gentle, accelerates as the warmup completes, and decays into a low-noise tail.

## Hyperparameter Cheat Sheet

### Theory

What real published transformers use:

| Model | $\eta_{\max}$ | $T_w$ | $T_d$ | $\eta_{\min} / \eta_{\max}$ |
|---|---|---|---|---|
| nanoGPT (124M) | $6 \times 10^{-4}$ | 100 | 5000 | 0.1 |
| GPT-3 (175B) | $6 \times 10^{-5}$ | 375 M tokens | 260 B tokens | 0.1 |
| LLaMA-1 (7B) | $3 \times 10^{-4}$ | 2000 | 1 T tokens | 0.1 |
| LLaMA-2 (70B) | $1.5 \times 10^{-4}$ | 2000 | 2 T tokens | 0.1 |

Patterns to notice:

- **$\eta_{\max}$ scales roughly with $1/\sqrt{N_{\text{params}}}$** — bigger models need smaller peak LRs because they have more parameters whose collective updates can destabilise the loss.
- **$T_w$ is roughly fixed at a few hundred to a few thousand steps** — independent of model size, because it is bounded below by Adam's bias-correction horizon $1/(1-\beta_2)$.
- **$\eta_{\min} / \eta_{\max} \approx 0.1$** is the universal default. Going lower wastes the tail of training; going higher leaves Adam too jittery to settle.

## Connection to Information Theory

### Theory

The schedule has a clean reading in terms of the bit budget from [Lesson 03](03-cross-entropy-loss.md):

- **Warmup:** the model has not yet committed to *any* hypothesis about the data distribution. Its initial cross-entropy is roughly $\log_2 |\mathcal{V}|$ bits per token (a uniform-prior baseline). Big steps would commit hard to whichever direction the first noisy gradients point — squandering bits to fit noise. Warmup keeps the model close to its uniform prior until enough minibatches have accumulated to give a robust gradient direction.
- **Cosine plateau:** once $\hat{\mathbf{v}}$ has stabilised, the model is in the regime where each gradient step transmits genuine signal (a low *Bayes risk* update). Big steps near the peak LR pay off: each update reduces real cross-entropy quickly.
- **Cosine decay tail:** late in training, the gradient is mostly minibatch noise plus a small genuine component. The signal-to-noise ratio of any single step is low. Smaller steps mean the optimiser averages over more noisy gradients per unit-of-distance moved, lowering the risk of overshooting the true minimum and locking in spurious bits.

Read this way, an LR schedule is not a heuristic — it is a *time-varying step size* that matches the optimiser's cautiousness to the current signal-to-noise ratio of the gradient.

## Key Takeaways

- **Warmup** ramps $\eta$ from 0 to $\eta_{\max}$ over the first hundreds-to-thousands of steps so Adam's moment estimates stabilise before any aggressive update happens.
- **Cosine decay** smoothly transitions $\eta$ from peak to floor over the remaining steps, with zero derivative at both endpoints — no discontinuities that would shock Adam.
- The combination `linear warmup + cosine decay` is the de-facto LLM standard; the only knobs that change between papers are $\eta_{\max}$, $T_w$, $T_d$, and the floor ratio $\eta_{\min}/\eta_{\max}$.
- Warmup is bounded below by Adam's bias-correction horizon $1/(1-\beta_2) \approx 1000$ steps; LRs *higher* than the peak in early training are usually divergent.
- $\eta_{\min}/\eta_{\max} \approx 0.1$ is universal across published models.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `lr_schedule.r` | the warmup + cosine schedule curve and its three phases (warmup, plateau, decay) |
| `schedule_vs_constant.r` | three simulated training runs (high-LR, low-LR, scheduled) on a noisy 2D bowl; loss curves overlaid |

Run all with `make lesson-17` (or `rustlab run lessons/17-learning-rate-scheduling/<name>.r`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `eta_sched(1)` | `0` (cold start) |
| `eta_sched(T_warmup + 1)` | $\eta_{\max} = 6 \times 10^{-4}$ |
| `eta_sched(T_total + 1)` | $\eta_{\min} = 6 \times 10^{-5}$ (the floor) |
| `loss_high(end)` (constant high LR) | high variance, often above the schedule's |
| `loss_low(end)` (constant low LR) | smooth but slow — clearly above schedule's |
| `loss_sched(end)` (schedule) | lowest of the three, with low tail variance |

## Exercises

1. **Why a smooth ramp.** Replace the warmup ramp $\eta_{\text{warmup}}(t) = \eta_{\max} \cdot t / T_w$ with a step jump (LR = 0 for $t < T_w/2$, then $\eta_{\max}$). What goes wrong with Adam's first update after the jump?
2. **The cosine endpoints.** Verify algebraically that $\eta_{\text{decay}}(T_w) = \eta_{\max}$ and $\eta_{\text{decay}}(T_d) = \eta_{\min}$.
3. **Warmup length scales with $\beta_2$.** Adam's bias correction $1 - \beta_2^t$ reaches 0.99 at $t \approx \log(0.01) / \log(\beta_2)$. Compute this for $\beta_2 \in \{0.99, 0.999, 0.9999\}$. What does each tell you about the minimum sensible $T_w$?
4. **Re-run schedule_vs_constant.r** with $\sigma = 1.0$ instead of $0.3$. Which schedule (high-LR vs scheduled) is most robust to higher gradient noise? Why?
5. **What if you skip cosine.** Modify `lr_schedule.r` to keep $\eta = \eta_{\max}$ flat after warmup (no decay). Run on a real loss surface (e.g. import `optimizer_comparison.r` from Lesson 16). Does it ever converge? Why or why not?

## What's next

Lesson 18 stitches together everything in Phase 6: backprop (Lesson 15) computes the gradient, AdamW (Lesson 16) consumes it, and the schedule (this lesson) modulates the step size. The result is the full **training loop** — minibatch sample, forward pass, backward pass, optimiser step, log loss, repeat — with the diagnostic curves (train vs validation loss, gradient norm) that tell you whether the run is healthy.
