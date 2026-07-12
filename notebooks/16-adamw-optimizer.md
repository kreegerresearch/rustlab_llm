# Lesson 16: The AdamW Optimizer

[Lesson 06](06-linear-layers-and-gradient-descent.md) used vanilla SGD on a convex paraboloid: pick a learning rate, step in the direction $-\nabla L$, and watch the loss decrease monotonically. Transformer loss surfaces are *not* convex paraboloids — they are anisotropic, full of narrow ravines and broad plateaus, and the gradient signal at any one step is a noisy estimate from a minibatch. **Adam** and its weight-decayed variant **AdamW** combine three ideas — momentum, per-parameter adaptive learning rates, and decoupled weight decay — and are the default optimisers for every modern LLM.

## Learning Objectives

- Diagnose the failure modes of **vanilla SGD** on anisotropic loss surfaces (oscillation in steep directions, slow drift in flat directions).
- Define **momentum** as an exponential moving average of the gradient, and explain why it accelerates convergence.
- Derive the **Adam** update with first-moment $\mathbf{m}_t$, second-moment $\mathbf{v}_t$, and bias correction.
- Distinguish **L2-coupled weight decay** (Adam) from **decoupled weight decay** (AdamW) and explain why AdamW is the LLM default.
- Read three optimiser trajectories on the same 2D loss surface and identify which optimiser is which.

## Background

Gradient descent and the loss landscape from [Lesson 06](06-linear-layers-and-gradient-descent.md). Backpropagation as the source of the gradient $\nabla L$ from [Lesson 15](15-backpropagation.md). The loss as a scalar function of all model parameters; the optimiser as the rule that maps gradient → parameter update.

## Notation

| Symbol | Meaning |
|---|---|
| $\theta_t$ | parameter vector at step $t$ |
| $\mathbf{g}_t = \nabla_\theta L(\theta_{t-1})$ | gradient at the current iterate |
| $\eta$ | learning rate (a.k.a. step size) |
| $\beta_1, \beta_2$ | exponential-decay rates for first and second moment (typical: 0.9, 0.999) |
| $\varepsilon$ | small constant for numerical stability (typical: $10^{-8}$) |
| $\lambda$ | weight-decay coefficient |

A **step** consumes one gradient (from one minibatch) and produces one new parameter vector.

## The Failure Mode of Vanilla SGD

### Theory

Plain SGD applies $\theta_t = \theta_{t-1} - \eta \mathbf{g}_t$. On an isotropic quadratic bowl this works beautifully (Lesson 06). On an **anisotropic** quadratic — e.g. $L(\theta) = \tfrac{1}{2}(a\theta_1^2 + b\theta_2^2)$ with $a \gg b$ — the picture is grim. The Hessian has eigenvalues $a$ and $b$; the largest stable learning rate is $\eta < 2/a$, so any step that prevents oscillation along the steep direction $\theta_1$ is far too small to make progress along the flat direction $\theta_2$. The optimiser zigzags across the ravine, advancing slowly along its floor.

### Example — SGD on an elongated bowl

Define $L(\theta_1, \theta_2) = \tfrac{1}{2}(200\theta_1^2 + \theta_2^2)$ (condition number 200) and run SGD from $(-2, 4)$ with $\eta = 0.009$ for 60 steps. The steep direction $\theta_1$ caps the stable learning rate at $\eta < 2/200 = 0.01$; anything larger oscillates and diverges.

```rustlab
% Loss and analytic gradient
function L = loss(theta)
  L = 0.5 * (200.0 * theta(1) ^ 2 + theta(2) ^ 2);
end
function g = grad(theta)
  g = [200.0 * theta(1), theta(2)];
end

theta_sgd = [-2.0, 4.0];
eta = 0.009;
n_steps = 60;
path_sgd = zeros(n_steps + 1, 2);
path_sgd(1, 1) = theta_sgd(1);
path_sgd(1, 2) = theta_sgd(2);

for k = 1:n_steps
  g = grad(theta_sgd);
  theta_sgd = theta_sgd - eta * g;
  path_sgd(k + 1, 1) = theta_sgd(1);
  path_sgd(k + 1, 2) = theta_sgd(2);
end

print("SGD final loss:", loss(theta_sgd));
print("SGD final theta:", theta_sgd);
```

After 60 steps SGD loss is ${loss(theta_sgd):%.4f}$. The steep coordinate $\theta_1$ has converged (it started at $-2$ and is now $\approx 0$), but $\theta_2$ is stuck: $\eta = 0.009$ shrinks it by only $0.9\%$ per step, so after 60 steps it has fallen only from $4$ to ${theta_sgd(2):%.3f}$ — a bit over half its starting value. Almost all of the residual loss is this un-converged flat direction, and no larger $\eta$ is available because the steep direction would blow up.

## Momentum: An EMA of the Gradient

### Theory

Add a velocity buffer $\mathbf{v}_t$ that accumulates past gradients with exponential decay:

$$\mathbf{v}_t = \mu \mathbf{v}_{t-1} + \mathbf{g}_t, \qquad \theta_t = \theta_{t-1} - \eta \mathbf{v}_t.$$

With $\mu \in [0, 1)$ (often 0.9), the recursion $\mathbf{v}_t = \mu \mathbf{v}_{t-1} + \mathbf{g}_t$ *accumulates* past gradients with a steady-state gain of $1/(1-\mu) \approx 10$ (it sums them, weighted by $\mu^k$, rather than averaging to a mean). In the steep direction the gradient oscillates sign and the buffer cancels itself; in the flat direction the gradient has a consistent sign and the buffer grows toward that $\approx 10\times$ gain, multiplying the effective step size. SGD-with-momentum drives along the ravine floor instead of bouncing off the walls.

### Example — Momentum vs plain SGD at a small learning rate

The payoff shows up precisely at a small, safe learning rate, where plain SGD's flat direction crawls. Run both on $L = \tfrac{1}{2}(20\theta_1^2 + \theta_2^2)$ from $(-2, 4)$ with $\eta = 0.01$ and $\mu = 0.9$ for 60 steps.

```rustlab
function Lm = loss20(th)
  Lm = 0.5 * (20.0 * th(1) ^ 2 + th(2) ^ 2);
end
function gm = grad20(th)
  gm = [20.0 * th(1), th(2)];
end

mu    = 0.9;
eta_m = 0.01;
nm    = 60;

% Plain SGD
th_s   = [-2.0, 4.0];
loss_s = zeros(nm + 1);
loss_s(1) = loss20(th_s);
for k = 1:nm
  th_s = th_s - eta_m * grad20(th_s);
  loss_s(k + 1) = loss20(th_s);
end

% SGD + heavy-ball momentum
th_m   = [-2.0, 4.0];
vel    = [0.0, 0.0];
loss_v = zeros(nm + 1);
loss_v(1) = loss20(th_m);
for k = 1:nm
  vel  = mu * vel + grad20(th_m);
  th_m = th_m - eta_m * vel;
  loss_v(k + 1) = loss20(th_m);
end

print("SGD      final loss:", loss_s(nm + 1));
print("Momentum final loss:", loss_v(nm + 1));
print("SGD / momentum ratio:", loss_s(nm + 1) / loss_v(nm + 1));

figure();
steps_m = 0:nm;
plot(steps_m, log10(loss_s + 1e-12), "color", "red",  "label", "SGD")
hold("on")
plot(steps_m, log10(loss_v + 1e-12), "color", "blue", "label", "SGD+momentum")
hold("off")
title("log10 loss: SGD vs momentum (eta=0.01, mu=0.9)")
xlabel("step")
ylabel("log10 L")
legend("SGD", "SGD+momentum")
```

Plain SGD ends at loss ${loss_s(nm + 1):%.4f}$ — the flat direction has barely moved — while momentum reaches ${loss_v(nm + 1):%.4f}$, about ${loss_s(nm + 1) / loss_v(nm + 1):%.0f}× lower. The velocity buffer turns the small, consistent $\theta_2$ gradient into a step roughly $1/(1-\mu) = 10\times$ larger, so the flat direction finally makes progress; meanwhile the sign-flipping steep-direction gradient averages toward zero and the oscillation damps. This is the same demo as `sgd_vs_momentum.rlab`.

## Adam: Per-Parameter Adaptive Learning Rates

### Theory

Adam keeps **two** EMAs:

- The first moment $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\mathbf{g}_t$ — like momentum, but normalised to a true mean.
- The second moment $\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\mathbf{g}_t^2$ — an EMA of squared gradients (per coordinate).

A parameter whose gradient has been large recently will have a large $v_t$ in that slot; dividing by $\sqrt{v_t}$ shrinks its effective step. Parameters that rarely receive a strong signal get a *bigger* step. The update is

$$\hat{\mathbf{m}}_t = \mathbf{m}_t / (1 - \beta_1^t), \qquad \hat{\mathbf{v}}_t = \mathbf{v}_t / (1 - \beta_2^t), \qquad \theta_t = \theta_{t-1} - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \varepsilon}.$$

The hats are **bias correction**: at $t = 1$, $\mathbf{m}_1 = (1-\beta_1)\mathbf{g}_1$ is heavily biased toward zero and $\mathbf{v}_1 = (1-\beta_2)\mathbf{g}_1^2$ even more so; dividing by $1 - \beta_1^t$ and $1 - \beta_2^t$ removes the bias exactly at $t = 1$ (the hats recover $\mathbf{g}_1$ and $\mathbf{g}_1^2$). The first-moment correction fades fast — $1 - \beta_1^{50} \approx 0.995$ — but the second-moment correction does **not**: $1 - \beta_2^{50} \approx 0.049$ is still a $\sim 20\times$ rescale, because $\beta_2 = 0.999$ has a half-life of about 693 steps. The $\hat{\mathbf{v}}_t$ correction stays material for hundreds of steps.

### Example — Adam step on the same anisotropic bowl

```rustlab
beta1 = 0.9;
beta2 = 0.999;
eps   = 1e-8;
eta_a = 0.09;

theta_adam = [-2.0, 4.0];
m = [0.0, 0.0];
v = [0.0, 0.0];
path_adam = zeros(n_steps + 1, 2);
path_adam(1, 1) = theta_adam(1);
path_adam(1, 2) = theta_adam(2);

for k = 1:n_steps
  g  = grad(theta_adam);
  m  = beta1 * m + (1 - beta1) * g;
  v  = beta2 * v + (1 - beta2) * (g .* g);
  m_hat = m / (1 - beta1 ^ k);
  v_hat = v / (1 - beta2 ^ k);
  theta_adam = theta_adam - eta_a * m_hat ./ (sqrt(v_hat) + eps);
  path_adam(k + 1, 1) = theta_adam(1);
  path_adam(k + 1, 2) = theta_adam(2);
end

print("Adam final loss:", loss(theta_adam));
```

Adam's per-coordinate rescaling lets both directions advance at similar speeds. It drives $\theta_2$ down to ${theta_adam(2):%.3f}$ and $\theta_1$ to ${theta_adam(1):%.3f}$, so the final loss is ${loss(theta_adam):%.4e}$ — roughly two orders of magnitude below SGD's ${loss(theta_sgd):%.4f}$ on the identical 60-step budget. Dividing each coordinate's step by $\sqrt{\hat{v}_t}$ is what breaks the condition-number bottleneck that traps SGD: the flat coordinate, whose gradient is small but consistent, gets a full-sized step instead of a $0.9\%$ nudge.

## Coupled vs Decoupled Weight Decay

### Theory

**L2 regularisation** and **weight decay** are two names for the same operation — but *only under plain SGD*. L2 regularisation adds a penalty $\tfrac{\lambda}{2}\|\theta\|^2$ to the loss; its gradient contributes $\lambda\theta$, so vanilla SGD becomes

$$\theta_t = \theta_{t-1} - \eta(\mathbf{g}_t + \lambda\theta_{t-1}) = (1 - \eta\lambda)\,\theta_{t-1} - \eta\,\mathbf{g}_t.$$

The refactored right-hand side shows the equivalence: adding $\lambda\theta$ to the gradient *is* multiplying $\theta$ by $(1 - \eta\lambda)$ each step.

Under **Adam** the two part ways. Fold $\lambda\theta$ into the gradient — call this **Adam-with-L2** — and it flows through both moment estimates. A large-magnitude parameter picks up a large $\lambda\theta$ in its $\mathbf{v}_t$, so its whole update is divided by a correspondingly larger $\sqrt{\hat{v}_t}$: the decay it feels is *rescaled per coordinate* by the same $1/\sqrt{\hat{v}_t}$ that scales the gradient. Loshchilov & Hutter (2017) argued this coupling is the bug — a regulariser ought to shrink every coordinate by the same fraction, not one modulated by each coordinate's gradient history. Their fix **decouples** the decay, applying it straight to the parameter, *outside* the moment machinery:

$$\boxed{\text{AdamW: } \theta_t = \theta_{t-1} - \eta\left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \varepsilon} + \lambda\theta_{t-1}\right).}$$

So AdamW is **not** "L2 regularisation done right under Adam." It is a genuinely *different* update from Adam-with-L2 — one that applies a uniform fractional shrinkage $\eta\lambda$ to every coordinate, whatever its curvature. The two agree only in the plain-SGD limit. Every published GPT (and most modern transformers) uses AdamW.

### Example — Adam vs AdamW with weight decay

A 2D problem whose *data* loss is minimised at $(1, 5)$, with weight decay pulling back toward the origin. The high-curvature coordinate $\theta_1$ and the low-curvature $\theta_2$ experience different effective decays under the coupled vs decoupled implementations, so the two settle at different points.

```rustlab
function g = grad_wd_demo(theta)
  % Loss = 0.5*(theta(1) - 1.0)^2 + 0.5*0.1*(theta(2) - 5.0)^2  + tiny noise
  % Gradient (no decay) is independent of theta(1)/theta(2) magnitudes here:
  g = [theta(1) - 1.0, 0.1 * (theta(2) - 5.0)];
end

lambda_wd = 0.1;
beta1_b = 0.9;
beta2_b = 0.999;
eps_b   = 1e-8;
eta_b   = 0.1;
n_b     = 200;

% Coupled (Adam-with-L2): add lambda*theta to gradient before moments
theta_c = [3.0, 3.0];
m_c = [0.0, 0.0];
v_c = [0.0, 0.0];
for k = 1:n_b
  g = grad_wd_demo(theta_c) + lambda_wd * theta_c;
  m_c = beta1_b * m_c + (1 - beta1_b) * g;
  v_c = beta2_b * v_c + (1 - beta2_b) * (g .* g);
  m_hat = m_c / (1 - beta1_b ^ k);
  v_hat = v_c / (1 - beta2_b ^ k);
  theta_c = theta_c - eta_b * m_hat ./ (sqrt(v_hat) + eps_b);
end

% Decoupled (AdamW): apply decay outside the moment estimates
theta_d = [3.0, 3.0];
m_d = [0.0, 0.0];
v_d = [0.0, 0.0];
for k = 1:n_b
  g = grad_wd_demo(theta_d);
  m_d = beta1_b * m_d + (1 - beta1_b) * g;
  v_d = beta2_b * v_d + (1 - beta2_b) * (g .* g);
  m_hat = m_d / (1 - beta1_b ^ k);
  v_hat = v_d / (1 - beta2_b ^ k);
  theta_d = theta_d - eta_b * (m_hat ./ (sqrt(v_hat) + eps_b) + lambda_wd * theta_d);
end

print("Coupled (Adam+L2) final theta:", theta_c);
print("Decoupled (AdamW)  final theta:", theta_d);
```

The two variants land at different points. The **coupled** run converges to $(${theta_c(1):%.3f}, ${theta_c(2):%.3f})$ — essentially the textbook L2 optimum $\arg\min\!\big(L + \tfrac{\lambda}{2}\|\theta\|^2\big) = (10/11,\ 5/2) \approx (0.909, 2.500)$. Measured against the data optimum $(1, 5)$, that is a $9\%$ shrink on the high-curvature $\theta_1$ but a full $50\%$ shrink on the low-curvature $\theta_2$: coupled L2 pulls the low-curvature coordinate far harder, because there the data gradient is too weak to resist the penalty. **AdamW** instead lands at $(${theta_d(1):%.3f}, ${theta_d(2):%.3f})$ — a near-uniform $3\%$ / $5\%$ shrink across the two coordinates. That curvature-independent per-step fractional decay is exactly what decoupling buys; it is a *different* endpoint from coupled L2, not the same one reached more accurately.

## Three Trajectories on One Surface

### Theory

Plot SGD, Adam, and AdamW paths over the same anisotropic bowl. SGD zigzags across the steep axis while crawling along the flat one; Adam glides down the ravine floor; AdamW glides the same way but its weight-decay term adds a constant pull toward the origin, so it settles even closer to it. (The bowl's minimum is itself the origin here, so that pull costs no extra loss.)

### Example — Optimiser-trajectory overlay

```rustlab
% AdamW path on the original elongated bowl, with weight decay pulling the
% optimum toward the origin.
lambda_aw = 0.05;
theta_aw = [-2.0, 4.0];
m_aw = [0.0, 0.0];
v_aw = [0.0, 0.0];
path_adamw = zeros(n_steps + 1, 2);
path_adamw(1, 1) = theta_aw(1);
path_adamw(1, 2) = theta_aw(2);
for k = 1:n_steps
  g = grad(theta_aw);
  m_aw = beta1 * m_aw + (1 - beta1) * g;
  v_aw = beta2 * v_aw + (1 - beta2) * (g .* g);
  m_hat = m_aw / (1 - beta1 ^ k);
  v_hat = v_aw / (1 - beta2 ^ k);
  theta_aw = theta_aw - eta_a * (m_hat ./ (sqrt(v_hat) + eps) + lambda_aw * theta_aw);
  path_adamw(k + 1, 1) = theta_aw(1);
  path_adamw(k + 1, 2) = theta_aw(2);
end

figure();
plot(path_sgd(:, 1),   path_sgd(:, 2),   "color", "red",   "label", "SGD")
hold("on")
plot(path_adam(:, 1),  path_adam(:, 2),  "color", "blue",  "label", "Adam")
plot(path_adamw(:, 1), path_adamw(:, 2), "color", "green", "label", "AdamW (λ=0.05)")
hold("off")
title("Optimiser trajectories on  L = ½(200 θ₁² + θ₂²)")
xlabel("θ₁  (steep direction)")
ylabel("θ₂  (flat direction)")
legend("SGD", "Adam", "AdamW")
```

SGD's path looks like a saw blade: with $\eta = 0.009$ the steep coordinate $\theta_1$ flips sign every step ($1 - \eta \cdot 200 = -0.8$), so it zigzags while $\theta_2$ barely drifts. Adam and AdamW both glide smoothly down the floor of the ravine. AdamW ends closer to the origin ($\|\theta\| \approx 0.024$) than Adam ($\|\theta\| \approx 0.165$) because of the constant pull from weight decay — and since the origin is also the loss minimum here, that extra shrinkage costs nothing.

## Choosing the Hyperparameters

### Theory

Three numbers dominate the optimiser's behaviour and they have surprisingly stable defaults across LLM training:

- $\beta_1 = 0.9$. Half-life $\log(1/2)/\log(\beta_1) \approx 6.6$ steps. Captures short-term momentum without going stale.
- $\beta_2 = 0.999$ is the generic Adam default (half-life $\approx 693$ steps). It tracks per-parameter scale over a long horizon — a parameter's typical gradient magnitude is roughly stationary across many minibatches even when individual gradients are noisy. Large-scale **LLM** training usually lowers it to $\beta_2 = 0.95$ (GPT-3, OPT, LLaMA, nanoGPT): at very large batch sizes the long-memory $0.999$ can destabilise, and $0.95$ (half-life $\approx 14$ steps) reacts faster to shifts in gradient scale.
- $\varepsilon = 10^{-8}$. Prevents division by zero on parameters that have never received a non-trivial gradient (e.g. unused embedding rows for rare tokens).

The learning rate $\eta$ is the only hyperparameter that requires per-task tuning — and even that is constrained by the warmup-and-decay *schedule* covered in [Lesson 17](17-learning-rate-scheduling.md). Weight decay $\lambda$ is typically $0.1$ for transformer language models (small but non-zero — it does real regularisation work on the embedding and projection matrices).

## Information-Theoretic Framing

### Theory

The optimiser is the credit-assignment loop: backprop measures how each parameter contributed to the bit overshoot in [Lesson 03](03-cross-entropy-loss.md)'s cross-entropy, and the optimiser turns those measurements into parameter updates. Two questions to keep in mind:

- **Adam's per-parameter learning rate behaves like a signal-to-noise ratio of the gradient** — an SNR-*like* quantity in Kingma & Ba's framing, not a formal maximum-likelihood estimate. Dividing by $\sqrt{v_t}$ effectively says "trust the sign of $m_t$ in proportion to how reliably it has been the sign of recent gradients." Parameters whose minibatch gradients are noisy (high variance / low signal) are updated cautiously; parameters whose gradients are consistent are updated boldly.
- **Decoupled weight decay is a uniform prior on parameter magnitudes.** It corresponds to a Gaussian prior $\mathcal{N}(0, 1/\lambda)$ over $\theta$, applied identically to every coordinate. Coupled L2 inside Adam corrupts that prior — the per-parameter $1/\sqrt{v_t}$ term changes how strongly each coordinate is pulled toward zero. AdamW cleanly separates "what does the data want?" (the gradient) from "what does the prior want?" (uniform shrinkage).

## Key Takeaways

- Vanilla SGD is unstable on anisotropic loss surfaces; the safest learning rate in the steep direction is the wrong learning rate for the flat direction.
- **Momentum** smooths sign-flipping gradients along steep axes and accumulates them along flat axes.
- **Adam** maintains a first-moment EMA $\mathbf{m}_t$ (momentum) and a second-moment EMA $\mathbf{v}_t$ (per-parameter scale), with **bias correction** for the early steps.
- **AdamW** decouples weight decay from the moment estimates: $\lambda\theta$ is subtracted directly from the parameter, not folded into $\mathbf{v}_t$. This restores the textbook behaviour of L2 regularisation under an adaptive optimiser.
- The hyperparameters $\beta_1 = 0.9$, $\varepsilon = 10^{-8}$, $\lambda = 0.1$ are stable defaults across transformer training. $\beta_2 = 0.999$ is the *generic* Adam default, but published large-scale LLM training (GPT-3, OPT, LLaMA, nanoGPT) uses $\beta_2 = 0.95$ for stability at large batch sizes.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `sgd_vs_momentum.rlab` | SGD vs SGD-with-momentum on the elongated bowl; loss curves and trajectories |
| `adam_step.rlab` | one Adam step in detail with bias correction; shows the $\hat{\mathbf{m}}$ factor reaching $\approx 1$ within a few steps while the $\hat{\mathbf{v}}$ factor lags ($\beta_2$ half-life $\approx 693$ steps) |
| `optimizer_comparison.rlab` | full SGD vs Adam vs AdamW trajectory overlay on the anisotropic loss surface |

Run all with `make lesson-16` (or `rustlab run lessons/16-adamw-optimizer/<name>.rlab`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `loss(theta_sgd)` after 60 SGD steps | ≈ `2.70` ($\theta_2$ stuck near `2.33`; $\theta_1$ converged) |
| `loss(theta_adam)` after 60 Adam steps | ≈ `0.017` (~160× below SGD) |
| Bias correction at $t = 1$ ($1 - \beta_1$) | `0.1` (big correction) |
| Bias correction at $t = 100$ | $\approx 1$ (negligible) |
| Decoupled vs coupled `theta_d - theta_c` | non-zero, especially in the small-curvature direction |

## Exercises

1. **Why bias correction matters.** At $t = 1$, write $\mathbf{m}_1$ and $\mathbf{v}_1$ in terms of $\mathbf{g}_1$ and confirm $\hat{\mathbf{m}}_1 = \mathbf{g}_1$, $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$. Now suppose you skipped **both** corrections: show that the raw first step $\mathbf{m}_1 / \sqrt{\mathbf{v}_1}$ exceeds the corrected step $\hat{\mathbf{m}}_1 / \sqrt{\hat{\mathbf{v}}_1}$ by a factor $(1 - \beta_1)/\sqrt{1 - \beta_2}$. Evaluate it at $(\beta_1, \beta_2) = (0.9, 0.999)$ — is the uncorrected first step too large or too small, and by how much?
2. **Tuning $\beta_2$.** Re-run `adam_step.rlab` with $\beta_2 = 0.9$. Does the optimiser still converge as quickly? Why is a long second-moment horizon important for stable training?
3. **Why $\varepsilon$ inside the square root vs outside.** The original Adam paper uses $\hat{\mathbf{m}} / (\sqrt{\hat{\mathbf{v}}} + \varepsilon)$. What changes if you write $\hat{\mathbf{m}} / \sqrt{\hat{\mathbf{v}} + \varepsilon}$? On parameters with $\hat{v} \to 0$, which version gives a saner step size?
4. **Coupled vs decoupled, by hand.** Write out one AdamW step and one Adam-with-L2 step for a single parameter $\theta = 2.0$ with gradient $g = 1.0$ and $\lambda = 0.1$. Where exactly do the two formulas diverge?
5. **Loss surface intuition.** `optimizer_comparison.rlab` already runs at condition number 200. Push it further to $L = 0.5(1000\theta_1^2 + \theta_2^2)$: what is the new SGD stability bound $2/a$, and what learning rate must SGD use? Does Adam's $\eta$ need to change at all? Re-run and compare the final losses.

## What's next

Lesson 17 covers the **learning-rate schedule** that wraps every step of the optimiser. Real LLM training uses a short linear *warmup* (so the model sees small gradients while the moment estimates initialise) followed by a long **cosine decay** (so the optimiser slows down as it nears the minimum). Picking $\eta$ alone is not enough; *how $\eta$ moves over the course of training* is its own design problem.
