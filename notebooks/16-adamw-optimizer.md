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

Define $L(\theta_1, \theta_2) = \tfrac{1}{2}(20\theta_1^2 + \theta_2^2)$ (condition number 20) and run SGD from $(-2, 4)$ with $\eta = 0.09$ for 60 steps.

```rustlab
% Loss and analytic gradient
function L = loss(theta)
  L = 0.5 * (20.0 * theta(1) ^ 2 + theta(2) ^ 2);
end
function g = grad(theta)
  g = [20.0 * theta(1), theta(2)];
end

theta_sgd = [-2.0, 4.0];
eta = 0.09;
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

After 60 steps SGD loss is ${loss(theta_sgd):%.4f}$ — we are barely off the start in $\theta_2$ because $\eta = 0.09$ shrinks $\theta_2$ by only 9 % per step.

## Momentum: An EMA of the Gradient

### Theory

Add a velocity buffer $\mathbf{v}_t$ that accumulates past gradients with exponential decay:

$$\mathbf{v}_t = \mu \mathbf{v}_{t-1} + \mathbf{g}_t, \qquad \theta_t = \theta_{t-1} - \eta \mathbf{v}_t.$$

With $\mu \in [0, 1)$ (often 0.9), the buffer averages roughly the last $1/(1-\mu) \approx 10$ gradients. In the steep direction the gradient oscillates sign and the EMA cancels itself; in the flat direction the gradient has a consistent sign and the EMA *grows*, multiplying the effective step size. SGD-with-momentum drives along the ravine floor instead of bouncing off the walls.

## Adam: Per-Parameter Adaptive Learning Rates

### Theory

Adam keeps **two** EMAs:

- The first moment $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\mathbf{g}_t$ — like momentum, but normalised to a true mean.
- The second moment $\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\mathbf{g}_t^2$ — an EMA of squared gradients (per coordinate).

A parameter whose gradient has been large recently will have a large $v_t$ in that slot; dividing by $\sqrt{v_t}$ shrinks its effective step. Parameters that rarely receive a strong signal get a *bigger* step. The update is

$$\hat{\mathbf{m}}_t = \mathbf{m}_t / (1 - \beta_1^t), \qquad \hat{\mathbf{v}}_t = \mathbf{v}_t / (1 - \beta_2^t), \qquad \theta_t = \theta_{t-1} - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \varepsilon}.$$

The hats are **bias correction**: at $t = 1$, $\mathbf{m}_1 = (1-\beta_1)\mathbf{g}_1$ would be heavily biased toward zero; dividing by $1 - \beta_1^t$ undoes that. After ~50 steps the correction is negligible.

### Example — Adam step on the same anisotropic bowl

```rustlab
beta1 = 0.9;
beta2 = 0.999;
eps   = 1e-8;
eta_a = 0.5;

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

Adam's per-coordinate rescaling makes both directions advance at similar speeds — final loss is now ${loss(theta_adam):%.4e}$, several orders below SGD.

## Coupled vs Decoupled Weight Decay

### Theory

**L2 regularisation**, also called weight decay, adds a penalty $\tfrac{\lambda}{2}\|\theta\|^2$ to the loss. With *vanilla* SGD this is equivalent to subtracting $\eta\lambda\theta$ from each parameter:

$$\theta_t = \theta_{t-1} - \eta(\mathbf{g}_t + \lambda\theta_{t-1}).$$

Plug the same recipe into Adam and the regularisation gradient $\lambda\theta$ flows into the second-moment estimate $\mathbf{v}_t$. Parameters with large magnitude end up with large $v_t$, which *shrinks* their decay rate — exactly the opposite of what regularisation should do. Loshchilov & Hutter (2017) noticed the bug and decoupled the two:

$$\boxed{\text{AdamW: } \theta_t = \theta_{t-1} - \eta\left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \varepsilon} + \lambda\theta_{t-1}\right).}$$

The decay term is applied to the parameter directly, *outside* the moment estimates. AdamW recovers the textbook behaviour of L2 regularisation in the presence of an adaptive optimiser. Every published GPT (and most modern transformers) uses AdamW.

### Example — Adam vs AdamW with weight decay

A 2D problem where the true minimum is at the origin and weight decay tries to pull us there. Different parameters experience different effective decays under coupled vs decoupled implementations.

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

The coupled and decoupled variants converge to different points: the L2-coupled version effectively penalises high-curvature parameters less than low-curvature ones; AdamW applies the same fractional shrinkage to every parameter as the textbook L2 says it should.

## Three Trajectories on One Surface

### Theory

Plot SGD, Adam, and AdamW paths over the same anisotropic bowl. SGD zigzags, Adam glides, and AdamW glides toward a slightly more shrunk minimum because of its weight-decay pull.

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

figure()
plot(path_sgd(:, 1),   path_sgd(:, 2),   "color", "red",   "label", "SGD")
hold("on")
plot(path_adam(:, 1),  path_adam(:, 2),  "color", "blue",  "label", "Adam")
plot(path_adamw(:, 1), path_adamw(:, 2), "color", "green", "label", "AdamW (λ=0.05)")
hold("off")
title("Optimiser trajectories on  L = ½(20 θ₁² + θ₂²)")
xlabel("θ₁  (steep direction)")
ylabel("θ₂  (flat direction)")
legend("SGD", "Adam", "AdamW")
```

SGD's path looks like a saw blade: large $\theta_1$ overshoots cause oscillation. Adam and AdamW both glide smoothly down the floor of the ravine. AdamW lands slightly inside the origin because of the constant pull from weight decay.

## Choosing the Hyperparameters

### Theory

Three numbers dominate the optimiser's behaviour and they have surprisingly stable defaults across LLM training:

- $\beta_1 = 0.9$. Half-life $\log(1/2)/\log(\beta_1) \approx 6.6$ steps. Captures short-term momentum without going stale.
- $\beta_2 = 0.999$. Half-life $\approx 693$ steps. Tracks per-parameter scale over a long horizon — important because a parameter's typical gradient magnitude is roughly stationary across many minibatches, even if individual gradients are noisy.
- $\varepsilon = 10^{-8}$. Prevents division by zero on parameters that have never received a non-trivial gradient (e.g. unused embedding rows for rare tokens).

The learning rate $\eta$ is the only hyperparameter that requires per-task tuning — and even that is constrained by the warmup-and-decay *schedule* covered in [Lesson 17](17-learning-rate-scheduling.md). Weight decay $\lambda$ is typically $0.1$ for transformer language models (small but non-zero — it does real regularisation work on the embedding and projection matrices).

## Information-Theoretic Framing

### Theory

The optimiser is the credit-assignment loop: backprop measures how each parameter contributed to the bit overshoot in [Lesson 03](03-cross-entropy-loss.md)'s cross-entropy, and the optimiser turns those measurements into parameter updates. Two questions to keep in mind:

- **Adam's per-parameter learning rate is a maximum-likelihood estimate of the gradient's signal-to-noise ratio.** Dividing by $\sqrt{v_t}$ effectively says "trust the sign of $m_t$ proportional to how reliably it has been the sign of recent gradients." Parameters whose minibatch gradients are noisy (high variance / low signal) are updated cautiously; parameters whose gradients are consistent are updated boldly.
- **Decoupled weight decay is a uniform prior on parameter magnitudes.** It corresponds to a Gaussian prior $\mathcal{N}(0, 1/\lambda)$ over $\theta$, applied identically to every coordinate. Coupled L2 inside Adam corrupts that prior — the per-parameter $1/\sqrt{v_t}$ term changes how strongly each coordinate is pulled toward zero. AdamW cleanly separates "what does the data want?" (the gradient) from "what does the prior want?" (uniform shrinkage).

## Key Takeaways

- Vanilla SGD is unstable on anisotropic loss surfaces; the safest learning rate in the steep direction is the wrong learning rate for the flat direction.
- **Momentum** smooths sign-flipping gradients along steep axes and accumulates them along flat axes.
- **Adam** maintains a first-moment EMA $\mathbf{m}_t$ (momentum) and a second-moment EMA $\mathbf{v}_t$ (per-parameter scale), with **bias correction** for the early steps.
- **AdamW** decouples weight decay from the moment estimates: $\lambda\theta$ is subtracted directly from the parameter, not folded into $\mathbf{v}_t$. This restores the textbook behaviour of L2 regularisation under an adaptive optimiser.
- The hyperparameters $\beta_1 = 0.9, \beta_2 = 0.999, \varepsilon = 10^{-8}, \lambda = 0.1$ are the de-facto defaults for every published GPT.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `sgd_vs_momentum.rlab` | SGD vs SGD-with-momentum on the elongated bowl; loss curves and trajectories |
| `adam_step.rlab` | one Adam step in detail with bias correction; verifies that $\hat{\mathbf{m}} \to \mathbf{m}$ as $t$ grows |
| `optimizer_comparison.rlab` | full SGD vs Adam vs AdamW trajectory overlay on the anisotropic loss surface |

Run all with `make lesson-16` (or `rustlab run lessons/16-adamw-optimizer/<name>.rlab`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `loss(theta_sgd)` after 60 SGD steps | ~$0.5 \cdot \theta_2^2$ (slow $\theta_2$ decay, small $\theta_1$) |
| `loss(theta_adam)` after 60 Adam steps | $< 10^{-3}$ (well below SGD) |
| Bias correction at $t = 1$ ($1 - \beta_1$) | `0.1` (big correction) |
| Bias correction at $t = 100$ | $\approx 1$ (negligible) |
| Decoupled vs coupled `theta_d - theta_c` | non-zero, especially in the small-curvature direction |

## Exercises

1. **Why bias correction matters.** At $t = 1$, what does $\mathbf{m}_1$ equal in terms of $\mathbf{g}_1$? What does $\hat{\mathbf{m}}_1$ equal? Why would Adam underestimate the first step's update without the correction?
2. **Tuning $\beta_2$.** Re-run `adam_step.rlab` with $\beta_2 = 0.9$. Does the optimiser still converge as quickly? Why is a long second-moment horizon important for stable training?
3. **Why $\varepsilon$ inside the square root vs outside.** The original Adam paper uses $\hat{\mathbf{m}} / (\sqrt{\hat{\mathbf{v}}} + \varepsilon)$. What changes if you write $\hat{\mathbf{m}} / \sqrt{\hat{\mathbf{v}} + \varepsilon}$? On parameters with $\hat{v} \to 0$, which version gives a saner step size?
4. **Coupled vs decoupled, by hand.** Write out one AdamW step and one Adam-with-L2 step for a single parameter $\theta = 2.0$ with gradient $g = 1.0$ and $\lambda = 0.1$. Where exactly do the two formulas diverge?
5. **Loss surface intuition.** On `optimizer_comparison.rlab`, change the bowl to $L = 0.5 (200\theta_1^2 + \theta_2^2)$ (condition number 200). What learning rate keeps SGD stable now? What does Adam need?

## What's next

Lesson 17 covers the **learning-rate schedule** that wraps every step of the optimiser. Real LLM training uses a short linear *warmup* (so the model sees small gradients while the moment estimates initialise) followed by a long **cosine decay** (so the optimiser slows down as it nears the minimum). Picking $\eta$ alone is not enough; *how $\eta$ moves over the course of training* is its own design problem.
