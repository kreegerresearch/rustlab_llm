---
title: "Lesson 06 — Linear Layers & Gradient Descent"
order: 6
---

# Lesson 06 — Linear Layers & Gradient Descent

The bigram model ([Lesson 05](05-bigram-language-model.md)) uses fixed counts. To learn from data we need
**learnable parameters** and a method to improve them. This lesson introduces the
**linear layer** $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ and **gradient
descent** — the engine that trains every neural network.

---

## The Linear Layer

A **linear layer** maps input $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$ to output
$\mathbf{y} \in \mathbb{R}^{d_{\text{out}}}$:

$$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

- $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ — the **weight matrix** (learnable)
- $\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}$ — the **bias vector** (learnable)

The embedding lookup from [Lesson 04](04-embeddings-and-similarity.md) is a special case: multiplying the embedding
matrix by a one-hot input selects one row — exactly a linear layer.

## Loss Function: Mean Squared Error

For a dataset $\{(x_i, y_i)\}$ the **MSE loss** is:

$$\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^{N} (w x_i + b - y_i)^2$$

For language models the loss is cross-entropy ([Lesson 03](03-cross-entropy-loss.md)), but MSE makes the geometry
of the loss landscape transparent.

---

## The Loss Landscape

The loss $\mathcal{L}(w, b)$ defines a surface over the $(w, b)$ plane. For MSE with
a linear model this surface is a **convex paraboloid** — a bowl with a unique minimum.

Let's visualise this for the dataset $x = [1,2,3,4]$, $y = [2,4,6,8]$ where the true
relationship is $y = 2x$ (so $w^* = 2$, $b^* = 0$):

```rustlab
x = [1.0, 2.0, 3.0, 4.0];
y = [2.0, 4.0, 6.0, 8.0];

% Loss at true parameters (w=2, b=0) and at initial parameters (w=0, b=0)
L_true = real(mean((2.0 * x + 0.0 - y) .^ 2));
L_init = real(mean((0.0 * x + 0.0 - y) .^ 2));
```

At the optimum, $\mathcal{L}(2, 0) = ${L_true:%.3f}$ — zero loss because
$y = 2x$ exactly. Starting from $(w, b) = (0, 0)$ the loss is
${L_init:%.2f}$, the distance we need gradient descent to close.

The analytic expansion for this dataset is:

$$\mathcal{L}(w,b) = 7.5w^2 + b^2 + 5wb - 30w - 10b + 30$$

<!-- hide -->
```rustlab
% Verify: L(2, 0) = 7.5*4 - 60 + 30 = 0
L_check = 7.5 * 4.0 + 0.0 + 5.0 * 2.0 * 0.0 - 30.0 * 2.0 - 10.0 * 0.0 + 30.0;
```

```rustlab
% Build a 40x40 grid of L(w, b) using outer products
n_grid = 40;
w_grid = linspace(-0.5, 3.5, n_grid);
b_grid = linspace(-3.0, 3.0, n_grid);

term_w2 = outer(ones(n_grid), 7.5 * w_grid .^ 2);
term_b2 = outer(b_grid .^ 2, ones(n_grid));
term_wb = 5.0 * outer(b_grid, w_grid);
term_w  = outer(ones(n_grid), -30.0 * w_grid);
term_b  = outer(-10.0 * b_grid, ones(n_grid));

L_matrix = term_w2 + term_b2 + term_wb + term_w + term_b + 30.0;
min_loss_flat = min(reshape(L_matrix, 1, n_grid * n_grid));
```

Analytic check: $\mathcal{L}(2, 0) = ${L_check:%.3f}$ from the expanded
formula. The minimum over the $40 \times 40$ grid is
${min_loss_flat:%.4f}$ — a hair above zero because the grid doesn't
land exactly on $(2, 0)$.

```rustlab
figure()
imagesc(L_matrix, "viridis")
title("MSE Loss L(w,b): y=2x  minimum at (w=2, b=0)")
```

The dark region (minimum loss) is centred at $(w, b) \approx (2, 0)$. The elliptical
contours show the loss is more sensitive to $w$ than $b$.

---

## Gradient Descent

The **gradient** $\nabla \mathcal{L} = [\partial \mathcal{L}/\partial w,\; \partial \mathcal{L}/\partial b]$
points toward steepest ascent. We move in the opposite direction:

$$w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}, \qquad b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}$$

where $\eta > 0$ is the **learning rate**. The partial derivatives are:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) \cdot x_i, \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$

### Running gradient descent

Starting at $(w_0, b_0) = (0, 0)$ with $\eta = 0.05$:

```rustlab
npts = 4.0;
lr = 0.05;
n_steps = 200;

w = 0.0;
b = 0.0;

w_path    = zeros(n_steps + 1);
b_path    = zeros(n_steps + 1);
loss_path = zeros(n_steps + 1);

w_path(1)    = w;
b_path(1)    = b;
loss_path(1) = mean((w * x + b - y) .^ 2);

for step = 1:n_steps
  pred     = w * x + b;
  residual = pred - y;
  dw = (2.0 / npts) * sum(residual .* x);
  db = (2.0 / npts) * sum(residual);
  w -= lr * dw;
  b -= lr * db;
  w_path(step + 1)    = w;
  b_path(step + 1)    = b;
  loss_path(step + 1) = mean((w * x + b - y) .^ 2);
end
```

After ${n_steps} steps with $\eta = ${lr}$: $w = ${w:%.4f}$ (true
$w^* = 2$), $b = ${b:%.4f}$ (true $b^* = 0$),
$\mathcal{L} = ${loss_path(n_steps + 1):%.2e}$ — effectively zero.

The loss decreases monotonically — guaranteed for MSE with a suitable learning rate.
Early steps are large (steep gradients far from the minimum); later steps are small.

```rustlab
figure()
plot(loss_path, "color", "blue", "label", "MSE loss")
hold("on")
hline(0.0, "gray", "minimum")
title("Gradient Descent: Loss vs. Step")
xlabel("Step")
ylabel("MSE Loss")
legend()
```

The trajectory in $(w, b)$ space curves — it does not go straight to the minimum
because the loss surface has different curvature in the $w$ and $b$ directions:

```rustlab
figure()
scatter(w_path, b_path, "Gradient Descent Path in (w,b) Space - converges to (2, 0)")
```

---

## Learning Rate Sensitivity

| $\eta$ | Behaviour |
|--------|-----------|
| Too large | Overshoots the minimum; loss oscillates or diverges |
| Too small | Convergence is slow; many iterations needed |
| Well-chosen | Loss decreases smoothly to the minimum |

### Verify the first step by hand

- Initial: $w_0 = 0$, $b_0 = 0$, predictions $= [0,0,0,0]$, residuals $= [-2,-4,-6,-8]$
- $\partial \mathcal{L}/\partial w = 0.5 \times ((-2)(1) + (-4)(2) + (-6)(3) + (-8)(4)) = -30$
- $\partial \mathcal{L}/\partial b = 0.5 \times (-2-4-6-8) = -10$
- $w_1 = 0 - 0.05 \times (-30) = 1.5$, $\;\; b_1 = 0 - 0.05 \times (-10) = 0.5$

---

## Key Takeaways

- Every component of a language model — embedding lookup to output projection — is a
  linear layer (or a composition with non-linear activations).
- Training means adjusting $\mathbf{W}$ and $\mathbf{b}$ to minimise the loss via
  gradient descent.
- For MSE the loss landscape is a convex paraboloid with a unique global minimum.
  Language model losses (cross-entropy through deep networks) are non-convex — but
  gradient descent remains the practical engine.
- A linear layer alone cannot learn complex functions. Stacking them without
  non-linearity is equivalent to a single linear layer. The activation function
  (Lesson 11) is what makes deep networks expressive.

---

← [Lesson 05 — The Bigram Language Model](05-bigram-language-model.md)
