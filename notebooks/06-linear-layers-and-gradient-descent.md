# Lesson 06: Linear Layers & Gradient Descent

The bigram model ([Lesson 05](05-bigram-language-model.md)) uses fixed counts. To learn from data we need **learnable parameters** and a method to improve them. This lesson introduces the **linear layer** $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ and **gradient descent** — the engine that trains every neural network.

## Learning Objectives

- Write the equation for a **linear layer** $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ and identify what each component represents.
- Define a **loss function** (mean squared error) and explain how it measures prediction quality.
- Derive the **gradient** of the loss with respect to $w$ and $b$ step by step.
- Apply one step of **gradient descent** by hand and verify the loss decreases.
- Read a **2-D loss landscape heatmap** and identify the minimum, gradient direction, and descent path.

## Background

Matrix-vector multiplication and dot products from linear algebra. The chain rule from calculus: $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$. Cross-entropy loss and the concept of a scalar loss from [Lesson 03](03-cross-entropy-loss.md). Embeddings as dense vector representations from [Lesson 04](04-embeddings-and-similarity.md).

## The Linear Layer

A **linear layer** (also called a fully-connected layer) maps input $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$ to output $\mathbf{y} \in \mathbb{R}^{d_{\text{out}}}$:

$$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}.$$

- $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ — the **weight matrix** (learnable).
- $\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}$ — the **bias vector** (learnable).
- $\mathbf{y}$ is a linear combination of the input features, one per output dimension.

The embedding lookup from [Lesson 04](04-embeddings-and-similarity.md) is a special case: multiplying the embedding matrix by a one-hot input selects one row — exactly a linear layer applied to a one-hot input.

## Loss Function: Mean Squared Error

For regression tasks (fitting a curve to data) the **mean squared error** loss is

$$\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 = \frac{1}{N} \sum_{i=1}^{N} (w x_i + b - y_i)^2,$$

where $(x_i, y_i)$ are data points and $\hat{y}_i = w x_i + b$ is the model's prediction. For language models the loss is cross-entropy ([Lesson 03](03-cross-entropy-loss.md)), but MSE makes the geometry of the loss landscape transparent and is ideal for building intuition.

## The Loss Landscape

The loss $\mathcal{L}(w, b)$ defines a 2-D surface over the $(w, b)$ plane. For MSE with a linear model this surface is a **convex paraboloid** — a bowl with a unique global minimum and no local minima. Gradient descent is guaranteed to converge to the global optimum.

Visualise it for the dataset $x = [1,2,3,4]$, $y = [2,4,6,8]$ where the true relationship is $y = 2x$ (so $w^* = 2$, $b^* = 0$):

```rustlab
x = [1.0, 2.0, 3.0, 4.0];
y = [2.0, 4.0, 6.0, 8.0];

% Loss at true parameters (w=2, b=0) and at initial parameters (w=0, b=0)
L_true = real(mean((2.0 * x + 0.0 - y) .^ 2));
L_init = real(mean((0.0 * x + 0.0 - y) .^ 2));
```

At the optimum, $\mathcal{L}(2, 0) = ${L_true:%.3f}$ — zero loss because $y = 2x$ exactly. Starting from $(w, b) = (0, 0)$ the loss is ${L_init:%.2f}$, the distance we need gradient descent to close.

The analytic expansion for this dataset is

$$\mathcal{L}(w,b) = 7.5w^2 + b^2 + 5wb - 30w - 10b + 30.$$

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

Analytic check: $\mathcal{L}(2, 0) = ${L_check:%.3f}$ from the expanded formula. The minimum over the $40 \times 40$ grid is ${min_loss_flat:%.4f}$ — a hair above zero because the grid doesn't land exactly on $(2, 0)$.

```rustlab
figure()
imagesc(L_matrix, "viridis")
title("MSE Loss L(w,b): y=2x  minimum at (w=2, b=0)")
```

The dark region (minimum loss) is centred at $(w, b) \approx (2, 0)$. The elliptical contours show the loss is more sensitive to $w$ than $b$.

### The Loss Surface in 3D

`meshgrid` builds coordinate matrices aligned with the loss grid, and `surf` renders it as a rotatable 3-D paraboloid — the "bowl" gradient descent is rolling toward.

```rustlab
[W_mesh, B_mesh] = meshgrid(w_grid, b_grid);

figure()
surf(W_mesh, B_mesh, L_matrix, "viridis")
title("MSE Loss Surface L(w,b)")
xlabel("w")
ylabel("b")
```

The surface is a **convex paraboloid**: one global minimum, no local minima, no plateaus. Anisotropy is visible as elongation along the $b$ axis — the bowl is steeper in $w$ than in $b$, which is why the gradient-descent trajectory curves rather than heading straight for $(2, 0)$.

## Gradient Descent

The **gradient** $\nabla \mathcal{L} = [\partial \mathcal{L}/\partial w,\; \partial \mathcal{L}/\partial b]$ points toward steepest ascent. We move in the opposite direction:

$$w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}, \qquad b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b},$$

where $\eta > 0$ is the **learning rate**. The partial derivatives, by the chain rule, are

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) \cdot x_i, \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i).$$

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

After ${n_steps} steps with $\eta = ${lr}$: $w = ${w:%.4f}$ (true $w^* = 2$), $b = ${b:%.4f}$ (true $b^* = 0$), $\mathcal{L} = ${loss_path(n_steps + 1):%.2e}$ — effectively zero.

The loss decreases monotonically — guaranteed for MSE with a suitable learning rate. Early steps are large (steep gradients far from the minimum); later steps are small.

```rustlab
figure()
plot(loss_path, "color", "blue", "label", "MSE loss")
hold("on")
hline(0.0, "gray", "minimum")
title("Gradient Descent: Loss vs. Step")
xlabel("Step")
ylabel("MSE Loss")
legend()
hold("off")
```

The trajectory in $(w, b)$ space curves — it does not go straight to the minimum because the loss surface has different curvature in the $w$ and $b$ directions:

```rustlab
figure()
scatter(w_path, b_path, "Gradient Descent Path in (w,b) Space - converges to (2, 0)")
```

## Learning Rate Sensitivity

| $\eta$ | Behaviour |
|--------|-----------|
| Too large | Overshoots the minimum; loss oscillates or diverges |
| Too small | Convergence is slow; many iterations needed |
| Well-chosen | Loss decreases smoothly to the minimum |

### Verify the first step by hand

- Initial: $w_0 = 0$, $b_0 = 0$, predictions $= [0,0,0,0]$, residuals $= [-2,-4,-6,-8]$.
- $\partial \mathcal{L}/\partial w = 0.5 \times ((-2)(1) + (-4)(2) + (-6)(3) + (-8)(4)) = -30$.
- $\partial \mathcal{L}/\partial b = 0.5 \times (-2-4-6-8) = -10$.
- $w_1 = 0 - 0.05 \times (-30) = 1.5,\;\; b_1 = 0 - 0.05 \times (-10) = 0.5$.

## Key Takeaways

- Every component of a language model — embedding lookup to output projection — is a linear layer (or a composition with non-linear activations).
- Training means adjusting $\mathbf{W}$ and $\mathbf{b}$ to minimise the loss via gradient descent.
- For MSE the loss landscape is a convex paraboloid with a unique global minimum. Language model losses (cross-entropy through deep networks) are non-convex — but gradient descent remains the practical engine.
- A linear layer alone cannot learn complex functions. Stacking them without non-linearity is equivalent to a single linear layer. The activation function (Lesson 11) is what makes deep networks expressive.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `loss_landscape.r` | the MSE loss as a 40×40 heatmap and rotatable 3-D `surf` over $(w, b)$ |
| `gradient_descent.r` | 200 steps of gradient descent on the same dataset; loss curve and $(w, b)$ trajectory |

Run all with `make lesson-06` (or `rustlab run lessons/06-linear-layers-and-gradient-descent/<name>.r`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `L_true` (= $\mathcal{L}(2, 0)$) | `0.000` |
| `L_init` (= $\mathcal{L}(0, 0)$) | `30.0` |
| `L_check` (analytic) | `0.000` |
| `min_loss_flat` (over grid) | ≈ `0.06` |
| Final `w` after 200 steps | ≈ `2.0` |
| Final `b` after 200 steps | ≈ `0.0` |
| Final `loss_path(end)` | ≈ `0` (machine epsilon) |
| Step-1 hand check $w_1$ | `1.5` |
| Step-1 hand check $b_1$ | `0.5` |

## Exercises

1. **Learning rate sensitivity.** Modify `gradient_descent.r` to use $\eta = 0.1$. Does the loss still decrease? What about $\eta = 0.2$? Find the threshold above which the algorithm diverges.
2. **Gradient at the minimum.** At $(w^*, b^*) = (2, 0)$, compute $\partial \mathcal{L}/\partial w$ and $\partial \mathcal{L}/\partial b$ by hand. Confirm both are zero.
3. **Non-zero bias.** Change the dataset to $y = [3, 5, 7, 9]$ (true relationship $y = 2x + 1$). Re-run `gradient_descent.r`. Where does the algorithm converge? What are the new values of $w^*$ and $b^*$?
4. **Counting parameters.** A language model uses a linear layer to project from embedding dimension $d = 512$ to vocabulary size $|\mathcal{V}| = 50{,}000$. How many parameters does this output linear layer have (weights + biases)? What fraction of GPT-2-small's 117M parameters does this represent?
5. **Convexity.** The MSE loss for a linear model is convex. Give an intuitive argument for why adding a non-linear activation (e.g., ReLU) between two linear layers makes the loss non-convex. Why does non-convexity matter for training?

## What's next

Lesson 07 returns to the language-modelling thread: it gives the bigram model **more context** by averaging the embedding vectors of all tokens seen so far — the simplest form of attention. This sets up Lesson 08, which replaces the uniform average with **learned attention weights** computed from queries, keys, and values.
