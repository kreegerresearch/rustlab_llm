# Lesson 06 — Linear Layers & Gradient Descent

## Learning Objectives

By the end of this lesson you will be able to:

- Write the equation for a **linear layer** $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ and identify what each component represents.
- Define a **loss function** (mean squared error) and explain how it measures prediction quality.
- Derive the **gradient** of the loss with respect to $\mathbf{W}$ and $\mathbf{b}$ step by step.
- Apply one step of **gradient descent** by hand and verify the loss decreases.
- Read a **2D loss landscape heatmap** and identify the minimum, gradient direction, and descent path.

---

## Background

This lesson assumes:

- Matrix-vector multiplication and dot products from linear algebra.
- The chain rule from calculus: $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$.
- Cross-entropy loss and the concept of a scalar loss from Lesson 03.
- Embeddings as dense vector representations from Lesson 04.

---

## Theory

### The Linear Layer

A **linear layer** (also called a fully-connected layer) maps an input vector $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$ to an output vector $\mathbf{y} \in \mathbb{R}^{d_{\text{out}}}$:

$$
\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}
$$

where:
- $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is the **weight matrix** — the learnable parameters.
- $\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}$ is the **bias vector** — a learnable offset.
- The output $\mathbf{y}$ is a linear combination of the input features, one per output dimension.

**Connection to embeddings.** The embedding lookup from Lesson 04 is a special case: the embedding matrix $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$ is a weight matrix, and multiplying by a one-hot vector selects one row — exactly a linear layer applied to a one-hot input.

### Loss Function: Mean Squared Error

For regression tasks (fitting a curve to data), the **mean squared error** (MSE) loss is:

$$
\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 = \frac{1}{N} \sum_{i=1}^{N} (w x_i + b - y_i)^2
$$

where $(x_i, y_i)$ are data points and $\hat{y}_i = w x_i + b$ is the model's prediction. For language models the loss is cross-entropy (Lesson 03), but MSE makes the geometry of the loss landscape transparent and is ideal for building intuition.

### Gradient Derivation

To minimise the loss, we need its partial derivatives with respect to $w$ and $b$. Apply the chain rule:

**Partial derivative with respect to $w$:**

$$
\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} 2(\hat{y}_i - y_i) \cdot \frac{\partial \hat{y}_i}{\partial w} = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) \cdot x_i
$$

**Partial derivative with respect to $b$:**

$$
\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} 2(\hat{y}_i - y_i) \cdot \frac{\partial \hat{y}_i}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)
$$

The gradient vector $\nabla \mathcal{L} = [\partial \mathcal{L}/\partial w,\; \partial \mathcal{L}/\partial b]$ points in the direction of **steepest ascent** of the loss surface. Moving in the opposite direction descends the loss.

### Gradient Descent Update Rule

$$
w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}, \qquad b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}
$$

where $\eta > 0$ is the **learning rate** — a hyperparameter that controls step size.

- **Too large $\eta$:** The update overshoots the minimum; loss oscillates or diverges.
- **Too small $\eta$:** Convergence is slow; many iterations needed.
- **Well-chosen $\eta$:** The loss decreases smoothly and converges to the minimum.

### The Loss Landscape

The loss $\mathcal{L}(w, b)$ defines a 2D surface over the $(w, b)$ plane. For MSE with a linear model, this surface is a **convex paraboloid** — it has a unique global minimum and no local minima. Gradient descent is guaranteed to converge to the global optimum.

The minimum is at:

$$
(w^*, b^*) = \arg\min_{w, b} \mathcal{L}(w, b)
$$

which can be found analytically via the **normal equations** or iteratively via gradient descent. For language model training (non-convex losses), gradient descent is the only practical option.

---

## Core Concepts

Every component of a language model — from the embedding lookup to the final output projection — is a linear layer (or a composition of linear layers with non-linear activations). Training the model means adjusting the weight matrices $\mathbf{W}$ and bias vectors $\mathbf{b}$ to minimise the loss. Gradient descent, driven by backpropagation (Lesson 15), is the engine that performs this adjustment.

Understanding the loss landscape geometry is essential for understanding training dynamics: why models sometimes get stuck, why learning rates matter, and why techniques like gradient clipping (Lesson 18) are necessary. The simple 2D MSE case makes this geometry visible and verifiable.

**Common misconception:** A linear layer is not the same as a neural network — it lacks the non-linearity needed to learn complex functions. Stacking linear layers without a non-linear activation is equivalent to a single linear layer (the composition of linear maps is linear). The non-linearity (GELU in Lessons 11–13) is what makes deep networks expressive.

---

## Simulations

### `loss_landscape.r` — 2D Loss Surface Heatmap

**What it computes:**
Evaluates the MSE loss $\mathcal{L}(w, b)$ on a $40 \times 40$ grid over the $(w, b)$ plane, using the dataset $x = [1,2,3,4]$, $y = [2,4,6,8]$ (true relationship: $y = 2x$, so $w^* = 2$, $b^* = 0$). Saves the loss surface as a heatmap.

**What to observe:**
- The dark region (minimum loss) is centred at $(w, b) \approx (2, 0)$.
- The surface is bowl-shaped — a convex paraboloid.
- Loss increases rapidly as $w$ moves away from 2, especially in the $w$ direction (because $w$ is multiplied by $x$ which has larger magnitude).
- Elliptical contours show that the loss is more sensitive to $w$ than to $b$.

**Verify by hand:**
Evaluate $\mathcal{L}(2.0, 0.0)$ manually. Confirm it equals 0 (the model perfectly predicts all four data points).

---

### `gradient_descent.r` — Gradient Descent Path

**What it computes:**
Runs 15 steps of gradient descent starting at $(w_0, b_0) = (0, 0)$ with learning rate $\eta = 0.02$. At each step, computes the predictions, residuals, gradients, and updated parameters. Prints the loss at each step. Plots the $(w, b)$ trajectory as a scatter plot.

**What to observe:**
- The loss decreases monotonically (guaranteed for MSE with a suitable learning rate).
- The path converges toward $(w^*, b^*) = (2, 0)$.
- Early steps are large (steep gradients far from the minimum); later steps are small (gentle gradients near the minimum).
- The trajectory curves — it does not go straight to the minimum because the loss surface is anisotropic (different curvature in the $w$ and $b$ directions).

**Verify by hand:**
Compute the first gradient descent step manually ($\eta = 0.05$):
- Initial: $w_0 = 0$, $b_0 = 0$, $\hat{y} = [0,0,0,0]$, residuals $= [-2,-4,-6,-8]$.
- $\partial \mathcal{L}/\partial w = 0.5 \times ((-2)(1) + (-4)(2) + (-6)(3) + (-8)(4)) = 0.5 \times (-60) = -30$.
- $\partial \mathcal{L}/\partial b = 0.5 \times (-2-4-6-8) = 0.5 \times (-20) = -10$.
- $w_1 = 0 - 0.05 \times (-30) = 1.5$, $b_1 = 0 - 0.05 \times (-10) = 0.5$.
- Compare to the printed Step 1 output.

---

## Exercises

1. **Learning rate sensitivity.** Modify `gradient_descent.r` to use $\eta = 0.1$. Does the loss still decrease? What about $\eta = 0.2$? Find the threshold learning rate above which the algorithm diverges.

2. **Gradient at the minimum.** At $(w^*, b^*) = (2, 0)$, compute $\partial \mathcal{L}/\partial w$ and $\partial \mathcal{L}/\partial b$ by hand. Confirm both are zero, as expected at a minimum.

3. **Non-zero bias.** Change the dataset to $y = [3, 5, 7, 9]$ (true relationship: $y = 2x + 1$). Re-run `gradient_descent.r`. Where does the algorithm converge? What are the new values of $w^*$ and $b^*$?

4. **Counting parameters.** A language model uses a linear layer to project from the embedding dimension $d = 512$ to the vocabulary size $|\mathcal{V}| = 50{,}000$. How many parameters does this output linear layer have (weights + biases)? What fraction of a total GPT-2-small parameter count (117M) does this represent?

5. **Convexity.** The MSE loss for a linear model is convex. Give an intuitive argument for why adding a non-linear activation (e.g., ReLU) between two linear layers makes the loss non-convex. Why does non-convexity matter for training?
