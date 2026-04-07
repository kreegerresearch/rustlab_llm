# Script:  loss_landscape.r
# System:  MSE loss surface for a 1D linear model  y = w*x + b
# Concept: 2D loss landscape L(w,b) = (1/N) * sum((w*x + b - y)^2)
# Equations: L(w,b) = 7.5w^2 + b^2 + 5wb - 30w - 10b + 30  (analytic for this dataset)
# Units:     w and b are dimensionless model parameters; L is in squared output units

# === Dataset ===
# x = [1, 2, 3, 4],  y = [2, 4, 6, 8]  →  true relationship y = 2x  (w*=2, b*=0)
x = [1.0, 2.0, 3.0, 4.0];
y = [2.0, 4.0, 6.0, 8.0];

# === Verify true and initial parameters ===
y_hat_true = 2.0 * x + 0.0;
L_true = real(mean((y_hat_true - y) .^ 2));
print("Loss at true parameters (w=2, b=0):", L_true, "  (should be 0)");

y_hat_init = 0.0 * x + 0.0;
L_init = real(mean((y_hat_init - y) .^ 2));
print("Loss at initial parameters (w=0, b=0):", L_init);

# === Analytic expansion of L(w,b) ===
# L(w,b) = (1/4) * sum_k (w*x_k + b - y_k)^2
# For x=[1,2,3,4], y=[2,4,6,8] this expands to:
#   L(w,b) = 7.5*w^2 + b^2 + 5*w*b - 30*w - 10*b + 30
# Verify at (w=2, b=0): 7.5*4 + 0 + 0 - 60 - 0 + 30 = 30 - 60 + 30 = 0  ✓
L_check = 7.5 * 4.0 + 0.0 + 5.0 * 2.0 * 0.0 - 30.0 * 2.0 - 10.0 * 0.0 + 30.0;
print("Analytic formula at (w=2, b=0):", L_check, "  (should be 0)");

# === 2D Loss Grid using outer products ===
# Build L[i,j] = L(w=w_grid[j], b=b_grid[i]) for a 40×40 grid.
# L[i,j] = 7.5*w[j]^2 + b[i]^2 + 5*w[j]*b[i] - 30*w[j] - 10*b[i] + 30
#
# Each term as a matrix (rows = b index, cols = w index):
#   term_w2 [i,j] = 7.5 * w[j]^2          → outer(ones, 7.5*w^2)
#   term_b2 [i,j] = b[i]^2                 → outer(b^2, ones)
#   term_wb [i,j] = 5 * w[j] * b[i]       → 5 * outer(b, w)
#   term_w  [i,j] = -30 * w[j]             → outer(ones, -30*w)
#   term_b  [i,j] = -10 * b[i]             → outer(-10*b, ones)

n_grid = 40;
w_grid = linspace(-0.5, 3.5, n_grid);
b_grid = linspace(-3.0, 3.0, n_grid);

term_w2 = outer(ones(n_grid), 7.5 * w_grid .^ 2);
term_b2 = outer(b_grid .^ 2, ones(n_grid));
term_wb = 5.0 * outer(b_grid, w_grid);
term_w  = outer(ones(n_grid), -30.0 * w_grid);
term_b  = outer(-10.0 * b_grid, ones(n_grid));

L_matrix = term_w2 + term_b2 + term_wb + term_w + term_b + 30.0;

print("Loss surface computed on 40x40 grid.");
min_loss_flat = min(reshape(L_matrix, 1, n_grid * n_grid));
print("Minimum loss on grid (should be ≈ 0):", min_loss_flat);

# === Heatmap ===
# Dark = low loss (near w=2, b=0), bright = high loss
# Rows = b values (row 1 = most negative b = -3)
# Cols = w values (col 1 = w = -0.5)
saveimagesc(L_matrix, "outputs/loss_landscape.svg", "MSE Loss L(w,b): y=2x  minimum at (w=2, b=0)", "viridis")
print("Saved outputs/loss_landscape.svg");
