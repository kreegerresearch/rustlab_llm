# Script:  softmax_ce_grad.r
# System:  Softmax over K logits + cross-entropy loss against a one-hot target
# Concept: The softmax+CE composition has a famously simple gradient
#          dL/dz = p - 1_y  (the prediction error vector).
#          Verify this entry by entry against a centred finite-difference.
# Equations:
#   p   = softmax(z)
#   L   = -log p_y
#   dL/dz_j = p_j - delta_{j,y}
# Units:   Logits z are dimensionless; loss in nats.

# === Setup ===
K = 5;                          # vocabulary size
z = [2.0, 1.0, 0.5, -0.5, 0.2];  # arbitrary logits
y_true = 2;                      # target class index (1-based)

p = softmax(z);
L_ce = -log(p(y_true));

print("Logits z:", z);
print("Probabilities p = softmax(z):", p);
print("p sums to 1:", sum(p));
print("Cross-entropy loss L = -log p(y_true):", L_ce);

# === Analytical gradient: dL/dz = p - 1_y ===
# Use zeros(K) (vector) so p - e_y stays vector + vector.
e_y = zeros(K);
e_y(y_true) = 1.0;
dL_dz_analytic = p - e_y;
print("Analytical dL/dz (= p - 1_y):", dL_dz_analytic);

# === Finite-difference check across every logit ===
eps = 1.0e-6;
max_err = 0.0;
for j = 1:K
  zp = z; zp(j) = z(j) + eps;
  zm = z; zm(j) = z(j) - eps;
  Lp = -log(softmax(zp)(y_true));
  Lm = -log(softmax(zm)(y_true));
  fd = (Lp - Lm) / (2.0 * eps);
  err = abs(fd - dL_dz_analytic(j));
  if err > max_err
    max_err = err;
  end
  print("  j =", j, "  analytic =", dL_dz_analytic(j), "  fd =", fd);
end
print("Max |fd - analytic| (should be ~1e-7):", max_err);

# === Sanity: gradient sums to zero ===
# Because both p and 1_y sum to 1, their difference sums to 0.
# A nudge that adds the same constant to every logit cannot change softmax,
# so the gradient must be orthogonal to the all-ones direction.
print("sum(dL/dz) (should be ~0):", sum(dL_dz_analytic));
