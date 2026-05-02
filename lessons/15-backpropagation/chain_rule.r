# Script:  chain_rule.r
# System:  Two-layer MLP  y = tanh(x W1) W2  with squared-error loss
# Concept: Apply the chain rule right-to-left to compute every parameter
#          gradient; verify each entry against a centred finite-difference
#          (which is independent of the analytical derivation).
# Equations:
#   y_hat = tanh(x W1) W2
#   L     = 1/2 (y_hat - t)^2
#   dL/dW2 = a1' * (y_hat - t)
#   dL/dW1 = x'  * ((y_hat - t) W2' .* (1 - a1.^2))
#   dL/dx  = ((y_hat - t) W2' .* (1 - a1.^2)) W1'
# Units:   All quantities dimensionless; loss in squared output units.

# === Setup ===
seed(15);
d_in = 4;
d_h  = 3;
W1 = randn(d_in, d_h) * 0.5;
W2 = randn(d_h,  1)   * 0.5;

x = [0.5, -0.2, 0.1, 0.3];
t = 1.0;

# === Forward ===
# y_hat is a 1x1 matrix; rustlab arithmetic auto-coerces it with the scalar t.
z1    = x * W1;
a1    = tanh(z1);
y_hat = a1 * W2;
L_mat = 0.5 * (y_hat - t) .^ 2;
L     = sum(L_mat);

print("Loss L:", L);
print("Prediction y_hat:");
print(y_hat);

# === Backward (right to left) ===
# Scalarise (y_hat - t) and use M(1) to coerce 1xN matrices to vectors when
# we need vector-vector elementwise ops.  See AGENTS.md "Vector vs. 1xN
# matrix type distinction".
dL_dy     = sum(y_hat - t);             # scalar
dL_dW2    = a1' * dL_dy;                # d_h × 1
dL_da1_m  = dL_dy * W2';                # 1 × d_h matrix
dL_da1    = dL_da1_m(1);                # row 1 -> vector of length d_h
dL_dz1    = dL_da1 .* (1 - a1 .^ 2);    # vector .* vector = vector
dL_dW1    = x' * dL_dz1;                # d_in × d_h
dL_dx     = dL_dz1 * W1';               # 1 × d_in

print("dL/dW2:");
print(dL_dW2);
print("dL/dW1:");
print(dL_dW1);

# === Finite-difference check at every entry of W1 ===
# Independently compute (L_+ - L_-) / (2 eps) and compare.
eps = 1.0e-5;
max_err = 0.0;
for i = 1:d_in
  for j = 1:d_h
    W1p = W1; W1p(i, j) = W1(i, j) + eps;
    W1m = W1; W1m(i, j) = W1(i, j) - eps;
    Lp = sum(0.5 * (tanh(x * W1p) * W2 - t) .^ 2);
    Lm = sum(0.5 * (tanh(x * W1m) * W2 - t) .^ 2);
    fd = (Lp - Lm) / (2.0 * eps);
    err = abs(fd - dL_dW1(i, j));
    if err > max_err
      max_err = err;
    end
  end
end
print("Max |fd - analytic| over W1 entries (should be ~1e-9):", max_err);

# === Finite-difference check at every entry of W2 ===
max_err2 = 0.0;
for j = 1:d_h
  W2p = W2; W2p(j, 1) = W2(j, 1) + eps;
  W2m = W2; W2m(j, 1) = W2(j, 1) - eps;
  Lp = sum(0.5 * (tanh(x * W1) * W2p - t) .^ 2);
  Lm = sum(0.5 * (tanh(x * W1) * W2m - t) .^ 2);
  fd = (Lp - Lm) / (2.0 * eps);
  err = abs(fd - dL_dW2(j, 1));
  if err > max_err2
    max_err2 = err;
  end
end
print("Max |fd - analytic| over W2 entries (should be ~1e-9):", max_err2);
