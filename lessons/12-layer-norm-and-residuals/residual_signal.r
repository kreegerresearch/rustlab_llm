# Script:  residual_signal.r
# System:  Deep stack of GELU sublayers, with and without residual connections
# Concept: Show that residuals preserve activation magnitude across depth — a
#          forward-pass proxy for the gradient-preservation argument that
#          motivates residual connections.
# Equations:
#   no residual:   x_{L+1} = GELU(W_L * x_L)
#   with residual: x_{L+1} = x_L + alpha * GELU(W_L * x_L),  alpha = 0.1
# Units:   activations dimensionless, "magnitude" = L2 norm

# === Setup ===
seed(13);
d = 32;
n_layers = 24;
alpha = 0.1;             % small residual step keeps the identity dominant

# === Stack of weight matrices, packed into one tall matrix ===
W_stack = zeros(n_layers * d, d);
for L = 1:n_layers
  W_L = randn(d, d) * (1.0 / sqrt(d));            % spectral norm ≈ 1
  for r = 1:d
    W_stack((L - 1) * d + r) = W_L(r);
  end
end

# === Forward sweep, two flavours ===
x0 = randn(d);
mag_no_res = zeros(n_layers + 1);
mag_res    = zeros(n_layers + 1);
mag_no_res(1) = norm(x0);
mag_res(1)    = norm(x0);

x_plain = x0;
x_resi  = x0;
for L = 1:n_layers
  W_L = zeros(d, d);
  for r = 1:d
    W_L(r) = W_stack((L - 1) * d + r);
  end

  % Use x * W' (right-multiply) so the result stays a vector, not a 1xN matrix.
  % See AGENTS.md Rustlab Recommendations — (W*x')' returns a matrix and
  % breaks the residual addition further down.
  f_plain = gelu(x_plain * W_L');
  x_plain = f_plain;
  mag_no_res(L + 1) = norm(x_plain);

  f_resi  = gelu(x_resi * W_L');
  x_resi  = x_resi + alpha * f_resi;
  mag_res(L + 1) = norm(x_resi);
end

print("Layer  0   magnitude (both):  ", mag_no_res(1));
print("Layer 24   magnitude (no res):", mag_no_res(n_layers + 1));
print("Layer 24   magnitude (w/ res):", mag_res(n_layers + 1));
print("Ratio (no_res / w_res):       ", mag_no_res(n_layers + 1) / mag_res(n_layers + 1));

# === Plot ===
figure()
hold("on")
plot(0:n_layers, mag_no_res, "color", "red",  "label", "no residual")
plot(0:n_layers, mag_res,    "color", "blue", "label", "with residual")
title("Activation magnitude vs. depth (24 random GELU sublayers, d=32)")
xlabel("layer index")
ylabel("|x|")
legend()
hold("off")
savefig("residual_signal.svg")
