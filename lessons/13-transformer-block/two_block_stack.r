# Script:  two_block_stack.r
# System:  Stack of 2 Pre-LN transformer blocks
# Concept: Demonstrate that block output shape feeds straight into the next
#          block as input, and the residual stream stays bounded.
# Equations:  same as block_forward.r, applied twice with different weights
# Units:   token features, dimensionless

seed(13);
T = 4;
d_model = 8;
H_heads = 2;
d_k = d_model / H_heads;
d_ff = 4 * d_model;
NEG_INF = -1.0e9;
scale = 1.0 / sqrt(d_k);

# === Causal mask (shared) ===
M_mask = zeros(T, T);
for i = 1:T
  for j = (i + 1):T
    M_mask(i, j) = NEG_INF;
  end
end

# === Helper: one Pre-LN block forward pass ===
function H_out = block_fwd(H_in, W_Q, W_K, W_V, W_O, W_ff1, W_ff2, T, d_model, H_heads, d_k, d_ff, scale, M_mask)
  H_norm1 = zeros(T, d_model);
  for t = 1:T
    H_norm1(t) = layernorm(H_in(t));
  end
  Q = H_norm1 * W_Q;
  K = H_norm1 * W_K;
  V = H_norm1 * W_V;

  out_concat = zeros(T, d_model);
  for h = 1:H_heads
    c_lo = (h - 1) * d_k + 1;
    Q_h = zeros(T, d_k);
    K_h = zeros(T, d_k);
    V_h = zeros(T, d_k);
    for t = 1:T
      for k = 1:d_k
        Q_h(t, k) = Q(t, c_lo + k - 1);
        K_h(t, k) = K(t, c_lo + k - 1);
        V_h(t, k) = V(t, c_lo + k - 1);
      end
    end
    S = Q_h * K_h' * scale + M_mask;
    A_h = zeros(T, T);
    for t = 1:T
      A_h(t) = softmax(S(t));
    end
    O_h = A_h * V_h;
    for t = 1:T
      for k = 1:d_k
        out_concat(t, c_lo + k - 1) = O_h(t, k);
      end
    end
  end
  A_out = out_concat * W_O;
  H_mid = H_in + A_out;

  H_norm2 = zeros(T, d_model);
  for t = 1:T
    H_norm2(t) = layernorm(H_mid(t));
  end
  F_out = gelu(H_norm2 * W_ff1) * W_ff2;
  H_out = H_mid + F_out;
end

# === Inputs and per-block weights ===
H_in = randn(T, d_model);

W_Q1 = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_K1 = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_V1 = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_O1 = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_ff11 = randn(d_model, d_ff) * sqrt(2.0 / d_model);
W_ff21 = randn(d_ff, d_model) * sqrt(2.0 / d_ff);

# Different seed for block 2 so the per-block params actually differ
seed(14);
W_Q2 = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_K2 = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_V2 = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_O2 = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_ff12 = randn(d_model, d_ff) * sqrt(2.0 / d_model);
W_ff22 = randn(d_ff, d_model) * sqrt(2.0 / d_ff);

# === Forward through 2 blocks ===
H1 = block_fwd(H_in, W_Q1, W_K1, W_V1, W_O1, W_ff11, W_ff21, T, d_model, H_heads, d_k, d_ff, scale, M_mask);
H2 = block_fwd(H1,   W_Q2, W_K2, W_V2, W_O2, W_ff12, W_ff22, T, d_model, H_heads, d_k, d_ff, scale, M_mask);

print("|H_in| =", norm(H_in));
print("|H1|   =", norm(H1));
print("|H2|   =", norm(H2));
print("Output shape after 2 blocks:", size(H2));

# === Plot magnitudes ===
mags = [norm(H_in), norm(H1), norm(H2)];
labels = {"H_in", "H1", "H2"};
figure()
bar(labels, mags)
title("Residual stream magnitude after each block")
ylabel("|H|")
savefig("two_block_stack.svg")
