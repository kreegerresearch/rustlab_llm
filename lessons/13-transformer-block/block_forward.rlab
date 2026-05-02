# Script:  block_forward.r
# System:  One Pre-LN transformer block (MHA → residual → FFN → residual)
# Concept: Trace the (T, d_model) shape through every step; verify the block
#          preserves shape and operates as a function on the residual stream.
# Equations:
#   H_mid = H_in  + MHA(LN1(H_in))
#   H_out = H_mid + FFN(LN2(H_mid))
#   MHA(X) = concat_h( softmax(Q_h K_h^T / sqrt(d_k) + M) V_h ) W_O
#   FFN(X) = GELU(X W_ff1) W_ff2
# Units:   token features, dimensionless

# === Configuration ===
seed(13);
T = 4;
d_model = 8;
H_heads = 2;
d_k = d_model / H_heads;
d_ff = 4 * d_model;
NEG_INF = -1.0e9;

# === Init ===
H_in = randn(T, d_model);
W_Q = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_K = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_V = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_O = randn(d_model, d_model) * (1.0 / sqrt(d_model));
W_ff1 = randn(d_model, d_ff)    * sqrt(2.0 / d_model);
W_ff2 = randn(d_ff,    d_model) * sqrt(2.0 / d_ff);

print("==== Input ====");
print("H_in shape:                      ", size(H_in));

# === Sublayer 1: Pre-LN MHA ===
H_norm1 = zeros(T, d_model);
for t = 1:T
  H_norm1(t) = layernorm(H_in(t));
end
print("LN1(H_in) shape:                 ", size(H_norm1));

Q = H_norm1 * W_Q;
K = H_norm1 * W_K;
V = H_norm1 * W_V;

M_mask = zeros(T, T);
for i = 1:T
  for j = (i + 1):T
    M_mask(i, j) = NEG_INF;
  end
end

scale = 1.0 / sqrt(d_k);
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
print("MHA output shape:                ", size(A_out));

# === Residual 1 ===
H_mid = H_in + A_out;
print("H_mid shape (after residual 1):  ", size(H_mid));

# === Sublayer 2: Pre-LN FFN ===
H_norm2 = zeros(T, d_model);
for t = 1:T
  H_norm2(t) = layernorm(H_mid(t));
end
print("LN2(H_mid) shape:                ", size(H_norm2));

F_pre  = H_norm2 * W_ff1;
F_post = gelu(F_pre);
F_out  = F_post * W_ff2;
print("FFN hidden  shape (T, d_ff):     ", size(F_pre));
print("FFN output  shape (T, d_model):  ", size(F_out));

# === Residual 2 ===
H_out = H_mid + F_out;
print("==== Output ====");
print("H_out shape:                     ", size(H_out));
print("|H_in| =", norm(H_in), "  |H_mid| =", norm(H_mid), "  |H_out| =", norm(H_out));

# === Visualise the residual stream at each stage ===
figure()
subplot(3, 1, 1)
imagesc(H_in,  "viridis")
title("H_in (T=4, d_model=8)")
subplot(3, 1, 2)
imagesc(H_mid, "viridis")
title("H_mid (after MHA + residual)")
subplot(3, 1, 3)
imagesc(H_out, "viridis")
title("H_out (after FFN + residual)")
savefig("block_forward.svg")
