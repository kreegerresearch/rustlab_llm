# Script:  ffn_forward.r
# System:  Position-wise feed-forward sublayer of a transformer block
# Concept: 2-layer MLP applied independently to each token vector; verify per-position independence
# Equations:
#   FFN(x) = W2 * GELU(W1 * x + b1) + b2
#   Same W1, W2, b1, b2 applied to every row of H (position-wise)
# Units:   token features, dimensionless

# === Configuration ===
seed(11);
T = 5;
d_model = 4;
d_ff = 4 * d_model;

# === Inputs and weights (He init for the GELU pre-activation) ===
H  = randn(T, d_model);
W1 = randn(d_model, d_ff) * sqrt(2.0 / d_model);
b1 = zeros(d_ff);
W2 = randn(d_ff, d_model) * sqrt(2.0 / d_ff);
b2 = zeros(d_model);

# === Forward pass ===
ones_T = ones(T);
hidden_pre  = H * W1 + outer(ones_T, b1);
hidden_post = gelu(hidden_pre);
out         = hidden_post * W2 + outer(ones_T, b2);

print("input H shape:        ", size(H));
print("hidden pre  (T, d_ff):", size(hidden_pre));
print("hidden post (T, d_ff):", size(hidden_post));
print("output      (T, d):   ", size(out));

# === Per-position independence check ===
# Permute the rows of H using a permutation matrix P (perm = [3, 1, 2, 5, 4]),
# run FFN, and verify FFN(P*H) == P*FFN(H).
# Vector indexing M([3,1,2,5,4]) is not yet in rustlab — TODO: replace with M(idx) when available.
P_perm = [0, 0, 1, 0, 0;
          1, 0, 0, 0, 0;
          0, 1, 0, 0, 0;
          0, 0, 0, 0, 1;
          0, 0, 0, 1, 0];
H_perm = P_perm * H;
hidden_pre_perm  = H_perm * W1 + outer(ones_T, b1);
hidden_post_perm = gelu(hidden_pre_perm);
out_perm         = hidden_post_perm * W2 + outer(ones_T, b2);

shuffle_err = max(reshape(abs(out_perm - P_perm * out), 1, T * d_model));
print("max | FFN(P*H) - P*FFN(H) | =", shuffle_err);

# === Save a plot of input H so the artefact has something visible ===
figure()
imagesc(H, "viridis")
title("Random input H (T=5, d_model=4)")
xlabel("d_model")
ylabel("position t")
savefig("ffn_forward.svg")
