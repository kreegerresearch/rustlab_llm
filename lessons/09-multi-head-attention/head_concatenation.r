# Script:  head_concatenation.r
# System:  Multi-head attention — concatenation and output projection
# Concept: Each of H heads produces an independent output O_h ∈ R^(T × d_v).
#          Concatenate along the feature dimension to get T × (H * d_v), then
#          project with W_O ∈ R^(H*d_v × d_model) to produce the final output.
# Equations: O_h = A_h V_h;   Concat = [O_1, O_2, ..., O_H];   O = Concat * W_O
# Units:     Real-valued reals

# === Dimensions ===
T = 4;           # sequence length
d_model = 4;     # model dimension
H = 2;           # number of heads
d_k = 2;         # per-head key/query dim   (= d_model / H)
d_v = 2;         # per-head value dim       (= d_model / H)
scale = 1.0 / sqrt(d_k);

# === Input X  (T × d_model) ===
X = [ 1.0, 0.0, 0.0, 0.0;
      0.0, 1.0, 0.0, 0.0;
      0.0, 0.0, 1.0, 0.0;
      0.0, 0.0, 0.0, 1.0 ];

# === Head 1 — "self" pattern ==========================================
# K_1[i] and Q_1[t] encode position on the unit circle, so dot product
# peaks at i = t.  V_1 is taken as the first d_v columns of X.
K1 = zeros(T, d_k);
Q1 = zeros(T, d_k);
for t = 1:T
  K1(t, 1) = 3.0 * cos(2.0 * pi * t / T);
  K1(t, 2) = 3.0 * sin(2.0 * pi * t / T);
  Q1(t, 1) = K1(t, 1);
  Q1(t, 2) = K1(t, 2);
end

V1 = zeros(T, d_v);
for t = 1:T
  V1(t, 1) = X(t, 1);
  V1(t, 2) = X(t, 2);
end

# === Head 2 — "uniform over past" =====================================
# All-zero Q, K means every score is 0 → softmax gives uniform weights.
# V_2 is the last d_v columns of X.
Q2 = zeros(T, d_k);
K2 = zeros(T, d_k);

V2 = zeros(T, d_v);
for t = 1:T
  V2(t, 1) = X(t, 3);
  V2(t, 2) = X(t, 4);
end

# === Shared causal mask ===
NEG_INF = -1.0e9;
M = zeros(T, T);
for i = 1:T
  for j = (i + 1):T
    M(i, j) = NEG_INF;
  end
end

# === Per-head attention weights ===
function A = attn_weights(Q, K, scale, M, T)
  S = Q * K' * scale;
  S_masked = S + M;
  A = zeros(T, T);
  for t = 1:T
    row = softmax(S_masked(t));
    for j = 1:T
      A(t, j) = row(j);
    end
  end
end

A1 = attn_weights(Q1, K1, scale, M, T);
A2 = attn_weights(Q2, K2, scale, M, T);

print("Head 1 attention weights:");
print(A1);
print("Head 2 attention weights (uniform over past):");
print(A2);

# === Per-head outputs O_h = A_h V_h  (T × d_v) ===
O1 = A1 * V1;
O2 = A2 * V2;

print("O_1 = A_1 V_1  (T × d_v):");
print(O1);
print("O_2 = A_2 V_2  (T × d_v):");
print(O2);
print("Shape of O_1:", size(O1));
print("Shape of O_2:", size(O2));

# === Concatenate along the feature dimension ===
# [O_1, O_2]  →  T × (H * d_v)  =  T × d_model
O_concat = [O1, O2];
print("Shape of concat:", size(O_concat));
print("Concat:");
print(O_concat);

# === Output projection W_O  (H*d_v × d_model) ===
# Normally learned by gradient descent.  We hand-set a permutation that
# swaps the first two columns with the last two so the projection has a
# visible effect.
W_O = [ 0.0, 0.0, 1.0, 0.0;
        0.0, 0.0, 0.0, 1.0;
        1.0, 0.0, 0.0, 0.0;
        0.0, 1.0, 0.0, 0.0 ];

O = O_concat * W_O;
print("Final MHA output  O = Concat * W_O  (T × d_model):");
print(O);
print("Shape of O:", size(O));

# === Parameter count for the full block ===
# Packing per-head projections into a single d_model × d_model matrix
# (each head is a slice of size d_model × d_k), the total parameters are:
#   Q, K, V projections : 3 * d_model^2
#   Output projection   : d_model^2
#   Full MHA block      : 4 * d_model^2
n_qkv = 3 * d_model * d_model;
n_wo  = d_model * d_model;
n_tot = n_qkv + n_wo;

print("--- Parameter counts at d_model =", d_model, ", H =", H, ", d_k =", d_k, "---");
print("  Q + K + V projections (3 * d_model^2):", n_qkv);
print("  Output projection W_O (d_model^2)   :", n_wo);
print("  Full MHA block (4 * d_model^2)      :", n_tot);
print("");
print("The count is independent of H when d_k = d_model / H.  Heads get narrower");
print("as H grows, but the total width (H * d_k) and parameter count are unchanged.");

# === Plots ===
figure()
subplot(1, 2, 1)
imagesc(O_concat, "viridis")
title("Concatenated head outputs (T × H*d_v)")

subplot(1, 2, 2)
imagesc(O, "viridis")
title("Final output O = Concat * W_O")
savefig("outputs/head_concatenation.svg")
print("Saved outputs/head_concatenation.svg");
