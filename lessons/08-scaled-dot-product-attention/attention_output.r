# Script:  attention_output.r
# System:  End-to-end scaled dot-product attention  X → Q, K, V → A → O
# Concept: Three linear projections turn each input embedding into a query,
#          key, and value vector.  Attention weights combine the value vectors
#          into the context-aware output O = A V.  Demonstrates the full
#          pipeline and counts learnable parameters for one attention block.
# Equations: Q = X W_Q;  K = X W_K;  V = X W_V;
#            A = softmax_row((Q K^T) / sqrt(d_k) + M);  O = A V
# Units:     All values are dimensionless reals; A entries in [0,1]

# === Dimensions ===
T = 5;          # sequence length
d_model = 6;    # input embedding dimension
d_k = 4;        # key / query dimension
d_v = 4;        # value dimension
scale = 1.0 / sqrt(d_k);

# === Input embeddings X (T × d_model) ===
# Hand-set so token 1 is the "keyword", later tokens mix in various ways.
X = [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
      0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
      0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
      0.5, 0.5, 0.0, 0.0, 0.0, 0.0;
      0.0, 0.0, 0.0, 1.0, 1.0, 1.0 ];

print("X shape (T × d_model):", size(X));

# === Projection matrices (d_model × {d_k, d_k, d_v}) ===
# Hand-set values keep the arithmetic reproducible.  In a real model these
# are learned by gradient descent.
W_Q = [ 1.0, 0.0, 0.0, 0.0;
        0.0, 1.0, 0.0, 0.0;
        0.0, 0.0, 1.0, 0.0;
        0.0, 0.0, 0.0, 1.0;
        0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0 ];

W_K = [ 1.0, 0.0, 0.0, 0.0;
        0.0, 1.0, 0.0, 0.0;
        0.0, 0.0, 1.0, 0.0;
        0.0, 0.0, 0.0, 1.0;
        0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0 ];

W_V = [ 1.0, 0.0, 0.0, 0.0;
        0.0, 1.0, 0.0, 0.0;
        0.0, 0.0, 1.0, 0.0;
        0.0, 0.0, 0.0, 1.0;
        1.0, 1.0, 0.0, 0.0;
        0.0, 0.0, 1.0, 1.0 ];

# === Linear projections ===
Q = X * W_Q;    # T × d_k
K = X * W_K;    # T × d_k
V = X * W_V;    # T × d_v

print("Q shape:", size(Q));
print("K shape:", size(K));
print("V shape:", size(V));

print("V (values to be mixed):");
print(V);

# === Attention weights A ===
S = Q * K' * scale;

NEG_INF = -1.0e9;
M = zeros(T, T);
for i = 1:T
  for j = (i + 1):T
    M(i, j) = NEG_INF;
  end
end
S_masked = S + M;

A = zeros(T, T);
for t = 1:T
  row = softmax(S_masked(t));
  for j = 1:T
    A(t, j) = row(j);
  end
end

print("Attention weights A:");
print(A);

# === Output O = A V  (T × d_v) ===
O = A * V;

print("Output O shape:", size(O));
print("O (context-aware representations):");
print(O);

# === Verify: row 1 of O equals row 1 of V ===
# Token 1 attends only to itself (A(1,1) = 1), so O(1) must equal V(1).
diff_row1 = max(abs(O(1) - V(1)));
print("max|O(1) - V(1)| (should be 0 — token 1 attends only to itself):", diff_row1);

# === Parameter count ===
# One self-attention block: W_Q, W_K, W_V   (no output projection yet — Lesson 09).
n_params_qkv = 3 * d_model * d_k;
print("Learnable parameters in W_Q, W_K, W_V:", n_params_qkv);
print("Formula: 3 * d_model * d_k  =  3 *", d_model, "*", d_k, "=", n_params_qkv);
print("Note: parameter count is independent of sequence length T.");

# === Plots ===
figure()
subplot(3, 1, 1)
imagesc(Q, "viridis")
title("Q = X W_Q (queries)")

subplot(3, 1, 2)
imagesc(K, "viridis")
title("K = X W_K (keys)")

subplot(3, 1, 3)
imagesc(V, "viridis")
title("V = X W_V (values)")
savefig("qkv_projections.svg")
print("Saved qkv_projections.svg");

figure()
subplot(2, 1, 1)
imagesc(A, "viridis")
title("Attention weights A")

subplot(2, 1, 2)
imagesc(O, "viridis")
title("Output O = A V")
savefig("attention_output.svg")
print("Saved attention_output.svg");
