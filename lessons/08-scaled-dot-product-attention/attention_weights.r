# Script:  attention_weights.r
# System:  Scaled dot-product attention — deriving the attention weight matrix
# Concept: Given hand-crafted Q, K, compute scores = Q K^T / sqrt(d_k), apply
#          a causal mask (upper triangle → -inf), then row-wise softmax to get
#          attention weights A.  Each row of A is a distribution over the first
#          t tokens; rows sum to 1; support is lower-triangular.
# Equations: S = Q K^T / sqrt(d_k);  M[t,i] = -inf if i > t else 0;
#            A[t, :] = softmax(S[t, :] + M[t, :])
# Units:     Q, K real-valued; A entries in [0, 1]

# === Small working example ===
T = 5;           # sequence length
d_k = 4;         # key / query dimension
scale = 1.0 / sqrt(d_k);    # = 0.5 for d_k = 4

# Hand-crafted Q, K designed so the attention pattern is interpretable.
# K row 1 has a large "keyword" feature along dim 1.
# Q rows 2 and 4 query for that keyword (dim 1), so they attend strongly to token 1.
# Q row 3 queries dim 2, matching K row 2.
K = [ 2.0,  0.0,  0.0,  0.0;
      0.0,  2.0,  0.0,  0.0;
      0.0,  0.0,  2.0,  0.0;
      1.0,  1.0,  0.0,  0.0;
      0.0,  0.0,  1.0,  1.0 ];

Q = [ 1.0,  0.0,  0.0,  0.0;
      1.0,  0.0,  0.0,  0.0;
      0.0,  1.0,  0.0,  0.0;
      1.0,  0.0,  0.0,  0.0;
      0.5,  0.5,  0.0,  0.0 ];

print("Q shape (T × d_k):", size(Q));
print("K shape (T × d_k):", size(K));

# === Raw scaled scores S = Q K^T / sqrt(d_k) ===
# Q: T × d_k,  K: T × d_k,  K' : d_k × T  =>  S : T × T
S = Q * K' * scale;
print("Scaled scores S (T × T):");
print(S);

# === Causal mask ===
# M[t, i] = -1e9 for i > t;  0 otherwise.
# Large negative (not literal -inf) avoids NaN in exp().
NEG_INF = -1.0e9;
M = zeros(T, T);
for i = 1:T
  for j = (i + 1):T
    M(i, j) = NEG_INF;
  end
end
S_masked = S + M;

print("Masked scores (upper triangle → -1e9):");
print(S_masked);

# === Row-wise softmax ===
A = zeros(T, T);
for t = 1:T
  row = softmax(S_masked(t));
  for j = 1:T
    A(t, j) = row(j);
  end
end

print("Attention weights A = softmax_row(S_masked):");
print(A);

# === Verify row sums ===
row_sums = zeros(T);
for t = 1:T
  row_sums(t) = sum(A(t));
end
print("Row sums of A (each should be 1):", row_sums);

# === Verify causality: upper triangle is ~0 ===
max_upper = 0.0;
for i = 1:T
  for j = (i + 1):T
    if A(i, j) > max_upper
      max_upper = A(i, j);
    end
  end
end
print("Max attention weight in upper triangle (should be ~0):", max_upper);

# === Row 1 is deterministic: only token 1 visible ===
print("A(1, 1) (token 1 can only attend to itself — should be 1):", A(1, 1));

# === Plots ===
# Three-stage pipeline: raw scores → masked scores → attention weights
figure()
subplot(3, 1, 1)
imagesc(S, "viridis")
title("Scaled scores S = Q K^T / sqrt(d_k)")

subplot(3, 1, 2)
imagesc(S_masked, "viridis")
title("After causal mask (upper triangle → -∞)")

subplot(3, 1, 3)
imagesc(A, "viridis")
title("Attention weights A = softmax_row(S_masked)")
savefig("attention_pipeline.svg")
print("Saved attention_pipeline.svg");

# Standalone heatmap of the final weights
figure()
imagesc(A, "viridis")
title("Causal Attention Weights — lower triangular, rows sum to 1")
savefig("attention_weights.svg")
print("Saved attention_weights.svg");
