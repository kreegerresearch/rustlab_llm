# Script:  prefix_averaging.r
# System:  Causal prefix averaging of a sequence of token embeddings
# Concept: The naive "bag of past tokens" summary at position t is the average
#          of embeddings 1..t.  It can be written as a single matrix multiply
#          X̄ = W*X where W is lower-triangular with row t = [1/t, ..., 1/t, 0, ..., 0].
# Equations: x̄_t = (1/t) * sum_{i=1..t} x_i;   X̄ = W * X
# Units:     Embedding values are dimensionless reals; weights in [0, 1]; rows of W sum to 1

# === Sequence of Token Embeddings ===
# T = 6 tokens, embedding dimension d = 4.
# Hand-crafted so each token has a distinct signature, making the averaging
# visually obvious when plotted as a heatmap.
T = 6;
d = 4;

X = [ 1.0,  0.0,  0.0,  0.0;
      0.0,  1.0,  0.0,  0.0;
      0.0,  0.0,  1.0,  0.0;
      0.0,  0.0,  0.0,  1.0;
      1.0,  1.0,  0.0,  0.0;
      0.0,  0.0,  1.0,  1.0 ];

print("Input embeddings X (T=6 rows, d=4 cols):");
print(X);

# === Method 1: Loop-based Running Average ===
# Accumulate a running sum and divide by t to get x̄_t.
X_bar_loop = zeros(T, d);
running = zeros(d);
for t = 1:T
  running = running + X(t);
  for k = 1:d
    X_bar_loop(t, k) = running(k) / t;
  end
end

print("Prefix averages via loop (X̄_loop):");
print(X_bar_loop);

# === Method 2: Lower-Triangular Averaging Matrix ===
# W[t, i] = 1/t  if i <= t,  else 0.
# Then X̄ = W * X in a single matrix multiply.
W = zeros(T, T);
for t = 1:T
  for i = 1:t
    W(t, i) = 1.0 / t;
  end
end

print("Averaging matrix W (lower-triangular, rows sum to 1):");
print(W);

# Sanity check: upper triangle is exactly zero, rows sum to 1.
row_sums = zeros(T);
for t = 1:T
  row_sums(t) = sum(W(t));
end
print("Row sums of W (each should be 1):", row_sums);
print("W(1,2) (above diagonal — must be 0):", W(1, 2));
print("W(1,6) (far above diagonal — must be 0):", W(1, 6));

# === X̄ = W * X ===
X_bar_mm = W * X;

print("Prefix averages via matrix multiply (X̄_mm):");
print(X_bar_mm);

# === Verify the Two Methods Agree ===
diff = max(reshape(abs(X_bar_loop - X_bar_mm), 1, T * d));
print("max|X̄_loop - X̄_mm| (should be ~0):", diff);

# === Verify Row 3 by Hand ===
# x̄_3 = (x_1 + x_2 + x_3) / 3 = ([1,0,0,0] + [0,1,0,0] + [0,0,1,0]) / 3
#      = [1/3, 1/3, 1/3, 0]
x_bar_3_expected = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0];
print("Expected x̄_3 = [1/3, 1/3, 1/3, 0]:", x_bar_3_expected);
print("Computed x̄_3:", X_bar_mm(3));

# === Plot: Averaging Matrix W ===
figure()
imagesc(W, "viridis")
title("Causal Averaging Matrix W (T=6) — lower-triangular, row t = 1/t")
savefig("outputs/averaging_matrix.svg")
print("Saved outputs/averaging_matrix.svg");

# === Plot: Input vs Averaged Embeddings ===
figure()
subplot(2, 1, 1)
imagesc(X, "viridis")
title("Input Embeddings X (distinct per token)")

subplot(2, 1, 2)
imagesc(X_bar_mm, "viridis")
title("Prefix Averages X̄ = W*X (smoother; each row mixes all earlier tokens)")
savefig("outputs/prefix_averages.svg")
print("Saved outputs/prefix_averages.svg");
