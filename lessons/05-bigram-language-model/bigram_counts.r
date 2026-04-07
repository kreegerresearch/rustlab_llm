# Script:  bigram_counts.r
# System:  Character-level bigram language model
# Concept: Bigram count matrix C[i,j] = #times token j follows token i; row-normalise to P
# Equations: P_ij = C_ij / sum_k(C_ik);  P^smooth_ij = (C_ij + 1) / (sum_k C_ik + |V|)
# Units:     Counts are non-negative integers; probabilities are in [0, 1]

# === Corpus and Vocabulary ===
# Corpus: "abcbabcba"  (9 characters)
# Vocabulary: { a:1, b:2, c:3 }    vocab_size = 3
# Token sequence: [1, 2, 3, 2, 1, 2, 3, 2, 1]

vocab_size = 3;
seq = [1, 2, 3, 2, 1, 2, 3, 2, 1];
n_tokens = len(seq);
n_bigrams = n_tokens - 1;

print("Corpus: abcbabcba");
print("Vocabulary: a=1  b=2  c=3");
print("Number of bigrams:", n_bigrams);

# === Build Bigram Count Matrix from the sequence ===
# For each consecutive pair (seq[t], seq[t+1]), increment C[i, j]
C = zeros(vocab_size, vocab_size);
for t = 1:(n_tokens - 1)
  i = seq(t);
  j = seq(t + 1);
  C(i, j) = C(i, j) + 1;
end

print("Bigram count matrix C  (row=current, col=next):");
print(C);

# === Verify: total count equals n_bigrams ===
total_counts = sum(reshape(C, 1, vocab_size * vocab_size));
print("Total bigram count (should be 8):", total_counts);

# === Row-Normalise → Probability Matrix P ===
# P_ij = C_ij / row_sum_i
P = zeros(vocab_size, vocab_size);
for i = 1:vocab_size
  row_sum = sum(C(i));
  P(i, 1) = C(i, 1) / row_sum;
  P(i, 2) = C(i, 2) / row_sum;
  P(i, 3) = C(i, 3) / row_sum;
end

print("Normalised probability matrix P:");
print(P);

# Verify row sums
row_sums = [sum(P(1)), sum(P(2)), sum(P(3))];
print("Row sums (each should be 1):", row_sums);

# === Laplace-Smoothed Probability Matrix P_smooth ===
# P^smooth_ij = (C_ij + 1) / (row_sum_i + |V|)
# Ensures no zero probabilities for unseen bigrams
C_smooth = C + ones(vocab_size, vocab_size);
P_smooth = zeros(vocab_size, vocab_size);
for i = 1:vocab_size
  row_sum_s = sum(C_smooth(i));
  P_smooth(i, 1) = C_smooth(i, 1) / row_sum_s;
  P_smooth(i, 2) = C_smooth(i, 2) / row_sum_s;
  P_smooth(i, 3) = C_smooth(i, 3) / row_sum_s;
end

print("Laplace-smoothed probability matrix P_smooth:");
print(P_smooth);

min_smooth = min(reshape(P_smooth, 1, vocab_size * vocab_size));
print("Minimum value in P_smooth (should be > 0):", min_smooth);

# === Row entropy ===
# H(row) = -sum(p * log2(p))   [bits]
eps = 1e-12;
H = zeros(vocab_size);
for i = 1:vocab_size
  p = P(i);
  H(i) = max([0.0, -sum(p .* log2(p + eps))]);
end

print("Row entropy H(a) bits:", H(1), "  (0 = deterministic)");
print("Row entropy H(b) bits:", H(2), "  (1 = maximum for 2 options)");
print("Row entropy H(c) bits:", H(3), "  (0 = deterministic)");

# === Heatmaps ===
saveimagesc(C, "outputs/bigram_counts.svg", "Bigram Count Matrix C (a,b,c)", "viridis")
print("Saved outputs/bigram_counts.svg");

saveimagesc(P, "outputs/bigram_probabilities.svg", "Bigram Probability Matrix P (row-normalised)", "viridis")
print("Saved outputs/bigram_probabilities.svg");
