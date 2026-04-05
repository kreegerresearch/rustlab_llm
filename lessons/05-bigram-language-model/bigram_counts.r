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
n_tokens   = 9;
n_bigrams  = n_tokens - 1;   # 8 consecutive pairs

print("Corpus: abcbabcba");
print("Vocabulary: a=1  b=2  c=3");
print("Number of bigrams:", n_bigrams);

# === Bigram Count Matrix C ===
# Trace through "abcbabcba":
#   (a,b)=1->2  (b,c)=2->3  (c,b)=3->2  (b,a)=2->1
#   (a,b)=1->2  (b,c)=2->3  (c,b)=3->2  (b,a)=2->1
#
# Counts:  C[1,2] a->b = 2,  C[2,3] b->c = 2,  C[3,2] c->b = 2,  C[2,1] b->a = 2
# All other pairs = 0
#
# Matrix layout: row = current token, col = next token
#        a  b  c
# a:  [  0, 2, 0 ]
# b:  [  2, 0, 2 ]
# c:  [  0, 2, 0 ]
C = [0, 2, 0; 2, 0, 2; 0, 2, 0];

print("Bigram count matrix C  (row=current, col=next):");
print(C);

# === Verify: total count equals n_bigrams ===
total_counts = sum(reshape(C, 1, vocab_size * vocab_size));
print("Total bigram count (should be 8):", total_counts);

# === Row-Normalise → Probability Matrix P ===
# P_ij = C_ij / row_sum_i
row_sum_a = sum(C(1));   # sum of row a
row_sum_b = sum(C(2));   # sum of row b
row_sum_c = sum(C(3));   # sum of row c

P_a = C(1) / row_sum_a;
P_b = C(2) / row_sum_b;
P_c = C(3) / row_sum_c;

P = [P_a; P_b; P_c];

print("Normalised probability matrix P:");
print(P);
print("Row sums (each should be 1):");
row_sums_P = [sum(P(1)), sum(P(2)), sum(P(3))];
print(row_sums_P);

# === Laplace-Smoothed Probability Matrix P_smooth ===
# P^smooth_ij = (C_ij + 1) / (row_sum_i + |V|)
# Ensures no zero probabilities for unseen bigrams
C_smooth = C + ones(vocab_size, vocab_size);

row_sum_a_s = sum(C_smooth(1));
row_sum_b_s = sum(C_smooth(2));
row_sum_c_s = sum(C_smooth(3));

P_a_s = C_smooth(1) / row_sum_a_s;
P_b_s = C_smooth(2) / row_sum_b_s;
P_c_s = C_smooth(3) / row_sum_c_s;

P_smooth = [P_a_s; P_b_s; P_c_s];

print("Laplace-smoothed probability matrix P_smooth:");
print(P_smooth);

# Verify no zeros (minimum value > 0)
min_smooth = min(reshape(P_smooth, 1, vocab_size * vocab_size));
print("Minimum value in P_smooth (should be > 0):", min_smooth);

# === Row entropy ===
# H(row) = -sum(p * log2(p))   [bits]
eps = 1e-12;
H_a = -sum(P_a .* log2(P_a + eps));
H_b = -sum(P_b .* log2(P_b + eps));
H_c = -sum(P_c .* log2(P_c + eps));

print("Row entropy H(a) bits:", H_a, "  (0 = deterministic)");
print("Row entropy H(b) bits:", H_b, "  (1 = maximum for 2 options)");
print("Row entropy H(c) bits:", H_c, "  (0 = deterministic)");

# === Heatmaps ===
saveimagesc(C, "outputs/bigram_counts.svg", "Bigram Count Matrix C (a,b,c)", "viridis")
print("Saved outputs/bigram_counts.svg");

saveimagesc(P, "outputs/bigram_probabilities.svg", "Bigram Probability Matrix P (row-normalised)", "viridis")
print("Saved outputs/bigram_probabilities.svg");
