# Script:  one_hot_encoding.r
# System:  One-hot encoding of a character sequence
# Concept: Mapping token indices to orthogonal basis vectors in R^|V|
# Equations: (e_i)_j = 1 if j == i, else 0;  e_i · e_j = delta_ij
# Units:   Binary matrix entries (0 or 1); no physical units

# === Vocabulary ===
# Encoding "hello" with vocabulary {e:1, h:2, l:3, o:4}
# vocab_size = 4 (four unique characters)
vocab_size = 4;

# === One-Hot Rows (each row is a basis vector in R^4) ===
# e → index 1 → [1, 0, 0, 0]
# h → index 2 → [0, 1, 0, 0]
# l → index 3 → [0, 0, 1, 0]
# o → index 4 → [0, 0, 0, 1]

oh_e = [1, 0, 0, 0];
oh_h = [0, 1, 0, 0];
oh_l = [0, 0, 1, 0];
oh_o = [0, 0, 0, 1];

# === Sequence Matrix X ∈ {0,1}^(T × |V|) ===
# "hello" encodes as token sequence [h, e, l, l, o] → [2, 1, 3, 3, 4]
# Stack one-hot rows:  row 1 = h, row 2 = e, row 3 = l, row 4 = l, row 5 = o
X = [oh_h; oh_e; oh_l; oh_l; oh_o];

print("One-hot matrix X for 'hello'  (5 tokens × 4 vocab):");
print(X);
print("Shape (rows=tokens, cols=vocab_size):", size(X));

# === Verify: each row sums to 1 (exactly one active character per token) ===
# Row sums — computed as X * ones(4) (matrix-vector product)
row_sums = X * ones(vocab_size)';
print("Row sums (each must equal 1):");
print(row_sums);

# === Verify: orthogonality — e_h · e_e = 0, e_l · e_l = 1 ===
dot_h_e = sum(oh_h .* oh_e);
dot_l_l = sum(oh_l .* oh_l);
print("Dot product h · e (should be 0):", dot_h_e);
print("Dot product l · l (should be 1):", dot_l_l);

# === Heatmap ===
# Bright = 1 (active character), dark = 0 (inactive)
# Rows: tokens in sequence order (h, e, l, l, o)
# Cols: vocabulary slots (e=1, h=2, l=3, o=4)
figure()
imagesc(X, "viridis")
title("One-Hot Matrix: 'hello' (5 tokens x 4 vocab)")
savefig("one_hot_matrix.svg")
print("Saved one_hot_matrix.svg")
