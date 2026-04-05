# Script:  embedding_matrix.r
# System:  Token embedding lookup via a learned embedding matrix
# Concept: E in R^(|V| x d); one-hot lookup selects a dense row vector
# Equations: h = E^T * e_i  =  E_i  (row i of the embedding matrix)
# Units:     Embedding values are dimensionless real numbers; no physical units

# === Parameters ===
vocab_size = 8;   # number of unique tokens in the vocabulary
d_embed    = 6;   # embedding dimension (d << vocab_size in practice)

# === Random Embedding Matrix Initialisation ===
# At the start of training, embeddings are drawn from a small normal distribution.
# Scale by 0.1 so initial activations are not too large (standard practice).
E = randn(vocab_size, d_embed) * 0.1;

print("Embedding matrix E  (shape: vocab_size x d_embed):");
print(size(E));
print(E);

# === Embedding Lookup via One-Hot Multiply ===
# Token index 3  →  one-hot vector e3 = [0,0,1,0,0,0,0,0]
# h = X * E  where X is the one-hot row vector
# Result should equal row 3 of E exactly.

e3 = [0, 0, 1, 0, 0, 0, 0, 0];    # one-hot for token index 3
h3 = e3 * E;                        # 1x8 * 8x6 = 1x6  (embedding lookup)

print("One-hot vector for token 3:");
print(e3);
print("Embedded representation of token 3  (= row 3 of E):");
print(h3);
print("Row 3 of E directly:");
print(E(3));

# === Verify: lookup result equals the row ===
# Max absolute difference should be 0 (or machine epsilon ~1e-16)
diff = max(abs(h3 - E(3)));
print("Max difference between lookup and direct row access (should be ~0):", diff);

# === Heatmap of Embedding Matrix ===
# Rows = tokens (0..vocab_size-1), Columns = embedding dimensions (0..d_embed-1)
# At random init all rows look similar; after training clusters would appear.
saveimagesc(E, "outputs/embedding_matrix.svg", "Embedding Matrix E  (8 tokens x 6 dims)  — random init", "viridis")
print("Saved outputs/embedding_matrix.svg")
