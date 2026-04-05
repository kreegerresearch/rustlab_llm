# Script:  cosine_similarity.r
# System:  Pairwise cosine similarity between token embedding vectors
# Concept: cos(a, b) = (a · b) / (||a|| ||b||);  S = E_norm * E_norm^T
# Equations: cos(a,b) = sum(a.*b) / (sqrt(sum(a.^2)) * sqrt(sum(b.^2)))
# Units:     Cosine similarity is dimensionless, in [-1, 1]

# === Hand-crafted Embeddings ===
# Four tokens in a 4-dimensional space.
# Dimensions encode: [royalty, femininity, age, authority]
# These are illustrative — in a real model these values are learned.
king  = [1.0,  0.1,  0.8,  0.9];
queen = [0.9,  0.9,  0.7,  0.8];
man   = [0.1,  0.1,  0.6,  0.4];
woman = [0.1,  0.9,  0.5,  0.3];

print("Embedding vectors (dim=4):");
print("king :", king);
print("queen:", queen);
print("man  :", man);
print("woman:", woman);

# === Cosine Similarity Helper ===
# cos(a, b) = (a · b) / (||a|| * ||b||)
# dot(a,b) = sum(a .* b)
# norm(a)  = sqrt(sum(a .^ 2))
function s = cos_sim(a, b)
  s = sum(a .* b) / (sqrt(sum(a .^ 2)) * sqrt(sum(b .^ 2)))
end

# === Pairwise Similarities ===
s_kk = cos_sim(king,  king);
s_kq = cos_sim(king,  queen);
s_km = cos_sim(king,  man);
s_kw = cos_sim(king,  woman);
s_qk = cos_sim(queen, king);
s_qq = cos_sim(queen, queen);
s_qm = cos_sim(queen, man);
s_qw = cos_sim(queen, woman);
s_mk = cos_sim(man,   king);
s_mq = cos_sim(man,   queen);
s_mm = cos_sim(man,   man);
s_mw = cos_sim(man,   woman);
s_wk = cos_sim(woman, king);
s_wq = cos_sim(woman, queen);
s_wm = cos_sim(woman, man);
s_ww = cos_sim(woman, woman);

print("Cosine similarity matrix (king, queen, man, woman):");
S = [s_kk, s_kq, s_km, s_kw; s_qk, s_qq, s_qm, s_qw; s_mk, s_mq, s_mm, s_mw; s_wk, s_wq, s_wm, s_ww];
print(S);

print("king  / queen (should be high — both royal):", s_kq);
print("king  / man   (moderate — same gender)      :", s_km);
print("queen / woman (moderate — same gender)      :", s_qw);
print("man   / woman (low  — different on most dims):", s_mw);

# === Verify Symmetry ===
# S_ij == S_ji  =>  max(|diff|) should be ~0
# Flatten the difference matrix to a vector so abs() and max() work element-wise
diff_flat = reshape(S - transpose(S), 1, 16);
sym_err = max(abs(diff_flat));
print("Symmetry check max|S - S'| (should be ~0):", sym_err);

# === Analogy Arithmetic: king - man + woman ≈ queen ===
# king - man + woman  (vector arithmetic in embedding space)
analogy = king - man + woman;
print("king - man + woman vector:", analogy);

# Cosine similarity of the analogy result to each token
sim_to_king  = cos_sim(analogy, king);
sim_to_queen = cos_sim(analogy, queen);
sim_to_man   = cos_sim(analogy, man);
sim_to_woman = cos_sim(analogy, woman);

print("Similarity of (king - man + woman) to:");
print("  king :", sim_to_king);
print("  queen:", sim_to_queen);
print("  man  :", sim_to_man);
print("  woman:", sim_to_woman);
print("Closest token should be 'queen'.");

# === Heatmap of Similarity Matrix ===
# Bright = similar (cos ~1), dark = dissimilar (cos ~0 or negative)
saveimagesc(S, "outputs/cosine_similarity.svg", "Cosine Similarity: king, queen, man, woman", "viridis")
print("Saved outputs/cosine_similarity.svg")
