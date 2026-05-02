# Script:  entropy.r
# System:  Shannon entropy of softmax probability distributions
# Concept: Entropy measures the uncertainty (spread) of a distribution
# Equations: H(p) = -sum_i p_i * log2(p_i)   [bits]
# Units:     Entropy is in bits (base-2 logarithm)

# === Logits ===
z = [2.0, 1.0, 0.5, -0.5];
vocab_size = 4;

# === Entropy at Multiple Temperatures ===
# Compute softmax at each temperature, then compute entropy H(p)

# Helper: entropy of a probability vector
# H = -sum(p .* log2(p))  — add small epsilon to avoid log(0)
eps = 1e-12;

p05 = softmax(z / 0.5);
p10 = softmax(z / 1.0);
p20 = softmax(z / 2.0);
p50 = softmax(z / 5.0);

H05 = -sum(p05 .* log2(p05 + eps));
H10 = -sum(p10 .* log2(p10 + eps));
H20 = -sum(p20 .* log2(p20 + eps));
H50 = -sum(p50 .* log2(p50 + eps));

print("Entropy at T=0.5:", H05, "bits");
print("Entropy at T=1.0:", H10, "bits");
print("Entropy at T=2.0:", H20, "bits");
print("Entropy at T=5.0:", H50, "bits");

# === Theoretical bounds ===
# Maximum entropy for vocab_size=4: log2(4) = 2.0 bits  (uniform distribution)
# Minimum entropy: 0 bits  (all mass on one token)
H_max = log2(vocab_size);
print("Maximum entropy (uniform, 4 tokens):", H_max, "bits");

# === Verify: uniform distribution achieves maximum entropy ===
p_uniform = ones(vocab_size) / vocab_size;
H_uniform = -sum(p_uniform .* log2(p_uniform + eps));
print("Entropy of uniform distribution:", H_uniform, "bits  (should equal log2(4) = 2.0)");

# === Verify: near-deterministic distribution has entropy ≈ 0 ===
# Almost all mass on token 1
p_det = [0.999, 0.0003, 0.0003, 0.0004];
H_det = -sum(p_det .* log2(p_det + eps));
print("Entropy of near-deterministic distribution:", H_det, "bits  (should be ≈ 0)");

# === Entropy vs Temperature bar chart ===
# Four representative temperatures
H_vec = [H05, H10, H20, H50];
figure()
bar(H_vec, "Entropy (bits) at T = 0.5, 1.0, 2.0, 5.0")
hold("on")
hline(log2(vocab_size), "red", "max = log2(4)")
savefig("entropy_vs_temperature.svg")
print("Saved entropy_vs_temperature.svg")
