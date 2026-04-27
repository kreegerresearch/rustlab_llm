# Script:  layernorm_distribution.r
# System:  LayerNorm sublayer used in every transformer block
# Concept: Standardise each token vector to mean 0 and std 1 across feature dim;
#          show the activation distribution before and after
# Equations:
#   LN(x) = (x - mu) / sqrt(sigma^2 + eps)        (pure standardisation)
#   mu = mean(x), sigma^2 = (1/d) sum (x - mu)^2  (population variance)
# Units:   token features, dimensionless

# === Build a deliberately off-distribution batch ===
seed(12);
T = 4;
d = 6;
H_pre = randn(T, d) * 3.0 + 1.5;        % mean ≈ 1.5, std ≈ 3.0

# === Per-row LayerNorm via loop (matrix overload not yet in rustlab) ===
H_ln = zeros(T, d);
for t = 1:T
  H_ln(t) = layernorm(H_pre(t));
end

# === Per-row stats ===
print("Per-row diagnostics:");
for t = 1:T
  mu_pre  = mean(H_pre(t));
  mu_post = mean(H_ln(t));
  sd_pre  = sqrt(mean((H_pre(t) - mu_pre)  .^ 2));
  sd_post = sqrt(mean((H_ln(t)  - mu_post) .^ 2));
  print("  row", t, "  mean pre=", mu_pre, "  std pre=", sd_pre, "  mean post=", mu_post, "  std post=", sd_post);
end

# === Histograms ===
pre_flat  = reshape(H_pre, 1, T * d);
post_flat = reshape(H_ln,  1, T * d);

figure()
subplot(2, 1, 1)
histogram(pre_flat)
title("Pre-LN activation distribution (mean ≈ 1.5, std ≈ 3.0)")
ylabel("count")

subplot(2, 1, 2)
histogram(post_flat)
title("Post-LN activation distribution (mean ≈ 0, std ≈ 1)")
xlabel("activation value")
ylabel("count")
savefig("layernorm_distribution.svg")
