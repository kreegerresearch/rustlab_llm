# Script:  gradient_flow.r
# System:  N-layer linear stack, each layer a random matrix scaled to a target
#          spectral radius rho.  Backprop the gradient from layer N to layer 1
#          and record its norm at each depth.
# Concept: When the typical Jacobian has spectral radius rho, gradient norm at
#          layer k is approximately ||g_N|| * rho^(N-k).
#          rho<1 -> vanishing,  rho=1 -> stable,  rho>1 -> exploding.
# Equations:
#   g_{k-1} = g_k * J_k'  with  J_k = (1/sqrt(d)) * G_k * rho,  G_k ~ N(0, I)
# Units:   Norms are dimensionless; log10 used to compress the dynamic range.

# === Setup ===
N = 12;
d = 32;
rhos = [0.7, 1.0, 1.3];
n_regimes = 3;

G_norms = zeros(n_regimes, N);

# === Backward through N random layers, three regimes ===
seed(33);
for r = 1:n_regimes
  rho = rhos(r);
  g = randn(1, d) * 0.1;       # upstream gradient at layer N
  for k = N:-1:1
    # 1/sqrt(d) keeps the random matrix near unit spectral radius;
    # multiplying by rho retargets it.
    J = (randn(d, d) / sqrt(d)) * rho;
    g = g * J';
    G_norms(r, k) = norm(g);
  end
end

# === Print numerical evidence ===
print("Per-layer ||grad||  (rho = 0.7 — vanish):");
for k = 1:N
  print("  L", k, ":", G_norms(1, k));
end

print("Per-layer ||grad||  (rho = 1.0 — stable):");
for k = 1:N
  print("  L", k, ":", G_norms(2, k));
end

print("Per-layer ||grad||  (rho = 1.3 — explode):");
for k = 1:N
  print("  L", k, ":", G_norms(3, k));
end

# Ratios L1 / L12 — vanish should be tiny, explode should be huge
print("L1/L12 ratio  (rho=0.7, expect << 1):", G_norms(1, 1) / G_norms(1, N));
print("L1/L12 ratio  (rho=1.0, expect ~ 1):", G_norms(2, 1) / G_norms(2, N));
print("L1/L12 ratio  (rho=1.3, expect >> 1):", G_norms(3, 1) / G_norms(3, N));

# === Heatmap (log10 to fit all three regimes on one scale) ===
# Rows are the three regimes; columns are layer depths L1..L12.  Categorical
# axis labels turn the heatmap into a direct lookup of "what is log10(||g||)
# at depth k for spectral-radius rho?".
regimes = {"rho=0.7  (vanish)", "rho=1.0  (stable)", "rho=1.3  (explode)"};
layers  = {"L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12"};

figure()
heatmap(layers, regimes, log10(G_norms + 1e-12), "log10(||grad||) per layer (rows: regime, cols: depth)", "viridis")
savefig("gradient_flow.svg")
print("Saved gradient_flow.svg");
