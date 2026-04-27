# Script:  pe_matrix.r
# System:  Sinusoidal positional encoding for a transformer's token stream
# Concept: Build the full T x d_model PE matrix and visualise its banded structure
# Equations:
#   PE(t, 2k)   = sin(t / 10000^(2k / d_model))
#   PE(t, 2k+1) = cos(t / 10000^(2k / d_model))
# Units:   positions (integer t), dimensions (integer i), encoding values dimensionless

# === Configuration ===
T = 64;
d_model = 32;

# === Build the PE matrix ===
PE = zeros(T, d_model);
for t = 1:T
  for i = 1:d_model
    pair_idx = floor((i - 1) / 2);
    div = 10000.0 ^ (2.0 * pair_idx / d_model);
    angle = t / div;
    if mod(i - 1, 2) == 0
      PE(t, i) = sin(angle);
    else
      PE(t, i) = cos(angle);
    end
  end
end

# === Sanity prints ===
print("PE shape:", size(PE));
print("PE(1, 1)  (sin(1)):     ", PE(1, 1));
print("PE(1, 2)  (cos(1)):     ", PE(1, 2));
print("PE(64, 1) (sin(64)):    ", PE(64, 1));
print("PE(t, end) ~= cos(0)=1: ", PE(1, d_model), PE(64, d_model));

# === Heatmap ===
figure()
imagesc(PE, "viridis")
title("Sinusoidal Positional Encoding (T=64, d=32)")
xlabel("Embedding dimension")
ylabel("Position t")
savefig("pe_matrix.svg")
