# Script:  pe_translation.r
# System:  Sinusoidal positional encoding — relative-position structure
# Concept: Verify that PE_t . PE_{t+k} depends only on the offset k, not on t
# Equations:
#   sim(t, k) = sum(PE(t) .* PE(t+k))
#   Translation property:  R_k(d) is a fixed rotation per (2k, 2k+1) pair
# Units:   integer positions, dimensionless similarity

# === Build the PE matrix (same scheme as pe_matrix.r) ===
T = 80;
d_model = 32;

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

# === Compute similarity vs. offset at two base positions ===
n_offsets = 40;
sims_short = zeros(n_offsets);
sims_far   = zeros(n_offsets);
for k = 0:(n_offsets - 1)
  sims_short(k + 1) = sum(PE(10) .* PE(10 + k));
  sims_far(k + 1)   = sum(PE(20) .* PE(20 + k));
end

drift = max(abs(sims_short - sims_far));
print("max |sim_t=10(k) - sim_t=20(k)| over k=0..39 =", drift);
print("sims_short(1) (self-similarity at t=10) =", sims_short(1));
print("sims_short(2) (offset 1)                 =", sims_short(2));
print("sims_short(end) (offset 39)              =", sims_short(n_offsets));

# === Plot ===
figure()
hold("on")
plot(0:(n_offsets - 1), sims_short, "color", "blue", "label", "base t=10")
plot(0:(n_offsets - 1), sims_far,   "color", "red",  "label", "base t=20")
title("PE Dot-Product Similarity vs. Offset k")
xlabel("Offset k (tokens)")
ylabel("Dot product")
legend()
hold("off")
savefig("pe_translation.svg")
