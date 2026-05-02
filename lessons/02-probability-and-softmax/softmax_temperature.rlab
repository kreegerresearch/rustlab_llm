# Script:  softmax_temperature.r
# System:  Softmax probability distribution over a vocabulary
# Concept: Temperature scaling of logits; sharpness vs. flatness of output distribution
# Equations: p_i(T) = exp(z_i / T) / sum_j exp(z_j / T)
# Units:     Logits are dimensionless real numbers; probabilities are in [0, 1]

# === Logits ===
# Four vocabulary tokens with raw scores (e.g., from a model's final layer)
z = [2.0, 1.0, 0.5, -0.5];
print("Logits z:", z);

# === Softmax at Three Temperatures ===
# T = 0.5  →  amplifies differences  →  peaked / confident distribution
# T = 1.0  →  standard softmax
# T = 2.0  →  shrinks differences   →  flatter / more uncertain distribution

p_cold    = softmax(z / 0.5);
p_neutral = softmax(z / 1.0);
p_warm    = softmax(z / 2.0);

print("Probabilities at T=0.5 (cold):");
print(p_cold);
print("Probabilities at T=1.0 (neutral):");
print(p_neutral);
print("Probabilities at T=2.0 (warm):");
print(p_warm);

# === Verify: all distributions sum to 1.0 ===
print("Sum at T=0.5 (should be 1):", sum(p_cold));
print("Sum at T=1.0 (should be 1):", sum(p_neutral));
print("Sum at T=2.0 (should be 1):", sum(p_warm));

# === Plot: overlaid bar charts for the three distributions ===
# Each subplot shows one temperature for clear comparison
figure()
subplot(3, 1, 1)
plot(p_cold, "color", "blue", "label", "T=0.5")
title("Softmax at T=0.5 (cold — peaked)")
ylabel("Probability")
ylim([0, 1])

subplot(3, 1, 2)
plot(p_neutral, "color", "green", "label", "T=1.0")
title("Softmax at T=1.0 (neutral)")
ylabel("Probability")
ylim([0, 1])

subplot(3, 1, 3)
plot(p_warm, "color", "red", "label", "T=2.0")
title("Softmax at T=2.0 (warm — flat)")
ylabel("Probability")
xlabel("Token index")
ylim([0, 1])

savefig("softmax_temperature.svg")
print("Saved softmax_temperature.svg")
