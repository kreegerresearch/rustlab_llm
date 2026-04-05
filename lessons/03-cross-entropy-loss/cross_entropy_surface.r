# Script:  cross_entropy_surface.r
# System:  Cross-entropy loss as a function of predicted probability
# Concept: L(p_c) = -log(p_c); convex loss that penalises low-confidence correct predictions
# Equations: L = -log(p_hat_c)   where p_hat_c is the predicted probability of the correct token
# Units:     p_hat_c is dimensionless in (0, 1]; loss is in nats (natural log base)

# === Loss Curve ===
# Sample p_hat_c across the interval (0.01, 1.0]
p_hat = linspace(0.01, 1.0, 200);

# Cross-entropy loss: L = -log(p_hat_c)
# Uses natural logarithm (nats) — standard in PyTorch / most frameworks
loss = -log(p_hat);

print("Cross-entropy loss at selected probabilities:");

# === Key reference points ===
# p=1/50000: baseline for a 50k-vocab model (uniform prediction)
p_uniform_50k = 1.0 / 50000.0;
L_50k = -log(p_uniform_50k);
print("  p=1/50000 (vocab 50k uniform baseline):", L_50k, "nats");

# p=1/4: uniform over 4 tokens (our toy vocab)
p_uniform_4 = 0.25;
L_4 = -log(p_uniform_4);
print("  p=0.25  (uniform over 4 tokens):", L_4, "nats  (= log(4))");
print("  log(4) =", log(4.0));

# p=0.5: model assigns 50% to the right answer
p_half = 0.5;
L_half = -log(p_half);
print("  p=0.50  :", L_half, "nats");

# p=0.9: model is fairly confident
p_high = 0.9;
L_high = -log(p_high);
print("  p=0.90  :", L_high, "nats");

# p=0.99: model is very confident and correct
p_vhigh = 0.99;
L_vhigh = -log(p_vhigh);
print("  p=0.99  :", L_vhigh, "nats");

# === Plot: L(p) curve ===
figure()
hold("on")
plot(p_hat, loss, "color", "blue", "label", "L = -log(p)")
title("Cross-Entropy Loss vs. Predicted Probability of Correct Token")
xlabel("Predicted probability of correct token (p_c)")
ylabel("Loss L = -log(p_c)  [nats]")
ylim([0, 6])
legend()
savefig("outputs/cross_entropy_surface.svg")
print("Saved outputs/cross_entropy_surface.svg")

# === Gradient of loss ===
# dL/dp_c = -1 / p_c
# At p_c = 0.01 the gradient magnitude is 100 — a very strong training signal
# At p_c = 0.99 the gradient magnitude is ~1.01 — nearly no update needed
grad_at_low  = 1.0 / 0.01;
grad_at_high = 1.0 / 0.99;
print("Gradient magnitude |dL/dp_c| at p=0.01:", grad_at_low);
print("Gradient magnitude |dL/dp_c| at p=0.99:", grad_at_high);
