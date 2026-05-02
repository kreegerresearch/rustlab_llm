# Script:  adam_step.r
# System:  Adam optimizer applied to the same anisotropic 2D quadratic
# Concept: Maintain a first-moment EMA m and second-moment EMA v of the
#          gradient.  Apply bias correction to both before the update.
#          Per-coordinate division by sqrt(v) gives an adaptive step size.
# Equations:
#   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
#   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t .* g_t
#   m_hat = m_t / (1 - beta1^t),  v_hat = v_t / (1 - beta2^t)
#   theta_t = theta_{t-1} - eta * m_hat / (sqrt(v_hat) + eps)
# Units:   theta is dimensionless; loss in squared output units.

a = 20.0;
b = 1.0;
function L = loss_aniso2(theta, a, b)
  L = 0.5 * (a * theta(1) ^ 2 + b * theta(2) ^ 2);
end
function g = grad_aniso2(theta, a, b)
  g = [a * theta(1), b * theta(2)];
end

beta1 = 0.9;
beta2 = 0.999;
eps   = 1.0e-8;
eta   = 0.5;
n_steps = 60;

theta = [-2.0, 4.0];
m = [0.0, 0.0];
v = [0.0, 0.0];
loss_curve = zeros(n_steps + 1);
loss_curve(1) = loss_aniso2(theta, a, b);

# === First step in detail to show bias correction ===
g1 = grad_aniso2(theta, a, b);
m_after = beta1 * m + (1 - beta1) * g1;
v_after = beta2 * v + (1 - beta2) * (g1 .* g1);
m_hat_t1 = m_after / (1 - beta1 ^ 1);   # divides by 0.1 -> blows up to g1
v_hat_t1 = v_after / (1 - beta2 ^ 1);   # divides by 0.001 -> blows up to g1.^2

print("Step 1 — raw m =", m_after);
print("Step 1 — bias-corrected m_hat =", m_hat_t1);
print("Step 1 — raw v =", v_after);
print("Step 1 — bias-corrected v_hat =", v_hat_t1);
print("Without bias correction the first step would be ~10x too small (1 - beta1 = 0.1).");

# Now run the full optimisation
for k = 1:n_steps
  g  = grad_aniso2(theta, a, b);
  m  = beta1 * m + (1 - beta1) * g;
  v  = beta2 * v + (1 - beta2) * (g .* g);
  m_hat = m / (1 - beta1 ^ k);
  v_hat = v / (1 - beta2 ^ k);
  theta = theta - eta * m_hat ./ (sqrt(v_hat) + eps);
  loss_curve(k + 1) = loss_aniso2(theta, a, b);
end

print("Final theta:", theta);
print("Final loss :", loss_aniso2(theta, a, b));

# Bias-correction factor at large t — should be near 1
print("Bias factor (1-beta1^60):", 1 - beta1 ^ 60);
print("Bias factor (1-beta2^60):", 1 - beta2 ^ 60);

# === Plot loss curve ===
figure()
steps = 0:n_steps;
plot(steps, log10(loss_curve + 1e-12), "color", "blue", "label", "Adam")
title("Adam log10 loss on anisotropic bowl")
xlabel("step")
ylabel("log10 L")
savefig("adam_step.svg")
print("Saved adam_step.svg");
