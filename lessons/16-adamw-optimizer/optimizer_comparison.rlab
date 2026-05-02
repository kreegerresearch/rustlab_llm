# Script:  optimizer_comparison.r
# System:  Three optimisers on the anisotropic 2D quadratic
#          L = 0.5 * (20 * theta1^2 + theta2^2).  Plot trajectories on top
#          of one another to show:
#            SGD   — zigzags across the ravine
#            Adam  — glides along the floor
#            AdamW — glides AND is pulled toward origin by decoupled decay
# Equations (AdamW):
#   m_t   = beta1 m + (1 - beta1) g
#   v_t   = beta2 v + (1 - beta2) g .* g
#   m_hat = m / (1 - beta1^t),  v_hat = v / (1 - beta2^t)
#   theta = theta - eta * (m_hat ./ (sqrt(v_hat) + eps) + lambda * theta)
# Units:   theta is dimensionless; loss in squared output units.

a = 20.0;
b = 1.0;
sigma = 0.5;        # minibatch gradient noise (sd) — closer to real training

function L = loss_q(theta, a, b)
  L = 0.5 * (a * theta(1) ^ 2 + b * theta(2) ^ 2);
end
function g = grad_q(theta, a, b)
  g = [a * theta(1), b * theta(2)];
end

n_steps = 80;
theta_init = [-2.0, 4.0];

seed(16);

# === SGD ===
theta_sgd = theta_init;
eta_sgd = 0.09;
path_sgd = zeros(n_steps + 1, 2);
path_sgd(1, 1) = theta_sgd(1); path_sgd(1, 2) = theta_sgd(2);
for k = 1:n_steps
  g = grad_q(theta_sgd, a, b) + sigma * randn(2);
  theta_sgd = theta_sgd - eta_sgd * g;
  path_sgd(k + 1, 1) = theta_sgd(1);
  path_sgd(k + 1, 2) = theta_sgd(2);
end

# === Adam ===
beta1 = 0.9;
beta2 = 0.999;
eps   = 1.0e-8;
eta_a = 0.5;
theta_adam = theta_init;
m_a = [0.0, 0.0]; v_a = [0.0, 0.0];
path_adam = zeros(n_steps + 1, 2);
path_adam(1, 1) = theta_adam(1); path_adam(1, 2) = theta_adam(2);
for k = 1:n_steps
  g = grad_q(theta_adam, a, b) + sigma * randn(2);
  m_a = beta1 * m_a + (1 - beta1) * g;
  v_a = beta2 * v_a + (1 - beta2) * (g .* g);
  m_hat = m_a / (1 - beta1 ^ k);
  v_hat = v_a / (1 - beta2 ^ k);
  theta_adam = theta_adam - eta_a * m_hat ./ (sqrt(v_hat) + eps);
  path_adam(k + 1, 1) = theta_adam(1);
  path_adam(k + 1, 2) = theta_adam(2);
end

# === AdamW ===
lambda_wd = 0.05;
theta_aw = theta_init;
m_aw = [0.0, 0.0]; v_aw = [0.0, 0.0];
path_adamw = zeros(n_steps + 1, 2);
path_adamw(1, 1) = theta_aw(1); path_adamw(1, 2) = theta_aw(2);
for k = 1:n_steps
  g = grad_q(theta_aw, a, b) + sigma * randn(2);
  m_aw = beta1 * m_aw + (1 - beta1) * g;
  v_aw = beta2 * v_aw + (1 - beta2) * (g .* g);
  m_hat = m_aw / (1 - beta1 ^ k);
  v_hat = v_aw / (1 - beta2 ^ k);
  theta_aw = theta_aw - eta_a * (m_hat ./ (sqrt(v_hat) + eps) + lambda_wd * theta_aw);
  path_adamw(k + 1, 1) = theta_aw(1);
  path_adamw(k + 1, 2) = theta_aw(2);
end

print("Final  SGD : theta =", theta_sgd,  "  loss =", loss_q(theta_sgd, a, b));
print("Final  Adam: theta =", theta_adam, "  loss =", loss_q(theta_adam, a, b));
print("Final AdamW: theta =", theta_aw,   "  loss =", loss_q(theta_aw,  a, b));

# === Trajectory overlay ===
figure()
plot(path_sgd(:, 1),   path_sgd(:, 2),   "color", "red",   "label", "SGD")
hold("on")
plot(path_adam(:, 1),  path_adam(:, 2),  "color", "blue",  "label", "Adam")
plot(path_adamw(:, 1), path_adamw(:, 2), "color", "green", "label", "AdamW (λ=0.05)")
hold("off")
title("SGD vs Adam vs AdamW on  L = ½(20 θ₁² + θ₂²)")
xlabel("θ₁  (steep)")
ylabel("θ₂  (flat)")
legend("SGD", "Adam", "AdamW")
savefig("optimizer_trajectories.svg")
print("Saved optimizer_trajectories.svg");
