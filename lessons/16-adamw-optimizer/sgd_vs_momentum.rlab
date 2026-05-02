# Script:  sgd_vs_momentum.r
# System:  Anisotropic 2D quadratic L = 0.5 * (a * theta1^2 + b * theta2^2)
# Concept: SGD with one fixed learning rate must trade speed in the flat
#          direction against stability in the steep direction.  Momentum
#          (an EMA of the gradient) cancels in the oscillating axis and
#          accumulates in the consistent axis.
# Equations:
#   SGD:       theta_{t} = theta_{t-1} - eta * g_t
#   Momentum:  v_t = mu * v_{t-1} + g_t;  theta_t = theta_{t-1} - eta * v_t
# Units:   theta is dimensionless; loss is in squared output units.

a = 20.0;
b = 1.0;

function L = loss_aniso(theta, a, b)
  L = 0.5 * (a * theta(1) ^ 2 + b * theta(2) ^ 2);
end

function g = grad_aniso(theta, a, b)
  g = [a * theta(1), b * theta(2)];
end

# === SGD ===
theta_sgd = [-2.0, 4.0];
eta = 0.09;            # near the stability bound 2/a = 0.1
n_steps = 60;
path_sgd = zeros(n_steps + 1, 2);
loss_sgd = zeros(n_steps + 1);
path_sgd(1, 1) = theta_sgd(1);
path_sgd(1, 2) = theta_sgd(2);
loss_sgd(1) = loss_aniso(theta_sgd, a, b);

for k = 1:n_steps
  g = grad_aniso(theta_sgd, a, b);
  theta_sgd = theta_sgd - eta * g;
  path_sgd(k + 1, 1) = theta_sgd(1);
  path_sgd(k + 1, 2) = theta_sgd(2);
  loss_sgd(k + 1) = loss_aniso(theta_sgd, a, b);
end

print("SGD final theta:", theta_sgd);
print("SGD final loss :", loss_aniso(theta_sgd, a, b));

# === SGD with momentum ===
theta_mom = [-2.0, 4.0];
v_mom = [0.0, 0.0];
mu = 0.9;
path_mom = zeros(n_steps + 1, 2);
loss_mom = zeros(n_steps + 1);
path_mom(1, 1) = theta_mom(1);
path_mom(1, 2) = theta_mom(2);
loss_mom(1) = loss_aniso(theta_mom, a, b);

for k = 1:n_steps
  g = grad_aniso(theta_mom, a, b);
  v_mom = mu * v_mom + g;
  theta_mom = theta_mom - eta * v_mom;
  path_mom(k + 1, 1) = theta_mom(1);
  path_mom(k + 1, 2) = theta_mom(2);
  loss_mom(k + 1) = loss_aniso(theta_mom, a, b);
end

print("SGD+mom final theta:", theta_mom);
print("SGD+mom final loss :", loss_aniso(theta_mom, a, b));

print("Speedup ratio (SGD final / momentum final):", loss_sgd(n_steps + 1) / loss_mom(n_steps + 1));

# === Plots: trajectory overlay and loss curves ===
figure()
plot(path_sgd(:, 1), path_sgd(:, 2), "color", "red",  "label", "SGD")
hold("on")
plot(path_mom(:, 1), path_mom(:, 2), "color", "blue", "label", "SGD+momentum")
hold("off")
title("Trajectories on  L = ½(20 θ₁² + θ₂²)")
xlabel("θ₁")
ylabel("θ₂")
legend("SGD", "SGD+momentum")
savefig("sgd_vs_momentum_path.svg")
print("Saved sgd_vs_momentum_path.svg");

figure()
steps = 0:n_steps;
plot(steps, log10(loss_sgd + 1e-12), "color", "red",  "label", "SGD")
hold("on")
plot(steps, log10(loss_mom + 1e-12), "color", "blue", "label", "SGD+momentum")
hold("off")
title("log10 loss vs step")
xlabel("step")
ylabel("log10 L")
legend("SGD", "SGD+momentum")
savefig("sgd_vs_momentum_loss.svg")
print("Saved sgd_vs_momentum_loss.svg");
