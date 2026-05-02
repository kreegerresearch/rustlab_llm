# Script:  schedule_vs_constant.r
# System:  Three simulated training runs on a noisy 2D quadratic.
# Concept: Compare a constant high LR (often unstable), a constant low LR
#          (slow but safe), and a warmup+cosine schedule (fast and stable).
# Equations:
#   gradient with noise: g_t = grad(theta) + sigma * z   where z ~ N(0, I)
#   surrogate loss: L = 0.5*(10*theta1^2 + theta2^2) + 1
# Units:   theta dimensionless; loss in surrogate units.

seed(17);

function L = sloss(theta)
  L = 0.5 * (10.0 * theta(1) ^ 2 + theta(2) ^ 2) + 1.0;
end
function g = sgrad(theta)
  g = [10.0 * theta(1), theta(2)];
end

n_train = 300;
sigma = 0.3;
init  = [-3.0, 2.0];

# === Constant high LR ===
theta = init;
loss_high = zeros(n_train + 1);
loss_high(1) = sloss(theta);
for t = 1:n_train
  g = sgrad(theta) + sigma * randn(2);
  theta = theta - 0.18 * g;
  loss_high(t + 1) = sloss(theta);
end

# === Constant low LR ===
theta = init;
loss_low = zeros(n_train + 1);
loss_low(1) = sloss(theta);
for t = 1:n_train
  g = sgrad(theta) + sigma * randn(2);
  theta = theta - 0.02 * g;
  loss_low(t + 1) = sloss(theta);
end

# === Warmup + cosine schedule ===
T_w  = 30;
T_d  = n_train;
eta_p = 0.18;
eta_f = 0.02;
theta = init;
loss_sched = zeros(n_train + 1);
loss_sched(1) = sloss(theta);
sched_curve = zeros(n_train + 1);
sched_curve(1) = 0.0;
for t = 1:n_train
  if t <= T_w
    eta_t = eta_p * (t / T_w);
  else
    progress = (t - T_w) / (T_d - T_w);
    eta_t = eta_f + 0.5 * (eta_p - eta_f) * (1 + cos(pi * progress));
  end
  sched_curve(t + 1) = eta_t;
  g = sgrad(theta) + sigma * randn(2);
  theta = theta - eta_t * g;
  loss_sched(t + 1) = sloss(theta);
end

print("Constant high LR  final loss:", loss_high(n_train + 1));
print("Constant low  LR  final loss:", loss_low(n_train + 1));
print("Warmup+cosine     final loss:", loss_sched(n_train + 1));

# === Loss curves overlay ===
figure()
steps = 0:n_train;
plot(steps, loss_high,  "color", "red",   "label", "constant high LR")
hold("on")
plot(steps, loss_low,   "color", "blue",  "label", "constant low LR")
plot(steps, loss_sched, "color", "green", "label", "warmup + cosine")
hold("off")
title("Loss vs step under three LR strategies (with noise sigma=0.3)")
xlabel("step")
ylabel("L")
legend("high", "low", "warmup+cosine")
savefig("schedule_vs_constant.svg")
print("Saved schedule_vs_constant.svg");

# === Companion: schedule curve ===
figure()
plot(steps, sched_curve, "color", "green", "label", "warmup+cosine")
title("eta(t) used in the scheduled run")
xlabel("step")
ylabel("eta")
savefig("schedule_curve.svg")
print("Saved schedule_curve.svg");
