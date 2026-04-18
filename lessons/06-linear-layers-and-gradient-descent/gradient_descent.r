# Script:  gradient_descent.r
# System:  Gradient descent on MSE loss for  y = w*x + b
# Concept: Iterative parameter update: w ← w - η * ∂L/∂w;  b ← b - η * ∂L/∂b
# Equations: ∂L/∂w = (2/N)*sum((w*x+b-y)*x);  ∂L/∂b = (2/N)*sum(w*x+b-y)
# Units:     w,b dimensionless; L in squared y-units; η (learning rate) dimensionless

# === Dataset ===
# y = 2x  (true relationship: w*=2, b*=0)
x = [1.0, 2.0, 3.0, 4.0];
y = [2.0, 4.0, 6.0, 8.0];
npts = 4.0;
lr = 0.05;
n_steps = 200;

# === Gradient descent loop ===
w = 0.0;
b = 0.0;

w_path    = zeros(n_steps + 1);
b_path    = zeros(n_steps + 1);
loss_path = zeros(n_steps + 1);

w_path(1)    = w;
b_path(1)    = b;
loss_path(1) = mean((w * x + b - y) .^ 2);

print("Starting gradient descent: w=0, b=0, lr=0.05");
print("True optimum: w*=2, b*=0");

for step = 1:n_steps
  pred     = w * x + b;
  residual = pred - y;
  dw = (2.0 / npts) * sum(residual .* x);
  db = (2.0 / npts) * sum(residual);
  w -= lr * dw;
  b -= lr * db;
  w_path(step + 1)    = w;
  b_path(step + 1)    = b;
  loss_path(step + 1) = mean((w * x + b - y) .^ 2);
end

print("After", n_steps, "steps:");
print("  w =", w, "  (true w* = 2)");
print("  b =", b, "  (true b* = 0)");
print("  Loss =", loss_path(n_steps + 1));

# === Loss curve ===
figure()
plot(loss_path, "color", "blue", "label", "MSE loss")
hold("on")
hline(0.0, "gray", "minimum")
title("Gradient Descent: Loss vs. Step")
xlabel("Step")
ylabel("MSE Loss")
legend()
savefig("outputs/loss_curve.svg")
print("Saved outputs/loss_curve.svg");

# === Gradient descent path in (w, b) space ===
figure()
scatter(w_path, b_path, "Gradient Descent Path in (w,b) Space — converges to (2, 0)")
savefig("outputs/gd_path.svg")
print("Saved outputs/gd_path.svg");
