# Script:  gelu_vs_relu.r
# System:  Activation functions inside the FFN sublayer
# Concept: Compare ReLU and GELU shapes and finite-difference derivatives near zero
# Equations:
#   ReLU(x) = max(0, x)
#   GELU(x) = x * Phi(x), Phi the standard-normal CDF
# Units:   x and y both dimensionless

# === Sample both activations on a wide range ===
xs = linspace(-3.0, 3.0, 200);
y_relu = relu(xs);
y_gelu = gelu(xs);

# === Plot both on the same axes ===
figure()
hold("on")
plot(xs, y_relu, "color", "blue", "label", "ReLU(x)")
plot(xs, y_gelu, "color", "red",  "label", "GELU(x)")
hline(0.0, "gray", "y=0")
title("ReLU vs GELU")
xlabel("x")
ylabel("activation")
legend()
hold("off")
savefig("gelu_vs_relu.svg")

# === Numerical derivative around zero ===
h_step = 1e-4;
xs_grad = -2:0.05:2;
n_grad = length(xs_grad);

dRelu = zeros(n_grad);
dGelu = zeros(n_grad);
for i = 1:n_grad
  x = xs_grad(i);
  dRelu(i) = (relu(x + h_step) - relu(x - h_step)) / (2.0 * h_step);
  dGelu(i) = (gelu(x + h_step) - gelu(x - h_step)) / (2.0 * h_step);
end

i_neg = round((-0.5 - xs_grad(1)) / 0.05) + 1;
print("d/dx ReLU(x = -0.5) =", dRelu(i_neg));
print("d/dx GELU(x = -0.5) =", dGelu(i_neg));
print("d/dx GELU(x =  0.0) =", dGelu(round((0.0 - xs_grad(1)) / 0.05) + 1));
print("d/dx GELU(x =  1.0) =", dGelu(round((1.0 - xs_grad(1)) / 0.05) + 1));
