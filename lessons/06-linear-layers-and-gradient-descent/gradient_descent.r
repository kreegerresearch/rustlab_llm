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

# === One gradient descent step ===
# Takes current (w, b); returns [w_new, b_new, loss_at_current_w_b]
function result = gd_step(w, b, x, y, lr, npts)
  pred     = w * x + b;
  residual = pred - y;
  loss     = mean(residual .^ 2);
  dw       = (2.0 / npts) * sum(residual .* x);
  db       = (2.0 / npts) * sum(residual);
  result   = [w - lr * dw,  b - lr * db,  loss];
end

# === Initial state ===
w0 = 0.0;
b0 = 0.0;
print("Starting gradient descent: w=0, b=0, lr=0.05");
print("True optimum: w*=2, b*=0");

# === 15 Unrolled Steps ===
# rustlab has no loop syntax — each step is written explicitly.
# s_k = [w_{k+1}, b_{k+1}, loss_at_step_k]
s0  = gd_step(w0,       b0,       x, y, lr, npts);
s1  = gd_step(s0(1),  s0(2),  x, y, lr, npts);
s2  = gd_step(s1(1),  s1(2),  x, y, lr, npts);
s3  = gd_step(s2(1),  s2(2),  x, y, lr, npts);
s4  = gd_step(s3(1),  s3(2),  x, y, lr, npts);
s5  = gd_step(s4(1),  s4(2),  x, y, lr, npts);
s6  = gd_step(s5(1),  s5(2),  x, y, lr, npts);
s7  = gd_step(s6(1),  s6(2),  x, y, lr, npts);
s8  = gd_step(s7(1),  s7(2),  x, y, lr, npts);
s9  = gd_step(s8(1),  s8(2),  x, y, lr, npts);
s10 = gd_step(s9(1),  s9(2),  x, y, lr, npts);
s11 = gd_step(s10(1), s10(2), x, y, lr, npts);
s12 = gd_step(s11(1), s11(2), x, y, lr, npts);
s13 = gd_step(s12(1), s12(2), x, y, lr, npts);
s14 = gd_step(s13(1), s13(2), x, y, lr, npts);
s15 = gd_step(s14(1), s14(2), x, y, lr, npts);

print("Step | w         | b         | Loss");
print("  0  | w=", w0,       "| b=", b0,       "| L=", s0(3));
print("  1  | w=", s0(1),    "| b=", s0(2),    "| L=", s1(3));
print("  2  | w=", s1(1),    "| b=", s1(2),    "| L=", s2(3));
print("  3  | w=", s2(1),    "| b=", s2(2),    "| L=", s3(3));
print("  4  | w=", s3(1),    "| b=", s3(2),    "| L=", s4(3));
print("  5  | w=", s4(1),    "| b=", s4(2),    "| L=", s5(3));
print("  6  | w=", s5(1),    "| b=", s5(2),    "| L=", s6(3));
print("  7  | w=", s6(1),    "| b=", s6(2),    "| L=", s7(3));
print("  8  | w=", s7(1),    "| b=", s7(2),    "| L=", s8(3));
print("  9  | w=", s8(1),    "| b=", s8(2),    "| L=", s9(3));
print(" 10  | w=", s9(1),    "| b=", s9(2),    "| L=", s10(3));
print(" 11  | w=", s10(1),   "| b=", s10(2),   "| L=", s11(3));
print(" 12  | w=", s11(1),   "| b=", s11(2),   "| L=", s12(3));
print(" 13  | w=", s12(1),   "| b=", s12(2),   "| L=", s13(3));
print(" 14  | w=", s13(1),   "| b=", s13(2),   "| L=", s14(3));
print(" 15  | w=", s14(1),   "| b=", s14(2),   "| L=", s15(3));

print("Final w (should approach 2):", s14(1));
print("Final b (should approach 0):", s14(2));

# === Collect the path for plotting ===
w_path    = [w0, s0(1), s1(1), s2(1), s3(1), s4(1), s5(1), s6(1), s7(1), s8(1), s9(1), s10(1), s11(1), s12(1), s13(1), s14(1)];
b_path    = [b0, s0(2), s1(2), s2(2), s3(2), s4(2), s5(2), s6(2), s7(2), s8(2), s9(2), s10(2), s11(2), s12(2), s13(2), s14(2)];
loss_path = [s0(3), s1(3), s2(3), s3(3), s4(3), s5(3), s6(3), s7(3), s8(3), s9(3), s10(3), s11(3), s12(3), s13(3), s14(3), s15(3)];

# === Loss curve ===
figure()
plot(loss_path, "color", "blue", "label", "MSE loss")
title("Gradient Descent: Loss vs. Step")
xlabel("Step")
ylabel("MSE Loss")
legend()
savefig("outputs/loss_curve.svg")
print("Saved outputs/loss_curve.svg");

# === Gradient descent path in (w, b) space ===
savescatter(w_path, b_path, "outputs/gd_path.svg", "Gradient Descent Path in (w,b) Space — converges to (2, 0)")
print("Saved outputs/gd_path.svg");
