# Script:  lr_schedule.r
# System:  Warmup + cosine-decay learning-rate schedule.
# Concept: eta(t) is a piecewise function:
#            t in [0, T_w]   linear ramp from 0 to eta_max
#            t in (T_w, T_d] cosine decay from eta_max to eta_min
# Equations:
#   eta_warmup(t) = eta_max * t / T_w
#   eta_decay(t)  = eta_min + 0.5*(eta_max - eta_min)*(1 + cos(pi * (t - T_w)/(T_d - T_w)))
# Units:   eta is dimensionless; t is step count.

T_total  = 5000;
T_warmup = 500;
T_decay  = T_total;
eta_max  = 6.0e-4;
eta_min  = 6.0e-5;

eta_sched = zeros(T_total + 1);
for t = 0:T_total
  if t < T_warmup
    eta_sched(t + 1) = eta_max * (t / T_warmup);
  else
    progress = (t - T_warmup) / (T_decay - T_warmup);
    eta_sched(t + 1) = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * progress));
  end
end

# === Endpoint sanity ===
print("eta(0) (should be 0):", eta_sched(1));
print("eta(T_warmup) (should be eta_max):", eta_sched(T_warmup + 1));
print("eta(T_total)  (should be eta_min):", eta_sched(T_total + 1));

# === Print sampled schedule values ===
print("Schedule at selected steps:");
print("  t=    0:", eta_sched(1));
print("  t=  100:", eta_sched(101));
print("  t=  250:", eta_sched(251));
print("  t=  500:", eta_sched(501));
print("  t= 1000:", eta_sched(1001));
print("  t= 2500:", eta_sched(2501));
print("  t= 4000:", eta_sched(4001));
print("  t= 5000:", eta_sched(5001));

# === Plot ===
figure()
steps = 0:T_total;
plot(steps, eta_sched, "color", "blue", "label", "warmup + cosine")
title("LR schedule:  T_w=500, T_total=5000, eta_max=6e-4")
xlabel("step")
ylabel("eta")
savefig("lr_schedule.svg")
print("Saved lr_schedule.svg");
