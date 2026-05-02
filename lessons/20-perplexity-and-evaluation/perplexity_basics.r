# Script:  perplexity_basics.r
# System:  Compute perplexity of three reference probability distributions.
# Concept: PPL = exp(H) where H = -sum(p * log(p)) is the entropy in nats.
#          A uniform distribution over K classes has PPL = K.  A peaked
#          distribution has PPL between 1 and K.  A perfectly-confident
#          distribution has PPL = 1.
# Equations:
#   H(p)   = -sum_i p_i log p_i      (nats)
#   PPL(p) = exp(H(p))
# Units:   probabilities dimensionless; H in nats; PPL dimensionless.

# === Three reference distributions ===
p_uniform = [0.25, 0.25, 0.25, 0.25];
p_peaked  = [0.70, 0.10, 0.10, 0.10];
p_sharp   = [0.97, 0.01, 0.01, 0.01];

function ppl = entropy_to_ppl(p)
  H = -sum(p .* log(p));
  ppl = exp(H);
end

function H = entropy(p)
  H = -sum(p .* log(p));
end

# === Print entropies and PPL values ===
H_u = entropy(p_uniform);
H_p = entropy(p_peaked);
H_s = entropy(p_sharp);

print("Distribution            H (nats)    PPL");
print("uniform 0.25/0.25/...   ", H_u, "    ", entropy_to_ppl(p_uniform));
print("peaked  0.70/0.10/...   ", H_p, "    ", entropy_to_ppl(p_peaked));
print("sharp   0.97/0.01/...   ", H_s, "    ", entropy_to_ppl(p_sharp));

# === Sanity checks ===
print("Uniform PPL should equal K = 4:", entropy_to_ppl(p_uniform));
print("Sharp distribution PPL should be near 1:", entropy_to_ppl(p_sharp));

# === Bridge from cross-entropy to PPL ===
# If a model assigns probability q_target to the correct class, the per-
# example cross-entropy is L = -log q_target and per-example PPL is 1/q_target.
% (This is just exp of a single log; PPL collapses to a reciprocal probability.)
q_targets = [0.5, 0.3, 0.1, 0.01];
print("Per-example PPL = 1 / q_target  (model probability on the correct class):");
for k = 1:length(q_targets)
  q = q_targets(k);
  L_q = -log(q);
  ppl_q = exp(L_q);
  print("  q =", q, "  L =", L_q, "  PPL =", ppl_q, "  (== 1/q =", 1/q, ")");
end

# === Plot: PPL as a function of (uniform-fraction-on-correct-class) ===
% A simple way to visualise PPL: plot exp(H) for distributions that put
% probability p on the correct class and (1-p)/(K-1) on each of K-1 other
% classes.  As p -> 1, PPL -> 1; as p -> 1/K, PPL -> K.
K = 4;
ps = linspace(1.0 / K, 0.999, 50);
ppl_curve = zeros(50);
for k = 1:50
  p = ps(k);
  rest = (1 - p) / (K - 1);
  q = [p, rest, rest, rest];
  ppl_curve(k) = exp(-sum(q .* log(q)));
end

figure()
plot(ps, ppl_curve, "color", "blue", "label", "PPL(p_correct)")
title("Perplexity as the model's confidence in the correct class grows")
xlabel("probability assigned to the correct class")
ylabel("PPL")
savefig("perplexity_basics.svg")
print("Saved perplexity_basics.svg");
