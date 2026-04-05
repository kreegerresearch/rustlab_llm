# Script:  bigram_sampling.r
# System:  Text generation via CDF-based sampling from a bigram language model
# Concept: Multinomial sampling using cumulative distribution + uniform threshold
# Equations: CDF_j = sum_{k=1}^{j} P_ik;  x_{t+1} = sum(CDF < u) + 1,  u ~ Uniform(0,1)
# Units:     Probabilities in [0,1]; token indices are integers in {1,...,|V|}

# === Probability Matrix (from bigram_counts.r) ===
# Corpus: "abcbabcba"   Vocab: a=1, b=2, c=3
#        a    b    c
# a:  [ 0,   1,   0 ]   → a always goes to b
# b:  [ 0.5, 0,  0.5]   → b goes to a or c with equal probability
# c:  [ 0,   1,   0 ]   → c always goes to b
P = [0.0, 1.0, 0.0; 0.5, 0.0, 0.5; 0.0, 1.0, 0.0];

# === CDF-based Sampling ===
# For a probability row p, the CDF is cumsum(p).
# A uniform draw u ∈ [0,1) maps to token index: sum(CDF < u) + 1
# This counts how many CDF bins are strictly below u, then adds 1 for the 1-based index.

print("Sampling mechanism demonstration:");
p_b = P(2);
cdf_b = cumsum(p_b);
print("  P(b)   =", p_b);
print("  CDF(b) =", cdf_b);
print("  u=0.3 → sum(CDF < 0.3) + 1 =", sum(cdf_b < 0.3) + 1, "  (expect 1 = a)");
print("  u=0.7 → sum(CDF < 0.7) + 1 =", sum(cdf_b < 0.7) + 1, "  (expect 3 = c)");

# === Generate a sequence using a loop ===
n_generate = 12;
generated = zeros(n_generate);
generated(1) = 1;   # start with token a

# Pre-draw all uniform samples for reproducibility
draws = rand(n_generate - 1);

for t = 1:(n_generate - 1)
  curr = generated(t);
  cdf = cumsum(P(curr));
  generated(t + 1) = sum(cdf < draws(t)) + 1;
end

print("Uniform draws:", draws);
print("Generated sequence (indices):", generated);

# === Training loss on the original corpus ===
# Bigrams: (a,b)(b,c)(c,b)(b,a)(a,b)(b,c)(c,b)(b,a) — 4 pairs each of two types
# P(b|a)=1.0 → log-prob=0;  P(c|b)=P(a|b)=0.5 → log-prob=log(0.5)
# Mean CE = -(4*log(1.0) + 4*log(0.5)) / 8 = -0.5*log(0.5)
log_probs = [log(1.0), log(0.5), log(1.0), log(0.5), log(1.0), log(0.5), log(1.0), log(0.5)];
mean_ce = -mean(log_probs);
print("Mean cross-entropy loss on training corpus (nats):", mean_ce);
print("Perplexity = exp(loss):", exp(mean_ce));

# === Plot row distributions ===
figure()
subplot(3, 1, 1)
plot(P(1), "color", "blue", "label", "P(next | a)")
title("P(next | a) — deterministic: always b")
ylabel("Probability")
ylim([0, 1])

subplot(3, 1, 2)
plot(P(2), "color", "green", "label", "P(next | b)")
title("P(next | b) — equal: a or c")
ylabel("Probability")
ylim([0, 1])

subplot(3, 1, 3)
plot(P(3), "color", "red", "label", "P(next | c)")
title("P(next | c) — deterministic: always b")
xlabel("Token index (1=a, 2=b, 3=c)")
ylabel("Probability")
ylim([0, 1])

savefig("outputs/bigram_row_distributions.svg")
print("Saved outputs/bigram_row_distributions.svg");
