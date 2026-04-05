# Script:  bigram_sampling.r
# System:  Text generation via CDF-based sampling from a bigram language model
# Concept: Multinomial sampling using cumulative distribution + uniform threshold
# Equations: CDF_j = sum_{k=1}^{j} P_ik;  x_{t+1} = min{ j : CDF_j >= u },  u ~ Uniform(0,1)
# Units:     Probabilities in [0,1]; token indices are integers in {1,...,|V|}

# === Probability Matrix (from bigram_counts.r) ===
# Corpus: "abcbabcba"   Vocab: a=1, b=2, c=3
#        a    b    c
# a:  [ 0,   1,   0 ]   → a always goes to b
# b:  [ 0.5, 0,  0.5]   → b goes to a or c with equal probability
# c:  [ 0,   1,   0 ]   → c always goes to b

P_a = [0.0, 1.0, 0.0];
P_b = [0.5, 0.0, 0.5];
P_c = [0.0, 1.0, 0.0];

# === CDF Vectors ===
# CDF_j = P(next token index <= j | current token)
# Used to map a uniform draw u ∈ [0,1) to a sampled token index.
CDF_a = cumsum(P_a);
CDF_b = cumsum(P_b);
CDF_c = cumsum(P_c);

print("CDFs for each starting token:");
print("  CDF(a):", CDF_a, "  → any u samples b (index 2)");
print("  CDF(b):", CDF_b, "  → u < 0.5 → a (index 1); u >= 0.5 → c (index 3)");
print("  CDF(c):", CDF_c, "  → any u samples b (index 2)");

# === Sampling demonstration: two specific uniform draws ===
# u=0.3 falls in the bin where CDF first exceeds 0.3
# u=0.7 falls in the bin where CDF first exceeds 0.7

print("Sampling from token b with u=0.3 (< 0.5):");
print("  CDF(b) = [0.5, 0.5, 1.0]");
print("  First index where CDF >= 0.3 is index 1 → sampled token: a");

print("Sampling from token b with u=0.7 (>= 0.5):");
print("  CDF(b) = [0.5, 0.5, 1.0]");
print("  First index where CDF >= 0.7 is index 3 → sampled token: c");

# === Traced 6-step sequence ===
# Starting token: a (index 1)
# Uniform draws chosen for illustration: u = [0.3, 0.7, 0.3, 0.7, 0.3, 0.7]
#
# Step 1: current=a, u=0.3 → CDF_a=[0,1,1], first >= 0.3 is index 2 → b
# Step 2: current=b, u=0.7 → CDF_b=[0.5,0.5,1.0], first >= 0.7 is index 3 → c
# Step 3: current=c, u=0.3 → CDF_c=[0,1,1], first >= 0.3 is index 2 → b
# Step 4: current=b, u=0.3 → CDF_b=[0.5,0.5,1.0], first >= 0.3 is index 1 → a
# Step 5: current=a, u=0.7 → CDF_a=[0,1,1], first >= 0.7 is index 2 → b
# Step 6: current=b, u=0.7 → first >= 0.7 is index 3 → c
#
# Generated sequence: a → b → c → b → a → b → c
# As indices:         1    2    3    2    1    2    3

seq = [1, 2, 3, 2, 1, 2, 3];
print("Generated sequence (token indices):", seq);
print("Generated sequence (tokens): a b c b a b c");

# === Training loss on the original corpus ===
# Corpus "abcbabcba" → bigrams (a,b)(b,c)(c,b)(b,a)(a,b)(b,c)(c,b)(b,a)
# Log-probabilities of each bigram under the trained model:
#   P(b|a) = 1.0  →  log(1.0) = 0
#   P(c|b) = 0.5  →  log(0.5) ≈ -0.693
#   P(b|c) = 1.0  →  log(1.0) = 0
#   P(a|b) = 0.5  →  log(0.5) ≈ -0.693

log_probs = [log(1.0), log(0.5), log(1.0), log(0.5), log(1.0), log(0.5), log(1.0), log(0.5)];
mean_ce = -mean(log_probs);

print("Mean cross-entropy loss on training corpus (nats):", mean_ce);
print("Expected: -0.5 * log(0.5) = 0.5 * 0.693 ≈ 0.347 nats");
print("Perplexity = exp(loss):", exp(mean_ce));
print("Perplexity interpretation: model is as uncertain as choosing from ~1.41 equally-likely tokens");

# === Bar chart: probability distributions from each starting token ===
figure()
subplot(3, 1, 1)
plot(P_a, "color", "blue", "label", "P(next | current=a)")
title("P(next token | a) — deterministic: always b")
ylabel("Probability")
ylim([0, 1])

subplot(3, 1, 2)
plot(P_b, "color", "green", "label", "P(next | current=b)")
title("P(next token | b) — equal: a or c")
ylabel("Probability")
ylim([0, 1])

subplot(3, 1, 3)
plot(P_c, "color", "red", "label", "P(next | current=c)")
title("P(next token | c) — deterministic: always b")
xlabel("Token index (1=a, 2=b, 3=c)")
ylabel("Probability")
ylim([0, 1])

savefig("outputs/bigram_row_distributions.svg")
print("Saved outputs/bigram_row_distributions.svg");
