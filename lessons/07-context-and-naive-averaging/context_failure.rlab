# Script:  context_failure.r
# System:  Bigram LM failure on ambiguous tokens that depend on prior context
# Concept: A Markov-1 model cannot disambiguate homonyms — its next-token
#          distribution depends only on the immediately previous token, so two
#          different histories that collapse to the same last token receive
#          identical predictions.
# Equations: P(x_{t+1} | x_t) = C[x_t, :] / sum(C[x_t, :])
# Units:     Probabilities in [0,1]; counts are non-negative integers

# === Tiny Corpus with an Ambiguous Token ===
# Sentence A: "river bank water"
# Sentence B: "money bank safe"
#
# The token 'bank' is ambiguous — its continuation depends on whether the
# preceding word was 'river' or 'money'. A bigram only sees 'bank'.
#
# Vocab:   river=1, bank=2, water=3, money=4, safe=5
vocab_size = 5;

# === Bigram Count Matrix ===
# C[i, j] = number of times token j immediately follows token i.
C = zeros(vocab_size, vocab_size);
C(1, 2) = 1;   # river -> bank     (sentence A)
C(2, 3) = 1;   # bank  -> water    (sentence A)
C(4, 2) = 1;   # money -> bank     (sentence B)
C(2, 5) = 1;   # bank  -> safe     (sentence B)

print("Bigram count matrix C (rows = prev token, cols = next token):");
print("   order: [river, bank, water, money, safe]");
print(C);

# === Row-Normalise to P(next | prev) ===
# Only rows with non-zero counts can be normalised; others stay all-zero.
P = zeros(vocab_size, vocab_size);
for i = 1:vocab_size
  row_sum = sum(C(i));
  if row_sum > 0
    for j = 1:vocab_size
      P(i, j) = C(i, j) / row_sum;
    end
  end
end

print("Probability matrix P (row-normalised):");
print(P);

# === The Failure: P(next | bank) is Identical in Both Contexts ===
# Whatever preceded 'bank', the model returns the same row.
p_after_bank = P(2);
print("P(next | bank) =", p_after_bank);
print("  P(water | bank) =", p_after_bank(3), "  (should be 0.5)");
print("  P(safe  | bank) =", p_after_bank(5), "  (should be 0.5)");
print("");
print("The bigram model gives the SAME distribution after 'bank'");
print("regardless of whether the history was 'river bank' or 'money bank'.");
print("A context-free model cannot disambiguate.");

# === Verify ===
row_sum_bank = sum(p_after_bank);
print("Row sum (should be 1):", row_sum_bank);

# === Plot the Ambiguous Row ===
figure()
labels = {"river", "bank", "water", "money", "safe"};
bar(labels, p_after_bank)
title("P(next | bank) — symmetric: water and safe are equally likely")
ylabel("Probability")
ylim([0, 1])
savefig("bank_distribution.svg")
print("Saved bank_distribution.svg");
