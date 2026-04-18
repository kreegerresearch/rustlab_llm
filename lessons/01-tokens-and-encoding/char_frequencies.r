# Script:  char_frequencies.r
# System:  Character-level tokenisation of a text corpus
# Concept: Vocabulary construction and relative character frequency distribution
# Equations: f_i = c_i / sum(c), where c_i is the count of character i
# Units:   Frequencies are dimensionless (probabilities summing to 1)

# === Corpus and Vocabulary ===
# Corpus: "to be or not to be" (18 characters including spaces)
# Vocabulary (sorted): ' '=1, 'b'=2, 'e'=3, 'n'=4, 'o'=5, 'r'=6, 't'=7
# Vocab size: 7 unique characters

vocab_size = 7;

# Raw character counts from the corpus "to be or not to be"
#   ' ' (space) : 5   b : 2   e : 2   n : 1   o : 4   r : 1   t : 3
counts = [5, 2, 2, 1, 4, 1, 3];

total = sum(counts);
print("Corpus: to be or not to be");
print("Total characters:", total);
print("Vocabulary size:", vocab_size);

# === Relative Frequencies ===
# f_i = c_i / sum(c)  —  each value is the probability of sampling that character
freqs = counts / total;

print("Character counts (space, b, e, n, o, r, t):");
print(counts);
print("Relative frequencies:");
print(freqs);
print("Sum of frequencies (should be 1.0):", sum(freqs));

# === Verification: most common character ===
max_freq = max(freqs);
print("Highest frequency:", max_freq);
# Expected: space appears 5/18 ≈ 0.2778 times

# === Bar Chart ===
# x-axis: character index (1=space, 2=b, 3=e, 4=n, 5=o, 6=r, 7=t)
# y-axis: relative frequency
figure()
bar(freqs, "Character Frequencies: 'to be or not to be'")
savefig("outputs/char_frequencies.svg")
print("Saved outputs/char_frequencies.svg")
