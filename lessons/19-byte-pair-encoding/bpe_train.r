# Script:  bpe_train.r
# System:  Byte-pair encoding on a small character corpus.
# Concept: One BPE step counts every adjacent pair, takes the argmax, and
#          rewrites the sequence by replacing each occurrence of that pair
#          with a fresh integer id.  Iterate to grow the vocab one token
#          per step; record the merge order.
# Equations:
#   counts[i, j] = #{ t : seq_t = i and seq_{t+1} = j }
#   (i*, j*) = argmax_{i, j} counts[i, j]
#   new_id   = vocab_size + 1
#   replace every (i*, j*) in seq with new_id; vocab_size += 1
# Units:   integer token ids; counts dimensionless.

# === Corpus: "abracadabra abracadabra abracadabra" ===
# Initial vocab:  a=1  b=2  c=3  d=4  r=5  ' '=6
seq = [1, 2, 5, 1, 3, 1, 4, 1, 2, 5, 1, ...
       6, ...
       1, 2, 5, 1, 3, 1, 4, 1, 2, 5, 1, ...
       6, ...
       1, 2, 5, 1, 3, 1, 4, 1, 2, 5, 1];
vocab_size = 6;

print("Initial corpus length:", length(seq));
print("Initial vocab size   :", vocab_size);

# === One BPE step (returns struct with new seq, merge pair, count, vocab) ===
function r = bpe_step(seq, vocab_size)
  V = vocab_size;
  L = length(seq);
  counts = zeros(V, V);
  for i = 1:(L - 1)
    a = seq(i); b = seq(i + 1);
    counts(a, b) = counts(a, b) + 1;
  end
  flat = reshape(counts, 1, V * V);
  idx  = argmax(flat);
  best_a = mod(idx - 1, V) + 1;        % column-major decode (matches reshape)
  best_b = floor((idx - 1) / V) + 1;
  best_c = counts(best_a, best_b);

  new_id = V + 1;
  new_seq = zeros(L);
  k = 1;
  i = 1;
  while i <= L
    matched = 0;
    if i < L
      if seq(i) == best_a
        if seq(i + 1) == best_b
          matched = 1;
        end
      end
    end
    if matched == 1
      new_seq(k) = new_id;
      k = k + 1;
      i = i + 2;
    else
      new_seq(k) = seq(i);
      k = k + 1;
      i = i + 1;
    end
  end
  trimmed = new_seq(1:(k - 1));
  r = struct("seq", trimmed, "a", best_a, "b", best_b, "count", best_c, "vocab", new_id);
end

# === Run 5 merges, printing each ===
n_merges = 5;
state = struct("seq", seq, "vocab", vocab_size);
merge_a = zeros(n_merges);
merge_b = zeros(n_merges);
merge_count = zeros(n_merges);

for m = 1:n_merges
  step = bpe_step(state.seq, state.vocab);
  merge_a(m) = step.a;
  merge_b(m) = step.b;
  merge_count(m) = step.count;
  print("Merge", m, ":  pair (", step.a, ",", step.b, ")  count =", step.count, ...
        "  new_id =", step.vocab, "  seq len =", length(step.seq));
  state.seq = step.seq;
  state.vocab = step.vocab;
end

print("Final corpus length:", length(state.seq));
print("Final vocab size   :", state.vocab);
print("Compression ratio  :", length(seq) / length(state.seq));

# === Bar chart of merge frequencies ===
labels = {"merge 1", "merge 2", "merge 3", "merge 4", "merge 5"};
figure()
bar(labels, merge_count)
title("Pair count chosen by each BPE merge step")
xlabel("merge step")
ylabel("count of merged pair")
savefig("bpe_merge_counts.svg")
print("Saved bpe_merge_counts.svg");
