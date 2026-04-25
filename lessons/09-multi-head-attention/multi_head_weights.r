# Script:  multi_head_weights.r
# System:  Multi-head self-attention — four heads with different attention patterns
# Concept: Run H independent scaled dot-product attentions in parallel on the
#          same sequence.  Each head has its own Q_h, K_h (and V_h), so each
#          can specialise in a different kind of relationship: first-token,
#          previous-token, self, or uniform.
# Equations: A_h = softmax_row((Q_h K_h^T) / sqrt(d_k) + M)    for h = 1..H
# Units:     Real-valued reals; A entries in [0, 1]

# === Dimensions ===
T = 6;         # sequence length
H = 4;         # number of heads
d_k = 2;       # per-head key/query dimension
scale = 1.0 / sqrt(d_k);

# Large-negative mask used by all heads (shared causal constraint).
NEG_INF = -1.0e9;
M = zeros(T, T);
for i = 1:T
  for j = (i + 1):T
    M(i, j) = NEG_INF;
  end
end

# === Hand-set Q, K for each head ===
# Each head encodes a different pattern.  In a real model Q_h = X*W_Q^h with
# learned W_Q^h — the hand-set values below mimic what learning would discover.

# --- Head 1: "attend to the first token" --------------------------------
# K row 1 alone carries a strong dim-1 signal; every query asks for dim 1.
K1 = zeros(T, d_k);
K1(1, 1) = 3.0;

Q1 = zeros(T, d_k);
for t = 1:T
  Q1(t, 1) = 1.0;
end

# --- Head 2: "attend to the previous token" -----------------------------
# Encode position i on the unit circle: K_i = 3 * [cos(2πi/T), sin(2πi/T)].
# Query at t asks for position t-1:     Q_t = 3 * [cos(2π(t-1)/T), sin(2π(t-1)/T)].
# Dot product is maximised when i = t-1.
K2 = zeros(T, d_k);
Q2 = zeros(T, d_k);
for i = 1:T
  K2(i, 1) = 3.0 * cos(2.0 * pi * i / T);
  K2(i, 2) = 3.0 * sin(2.0 * pi * i / T);
end
for t = 1:T
  Q2(t, 1) = 3.0 * cos(2.0 * pi * (t - 1) / T);
  Q2(t, 2) = 3.0 * sin(2.0 * pi * (t - 1) / T);
end

# --- Head 3: "attend to self" -------------------------------------------
# Same position encoding for K, and Q_t equals K_t — dot product max at i = t.
K3 = K2;
Q3 = K2;

# --- Head 4: "uniform over past" (the Lesson 07 default) ----------------
# All scores zero → softmax gives uniform weights across the causal triangle.
Q4 = zeros(T, d_k);
K4 = zeros(T, d_k);

# === Compute attention weights for each head ===
# Helper: we repeat the same 5-step recipe per head.
# (A function would be cleaner, but rustlab functions need a matrix return.)

function A = causal_attention_weights(Q, K, scale, M, T)
  S = Q * K' * scale;
  S_masked = S + M;
  A = zeros(T, T);
  for t = 1:T
    row = softmax(S_masked(t));
    for j = 1:T
      A(t, j) = row(j);
    end
  end
end

A1 = causal_attention_weights(Q1, K1, scale, M, T);
A2 = causal_attention_weights(Q2, K2, scale, M, T);
A3 = causal_attention_weights(Q3, K3, scale, M, T);
A4 = causal_attention_weights(Q4, K4, scale, M, T);

print("Head 1 — 'first-token' attention:");
print(A1);
print("Head 2 — 'previous-token' attention:");
print(A2);
print("Head 3 — 'self' attention:");
print(A3);
print("Head 4 — 'uniform-over-past' attention:");
print(A4);

# === Sanity checks ===
# Each row sums to 1; upper triangle is ~0.
for h = 1:H
  print("--- Head", h, "---");
  if h == 1
    A = A1;
  elseif h == 2
    A = A2;
  elseif h == 3
    A = A3;
  else
    A = A4;
  end
  row_sums = zeros(T);
  for t = 1:T
    row_sums(t) = sum(A(t));
  end
  print("  Row sums (each should be 1):", row_sums);
end

# === Verify pattern: Head 1 — every row concentrates on column 1 ===
# For rows 2..T, A1(t, 1) should be the largest entry in row t.
max_col_matches = 0;
for t = 2:T
  row = A1(t);
  if argmax(row) == 1
    max_col_matches = max_col_matches + 1;
  end
end
print("Head 1: rows whose max is at column 1 (should be", T - 1, "):", max_col_matches);

# === Verify pattern: Head 3 — row t (t > 1) concentrates on column t (self) ===
# For row t the strongest causal attendee should be t itself.
self_matches = 0;
for t = 2:T
  if argmax(A3(t)) == t
    self_matches = self_matches + 1;
  end
end
print("Head 3: rows whose max is at column t (self) (should be", T - 1, "):", self_matches);

# === Verify pattern: Head 4 — rows are uniform over their causal support ===
# Row t has t equal entries of 1/t.  Check row 4: should be [0.25, 0.25, 0.25, 0.25, 0, 0].
print("Head 4: row 4 (expect [0.25, 0.25, 0.25, 0.25, 0, 0]):", A4(4));

# === Plot 4 heads side-by-side ===
figure()
subplot(2, 2, 1)
imagesc(A1, "viridis")
title("Head 1 — first token")

subplot(2, 2, 2)
imagesc(A2, "viridis")
title("Head 2 — previous token")

subplot(2, 2, 3)
imagesc(A3, "viridis")
title("Head 3 — self")

subplot(2, 2, 4)
imagesc(A4, "viridis")
title("Head 4 — uniform")
savefig("multi_head_attention.svg")
print("Saved multi_head_attention.svg");
