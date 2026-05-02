# Script:  gpt_forward.r
# System:  Full GPT decoder forward pass on a toy config
# Concept: Wire token embedding + sinusoidal PE + N transformer blocks +
#          final LayerNorm + LM head into one end-to-end pipeline; print
#          the output shape and a sample next-token distribution.
# Equations:
#   H^{(0)} = E[ids] + PE
#   H^{(l)} = Block_l(H^{(l-1)}),   l = 1..N
#   logits  = LN_f(H^{(N)}) * W_U
#   probs   = softmax(logits(t))    per position t
# Units:   token features dimensionless; logits in pre-softmax units; probs in [0, 1]

# === Hyperparameters ===
seed(14);
vocab    = 50;
T        = 8;
d_model  = 64;
H_heads  = 4;
d_k      = d_model / H_heads;
d_ff     = 4 * d_model;
N_blocks = 4;
NEG_INF  = -1.0e9;
scale    = 1.0 / sqrt(d_k);

# === Token embedding ===
E_tok = randn(vocab, d_model) * 0.1;

# === Sinusoidal positional encoding ===
PE = zeros(T, d_model);
for t = 1:T
  for i = 1:d_model
    pair_idx = floor((i - 1) / 2);
    div = 10000.0 ^ (2.0 * pair_idx / d_model);
    angle = t / div;
    if mod(i - 1, 2) == 0
      PE(t, i) = sin(angle);
    else
      PE(t, i) = cos(angle);
    end
  end
end

# === Causal mask shared across blocks ===
M_mask = zeros(T, T);
for i = 1:T
  for j = (i + 1):T
    M_mask(i, j) = NEG_INF;
  end
end

# === One Pre-LN transformer block ===
function H_out = block_fwd(H_in, W_Q, W_K, W_V, W_O, W_ff1, W_ff2, T, d_model, H_heads, d_k, d_ff, scale, M_mask)
  H_norm1 = zeros(T, d_model);
  for t = 1:T
    H_norm1(t) = layernorm(H_in(t));
  end
  Q = H_norm1 * W_Q;
  K = H_norm1 * W_K;
  V = H_norm1 * W_V;

  out_concat = zeros(T, d_model);
  for h = 1:H_heads
    c_lo = (h - 1) * d_k + 1;
    Q_h = zeros(T, d_k);
    K_h = zeros(T, d_k);
    V_h = zeros(T, d_k);
    for t = 1:T
      for k = 1:d_k
        Q_h(t, k) = Q(t, c_lo + k - 1);
        K_h(t, k) = K(t, c_lo + k - 1);
        V_h(t, k) = V(t, c_lo + k - 1);
      end
    end
    S = Q_h * K_h' * scale + M_mask;
    A_h = zeros(T, T);
    for t = 1:T
      A_h(t) = softmax(S(t));
    end
    O_h = A_h * V_h;
    for t = 1:T
      for k = 1:d_k
        out_concat(t, c_lo + k - 1) = O_h(t, k);
      end
    end
  end
  H_mid = H_in + out_concat * W_O;

  H_norm2 = zeros(T, d_model);
  for t = 1:T
    H_norm2(t) = layernorm(H_mid(t));
  end
  H_out = H_mid + gelu(H_norm2 * W_ff1) * W_ff2;
end

# === Forward pass ===
ids = [7, 13, 7, 21, 13, 5, 9, 7];
print("Token ids:", ids);

# Embed + PE
H = zeros(T, d_model);
for t = 1:T
  H(t) = E_tok(ids(t)) + PE(t);
end
print("H^{(0)} shape:", size(H));

# Stack of N blocks (independently initialised)
for ell = 1:N_blocks
  W_Q_l = randn(d_model, d_model) * (1.0 / sqrt(d_model));
  W_K_l = randn(d_model, d_model) * (1.0 / sqrt(d_model));
  W_V_l = randn(d_model, d_model) * (1.0 / sqrt(d_model));
  W_O_l = randn(d_model, d_model) * (1.0 / sqrt(d_model));
  W_f1_l = randn(d_model, d_ff)    * sqrt(2.0 / d_model);
  W_f2_l = randn(d_ff,    d_model) * sqrt(2.0 / d_ff);
  H = block_fwd(H, W_Q_l, W_K_l, W_V_l, W_O_l, W_f1_l, W_f2_l, T, d_model, H_heads, d_k, d_ff, scale, M_mask);
end
print("H^{(N)} shape after", N_blocks, "blocks:", size(H));

# Final LN
H_f = zeros(T, d_model);
for t = 1:T
  H_f(t) = layernorm(H(t));
end

# LM head
W_U = randn(d_model, vocab) * (1.0 / sqrt(d_model));
logits = H_f * W_U;
print("logits shape:", size(logits));

# Sample distribution at the last position
probs_last = softmax(logits(T));
print("sum(probs_last):", sum(probs_last));
print("argmax next token id:", argmax(probs_last));
print("max prob value:      ", max(probs_last));

# Heatmap of logits across T positions
figure()
imagesc(logits, "viridis")
title("Logits (T=8, |V|=50)")
xlabel("vocab id")
ylabel("position t")
savefig("gpt_forward.svg")
