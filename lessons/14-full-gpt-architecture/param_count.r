# Script:  param_count.r
# System:  Full GPT decoder
# Concept: Parameter count by component for the toy config and for two
#          published GPT-2 sizes; bar chart of the toy config breakdown;
#          confirm the formula reproduces GPT-2 small's 124M parameters.
# Equations:
#   n ≈ N * 12 * d_model^2  +  2 * |V| * d_model + O(d_model)
#   per-block:  4 * d_model^2  (attention)  +  8 * d_model^2 (FFN)  +  small bias/LN terms
# Units:   integer parameter counts

function n = block_params(d_model, d_ff)
  n_attn = 4 * d_model * d_model;
  n_ffn  = 2 * d_model * d_ff;
  n_bias = d_ff + d_model;          % FFN biases
  n_ln   = 4 * d_model;             % two LN affines (γ, β each)
  n = n_attn + n_ffn + n_bias + n_ln;
end

function n = gpt_params(vocab, d_model, N_blocks, d_ff, T_max, tied)
  n_embed = vocab * d_model;
  n_pe    = T_max * d_model;        % learned PE; pass T_max=0 for sinusoidal (free)
  n_blks  = N_blocks * block_params(d_model, d_ff);
  n_ln_f  = 2 * d_model;
  if tied == 1
    n_lm  = 0;
  else
    n_lm  = d_model * vocab;
  end
  n = n_embed + n_pe + n_blks + n_ln_f + n_lm;
end

# === Toy config used in lesson 14 ===
vocab_toy = 50; d_toy = 64; N_toy = 4; d_ff_toy = 4 * d_toy;

n_embed_toy   = vocab_toy * d_toy;
n_per_block_t = block_params(d_toy, d_ff_toy);
n_blocks_toy  = N_toy * n_per_block_t;
n_ln_f_toy    = 2 * d_toy;
n_lm_toy      = d_toy * vocab_toy;
n_total_toy   = n_embed_toy + n_blocks_toy + n_ln_f_toy + n_lm_toy;

print("==== Toy config (V=50, d=64, N=4, H=4) ====");
print("Token embedding         :", n_embed_toy);
print("Per block               :", n_per_block_t);
print("All", N_toy, "blocks               :", n_blocks_toy);
print("Final LayerNorm         :", n_ln_f_toy);
print("LM head (untied)        :", n_lm_toy);
print("TOTAL (untied)          :", n_total_toy);
print("TOTAL (weight-tied)     :", n_total_toy - n_lm_toy);

# === GPT-2 small (Radford et al. 2019; published 124,439,808 params) ===
# GPT-2 uses learned PE with T_max=1024 and a weight-tied LM head.
print("");
print("==== GPT-2 small (V=50257, d=768, N=12, learned PE T_max=1024) ====");
n_gpt2_small_tied = gpt_params(50257, 768, 12, 4 * 768, 1024, 1);
print("Weight-tied total       :", n_gpt2_small_tied);
print("Reference (paper)       : 124,439,808");
print("Discrepancy             :", 124439808 - n_gpt2_small_tied);

# === GPT-2 medium (V=50257, d=1024, N=24, T_max=1024) — published ~355M ===
print("");
print("==== GPT-2 medium (V=50257, d=1024, N=24) ====");
n_gpt2_med_tied = gpt_params(50257, 1024, 24, 4 * 1024, 1024, 1);
print("Weight-tied total       :", n_gpt2_med_tied);
print("Reference (paper)       : 354,823,168");
print("Discrepancy             :", 354823168 - n_gpt2_med_tied);

# === Bar chart of toy-config breakdown ===
labels = {"embed", "blocks", "LN_f", "LM head"};
counts = [n_embed_toy, n_blocks_toy, n_ln_f_toy, n_lm_toy];
figure()
bar(labels, counts)
title("Toy GPT parameter count by component")
ylabel("parameters")
savefig("param_count.svg")
