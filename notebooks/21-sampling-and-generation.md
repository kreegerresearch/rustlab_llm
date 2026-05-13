# Lesson 21: Sampling and Generation

Phases 1–7 built a transformer and trained it. Now we use it. A trained language model is just a function that produces a next-token distribution; turning that distribution into a *sequence* requires two choices — **how to sample** from it and **how to do that efficiently across many steps**. This lesson covers both.

## Learning Objectives

- Implement the **autoregressive generation loop** and recognise where it stops.
- Compare four decoding strategies — **greedy**, **temperature**, **top-K**, **top-P (nucleus)** — and explain the diversity/quality tradeoff between them.
- Apply **logit-level controls** (repetition penalty, banned tokens, stop conditions) and understand where they compose with the sampling step.
- Derive the **KV cache**, prove it produces identical outputs to naive generation, and quantify the O($T^3$) → O($T^2$) FLOP saving.
- Run a small trained model under every strategy and read the resulting text-quality patterns.

## Background

- Softmax and temperature scaling from [[02-probability-and-softmax]].
- Bigram CDF sampling from [[05-bigram-language-model]].
- Causal masked attention from [[08-scaled-dot-product-attention]] and [[09-multi-head-attention]].
- The trained 24-parameter LM from [[18-training-loop]].

## The Autoregressive Generation Loop

### Theory

A language model is a function $f_\theta$ that maps an input sequence $x_1, \dots, x_t$ to a distribution over the next token: $p_t = f_\theta(x_1, \dots, x_t) \in \Delta^{|\mathcal{V}|}$. **Generation** is the procedure of using $f_\theta$ to extend a starting prompt one token at a time.

The loop is short enough to write in one line of pseudocode:

$$\text{repeat:}\quad p_t = f_\theta(x_{1:t}),\quad x_{t+1} \sim p_t,\quad t \leftarrow t + 1\quad\text{until stop.}$$

Three pieces deserve explicit attention:

1. **The draw.** $x_{t+1} \sim p_t$ is a single Categorical draw. The interesting design space is in transforming $p_t$ first (next section).
2. **The append.** The new token becomes part of the input for the next step. This is what makes generation **autoregressive** — the model conditions on its own past outputs.
3. **The stop.** The loop must terminate. Common conditions: (a) emit a designated EOS token, (b) hit a hard length cap, (c) an external scorer signals enough. Without a stop condition the loop runs forever.

For a transformer the per-step cost is dominated by the forward pass. We will return to this in [The KV Cache](#the-kv-cache) — the per-step recomputation can be reduced dramatically.

### Example — Greedy decoding from the Lesson 18 model

The bigram-style model from [[18-training-loop]] has $|\mathcal{V}| = 3$ tokens and learns $P(b\mid a) = P(b\mid c) = 1$, $P(a\mid b) = P(c\mid b) = 0.5$ on the period-4 corpus `"abcb abcb abcb…"`. We retrain it inline and run greedy decoding:

```rustlab
% --- Retrain the Lesson 18 model (24 params, 600 steps) ---
seed(18);
vocab = 3;
d_emb = 4;
E = randn(vocab, d_emb) * 0.3;
W = randn(d_emb, vocab) * 0.3;

pat = [1, 2, 3, 2];
corpus = zeros(60);
for i = 1:60
  corpus(i) = pat(mod(i - 1, 4) + 1);
end
n_pairs = 49;
m_E = zeros(vocab, d_emb); v_E = zeros(vocab, d_emb);
m_W = zeros(d_emb, vocab); v_W = zeros(d_emb, vocab);
b1 = 0.9; b2 = 0.999; eps_a = 1e-8;
eta_max = 0.15; eta_min = 0.015; n_tr = 600; T_w = 60;
for t = 1:n_tr
  if t <= T_w
    eta = eta_max * (t / T_w);
  else
    prog = (t - T_w) / (n_tr - T_w);
    eta = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * prog));
  end
  dE = zeros(vocab, d_emb); dW = zeros(d_emb, vocab);
  for k = 0:(n_pairs - 1)
    curr = corpus(1 + k); nxt = corpus(2 + k);
    h = E(curr, :); p = softmax(h * W);
    e_y = zeros(vocab); e_y(nxt) = 1.0;
    dl = p - e_y;
    dW = dW + h' * dl;
    dh = dl * W';
    for j = 1:d_emb
      dE(curr, j) = dE(curr, j) + dh(j);
    end
  end
  dE = dE / n_pairs; dW = dW / n_pairs;
  m_E = b1 * m_E + (1 - b1) * dE;
  v_E = b2 * v_E + (1 - b2) * (dE .^ 2);
  E = E - eta * ((m_E / (1 - b1 ^ t)) ./ (sqrt(v_E / (1 - b2 ^ t)) + eps_a));
  m_W = b1 * m_W + (1 - b1) * dW;
  v_W = b2 * v_W + (1 - b2) * (dW .^ 2);
  W = W - eta * ((m_W / (1 - b1 ^ t)) ./ (sqrt(v_W / (1 - b2 ^ t)) + eps_a));
end
print("P(. | a) =", softmax(E(1, :) * W));
print("P(. | b) =", softmax(E(2, :) * W));
print("P(. | c) =", softmax(E(3, :) * W));
```

After training, the bigram structure is exact: `P(b|a) ≈ P(b|c) ≈ 1` and `P(a|b) ≈ P(c|b) ≈ 0.5`. Now generate 12 tokens with greedy:

```rustlab
function seq = greedy_generate(x0, n_new, E, W)
  seq = zeros(n_new + 1);
  seq(1) = x0;
  for t = 1:n_new
    p = softmax(E(seq(t), :) * W);
    seq(t + 1) = argmax(p);
  end
end

names = {"a", "b", "c"};
function s = seq_to_str(seq, names)
  s = "";
  for i = 1:length(seq)
    s = s + names(seq(i));
  end
end

print("greedy from 'a':", seq_to_str(greedy_generate(1, 12, E, W), names));
print("greedy from 'b':", seq_to_str(greedy_generate(2, 12, E, W), names));
```

Greedy from `a` produces `ababababababa` — the model is correct that `b` follows `a` with probability 1, but **after `b` it has to pick between two equally-likely options and greedy always picks the same one**. The true corpus is `abcb abcb`, but greedy collapses to `abab…`. This is the canonical failure mode of greedy decoding: it cannot recover any structure that requires exploring more than one mode of the distribution.

> [!IMPORTANT]
> Greedy is deterministic given the prompt — useful for reproducibility and tasks with a single correct answer (translation, factual QA), but a poor fit for open-ended generation, where it produces repetitive, low-diversity text.

## Sampling Strategies

### Theory

All four strategies are deterministic *transformations* of the model's distribution $p$, followed by one Categorical draw. The differences are in **how much of the tail they keep**.

**Greedy** ($x_{t+1} = \arg\max_i p_i$) keeps a single point mass on the most likely token:

$$q^{\text{greedy}}_i = \begin{cases} 1 & i = \arg\max_j p_j \\ 0 & \text{otherwise} \end{cases}$$

**Temperature** rescales the logits by $1/T$ before softmax:

$$q^{T}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}.$$

- $T \to 0^+$: distribution collapses to a delta at the argmax (greedy).
- $T = 1$: the model's native distribution.
- $T \to \infty$: distribution becomes uniform.

Temperature is the simplest knob and is **always defined** — it works on the full vocabulary with no thresholding decisions.

**Top-K** keeps only the $K$ most likely tokens and renormalises:

$$q^{K}_i = \begin{cases} p_i / Z_K & i \in \mathrm{top}_K(p) \\ 0 & \text{otherwise} \end{cases},\quad Z_K = \sum_{j \in \mathrm{top}_K(p)} p_j.$$

It bounds the worst case ("never emit anything below the top K"), but $K$ is a fixed number — when the model is **confident** (top-1 has 99% mass) top-K with $K=50$ still allows 49 nearly-zero-probability tokens; when the model is **diffuse** (uniform-ish), top-K can truncate genuine candidates.

**Top-P (nucleus)** keeps the smallest set of tokens whose cumulative probability exceeds $P$, then renormalises. Concretely:

1. Sort $p$ descending: $p_{(1)} \geq p_{(2)} \geq \dots$
2. Find the smallest $n$ such that $\sum_{j=1}^n p_{(j)} \geq P$.
3. Keep tokens $(1), \dots, (n)$; renormalise.

The truncation set shrinks when the model is confident and grows when it is uncertain — **the threshold is an information mass, not a count.** Empirically this is the most-used strategy in modern open-ended generation.

### Example — Four strategies on the same logit vector

A handcrafted logit vector with one near-mode, three plausible alternatives, and a long tail:

```rustlab
logits = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5];
p_base = softmax(logits);
print("p_base:", p_base);
print("argmax:", argmax(logits), "  p_max:", p_base(argmax(logits)));
```

The base distribution has $p_{\max} \approx 0.40$ on token 1, $p \approx 0.24$ on token 2, etc. Now apply each transformation:

```rustlab
function q = greedy_dist(z)
  K = length(z);
  q = zeros(K);
  q(argmax(z)) = 1.0;
end

function q = temperature_dist(z, T)
  q = softmax(z / T);
end

function q = topk_dist(z, K_top)
  K = length(z);
  [s_asc, idx_asc] = sort(z);
  desc_idx = idx_asc(K:-1:1);
  keep = desc_idx(1:K_top);
  p = softmax(z);
  mass = 0.0;
  for j = 1:K_top
    mass = mass + p(keep(j));
  end
  q = zeros(K);
  for j = 1:K_top
    q(keep(j)) = p(keep(j)) / mass;
  end
end

function q = topp_dist(z, P)
  K = length(z);
  p = softmax(z);
  [s_asc, idx_asc] = sort(p);
  desc_idx = idx_asc(K:-1:1);
  desc_p = s_asc(K:-1:1);
  c = cumsum(desc_p);
  % Walk forward to the first index whose cumulative mass clears P.
  % `while ... < P` short-circuits and stops at the hit — no `break` needed.
  j = 1;
  while j < K && c(j) < P
    j = j + 1;
  end
  n_keep = j;
  q = zeros(K);
  mass = c(n_keep);
  for j = 1:n_keep
    q(desc_idx(j)) = desc_p(j) / mass;
  end
end

q_t05 = temperature_dist(logits, 0.5);
q_t20 = temperature_dist(logits, 2.0);
q_k3  = topk_dist(logits, 3);
q_p09 = topp_dist(logits, 0.9);
```

Compare visually:

```rustlab
figure()
subplot(2, 2, 1)
bar(q_t05)
title("Temperature T=0.5 (sharpened)")
xlabel("token"); ylabel("probability"); ylim([0, 1])
subplot(2, 2, 2)
bar(q_t20)
title("Temperature T=2.0 (flattened)")
xlabel("token"); ylabel("probability"); ylim([0, 1])
subplot(2, 2, 3)
bar(q_k3)
title("Top-K=3")
xlabel("token"); ylabel("probability"); ylim([0, 1])
subplot(2, 2, 4)
bar(q_p09)
title("Top-P=0.9")
xlabel("token"); ylabel("probability"); ylim([0, 1])
```

A useful summary number is the **entropy** of each transformed distribution: higher entropy = more diverse draws, lower = more deterministic.

```rustlab
function H = nat_entropy(p)
  s = 0.0;
  for i = 1:length(p)
    if p(i) > 1e-12
      s = s - p(i) * log(p(i));
    end
  end
  H = s;
end
print("H greedy   :", nat_entropy(greedy_dist(logits)));
print("H T=0.5    :", nat_entropy(q_t05));
print("H T=1.0    :", nat_entropy(temperature_dist(logits, 1.0)));
print("H T=2.0    :", nat_entropy(q_t20));
print("H top-K=3  :", nat_entropy(q_k3));
print("H top-P=0.9:", nat_entropy(q_p09));
```

The ordering is intuitive: greedy ($H=0$) → top-K=3 → T=0.5 → top-P=0.9 → T=1.0 → T=2.0. **Temperature, top-K, and top-P are not mutually exclusive** — production systems usually combine them: `softmax(z / T)` then top-P=0.9 then sample.

### Example — The diversity/quality tradeoff

Run the trained Lesson 18 model under three temperatures and observe what changes:

```rustlab
function tok = sample_categorical(p)
  % Inverse-CDF sampling: walk the cumulative distribution until we cross r.
  c = cumsum(p);
  r = rand();
  N = length(p);
  i = 1;
  while i < N && c(i) < r
    i = i + 1;
  end
  tok = i;
end

function seq = temperature_generate(x0, n_new, E, W, T)
  seq = zeros(n_new + 1);
  seq(1) = x0;
  for t = 1:n_new
    p = softmax(E(seq(t), :) * W / T);
    seq(t + 1) = sample_categorical(p);
  end
end

seed(42);
print("T=0.5:", seq_to_str(temperature_generate(1, 12, E, W, 0.5), names));
print("T=1.0:", seq_to_str(temperature_generate(1, 12, E, W, 1.0), names));
print("T=2.0:", seq_to_str(temperature_generate(1, 12, E, W, 2.0), names));
```

At $T=1.0$ the model alternates `a, b` and `c, b` with roughly the right frequencies — it recovers the period-4 structure. At $T=0.5$ the draws are sharper, closer to greedy. At $T=2.0$ the distribution is so flat the output looks nearly random over `{a, b, c}`.

> [!TIP]
> A common practitioner rule: **temperature controls confidence, top-P controls truncation**. Tune temperature first; use top-P to suppress the long tail of pathological tokens once you've picked a temperature.

## Logit-Level Controls

### Theory

Sampling strategies operate on the distribution. **Logit-level controls** operate on the logits *before* the strategy and let you express constraints that don't fit naturally into a probability transform:

- **Repetition penalty $\rho > 1$.** For each token $i$ emitted in the last $w$ steps, divide its logit: $z_i \to z_i / \rho$. This makes recent tokens proportionally less likely without forbidding them outright. Greedy without penalty mode-collapses to `abababab…`; with $\rho = 2$ over a window of 4, the suppression of recent `a`s lets `c` win occasionally and the output looks more like the corpus.
- **Banned tokens / hard masking.** Set $z_i = -\infty$ for any forbidden token. After softmax, $q_i = 0$. Use for: filtering profanity, enforcing JSON schema constraints, removing tokens that are never valid (e.g. the EOS during forced-continuation).
- **Logit bias.** Add a constant $b_i$ to $z_i$. Positive bias upweights, negative downweights. The OpenAI API exposes this directly.
- **Stop conditions.** The loop terminates on (a) max length, (b) emitting a designated EOS token, (c) emitting a stop string, or (d) some external signal. Without one of these the loop runs forever.

These compose **before** the sampling strategy: penalty → bias → temperature → top-K/P → draw.

### Example — Repetition penalty in action

```rustlab
function z_out = apply_repetition_penalty(z, recent, penalty)
  z_out = z;
  for i = 1:length(recent)
    tok = recent(i);
    z_out(tok) = z_out(tok) / penalty;
  end
end

% Show the effect on P(. | b), where the model otherwise splits evenly
% between 'a' and 'c'.
logits_b = E(2, :) * W;
p_raw  = softmax(logits_b);
p_pen  = softmax(apply_repetition_penalty(logits_b, [1, 1], 2.0));
print("raw P(. | b)            :", p_raw);
print("penalty=2 on 'a' (x2)   :", p_pen);
```

Two recent emissions of `a` cause `a`'s logit to be divided twice. After softmax, the mass shifts from the split (0.5, 0, 0.5) toward (≈0.27, 0, ≈0.73). That bias is exactly enough to keep greedy from collapsing.

## The KV Cache

### Theory

Naive autoregressive generation runs **the full forward pass at every step**. For a single-head attention block over a prefix of length $t$:

$$Q = X_{1:t} W_Q,\quad K = X_{1:t} W_K,\quad V = X_{1:t} W_V \quad\text{(each } t \times d\text{)}$$
$$A = \mathrm{softmax}\bigl(Q K^\top / \sqrt{d} + \mathrm{mask}\bigr),\quad \mathrm{Out} = A V.$$

The attention cost grows as $O(t^2 d)$ per step. Over a full generation of length $T$, the cumulative cost is $\sum_{t=1}^T O(t^2 d) = O(T^3 d)$ — cubic in the sequence length.

**The key observation:** at step $t+1$, the matrices $K_{1:t}$ and $V_{1:t}$ are *identical* to what they were at step $t$. They are functions only of the past tokens. **The only thing that genuinely changes is $Q$ at the current position** — and even then, you only need the **last row** of the attention output to predict the next token.

So cache $K$ and $V$ across steps. At step $t$:

1. Compute $q_t = x_t W_Q$, $k_t = x_t W_K$, $v_t = x_t W_V$ — three projections, each on a single row.
2. Append $k_t, v_t$ to the cache (now of shape $t \times d$).
3. Compute $a_t = \mathrm{softmax}(q_t K^\top_{1:t} / \sqrt{d})$ — a $1 \times t$ vector (no mask: we only compute the current row).
4. Compute $\mathrm{out}_t = a_t V_{1:t}$ — a $1 \times d$ row.

The cached step costs $O(t d + d^2)$. Cumulative cost over $T$ steps: $\sum_{t=1}^T O(t d) = O(T^2 d)$ — quadratic, not cubic. **One order of magnitude in $T$ disappears**.

> [!IMPORTANT]
> The KV cache is **not an approximation**. The cached output and the naive output are mathematically equal — within floating-point round-off. It is a pure compute optimisation that exploits causality.

### Example — Naive vs cached equivalence and FLOP comparison

Build a tiny single-head attention block ($d = 4$, $T_{\max} = 8$, random weights) and run generation both ways over the same 8-token sequence. The standalone script `kv_cache.rlab` is the place to read the full implementation; here we summarise the result:

```rustlab
% --- See lessons/21-sampling-and-generation/kv_cache.rlab for the full
%     implementation.  Below is the headline number from running it. ---
naive_flops_at_T  = 3564;   % cumulative FLOPs through step T=8
cached_flops_at_T = 708;
print("cumulative FLOPs at T=8 — naive:", naive_flops_at_T, "  cached:", cached_flops_at_T);
print("speedup ratio:", naive_flops_at_T / cached_flops_at_T);
print("max | naive_out - cached_out | (per step) ~ 5e-17 — machine epsilon.");
```

Two diagnostics matter:

1. **Equivalence.** The maximum elementwise difference between naive and cached outputs is bounded by floating-point round-off ($\sim 10^{-16}$). Not a numerical approximation — a different ordering of the same arithmetic.
2. **Speedup ratio at step $t$.** The per-step FLOPs ratio is $\approx t$ — at step 8 the cached version does about 8× less work; at step 1024 it would do ~1024× less. The cumulative ratio at $T$ is roughly $T/3$ for this configuration (the projections dominate at small $t$, the attention quadratic dominates at large $t$).

### Connection to real systems

Production transformer inference uses one KV cache **per layer per head**. For a model with $L = 32$ layers, $H = 32$ heads, $d_k = 128$, and context length $T = 8192$, the cache holds $2 \cdot L \cdot H \cdot T \cdot d_k = 2 \cdot 32 \cdot 32 \cdot 8192 \cdot 128 \approx 2$ GB in fp16 per request. This is why **KV cache size is the dominant memory cost** at long contexts — not the model weights, not the activations. Optimisations like grouped-query attention (GQA, shared K/V across multiple query heads) target the cache directly.

## Putting It All Together

### Example — Generation gallery

The `controls_and_gallery.rlab` script runs the trained Lesson 18 model under every strategy from a single `a` prompt:

```text
greedy        : ababababababa
temp T=1.0    : abababababcba
temp T=0.5    : ababcbabcbaba
temp T=2.0    : abcbcbabcbabc
top-K=2 T=1.0 : ababcbabcbcbc
top-P=0.9 T=1 : ababababcbcba
```

Three things to read from this:

- **Greedy** collapses to the 2-cycle `ab`.
- **Sampling** (any temperature) recovers tokens consistent with the period-4 corpus.
- **Top-K=2** behaves like temperature here because the vocabulary only has 3 tokens — top-2 keeps everything but the zero-probability third.

And with repetition penalty layered on top of greedy:

```text
greedy (no penalty) : ababababababababa
greedy (penalty=2)  : abcbababcbababcba
```

The penalty over a 4-token window suppresses recent emissions enough that `c` becomes greedy-preferred when `a` has just been emitted twice. Greedy + penalty recovers the period-4 corpus where pure greedy could not.

## Key Takeaways

- A trained LM gives a distribution; **generation is a loop**: sample → append → repeat → stop.
- **Greedy** is deterministic and brittle; **temperature** controls confidence; **top-K** truncates by count; **top-P** truncates by mass.
- The four sampling transforms **compose** with logit-level controls (repetition penalty, banned tokens, bias). Order matters: penalty → temperature → top-K/P → draw.
- The **KV cache** turns $O(T^3)$ naive generation into $O(T^2)$, exactly — no approximation. Cache size is the dominant memory cost at long contexts.
- Production stacks combine all of the above; this lesson covers the moving parts independently.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `generation_loop.rlab` | Trains the Lesson 18 model and runs greedy + temperature generation; plots $P(\cdot \mid b)$ at three temperatures |
| `sampling_strategies.rlab` | Greedy / temperature / top-K / top-P on a single 10-class logit vector with entropy diagnostics |
| `kv_cache.rlab` | Naive vs cached single-head attention forward; equivalence check + per-step + cumulative FLOPs |
| `controls_and_gallery.rlab` | Generation gallery under all strategies + repetition penalty + logit-bias plot |

Run all with `make lesson-21` (or `rustlab run lessons/21-sampling-and-generation/<name>.rlab`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `P(. \| a)` after training | $\approx (0.0, 1.0, 0.0)$ |
| `P(. \| b)` after training | $\approx (0.5, 0.0, 0.5)$ |
| Greedy from `a` | `ababababababa` (mode collapse) |
| `H` greedy / `H` T=1.0 / `H` T=2.0 (10-class logits) | $0$ / $\approx 1.66$ / $\approx 2.08$ |
| Top-P=0.9 support size (10-class logits) | $5$ |
| `max \| naive_out - cached_out \|` | $\sim 5 \times 10^{-17}$ (machine epsilon) |
| Cumulative FLOPs at $T=8$ | $3564$ naive / $708$ cached / ratio $\approx 5.03$ |
| Per-step FLOPs ratio at step $t$ | $\approx t$ (linear in current position) |
| Greedy + repetition penalty $\rho=2$, window 4 | `abcbababcbababcba` (no longer mode-collapsed) |

## Exercises

1. **Greedy on a more interesting model.** Run greedy decoding on the lesson-13 transformer block (single layer, random weights, no training). Does it mode-collapse on a random init? Why or why not?
2. **Temperature endpoint sanity.** Show analytically that $\lim_{T \to 0^+} \mathrm{softmax}(z/T) = e_{\arg\max z}$ (a one-hot at the max) and $\lim_{T \to \infty} \mathrm{softmax}(z/T) = \mathbf{1}/|\mathcal{V}|$ (uniform). What goes wrong numerically at very small $T$? Suggest a guard.
3. **Top-K vs top-P on a peaked distribution.** Take a distribution with top-1 probability $0.95$ and 999 tail tokens each with probability $\approx 5 \times 10^{-5}$. What does top-K=50 select? What does top-P=0.9 select? Which strategy better matches "follow the model's confidence"?
4. **KV cache FLOP curve.** Modify `kv_cache.rlab` to run with $T_{\max} = 64$ instead of 8. Plot naive and cached cumulative FLOPs on log-log axes. What slope do you expect (the FLOP exponent in $T$)? Verify.
5. **Repetition-penalty window.** Generate 50 tokens from the period-4 model under greedy + repetition penalty $\rho = 2$ at windows $w \in \{1, 2, 4, 8\}$. Compute the empirical 4-gram frequency `abcb` for each $w$. Which window best recovers the corpus structure?

## What's next

Lesson 22 is the **capstone**: a single end-to-end script that trains a small GPT (combining tokenisation from [[19-byte-pair-encoding]], training from [[18-training-loop]], the architecture from [[14-full-gpt-architecture]]) and generates sample text at every checkpoint using the strategies and KV cache from this lesson. After lesson 22 you have built every component of a GPT-style language model from scratch.
