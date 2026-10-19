# Train your own Sherlock GPT

A standalone tutorial that trains, samples from, fine-tunes and distils a
character-level GPT on *The Adventures of Sherlock Holmes*, using only
rustlab and the pieces derived in the curriculum: the Lesson 14
architecture, the Lesson 15/22 backward pass, the Lesson 16–18 optimiser
and training loop, the Lesson 20 evaluation and the Lesson 21 sampler —
taken off the toy corpora and pointed at a real book.

Nothing here is a black box. The model is `lib/gpt.rlab` (≈ 460 lines,
every gradient analytical and finite-difference checked); the four stage
scripts in this directory are each short enough to read in one sitting.

## What you will build

Four stages, one script each, all driven by the values in `config.rlab`:

| Stage | Script | What it does | Writes |
|---|---|---|---|
| 0 | `prepare_corpus.py` | Text → integer ids + vocabulary (one-time, Python) | `data/ids.csv`, `data/vocab.rlab` |
| 1 | `train.rlab` | Pre-train the GPT: AdamW, warmup + cosine, global-norm clipping, held-out evaluation, checkpoints | `checkpoints/ckpt_latest.npz`, `train_curves.svg` |
| 2 | `generate.rlab` | Continue a prompt under greedy, temperature and top-k decoding | printed text |
| 3 | `sft.rlab` | Supervised fine-tuning on 46 hand-written Q/A pairs (`qa_pairs.rlab`) with prompt masking, replay of corpus windows, and a forgetting probe | `checkpoints/ckpt_sft.npz`, `sft_curves.svg` |
| 4 | `distill.rlab` | Train a 2-layer, $d = 64$ student on the teacher's softened logits, against a matched hard-target control | `checkpoints/ckpt_student.npz`, `distill_curves.svg` |
| — | `ask.rlab` | Ask the fine-tuned model one question from the command line, formatted exactly as SFT trained it, decoded greedily to the newline | a printed answer |
| — | `benchmark.rlab` | Model-size sweep, cost breakdown, `parmap` scaling and memory footprint of `lib/gpt.rlab` (random ids; no corpus needed) | `benchmark.svg` and the numbers behind [`docs/sherlock-performance-and-rustlab-requests.md`](../../docs/sherlock-performance-and-rustlab-requests.md) |

**What to expect, honestly.** The default model has 811,008 parameters
and sees the book about five and a half times in twenty minutes, ending
at 2.2 bits per character. At that point it has learned the Gutenberg
line length, capitalisation after a full stop, the short function words
(`and`, `with`, `his`, `were`, `you`, `my`) and the sound of English —
longer words are pronounceable inventions, and the names are not there
yet. This is the sample the measured run printed at step 3,000
($\tau = 0.8$, top-k 20):

```
Sherlock Holmes ears dushed present coult and
the stirned his bullogrusly benitoransed that with see you hains
were wear strared with hile bout as and whitteen with his lone of my dan
whitat ha wa ha deveate, and loy boven eveed in the cesinible befort,
whithists. Came wa gity tow a bovert the todrnised. Thell. Awa
```

Spelling, names and dialogue layout arrive between about 1.9 and 1.6
bits/char, which takes several times the default number of steps (the
held-out loss was still falling when the run stopped) and, ideally, more
text — see *Scaling up*. After SFT the model answers the questions it
was fine-tuned on, nearly verbatim; ask the same question in different
words and it drifts back into this kind of text. It does not reason, and
no amount of sampling cleverness will make it. What the tutorial shows is
the *pipeline* — the same pre-train → fine-tune → distil sequence that,
at $10^5 \times$ the parameters and $10^6 \times$ the data, produces a
chat assistant — end to end, at a scale where you can print every
intermediate.

## Prerequisites

- **rustlab ≥ 0.3.9** (`rustlab --version`). The scripts rely on
  `parmap`, multi-output functions, struct values, string arrays and
  NPZ `save` / `load`.
- **Python 3** for Stage 0 only. rustlab cannot read a text file or map a
  character to its code, so the corpus is converted once, outside
  rustlab, by a standard-library-only script. Nothing else uses Python.
- A multi-core machine. Training is data-parallel over the batch with
  `parmap` (default: 8 windows → 8 worker threads); the numbers below are
  from a 14-core Apple-silicon laptop.
- About 100 MB of disk for checkpoints at the default settings.

## Quick start

From the repository root, one command runs everything:

```sh
make sherlock-quick      # 3-5 minutes: a small model through every stage, then report.md
make sherlock            # the default model: about 30 minutes for all stages
```

Either one checks the toolchain (`sherlock-doctor`), prepares the data,
pre-trains, samples, fine-tunes, distils, and writes
`tutorial/sherlock/report.md` from the stage logs and figures. Afterwards:

```sh
make sherlock-ask Q="Who is the woman?"   # put a question to the fine-tuned model
make sherlock-status                      # progress report of a running or finished run
```

The stages are also separate targets, in this order:

```sh
make sherlock-doctor     # rustlab >= 0.3.9, python3, corpus, prepared data, preset, cores
make sherlock-data       # Stage 0: about a second. Writes data/ids.csv and data/vocab.rlab
make sherlock-train      # Stage 1: about 20 minutes at the defaults (3,000 steps, 3.1 M tokens)
make sherlock-generate   # Stage 2: three samples from checkpoints/ckpt_latest.npz
make sherlock-sft        # Stage 3: fine-tune on the Q/A pairs and probe forgetting
make sherlock-distill    # Stage 4: distil the trained model into a half-width, half-depth student
make sherlock-report     # assemble report.md from whichever stages have run
make sherlock-bench      # throughput at several model sizes (optional, a few minutes)
```

`make sherlock-quick` selects the 2–3 minute preset (`presets/quick.rlab`,
which trains into `checkpoints-quick/` so it never overwrites a long run)
and then runs `make sherlock`; `make sherlock-preset PRESET=default`
switches back — see *Presets* below.

Each stage target is
`cd tutorial/sherlock && rustlab run <script> --plot none 2>&1 | tee <stage>.log`
(the data target runs `python3 prepare_corpus.py data/adventures.txt`).
`--plot none` only suppresses the interactive plot window; `savefig`
still writes the figures. The `<stage>.log` files are what
`make sherlock-status` and `make sherlock-report` read, so a run can be
inspected or written up long after the terminal has scrolled away. Two
housekeeping targets: `make sherlock-test` runs the library self-test
(`test_gpt_lib.rlab`: flatten / unflatten round trip, finite-difference
gradient check on every parameter block, and a timing probe at the
default size), and `make sherlock-clean` deletes checkpoints, figures,
logs and `report.md` and resets the preset and the question (it keeps
`data/ids.csv`; delete that by hand if you want a full reset).

A complete default run is committed under `example-run/` (`train.log`,
`generate.log`, `sft.log`, `distill.log`, `ask.log`, the figures and the
assembled `report.md`), so you can read what to expect before spending
the twenty minutes.

Stage 0 prints the three numbers you will measure everything against:

```
characters      : 562,269
vocabulary size : 77
order-0 entropy : 4.473 bits/char  (log2 V = 6.267)
```

`prepare_corpus.py` strips the Project Gutenberg header and footer,
folds typographic quotes and dashes to ASCII, drops any remaining
non-ASCII character and collapses runs of blank lines, leaving 562,269
characters over a 77-symbol vocabulary. The **order-0 entropy** is the
entropy of the character-frequency distribution (Lesson 01): the loss a
model that knows only how often each character occurs would converge to.
$\log_2 V = 6.267$ bits is the loss of a model that knows nothing at all.

Stage 1 then prints its header and starts logging:

```
corpus: 562269 chars   vocab: 77    train: 534156   val: 28113
model: d = 128  L = 4  H = 4  d_ff = 512  T = 128   params = 811008
tokens per step: 1024   steps: 3000   total tokens: 3072000  (5.75 epochs)
progress is printed every 50 steps as [percent done  ETA min:sec]; held-out loss every 250 steps; checkpoints every 500 steps.
[  2%  ETA  18:41]  step    50 | loss 3.6777 nats (5.306 bits/char) | |g| 0.333 | lr 6.67e-04 |  2740 tok/s |    19 s
[  3%  ETA  18:22]  step   100 | loss 3.1019 nats (4.475 bits/char) | |g| 0.384 | lr 1.33e-03 |  2680 tok/s |    38 s
```

(The loss, gradient-norm and throughput values are from the measured
run in `example-run/train.log`; the percent-done and ETA columns were
added to the log line after that run, so there they appear without the
prefix.) The ETA is elapsed time divided by the fraction of steps done,
so it settles after the first few log lines and drifts up slightly when
the evaluation and checkpoint writes land.

Notice step 100: 4.475 bits/char is the order-0 entropy to three
figures. One hundred steps in, the model has learned the character
frequencies and nothing else. Everything after that is context.

## Every parameter (`config.rlab`)

All four stage scripts begin with `run "config.rlab"` and read the struct
`C`. Edit a value, re-run the stage; nothing else needs touching. (Stages 3
and 4 have a few knobs of their own — steps, learning rate, replay
fraction, student size, distillation temperature — in `C_sft` and `C_kd`
structs at the top of `sft.rlab` and `distill.rlab`; see those sections.) The two
right-hand columns are the two shipped presets — `quick`, a 2–3 minute
sanity run, and `large`, a run you leave going overnight; `—` means keep
the default.

| Field | Meaning | Default | `quick` preset | `large` preset |
|---|---|---|---|---|
| `ids_csv` | Corpus ids written by Stage 0 | `"data/ids.csv"` | — | — (point at a bigger corpus; see *Scaling up*) |
| `val_frac` | Fraction of the corpus (from the end) held out for evaluation | `0.05` | — | — |
| `max_chars` | Truncate the corpus to this many characters; `0` = use all | `0` | — | — |
| `d_model` | Residual-stream width $d$ (Lesson 14) | `128` | `64` | `192` |
| `n_layer` | Number of transformer blocks $L$ | `4` | `2` | `6` |
| `n_head` | Attention heads per block; $d_k = d / H$ must be an integer | `4` | `4` | `6` |
| `d_ff` | FFN hidden width; the GPT-2 convention is $4d$ | `512` | `256` | `768` |
| `T` | Context length in characters (each window is $T + 1$ ids, $T$ predictions) | `128` | `64` | — (`256` would cost $\approx 2\times$ per token in attention) |
| `init_std` | Gaussian init scale; residual projections are scaled by $1/\sqrt{2L}$ (nanoGPT) | `0.02` | — | — |
| `n_steps` | Optimiser steps | `3000` | `400` | `20000` |
| `batch` | Windows per step, each on its own `parmap` worker | `8` | — | `14` (one per logical CPU on the reference machine) |
| `eta_max` | Peak learning rate (Lesson 17) | `2.0e-3` | `3.0e-3` | `1.5e-3` |
| `eta_min` | Cosine floor | `2.0e-4` | `3.0e-4` | `1.5e-4` |
| `warmup` | Linear warmup steps | `150` | `40` | `500` |
| `beta1`, `beta2` | Adam moment horizons; $\beta_2 = 0.95$ is the LLM convention (Lesson 16) | `0.9`, `0.95` | — | — |
| `eps_adam` | Adam denominator floor | `1e-8` | — | — |
| `weight_decay` | Decoupled decay, weight matrices only (`gpt_wd_mask`) | `0.1` | — | — |
| `clip_norm` | Global gradient-norm clip (Lesson 18 sidebar) | `1.0` | — | — |
| `log_every` | Print the running train loss and throughput | `50` | `25` | — |
| `eval_every` | Held-out loss on `eval_windows` fixed windows | `250` | `100` | `500` |
| `eval_windows` | Evenly spaced windows of the validation split used for evaluation | `16` | — | — |
| `ckpt_every` | Save `ckpt_latest.npz` (with Adam state) and a numbered snapshot | `500` | `200` | `1000` |
| `ckpt_dir` | Checkpoint directory | `"checkpoints"` | — | — |
| `resume` | `true`: continue from `ckpt_latest.npz` (see *Resuming*) | `false` | — | — |
| `seed` | RNG seed for init and window sampling | `1` | — | — |
| `prompt` | Text the sampler continues (Stage 2) | `"Sherlock Holmes "` | — | — |
| `gen_tokens` | Characters to generate | `300` | `200` | — |
| `temperature` | Softmax temperature $\tau$; `<= 0` means greedy | `0.8` | — | — |
| `top_k` | Keep the $k$ most probable characters before sampling; `0` disables | `20` | — | — |

Parameter count follows the Lesson 14 formula $n \approx 12 L d^2 + 2 V d$
(the exact count is printed at start-up): 811 k at the defaults, ≈ 110 k
for `quick`, ≈ 2.7 M for `large`. Tokens per run are
`batch × T × n_steps`; divide by the measured tok/s for the wall time:

| Preset | Params | Tokens | Measured / extrapolated rate | Wall time | Epochs of this book |
|---|---|---|---|---|---|
| `quick` | ≈ 110 k | 205 k (400 × 8 × 64) | > 2,500 tok/s (smaller model, same per-step overhead) | 2–3 min | ≈ 0.4 |
| `default` | 811 k | 3.07 M | 2,527 tok/s averaged over a full run (2,100–2,900 per window; up to 3,600 in short probes) | 20 min (1,216 s measured) | ≈ 5.5 |
| `large` | ≈ 2.7 M | 35.8 M (20,000 × 14 × 128) | ≈ 800–1,200 tok/s *(extrapolated: cost ∝ params, 14 workers)* | ≈ 8–12 h | ≈ 67 — far too many; use the full canon and cut `n_steps` |

The `large` preset only makes sense with a larger corpus: sixty-odd
passes over one book is memorisation, and the held-out curve will tell
you so (Lesson 20). With the full canon (≈ 3.2 M characters) the same
35.8 M tokens are ≈ 12 epochs. Run `make sherlock-bench` to replace the
extrapolated rate with a measured one before committing an evening.

### Presets

`config.rlab` ends with `run "preset.rlab"`, so whatever that file sets
has the last word. `presets/` holds three files: `default.rlab` (empty —
the defaults above), `quick.rlab` and `large.rlab` (the two right-hand
columns). Select one for every stage with

```sh
make sherlock-preset PRESET=quick     # or large, or default
make sherlock-quick                   # the same as PRESET=quick
```

which copies the file to `preset.rlab`; `train`, `generate`, `sft`,
`distill` and `ask` all read it. Your own experiment is a fourth file in
`presets/`, or a few lines in `preset.rlab` by hand. Switching presets
changes the model dimensions, so the checkpoint rules under *Resuming*
and *Troubleshooting* apply; `make sherlock-clean` resets the preset to
`default`.

The `quick` preset also sets `C.ckpt_dir = "checkpoints-quick"`, so a quick demo and a long default run keep separate checkpoints and separate SFT / student models; the figures, logs and `report.md` are shared, so `make sherlock-report` describes whichever ran last. Three optional fields let a preset scale the later stages too: `C.sft_steps` and `C.sft_eta` override `C_sft.n_steps` / `C_sft.eta` in `sft.rlab`, and `C.kd_steps` overrides `C_kd.n_steps` in `distill.rlab` (`quick` sets `sft_steps = 800`, because a 101 k-parameter model needs more passes to memorise the pairs).

What the quick preset delivers, measured: pre-training 400 steps in 18 s (11,600 tok/s) to 3.40 held-out bits/char — a little better than a bigram model — then SFT in 37 s to 29 / 46 exact answers ("Where does Holmes live?" → "221B Baker Street", but "Who is the woman?" → "I Irthe Adler") with +0.11 bits/char forgetting, and distillation into a 17.6 k-parameter student (3.53 vs 3.59 bits/char hard-target). It exercises every stage in a few minutes; the default preset is where the text starts to look like English and the answers come out clean.

## Resuming

Set `C.resume = true` in `config.rlab` and run `make sherlock-train`
again. The script loads `theta`, the Adam moments `m` and `v`, and the
step counter from `checkpoints/ckpt_latest.npz`, then runs **another**
`n_steps` steps. Three things to know:

- The learning-rate schedule restarts: warmup and cosine are computed
  over the new segment. That is a deliberate simplification; if you want
  one continuous schedule, set `warmup = 0` and a lower `eta_max` for the
  continuation.
- `train_curves.svg` and `train_history.npz` cover the new segment only.
  The numbered snapshots `ckpt_000500.npz`, `ckpt_001000.npz`, … keep
  the global step count, so nothing is lost.
- `train.rlab` trusts the model dimensions in `config.rlab`. If you
  changed `d_model`, `n_layer`, `n_head`, `d_ff` or the vocabulary since
  the checkpoint was written, `gpt_unflatten` fails with a reshape
  error — see *Troubleshooting*. (`generate.rlab` reads the dimensions
  from the checkpoint's `cfgv` field instead, so it survives config
  edits.)

## Reading the printed lines

```
step   100 | loss 3.1019 nats (4.475 bits/char) | |g| 0.384 | eta 1.33e-03 |   2680 tok/s |     38 s
         held-out loss 2.9850 nats (4.306 bits/char)
```

| Column | Meaning |
|---|---|
| `[ 33%  ETA 12:42]` | Prefix added to the current `train.rlab` after the run quoted here: fraction of `n_steps` done and the remaining time at the current rate |
| `step` | Optimiser step (global; continues across a resume) |
| `loss … nats` | Mean training loss over the last `log_every` steps, in nats per character: $\mathcal{L} = -\frac{1}{T}\sum_t \ln P(x_{t+1} \mid x_{\le t})$ |
| `(… bits/char)` | The same divided by $\ln 2$. Bits per character is the natural unit for a character model: it is the compressed size of the text under the model (Lesson 03), and $2^{\text{bits/char}}$ is the per-character perplexity (Lesson 20) |
| `\|g\|` | Global $\ell_2$ norm of the averaged gradient on the most recent step, **before** clipping. If it sits above `clip_norm` every step, clipping is permanently active and the effective learning rate is lower than `eta` |
| `lr` (`eta` in the log quoted here) | Learning rate at this step (linear warmup, then cosine to `eta_min`) |
| `tok/s` | `batch × T × log_every` divided by the wall time of the last `log_every` steps. The window after an evaluation or checkpoint includes that time, so it dips |
| `s` | Wall-clock seconds since the run started |
| `held-out loss` | Mean loss over `eval_windows` fixed, evenly spaced windows of the last 5 % of the corpus — never trained on, never sampled from |

What good numbers look like, in bits per character:

| Model | bits/char | Where it comes from |
|---|---|---|
| Knows nothing | 6.267 | $\log_2 77$ — the uniform distribution; the loss at step 0 |
| Character frequencies | 4.473 | Order-0 entropy of this corpus; reached by step ≈ 100 |
| Bigram (previous character only) | ≈ 3.3 | The Lesson 05 model trained on this book |
| This tutorial's default run | 2.20 | 811 k parameters, 3,000 steps, 20 minutes; held-out, still improving |
| A good small character GPT | ≈ 1.5–1.8 | Held-out, after a few epochs |
| Published character models on large English corpora | ≈ 1.0–1.3 | Hundreds of millions of parameters, hundreds of MB of text |

The run quoted above passed 3 bits/char at step 900, 2.5 at step 1,800,
and finished at **2.235 train / 2.203 held-out bits/char** after 3,000
steps (1,216 s; 2,527 tok/s averaged over the whole run, with the
per-window figure between 2,100 and 2,900). The held-out curve was still
falling at the end — 2.395 → 2.317 → 2.266 → 2.240 → 2.203 over the last
four evaluations — so the default run is *under-trained*, not
over-fitted: the 1.5–1.8 range needs several times more steps and,
ideally, more text (see *Scaling up*). The gradient norm sat between 0.8
and 1.2 for most of the run, so clipping at 1.0 was active on roughly
half the steps; that is normal for this learning rate. A spike to
several times that, coinciding with a loss spike, means `eta_max` is too
high or `warmup` too short. If the held-out loss turns upward while the
training loss keeps falling, the model is memorising the book; stop, or
get more books.

## Reading `train_curves.svg`

Left panel: training loss per step (blue, noisy — it is a single batch
of eight windows) and held-out loss at each evaluation (red), both in
bits per character, with the uniform floor $\log_2 V$ drawn in gray. The
shape you want is a fast fall to ≈ 4.5 (the frequency floor), a slower
fall through ≈ 3 (the bigram floor), then a long, flattening tail. The
red curve should track the blue one closely; a gap that widens with
time, red above blue, is overfitting (Lesson 20). In the default run
the red curve sits slightly *below* the blue — the last 5 % of the book
is a little more predictable than average, and the blue line is a noisy
single-batch estimate — and both are still falling at step 3,000: no
overfitting at 5.5 epochs.

Right panel: gradient norm per step on a log axis (pre-clip value).
Large during the first few dozen steps while the embeddings and LM head
move from their random init, then settling. Isolated spikes that do not
coincide with loss spikes are harmless — clipping handles them;
persistent growth is not.

The raw arrays are in `checkpoints/train_history.npz` (`step`, `loss`,
`gnorm`, `val_step`, `val_loss`) if you want to re-plot them.

## Generation controls

`make sherlock-generate` loads `checkpoints/ckpt_latest.npz` and prints
three continuations of `C.prompt`:

1. **Greedy** (`tau <= 0`): always the most probable character. The
   output is deterministic and, as Lesson 21 predicts, falls into a loop
   within a sentence or two.
2. **`C.temperature` with `C.top_k`** (defaults $\tau = 0.8$, $k = 20$):
   sharpened distribution, tail removed. This is the setting that reads
   best.
3. **$\tau = 1.2$, no truncation**: the model's raw distribution, slightly
   flattened. More variety, more misspellings — you are sampling the
   tail the model is unsure about.

Edit `C.prompt`, `C.gen_tokens`, `C.temperature` and `C.top_k` in
`config.rlab`. Prompts are encoded with `gpt_encode`, which silently skips
characters not in the vocabulary (a curly quote, for instance). rustlab
string literals have no escape sequences, so a newline in a prompt is
`"line one" + sprintf("%c", 10) + "line two"`.

Two things to know about speed and context: `lib/gpt.rlab` has no KV
cache (Lesson 21 derives one; it is on the *next steps* list), so every
generated character re-runs the forward pass over the current context,
and a 300-character sample takes tens of seconds. The context is the
last $T + 1$ characters; anything earlier has no influence on the next
one.

`generate.rlab` always reads `C.ckpt_dir + "/ckpt_latest.npz"`. To
sample from a numbered snapshot, the SFT model
(`checkpoints/ckpt_sft.npz`) or the student, copy that file over
`ckpt_latest.npz` (keep a copy of the original) or change the one
`ck = …` line at the top of `generate.rlab`. Every checkpoint carries its
own `cfgv`, so a student with different dimensions loads without
touching `config.rlab`. For the SFT model use a prompt in its training
template — `"Q: Where does Holmes live?" + sprintf("%c", 10) + "A: "` —
and read the first line of the greedy sample; or let `ask.rlab` do
exactly that:

```sh
make sherlock-ask Q="Where does Holmes live?"
```

It writes `question.rlab` (`C.question = "…";`), loads
`checkpoints/ckpt_sft.npz`, decodes greedily until the newline (80
characters at most) and prints the answer. Set `C.use_base_model = true`
in `preset.rlab` to put the same question to the pre-trained model
instead. The question cannot contain a double quote (rustlab literals
have no escapes).

## Stage 3 — supervised fine-tuning (`sft.rlab`)

The Lesson 25 recipe on a real model. `qa_pairs.rlab` holds 46
hand-written question/answer pairs whose answers are stated literally in
the stories ("Where does Holmes live?" → "221B Baker Street"; "What is
the name of the mastiff at the Copper Beeches?" → "Carlo"). Each pair is
one character sequence in a fixed template,

```
Q: Where does Holmes live?
A: 221B Baker Street
```

and its loss mask is zero on the prompt characters and one on the answer
characters and the closing newline, so the model is trained to
*respond*, not to predict questions. `qa_pairs.rlab` checks every example
against the vocabulary and the mask length when it loads; run it on its
own to see the table.

The script loads `checkpoints/ckpt_latest.npz`, measures the held-out
loss on the book — the **forgetting baseline** — prints the base model's
greedy answers to three training questions and to one paraphrase it will
never see ("Who is the narrator of the stories?"), then fine-tunes for
300 steps with a fresh AdamW at a constant $10^{-3}$ (half the
pre-training peak — $3 \times 10^{-4}$ learns too slowly in 300 steps),
no weight decay, batch 8. Its knobs are the `C_sft` struct at the top of the file:

| Field | Meaning | Default |
|---|---|---|
| `n_steps`, `eta`, `batch` | Steps; constant learning rate; sequences per step (one `parmap` worker each) | `300`, `1.0e-3`, `8` |
| `replay`, `replay_frac` | Mix pre-training windows into every batch; the fraction of the batch that is corpus windows | `true`, `0.5` |
| `probe_every`, `gen_max` | Print greedy answers every so many steps; characters generated per answer (stops at the first newline) | `50`, `40` |
| `held_in`, `held_out` | Which training questions the probe prints; the paraphrase that is never trained on | `[1, 3, 31]`, `"Who is the narrator of the stories?"` |
| `eval_windows` | Held-out windows for the forgetting probe | `16` |
| `ckpt_in`, `ckpt_out` | Input and output checkpoints | `checkpoints/ckpt_latest.npz`, `checkpoints/ckpt_sft.npz` |

Every 25 steps it prints the masked SFT loss and the replay loss; at the
end it greedy-decodes all 46 questions, counts a hit when the output
*starts with* the reference answer, and prints

```
SFT accuracy: N / 46 = … %   (4 replay windows per batch, 300 steps, … s)
forgetting probe: held-out corpus loss a -> b nats  (… -> … bits/char)  delta …
```

before saving `checkpoints/ckpt_sft.npz` and `sft_curves.svg` (SFT loss
and replay loss in bits/char against the pre-SFT held-out line).

Measured on the recorded default run (`example-run/sft.log`, 113 s):

```
SFT accuracy: 43 / 46 = 93.5 %   (4 replay windows per batch, 300 steps, 113 s)
forgetting probe: held-out corpus loss 1.5269 -> 1.6950 nats  (2.203 -> 2.445 bits/char)  delta +0.1681 nats (+0.243 bits/char)
```

and afterwards (`example-run/ask.log`): "Who is the woman?" → "Irene
Adler", "Where does Holmes live?" → "221B Baker Street", "Who narrates
the stories?" → "Dr. Watson", "What is the speckled band?" → "A swamp
adder"; but "What does Holmes play?" → "The polloker Street" and "Who is
Holmes' brother?" (not in the pairs) → "Mrs. A swamp adder". The same
recipe without replay, at a learning rate of 3e-4, reached 11 / 46 and
lost 1.9 bits/char on the book — textbook catastrophic forgetting.

What it demonstrates:

- **Context shorter than an example.** The longest pair is 82
  characters. With a preset whose context `T + 1` is shorter than that
  (`quick`: 65), `sft.rlab` keeps the *tail* of the example — the answer
  and the end of the question — and prints how many pairs it truncated.
  Generation already crops the prompt to the last `T + 1` characters.
- **Prompt masking** is just the `mask` argument every forward call in
  `lib/gpt.rlab` already takes — all ones during pre-training, zero on
  the prompt here. Each worker returns the mean loss over its *own*
  scored positions, so a 15-character answer and a 128-character replay
  window weigh the same in the gradient.
- **Memorisation, not understanding.** With 46 examples and 300 steps
  (each pair is seen about 27 times) the accuracy on the training
  questions should climb toward 100 %, while the paraphrase stays wrong
  or drifts: forty-six strings cannot teach a mapping from questions to
  facts.
- **Catastrophic forgetting and its cheapest fix.** The delta in the
  forgetting probe is what SFT costs the pre-training distribution. Set
  `C_sft.replay = false` and run again: the delta grows, visibly. With
  half of every batch drawn from the book it should stay within a few
  hundredths of a bit.

## Stage 4 — distillation (`distill.rlab`)

Knowledge distillation (Hinton, Vinyals and Dean, 2015): train a small
student to match the trained teacher's *distribution* over the next
character rather than the one-hot next character. The student is
$L = 2$, $d = 64$, 4 heads, $d_{\text{ff}} = 256$ — about 110 k
parameters, one-seventh of the teacher — and the teacher is
`checkpoints/ckpt_latest.npz`.

Both logit vectors are softened with a temperature $\tau$ before the
softmax, $p_\tau = \mathrm{softmax}(z / \tau)$, and the student minimises

$$\mathcal{L} = \alpha\,\tau^2\,\mathrm{KL}\!\left(p_\tau^{\text{teacher}} \,\|\, p_\tau^{\text{student}}\right) + (1 - \alpha)\,\mathrm{CE}(\text{one-hot}, p^{\text{student}}).$$

At $\tau = 1$ a confident teacher puts almost all its mass on one
character and the student learns little beyond the hard label; at
$\tau = 2$ the teacher's ranking of the *wrong* answers — that `q` is
followed by `u`, that `.` is followed by a space or a newline but never
a lowercase letter — becomes visible, and that is what the student
copies ("dark knowledge"). The gradient at the student's logits is

$$\frac{\partial \mathcal{L}}{\partial z} = \frac{\alpha\,\tau}{n}\left(p_\tau^{\text{student}} - p_\tau^{\text{teacher}}\right) + \frac{1 - \alpha}{n}\left(p^{\text{student}} - \text{one-hot}\right),$$

the same $p - q$ shape as the hard-label case (Lesson 03), which is why
only `gpt_dlogits` is replaced (by `kd_dlogits`) and `gpt_backward` is
untouched. The script checks `kd_dlogits` against a central difference
before training anything, in the spirit of Lesson 22.

Knobs are the `C_kd` struct at the top of the file:

| Field | Meaning | Default |
|---|---|---|
| `d_model`, `n_layer`, `n_head`, `d_ff` | The student | `64`, `2`, `4`, `256` |
| `n_steps`, `batch` | Steps; windows per step (one `parmap` worker each; every worker runs the teacher forward and the student forward + backward) | `600`, `8` |
| `eta_max`, `eta_min`, `warmup` | Warmup + cosine, as in pre-training | `2.0e-3`, `2.0e-4`, `60` |
| `T_kd` | Softening temperature $\tau$ | `2.0` |
| `alpha` | Weight of the soft (teacher) term; `0` is hard targets only | `0.5` |
| `log_every`, `eval_every`, `eval_windows` | Logging and held-out evaluation | `50`, `100`, `16` |
| `ckpt_in`, `ckpt_out` | Teacher in, student out | `checkpoints/ckpt_latest.npz`, `checkpoints/ckpt_student.npz` |

The script trains **two** students from the same seed, the same
initialisation and the same sequence of windows: the distilled one
($\alpha = 0.5$) and a hard-target control ($\alpha = 0$). It logs the
student's hard cross-entropy and the mixed loss every 50 steps, the
held-out loss every 100, and ends with

```
model                     params  nats/char  bits/char
teacher                   811008     …          …
distilled student         ~110000    …          …
hard-target student       ~110000    …          …
distillation gain: +… nats (+… bits/char) vs hard targets; gap to teacher … bits/char
```

then saves `checkpoints/ckpt_student.npz` (loadable by `generate.rlab`,
since it carries its own `cfgv`) and `distill_curves.svg` — train and
held-out curves of both students with the teacher's held-out loss as a
horizontal line.

Measured on the recorded default run (`example-run/distill.log`, 82 s +
42 s for the control): teacher 2.203, distilled student 3.420,
hard-target student 3.439 bits/char — a gain of 0.020 bits/char, and a
1.2 bits/char gap to the 7.4× larger teacher.

What to expect: a positive distillation gain of a few hundredths of a
bit — the student learns faster from a soft target because every
position carries 77 numbers of supervision instead of one — and a clear
gap to the teacher that no amount of distillation closes: a 2-layer,
$d = 64$ model cannot represent what a 4-layer, $d = 128$ one knows.
Two experiments worth the minutes they take: `alpha = 1.0` (pure
distillation, no hard labels) and `T_kd = 1.0` (no softening); both
should shrink the gain. Distillation is not a lesson in the curriculum;
this stage is the only place it appears.

## Benchmark and the performance document

`make sherlock-bench` runs `benchmark.rlab`, which needs no corpus (the
ids are random) and prints its results as markdown tables:

1. **Model-size sweep** — forward, backward and flatten time, serial and
   `parmap(8)` tokens per second, and achieved GFLOP/s for
   $(d, L) = (64, 2), (128, 4), (192, 6), (256, 8)$; a size whose single
   window takes more than 5 s is reported and the larger ones skipped.
2. **Cost breakdown of the default model** — time per forward and
   backward part, matmul FLOPs against wall time, and the cost of each
   primitive at the tutorial's shapes (built-in `layernorm`, `softmax`,
   `gelu`, two matmul sizes, one attention head, the embedding gather and
   scatter-add, a bare loop iteration).
3. **`parmap` scaling** — seconds per step, tok/s, speed-up and
   efficiency as the batch grows, plus the per-step overhead outside the
   windows.
4. **Memory** per configuration.
5. `benchmark.svg` — tokens per second against parameter count, serial
   and parallel, on log axes.

The measured numbers, what they mean (how much of a step is dense matmul
and how much is interpreter overhead and thread start-up), and the list
of rustlab features that would make this tutorial faster are written up
in
[`docs/sherlock-performance-and-rustlab-requests.md`](../../docs/sherlock-performance-and-rustlab-requests.md).
Read it before changing `batch` or buying a bigger machine.

## Scaling up

**More books.** `prepare_corpus.py` concatenates any number of plain-text
files:

```sh
cd tutorial/sherlock
python3 prepare_corpus.py data/adventures.txt data/memoirs.txt data/return.txt data/hound.txt
```

The Holmes canon is public domain and on Project Gutenberg: *A Study in
Scarlet*, *The Sign of the Four*, *The Adventures* (#1661, the one
committed here), *The Memoirs*, *The Hound of the Baskervilles*, *The
Return*, *The Valley of Fear*, *His Last Bow*, *The Case-Book*. Together
they are ≈ 3.4 MB of text, ≈ 3.2 M characters after cleaning — six times
this corpus. Check each book's catalogue number on gutenberg.org and
download the "Plain Text UTF-8" version; the Gutenberg header and footer
are stripped automatically. Two consequences of changing the corpus:

- The vocabulary may change (a book with a `[` or a `%` adds a symbol).
  `vocab.rlab` is regenerated, and every existing checkpoint becomes
  incompatible (its embedding has the old $V$ rows). Delete
  `checkpoints/` or point `ckpt_dir` somewhere new. `--lower` is worth
  trying: it folds the vocabulary to ≈ 50 symbols and the model learns
  noticeably faster at a small cost in fidelity.
- Time scales with tokens, not with corpus size. At the default model's
  ≈ 2,700 tok/s one epoch of the canon is ≈ 20 minutes; five epochs is
  under two hours. The 2.7 M-parameter overnight configuration at the
  extrapolated ≈ 800 tok/s takes ≈ 1.1 hours per epoch, so 20,000 steps
  is ≈ 6.5 epochs and ≈ 7 hours.

**A bigger model.** `d_model`, `n_layer`, `d_ff` and `T` are the knobs,
in that order of effect. Per-token cost grows roughly linearly with
parameters and, through attention, quadratically with `T`. Keep
`d_model / n_head` at 32 or 48. Lower `eta_max` as the model grows
(nanoGPT uses $6 \times 10^{-4}$ at 124 M).

**Faster.** `batch` is the only parallelism; it should not exceed your
physical cores, and beyond about 8 the per-step overhead of starting
`parmap` workers and summing eight 811 k-element gradient rows stops
paying for itself. What rustlab would need to go meaningfully faster — and
what the benchmark says about where the time goes now — is in the
performance document linked above.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `load` fails on `data/ids.csv`, or `run "data/vocab.rlab"` cannot find the file | Stage 0 has not been run (both files are gitignored) | `make sherlock-data` |
| `python3: command not found` | No Python 3 on the path | Install Python 3; the script needs only the standard library |
| Reshape or size error inside `gpt_unflatten` right after "resumed from step …" | `checkpoints/ckpt_latest.npz` was written with different `d_model` / `n_layer` / `n_head` / `d_ff` / vocabulary than `config.rlab` plus `preset.rlab` now specify (switching presets does this) | Delete `checkpoints/`, or set `resume = false`, or point `ckpt_dir` at a fresh directory |
| Same error in `sft.rlab` / `distill.rlab` after re-running Stage 0 with more books | Vocabulary size changed, so the embedding and LM head shapes changed | Retrain, or regenerate the corpus with the original single book |
| Loss stuck near 4.47 bits/char for hundreds of steps | Learning rate too low, or warmup too long for a short run | Check `eta` in the log; raise `eta_max`, shorten `warmup` |
| Loss spikes or goes `NaN` | Learning rate too high for the model size | Lower `eta_max`, lengthen `warmup`; `clip_norm = 1.0` should already be set |
| `parmap` result is wrong, or an edit inside the training loop "does nothing" | The lambda passed to `parmap` must be pure: it receives copies of `W`, `mask`, `P`, `cfg` and returns one row; writes to outer variables and RNG calls inside it do not propagate back | Keep all state changes outside the `parmap` call; draw random numbers before it (`sample_windows` does) |
| tok/s is much lower than ≈ 2,500 | Fewer cores than `batch`, another heavy process, or a laptop on battery | Lower `batch` to the physical core count; check Activity Monitor |
| Generated text contains characters you did not expect, or your prompt was shortened | `gpt_encode` skips characters outside the 77-symbol vocabulary | Use plain ASCII quotes and dashes in `C.prompt` |
| `error: line 124: runtime error: No such file or directory` in `train.rlab` | `save` does not create directories and `ckpt_dir` does not exist (a hand-edited preset) | `mkdir -p tutorial/sherlock/<ckpt_dir>`; the `make` targets create `checkpoints/` and `checkpoints-quick/` themselves |
| `ask.rlab` cannot load `checkpoints/ckpt_sft.npz` | Stage 3 has not been run | `make sherlock-sft`, or set `C.use_base_model = true` in `preset.rlab` to query the pre-trained model |
| `make sherlock-ask` fails while reading `question.rlab` | The question contains a double quote; rustlab string literals have no escapes | Rephrase without `"` (an apostrophe is fine) |
| `savefig` writes nothing | Running with `--plot none` is fine; a missing figure means the run died before section 5 of `train.rlab` | Look for the error above the last `step` line |

## File map

```
tutorial/sherlock/
  README.md            this page
  config.rlab          every knob, one struct C, read by all four stage scripts
  prepare_corpus.py    Stage 0 (Python, stdlib only): text -> data/ids.csv + data/vocab.rlab
  train.rlab           Stage 1: pre-training
  generate.rlab        Stage 2: sampling from checkpoints/ckpt_latest.npz
  sft.rlab             Stage 3: supervised fine-tuning with prompt masking, replay, forgetting probe (knobs: C_sft)
  qa_pairs.rlab        the 46 Q/A pairs Stage 3 trains on: qa_prompt(), qa_table(), qa_example()
  ask.rlab             ask the fine-tuned model one question (make sherlock-ask Q="...")
  question.rlab        C.question = "..."; written by make sherlock-ask, read by ask.rlab
  preset.rlab          the selected preset; run last by config.rlab (make sherlock-preset PRESET=...)
  presets/             default.rlab (empty), quick.rlab (2-3 min demo), large.rlab (overnight)
  distill.rlab         Stage 4: teacher -> half-width, half-depth student on softened logits, vs a hard-target control (knobs: C_kd)
  benchmark.rlab       throughput at several sizes (feeds the performance document)
  test_gpt_lib.rlab    self-test for lib/gpt.rlab: round trip, gradient check, timing
  report.py            make sherlock-report: assembles report.md from the stage logs and figures
  report.md            the assembled report of the last run (generated, gitignored)
  <stage>.log          train / generate / sft / distill / ask / benchmark output, written by the make targets (gitignored)
  example-run/         a complete default run, committed: the logs, the figures and its report.md
  data/
    adventures.txt     Project Gutenberg #1661, public domain (committed, 607 KB)
    ids.csv            one 1-based character id per line (generated, gitignored)
    vocab.rlab         vocab_chars, vocab_size, corpus_chars (generated, gitignored)
  checkpoints/         ckpt_latest.npz (theta + Adam state), ckpt_NNNNNN.npz, ckpt_sft.npz, ckpt_student.npz,
                       train_history.npz (gitignored)
  checkpoints-quick/   the same for the quick preset (gitignored)
  train_curves.svg     loss and gradient-norm curves from the last run (gitignored)
  sft_curves.svg       Stage 3 curves (gitignored)
  distill_curves.svg   Stage 4 curves (gitignored)
  benchmark.svg        throughput against model size (gitignored)

lib/gpt.rlab           the model: config, init, flat parameter view, forward, backward,
                       AdamW, schedule, clipping, generation, encode / decode
docs/sherlock-performance-and-rustlab-requests.md
                       measurements and what rustlab needs to go faster
```

`train.rlab` is longer than the curriculum's 60-line guideline for
`lessons/` scripts; scripts under `tutorial/` are allowed to be, because
each is a complete program rather than one concept.

## Where each piece comes from

| In this tutorial | Lesson |
|---|---|
| Characters → ids, vocabulary, order-0 entropy | 01 Tokens and Encoding |
| Softmax, temperature | 02 Probability and Softmax |
| Cross-entropy in nats and bits; soft-target loss (distillation) | 03 Cross-Entropy Loss |
| Token embedding `E` | 04 Embeddings and Similarity |
| The bigram floor (≈ 3.3 bits/char) | 05 The Bigram Language Model |
| Causal multi-head attention (`Wqkv`, `Wo`) | 08 Scaled Dot-Product Attention, 09 Multi-Head Attention |
| Sinusoidal positions `PE` | 10 Positional Encoding |
| GELU FFN (`W1`, `b1`, `W2`, `b2`) | 11 Feed-Forward Block |
| Pre-LN blocks and residual stream | 12 Layer Norm and Residuals, 13 The Transformer Block |
| The full stack, parameter count, nanoGPT init | 14 Full GPT Architecture |
| The analytical backward pass, finite-difference check (`test_gpt_lib.rlab`) | 15 Backpropagation, 22 Full Backprop Through the Block |
| AdamW with decoupled decay and bias correction | 16 AdamW Optimizer |
| Warmup + cosine (`gpt_lr`) | 17 Learning Rate Scheduling |
| The loop, gradient clipping, logging, checkpoints | 18 The Complete Training Loop |
| Held-out loss, bits/char, perplexity, overfitting diagnosis | 20 Perplexity and Evaluation |
| Greedy / temperature / top-k, the autoregressive loop, why a KV cache | 21 Sampling and Generation |
| The end-to-end pipeline this tutorial scales up | 23 Putting It All Together |
| SFT with prompt masking, catastrophic forgetting | 25 Fine-Tuning — SFT and DPO |
| Distillation | not in the curriculum — this stage only |
| Running the trained model in fixed point (not done here; a natural next step) | 26 Quantization and Fixed-Point Inference |
