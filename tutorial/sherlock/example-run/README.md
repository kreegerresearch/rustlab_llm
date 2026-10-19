# Example run: the default preset, recorded

This folder is one complete pass of the tutorial with the default
`config.rlab` (no preset overrides), kept so you can read what to expect
before running anything. Nothing here is needed to run the tutorial;
`make sherlock-clean` never touches it.

| | |
|---|---|
| Date | 2026-10-18 |
| Machine | Apple silicon M4 Max laptop, 14 logical cores (`parmap` uses them for the batch) |
| rustlab | 0.3.9 |
| Corpus | `data/adventures.txt` (Project Gutenberg #1661), 562,269 characters, 77-symbol vocabulary, order-0 entropy 4.473 bits/char |
| Model | `d_model 128`, 4 layers, 4 heads, `d_ff 512`, context 128 → 811,008 parameters |
| Pre-training | 3,000 steps × 8 windows × 128 tokens = 3.07 M tokens (5.75 epochs), 1,216 s, 2,527 tok/s average |
| Result | held-out **2.203 bits/char** (1.527 nats), train 2.235; still falling at the end |

## Files

| File | Written by | What it shows |
|---|---|---|
| `train.log` | `make sherlock-train` | The header, one line per 50 steps (loss in nats and bits/char, gradient norm, learning rate, tokens/s, elapsed), held-out loss every 250 steps, the final sample. This run predates the `[percent done  ETA]` prefix and the `lr` label, so its lines read `step … eta …`. |
| `train_curves.svg` | `train.rlab` | Left: train and held-out loss in bits/char against the uniform line. Right: gradient norm per step. |
| `generate.log` | `make sherlock-generate` | Three continuations of the prompt `Sherlock Holmes` (with a trailing space): greedy (degenerates into a repeated 4-gram), τ = 0.8 with top-k 20, τ = 1.2 untruncated. |
| `sft.log` | `make sherlock-sft` | Supervised fine-tuning on the 46 Q/A pairs: loss every 25 steps, greedy probes every 50, the final per-question table, **43 / 46 = 93.5 %** exact answers in 113 s, and the forgetting probe (2.203 → 2.445 held-out bits/char, +0.243). |
| `sft_curves.svg` | `sft.rlab` | SFT loss and replay loss in bits/char against the pre-SFT held-out line. |
| `ask.log` | `make sherlock-ask Q=…` | Six questions put to the fine-tuned model: four answered exactly, one misspelled, one not in the pairs and answered nonsensically. |
| `distill.log` | `make sherlock-distill` | Distillation into the 109,440-parameter student (τ = 2, α = 0.5, 600 steps, 82 s) and the hard-target control (42 s): teacher 2.203, distilled **3.420**, hard-target 3.439 bits/char. |
| `distill_curves.svg` | `distill.rlab` | Held-out bits/char of both students over training. |
| `benchmark.log` | `make sherlock-bench` | Forward / backward time and tokens/s for four model sizes, the cost breakdown of the default model, `parmap` scaling over the batch, memory per configuration. Analysed in `docs/sherlock-performance-and-rustlab-requests.md`. |
| `benchmark.svg` | `benchmark.rlab` | Tokens/s against parameter count, serial and `parmap(8)`. |
| `report.md` | `make sherlock-report` | All of the above assembled into one page by `report.py`. Its figure links point at the sibling `.svg` files in this folder. |

## Reproducing it

```sh
make sherlock            # default preset: data, train, generate, sft, distill, report (~30 min)
make sherlock-ask Q="Who is the woman?"
make sherlock-bench
```

Pre-training is seeded (`C.seed = 1`), but `parmap` worker scheduling and
floating-point reduction order make the numbers differ in the last digits
from run to run; the held-out loss should land within a few hundredths of
a bit of the value above on the same hardware.
