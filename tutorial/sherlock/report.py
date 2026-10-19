#!/usr/bin/env python3
"""Assemble report.md from the stage logs and figures in this directory.

    make sherlock-report          (or: python3 report.py)

Every `make sherlock-*` stage tees its output to <stage>.log; this script
reads whichever logs exist and writes a single Markdown page with the
key numbers, the figures and the generated text, so a run can be shared
without re-running anything.  Standard library only.
"""
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def read(name):
    return open(name, encoding="utf-8", errors="replace").read() if os.path.exists(name) else ""


def grab(text, pattern, flags=re.M):
    m = re.findall(pattern, text, flags)
    return m


def section(title, body):
    return f"## {title}\n\n{body.strip()}\n\n" if body.strip() else ""


def fig(path, caption):
    return f"![{caption}]({path})\n\n*{caption}*\n" if os.path.exists(path) else f"*({path} not found: run the stage first)*\n"


out = ["# Sherlock GPT run report\n", f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `report.py` from the logs in `tutorial/sherlock/`.\n"]

# --- configuration actually used (from the training log header) --------
train = read("train.log") or read("train_run.log")
hdr = "\n".join(l for l in train.splitlines()[:6] if l.startswith(("corpus:", "model:", "tokens per step:")))
out.append(section("Configuration", "```\n" + hdr + "\n```" if hdr else ""))

# --- training ------------------------------------------------------------
if train:
    steps = grab(train, r"^\[.*?\]\s+step\s+(\d+) \| loss ([\d.]+) nats \(([\d.]+) bits/char\).*?\|\s*(\d+) tok/s \|\s*(\d+) s")
    if not steps:  # older log format without the progress prefix
        steps = grab(train, r"^step\s+(\d+) \| loss ([\d.]+) nats \(([\d.]+) bits/char\).*?\|\s*(\d+) tok/s \|\s*(\d+) s")
    held = grab(train, r"held-out loss ([\d.]+) nats \(([\d.]+) bits/char\)")
    done = grab(train, r"^done: (.*)$")
    rows = ["| step | train bits/char | tok/s | elapsed s |", "|---:|---:|---:|---:|"]
    keep = steps[:: max(1, len(steps) // 10)] + ([steps[-1]] if steps and steps[-1] not in steps[:: max(1, len(steps) // 10)] else [])
    for st, nats, bits, tps, el in keep:
        rows.append(f"| {st} | {bits} | {tps} | {el} |")
    body = "\n".join(rows)
    if held:
        body += f"\n\nHeld-out loss: first {held[0][1]} bits/char, last **{held[-1][1]} bits/char** ({held[-1][0]} nats). "
        body += "For scale: uniform over the vocabulary is log2(V), the order-0 entropy of this corpus is 4.47 bits/char, a bigram model about 3.3.\n"
    if done:
        body += f"\nRun summary: `{done[-1]}`\n"
    body += "\n" + fig("train_curves.svg", "Training and held-out loss (bits/char) and gradient norm")
    sample = train.split("=== sample")[-1] if "=== sample" in train else ""
    if sample:
        sample = "\n".join(sample.splitlines()[1:]).split("Saved train_curves")[0]
        body += "\n### Sample at the end of training\n\n```text\n" + sample.strip() + "\n```\n"
    out.append(section("Pre-training (`train.rlab`)", body))

# --- generation ----------------------------------------------------------
gen = read("generate.log")
if gen:
    out.append(section("Generation (`generate.rlab`)", "```text\n" + gen.strip() + "\n```"))

# --- SFT ---------------------------------------------------------------------
sft = read("sft.log")
if sft:
    acc = grab(sft, r"(accuracy[^\n]*)", re.I)
    forget = grab(sft, r"((?:forgetting|held-out)[^\n]*)", re.I)
    body = ""
    if acc:
        body += "Exact-match accuracy on the training questions (greedy decoding):\n\n- " + "\n- ".join(dict.fromkeys(acc)) + "\n\n"
    if forget:
        body += "Forgetting probe (held-out corpus loss before/after):\n\n- " + "\n- ".join(dict.fromkeys(forget[-6:])) + "\n\n"
    body += fig("sft_curves.svg", "SFT loss")
    probes = grab(sft, r"^(Q: .*)$")
    tail = "\n".join(sft.strip().splitlines()[-40:])
    body += "\n### Log tail\n\n```text\n" + tail + "\n```\n"
    out.append(section("Supervised fine-tuning (`sft.rlab`)", body))

# --- ask ---------------------------------------------------------------
ask = read("ask.log")
if ask:
    out.append(section("Questions asked (`ask.rlab`)", "```text\n" + ask.strip() + "\n```"))

# --- distillation ------------------------------------------------------
kd = read("distill.log")
if kd:
    tbl = [l for l in kd.splitlines() if l.startswith("|") or re.search(r"bits/char", l)]
    body = "```text\n" + "\n".join(tbl[-12:]) + "\n```\n\n" + fig("distill_curves.svg", "Distilled vs hard-target student")
    out.append(section("Distillation (`distill.rlab`)", body))

# --- benchmark ---------------------------------------------------------
bench = read("benchmark.log")
if bench:
    body = "```text\n" + "\n".join(bench.strip().splitlines()[-60:]) + "\n```\n\n" + fig("benchmark.svg", "Throughput vs model size")
    body += "\nSee `docs/sherlock-performance-and-rustlab-requests.md` for the analysis and the rustlab request list.\n"
    out.append(section("Benchmark (`benchmark.rlab`)", body))

if len(out) <= 2:
    out.append("No stage logs found. Run `make sherlock-quick` or `make sherlock` first.\n")

open("report.md", "w").write("\n".join(out))
print("wrote", os.path.join(HERE, "report.md"), f"({sum(len(x) for x in out):,} chars)")
