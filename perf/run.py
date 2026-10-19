#!/usr/bin/env python3
"""One turn of the rustlab performance loop (see perf/README.md).

    python3 perf/run.py --label baseline        # or: make perf LABEL=baseline
    python3 perf/run.py --label iter01 --note "dgemm via Accelerate"

Runs bench_suite.rlab, gate.rlab and profile_step.rlab (under --profile),
records everything with the rustlab version and source revision in
perf/history/NN-<label>.json, prints and writes a comparison against the
baseline and the previous run to perf/reports/NN-<label>.md, and exits
non-zero if the correctness gate fails.  Standard library only.
"""
import argparse
import datetime
import glob
import json
import os
import platform
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(HERE, "history")
REPORTS = os.path.join(HERE, "reports")
REF = os.path.join(HERE, "reference", "gate.json")
# The binary under test: $RUSTLAB (e.g. ../rustlab/target/release/rustlab) or `rustlab` on PATH.
BIN = os.environ.get("RUSTLAB", "rustlab")
GATE_TOL = 1e-6          # relative tolerance on the loss trajectory
GRADCHECK_MAX = 1e-6     # finite-difference agreement
REGRESSION = 0.10        # flag keys more than 10 % slower than the previous run

# Keys whose rate (not time) is the headline, in report order.
HEADLINE = [
    ("e2e.train_step_B8", "training step, B = 8"),
    ("par.grad_flat_B8", "parmap gradient, B = 8"),
    ("model.grad_flat", "one window forward + backward (serial)"),
    ("prim.matmul_512x512x512", "matmul 512^3"),
    ("prim.matmul_129x128x512", "matmul at the FFN shape"),
]


def sh(cmd, cwd=HERE):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def environment():
    env = {"date": datetime.datetime.now().isoformat(timespec="seconds"),
           "machine": platform.machine(), "os": platform.platform()}
    _, out, _ = sh([BIN, "--version"])
    env["rustlab_version"] = out.strip()
    _, path, _ = sh(["sh", "-c", f"command -v '{BIN}'"])
    path = os.path.abspath(path.strip()) if path.strip() else ""
    env["rustlab_binary"] = path
    if path and os.path.exists(path):
        env["rustlab_binary_mtime"] = datetime.datetime.fromtimestamp(
            os.path.getmtime(os.path.realpath(path))).isoformat(timespec="seconds")
    # Source revision: the checkout that holds the binary under test, else ../rustlab.
    src = os.path.normpath(os.path.join(HERE, "..", "..", "rustlab"))
    if path and "/target/" in path:
        src = path.split("/target/")[0]
    if os.path.isdir(os.path.join(src, ".git")):
        _, sha, _ = sh(["git", "-C", src, "rev-parse", "--short", "HEAD"])
        _, dirty, _ = sh(["git", "-C", src, "status", "--porcelain", "--untracked-files=no"])
        _, subj, _ = sh(["git", "-C", src, "log", "-1", "--format=%s"])
        env["rustlab_source"] = {"sha": sha.strip(), "dirty": bool(dirty.strip()), "subject": subj.strip()}
    for key, cmd in (("cpu", ["sysctl", "-n", "machdep.cpu.brand_string"]),
                     ("cores", ["sysctl", "-n", "hw.logicalcpu"])):
        rc, out, _ = sh(cmd)
        if rc == 0:
            env[key] = out.strip()
    return env


def run_bench():
    rc, out, err = sh([BIN, "run", "bench_suite.rlab", "--plot", "none"])
    if rc != 0:
        sys.exit("bench_suite.rlab failed:\n" + err + out)
    res = {}
    for line in out.splitlines():
        m = re.match(r"PERF (\S+) (\S+) (\S+) (\S+)", line)
        if m:
            res[m.group(1)] = {"ms": float(m.group(2)), "rate": float(m.group(3)), "unit": m.group(4)}
    return res


def run_gate():
    rc, out, err = sh([BIN, "run", "gate.rlab", "--plot", "none"])
    if rc != 0:
        return {"error": (err + out)[-2000:]}
    g = {}
    for line in out.splitlines():
        m = re.match(r"GATE (\S+) (\S+)", line)
        if m:
            g[m.group(1)] = float(m.group(2))
    return g


def run_profile():
    rc, out, err = sh([BIN, "run", "--profile", "profile_step.rlab", "--plot", "none"])
    # rustlab <= 0.3.9: name calls total avg in out Mbit/s (7 fields).
    # rustlab >= 0.3.10: name calls total self avg in out Mbit/s (8 fields); nested
    # user functions, [top-level], [matmul] and parmap:<name> worker rows appear.
    rows = []
    num = r"([\d.]+)"
    p8 = re.compile(r"\s+(\S+)\s+(\d+)" + (r"\s+" + num) * 6 + r"\s*$")
    p7 = re.compile(r"\s+(\S+)\s+(\d+)" + (r"\s+" + num) * 5 + r"\s*$")
    for line in err.splitlines():
        m = p8.match(line)
        if m:
            row = {"fn": m.group(1), "calls": int(m.group(2)), "total_ms": float(m.group(3)),
                   "self_ms": float(m.group(4)), "in_kb": float(m.group(6)), "out_kb": float(m.group(7))}
        else:
            m = p7.match(line)
            if not m:
                continue
            row = {"fn": m.group(1), "calls": int(m.group(2)), "total_ms": float(m.group(3)),
                   "self_ms": None, "in_kb": float(m.group(5)), "out_kb": float(m.group(6))}
        if row["fn"] != "TOTAL":
            rows.append(row)
    return rows


def check_gate(g, record_reference=True):
    if "error" in g:
        return False, ["gate.rlab did not run: " + g["error"].strip().splitlines()[-1]]
    msgs, ok = [], True
    if g.get("gradcheck", 1) > GRADCHECK_MAX:
        ok = False
        msgs.append(f"gradient check {g['gradcheck']:.3g} > {GRADCHECK_MAX:g}")
    else:
        msgs.append(f"gradient check {g['gradcheck']:.3g} (limit {GRADCHECK_MAX:g})")
    if not os.path.exists(REF):
        if not record_reference:
            return ok, msgs + ["no reference yet (dry run: not recorded)"]
        os.makedirs(os.path.dirname(REF), exist_ok=True)
        json.dump(g, open(REF, "w"), indent=1)
        msgs.append("no reference yet: this run's trajectory is now perf/reference/gate.json")
        return ok, msgs
    ref = json.load(open(REF))
    worst, worst_key = 0.0, ""
    for k, rv in ref.items():
        if k == "gradcheck":
            continue
        if k not in g:
            return False, msgs + [f"missing gate value {k}"]
        rel = abs(g[k] - rv) / max(abs(rv), 1e-300)
        if rel > worst:
            worst, worst_key = rel, k
    if worst > GATE_TOL:
        ok = False
        msgs.append(f"trajectory differs from reference: {worst_key} rel {worst:.3g} > {GATE_TOL:g}")
    else:
        msgs.append(f"trajectory matches reference (worst rel diff {worst:.3g}, limit {GATE_TOL:g})")
    return ok, msgs


def load_history():
    runs = []
    for p in sorted(glob.glob(os.path.join(HIST, "*.json"))):
        runs.append(json.load(open(p)))
    return runs


def fmt_rate(r):
    if not r or r["rate"] == 0:
        return f"{r['ms']:.3f} ms" if r else "—"
    return f"{r['rate']:,.4g} {r['unit']}"


def speed(old, new):
    """>1 means faster.  Uses time, so it works for every key."""
    if not old or not new or new["ms"] <= 0:
        return None
    return old["ms"] / new["ms"]


def breakdown(b):
    """Where one B = 8 training step goes, from the measured pieces."""
    step = b.get("e2e.train_step_B8", {}).get("ms")
    if not step:
        return []
    par = b["par.grad_flat_B8"]["ms"]
    serial8 = 8 * b["model.grad_flat"]["ms"]
    parts = [
        ("model forward + backward, 8 windows (spread over the cores)", min(par, step)),
        ("unflatten theta -> struct", b["flat.unflatten_811k"]["ms"]),
        ("reduce 8 gradient rows", b["flat.batch_reduce_8x811k"]["ms"]),
        ("clip", b["flat.clip_811k"]["ms"]),
        ("AdamW on flat vectors", b["flat.adamw_811k"]["ms"]),
    ]
    rest = step - sum(p[1] for p in parts)
    parts.append(("other (sampling windows, interpreter)", max(rest, 0.0)))
    eff = serial8 / par / 8 if par else 0
    return [(name, ms, 100 * ms / step) for name, ms in parts] + [("parmap efficiency vs 8 x serial", eff * 100, None)]


def report(run, base, prev):
    b = run["bench"]
    lines = [f"# Perf loop run {run['index']:02d} — {run['label']}", ""]
    e = run["env"]
    src = e.get("rustlab_source", {})
    lines += [f"- Date: {e['date']}", f"- rustlab: {e.get('rustlab_version')} "
              f"(binary {e.get('rustlab_binary_mtime', '?')}; source {src.get('sha', '?')}"
              f"{' dirty' if src.get('dirty') else ''}: {src.get('subject', '')})",
              f"- Machine: {e.get('cpu', e['machine'])}, {e.get('cores', '?')} cores"]
    if run.get("note"):
        lines.append(f"- Note: {run['note']}")
    lines += ["", f"**Correctness gate: {'PASS' if run['gate_ok'] else 'FAIL'}** — " + "; ".join(run["gate_msgs"]), ""]
    lines += ["## Headline", "", "| workload | now | vs previous | vs baseline |", "|---|---:|---:|---:|"]
    for key, name in HEADLINE:
        sp = speed(prev["bench"].get(key) if prev else None, b.get(key))
        sb = speed(base["bench"].get(key) if base else None, b.get(key))
        lines.append(f"| {name} | {fmt_rate(b.get(key))} | {f'{sp:.2f}×' if sp else '—'} | {f'{sb:.2f}×' if sb else '—'} |")
    bd = breakdown(b)
    if bd:
        lines += ["", "## Where a training step goes", "", "| part | ms | % of step |", "|---|---:|---:|"]
        for name, ms, pct in bd:
            lines.append(f"| {name} | {ms:.1f} | {pct:.0f} % |" if pct is not None else f"| {name} | {ms:.0f} % | |")
    lines += ["", "## Every workload", "", "| key | ms | rate | vs previous | vs baseline |", "|---|---:|---:|---:|---:|"]
    flagged = []
    for key in sorted(b):
        sp = speed(prev["bench"].get(key) if prev else None, b[key])
        sb = speed(base["bench"].get(key) if base else None, b[key])
        mark = ""
        if sp and sp < 1 - REGRESSION:
            mark = " ⚠"
            flagged.append(key)
        rate = f"{b[key]['rate']:,.4g} {b[key]['unit']}" if b[key]["rate"] else ""
        lines.append(f"| `{key}` | {b[key]['ms']:.3f} | {rate} | {f'{sp:.2f}×' if sp else '—'}{mark} | {f'{sb:.2f}×' if sb else '—'} |")
    if flagged:
        lines += ["", f"⚠ More than {int(REGRESSION * 100)} % slower than the previous run: " + ", ".join(f"`{k}`" for k in flagged)]
    if run["profile"]:
        has_self = any(r.get("self_ms") is not None for r in run["profile"])
        lines += ["", "## `rustlab run --profile profile_step.rlab`", ""]
        if has_self:
            lines += ["Sorted by self time. `Total` is inclusive. `parmap:` rows are worker CPU time summed over threads.", "",
                      "| function | calls | total ms | self ms | in KB | out KB |", "|---|---:|---:|---:|---:|---:|"]
            for r in sorted(run["profile"], key=lambda r: -(r["self_ms"] or 0))[:20]:
                lines.append(f"| `{r['fn']}` | {r['calls']} | {r['total_ms']:.1f} | {r['self_ms']:.1f} | {r['in_kb']:,.0f} | {r['out_kb']:,.0f} |")
        else:
            lines += ["| function | calls | total ms | in KB | out KB |", "|---|---:|---:|---:|---:|"]
            for r in run["profile"][:12]:
                lines.append(f"| `{r['fn']}` | {r['calls']} | {r['total_ms']:.1f} | {r['in_kb']:,.0f} | {r['out_kb']:,.0f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-only", action="store_true", help="run only the correctness gate and compare it with the reference")
    ap.add_argument("--label", default="", help="short name for this run, e.g. baseline, iter01")
    ap.add_argument("--note", default="", help="what changed in rustlab since the previous run")
    ap.add_argument("--dry-run", action="store_true", help="measure and print, but do not record")
    a = ap.parse_args()
    if a.gate_only:
        ok, msgs = check_gate(run_gate(), record_reference=False)
        print(("GATE PASS: " if ok else "GATE FAIL: ") + "; ".join(msgs))
        return 0 if ok else 1
    if not a.label:
        ap.error("--label is required (except with --gate-only)")

    os.makedirs(HIST, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    hist = load_history()
    print("measuring ... (about 15 s)", flush=True)
    run = {"label": re.sub(r"[^A-Za-z0-9_.-]", "-", a.label), "note": a.note, "env": environment(),
           "bench": run_bench(), "gate": run_gate(), "profile": run_profile()}
    run["gate_ok"], run["gate_msgs"] = check_gate(run["gate"], not a.dry_run)
    run["index"] = (hist[-1]["index"] + 1) if hist else 0
    base = hist[0] if hist else None
    prev = hist[-1] if hist else None
    text = report(run, base, prev)
    if not a.dry_run:
        stem = f"{run['index']:02d}-{run['label']}"
        json.dump(run, open(os.path.join(HIST, stem + ".json"), "w"), indent=1)
        open(os.path.join(REPORTS, stem + ".md"), "w").write(text)
        print(f"recorded perf/history/{stem}.json and perf/reports/{stem}.md\n")
    print(text)
    return 0 if run["gate_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
