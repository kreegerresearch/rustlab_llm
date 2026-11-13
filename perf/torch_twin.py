"""PyTorch twin of perf/bench_suite.rlab's e2e.train_step_B8.

The yardstick for the performance flywheel: the same model and step in
PyTorch on the same machine.  Not a dependency of this repository; run it
in a throwaway environment (make perf-torch):

    uv run --with torch --with numpy python perf/torch_twin.py          # every config
    uv run --with torch --with numpy python perf/torch_twin.py float64    # filter by label

Same model as lib/gpt.rlab: V 77, d 128, 4 layers, 4 heads, d_ff 512,
pre-LN, tanh GELU, fixed sinusoidal PE, untied W_U, 811,008 parameters.
One step: B = 8 windows of 129 tokens (128 predictions each), mean loss,
backward, global-norm clip 1.0, AdamW (lr 1e-3, betas 0.9/0.95, wd 0.1).
Prints the median tok/s over timed steps for each configuration.
"""
import math, statistics, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

V, D, L, H, DFF, T, B = 77, 128, 4, 4, 512, 129, 8


def sinusoidal(t_max, d):
    k = torch.arange(d // 2, dtype=torch.float64)
    omega = 10000.0 ** (-2 * k / d)
    a = torch.outer(torch.arange(t_max, dtype=torch.float64), omega)
    pe = torch.zeros(t_max, d, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(a)
    pe[:, 1::2] = torch.cos(a)
    return pe


class Block(nn.Module):
    def __init__(self, sdpa):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.wo = nn.Linear(D, D, bias=False)
        self.ln2 = nn.LayerNorm(D)
        self.w1 = nn.Linear(D, DFF)
        self.w2 = nn.Linear(DFF, D)
        self.sdpa = sdpa

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (z.view(b, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        if self.sdpa:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            s = q @ k.transpose(-1, -2) / math.sqrt(D // H)
            mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
            o = torch.softmax(s.masked_fill(mask, float("-inf")), dim=-1) @ v
        x = x + self.wo(o.transpose(1, 2).reshape(b, t, D))
        return x + self.w2(F.gelu(self.w1(self.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, sdpa):
        super().__init__()
        self.E = nn.Embedding(V, D)
        self.register_buffer("PE", sinusoidal(T, D))
        self.blocks = nn.ModuleList(Block(sdpa) for _ in range(L))
        self.lnf = nn.LayerNorm(D)
        self.WU = nn.Linear(D, V, bias=False)

    def forward(self, ids):
        x = self.E(ids) + self.PE[: ids.shape[1]].to(self.E.weight.dtype)
        for blk in self.blocks:
            x = blk(x)
        return self.WU(self.lnf(x))


def run(dtype, threads, sdpa, compile_, device="cpu", steps=40, warm=8):
    torch.manual_seed(0)
    if device == "cpu":
        torch.set_num_threads(threads)
    model = GPT(sdpa).to(device=device, dtype=dtype)
    n = sum(p.numel() for p in model.parameters())
    assert n == 811008, n
    decay = [p for p in model.parameters() if p.dim() == 2 and p is not model.E.weight] + [model.E.weight]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": other, "weight_decay": 0.0}],
                            lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    fwd = torch.compile(model) if compile_ else model

    def step():
        w = torch.randint(0, V, (B, T), device=device)
        logits = fwd(w[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, V), w[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    for _ in range(warm):
        step()
    sync()
    ts = []
    for _ in range(steps):
        t0 = time.perf_counter()
        step()
        sync()
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    return med * 1e3, B * (T - 1) / med


if __name__ == "__main__":
    configs = [
        ("float64", torch.float64, 10, False, False, "cpu"),
        ("float64", torch.float64, 14, False, False, "cpu"),
        ("float64", torch.float64, 1, False, False, "cpu"),
        ("float32", torch.float32, 10, False, False, "cpu"),
        ("float32", torch.float32, 10, True, False, "cpu"),
        ("float32", torch.float32, 10, True, True, "cpu"),
        ("float32", torch.float32, 0, True, False, "mps"),
    ]
    only = sys.argv[1:]
    for name, dt, th, sdpa, comp, dev in configs:
        label = f"{dev} {name} threads={th if dev == 'cpu' else '-'} attn={'sdpa' if sdpa else 'manual'}{' compile' if comp else ''}"
        if only and not any(o in label for o in only):
            continue
        try:
            ms, tps = run(dt, th, sdpa, comp, dev)
            print(f"TORCH {label:58s} {ms:8.2f} ms/step {tps:10.0f} tok/s", flush=True)
        except Exception as e:  # compile may need a toolchain
            print(f"TORCH {label:58s} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
