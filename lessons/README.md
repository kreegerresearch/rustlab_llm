# Standalone `.r` scripts (per-lesson)

Each `<NN-topic-slug>/` directory holds shell-runnable `rustlab` scripts
that exercise the same algorithms as the lesson's notebook source. The
notebook source itself lives at
[`../notebooks/<NN-topic-slug>.md`](../notebooks/) — *not* here.

```
lessons/
└── 01-tokens-and-encoding/
    ├── char_frequencies.r
    ├── one_hot_encoding.r
    └── *.svg              # output from `rustlab run *.r` (gitignored)
```

Subdirectories track the notebook slugs one-for-one. As `.r` scripts
are added or rewritten for a lesson, drop them in the matching
`lessons/<slug>/` directory.

## Running

```sh
rustlab run lessons/01-tokens-and-encoding/char_frequencies.r
make lesson-01     # run every .r in lesson 01
```

Scripts call `savefig("foo.svg")` next to themselves; the canonical
rendered plots live in [`../site/`](../site/), so `.r` artefacts
(`*.svg`, `*.html`, `*.png` under each lesson dir) are gitignored.

## Why have both notebooks and scripts?

The notebook is the lesson — prose, math, and code interleaved, executed
by the renderer into the published site. The `.r` scripts are *parallel*
to the notebook's code blocks: each script maps to one or two
` ```rustlab ` blocks in the notebook source. Tinker with a script
without touching the notebook to explore one concept in isolation.

The duplication is intentional and small. When you change one, change
the other.
