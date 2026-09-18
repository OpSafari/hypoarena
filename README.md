# hypoarena

`hypoarena` is a **fully offline** scientific hypothesis-discovery workbench.
It decomposes the **mechanism layer** of generate–debate–evolve "AI co-scientist"
pipelines into individually testable components: a citation-bearing
hypothesis/evidence graph, a synthetic literature factory, span-level grounding
verification, pluggable agent adapters, Bradley–Terry / Elo tournaments,
paraphrase deduplication (TF-IDF, MinHash LSH), hypothesis evolution operators,
and Bayesian evidence accumulation.

At runtime the package depends only on NumPy. PyTorch (CPU-only) is an optional
extra used solely to demonstrate a small trainable ranker. Every default test
and example runs without network access, without calling real models, and
without downloading weights.

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"          # development env (hatchling, pytest, ruff, mypy)
.venv/bin/pip install -e ".[dev,torch]" \
    --extra-index-url https://download.pytorch.org/whl/cpu   # when the optional ranker is needed
```

## Quick start

```bash
.venv/bin/hypoarena demo --chains 2 --chain-length 2 --out /tmp/hya
```

This runs the complete pipeline offline over a synthetic corpus, prints whether
each planted causal chain was recovered, and writes `report.md` / `report.html`.
It is a mechanism demonstration and makes no claim about real scientific
discovery capability. A typical run reports every planted link (e.g.
`planted links: 6  recovered: 6  rate: 1.0` on the demo corpus) and ends with an
explicit note that it is a synthetic demonstration only.

## Common commands

| Command | Purpose |
| --- | --- |
| `make build` | Build a wheel with the local venv (`--no-isolation`) |
| `make test` | Fast test suite (excludes the `slow` mark) |
| `make test-all` | Full suite (includes `slow` / `model` tests) |
| `make format` / `make format-check` | ruff formatting and its check mode |
| `make lint` | ruff check + format --check |
| `make typecheck` | mypy static analysis |
| `make demo` | End-to-end offline demonstration |

## CLI

The `hypoarena` command-line tool exposes one subcommand per pipeline stage, in
pipeline order: `corpus`, `generate`, `verify`, `dedup`, `debate`, `rank`,
`evolve`, `accumulate`, `report`, and `demo`. All of them run offline over
synthetic corpora. One-shot demonstration:

```console
$ hypoarena demo --chains 2 --chain-length 2 --out /tmp/hya
```

Subcommand options, shared arguments, and exit codes are documented in
[docs/cli.md](docs/cli.md).

### What each stage does

- **corpus** — generate a deterministic synthetic literature with planted causal
  chains, competing hypotheses, paraphrase clusters, and controllable noise
  (distractors, contradictory findings); every corpus carries a provenance hash.
- **generate / debate / evolve** — run the generate–debate–evolve loop over the
  pluggable agent protocol (propose / critique / revise) with scripted, replay,
  or loopback-mock agents; evolution operators (scope narrowing, variable
  substitution, mechanism crossover, claim decomposition) preserve graph
  validity by construction.
- **verify** — span-level grounding checks: citation existence, entity overlap,
  polarity consistency, and numeric agreement, with graded flags for ungrounded
  or weakly grounded claims. Adversarial fixtures (fabricated citations,
  drifted numbers, negation flips) must be caught.
- **dedup** — normalized-text hashing, char/word n-gram Jaccard, TF-IDF cosine,
  and MinHash LSH with documented false-positive/false-negative behavior;
  planted paraphrase clusters are detected at measured recall/precision.
- **rank** — round-robin pairwise judging over rubric scores (novelty,
  testability, grounding, consistency) with Bradley–Terry/Elo updates, draw
  handling, and K-factor decay. Property tests assert that ratings recover a
  planted skill order within tolerance on seeded runs.
- **accumulate** — Bayesian belief updating from graded evidence with prior
  sensitivity analysis, contradiction policies, and golden numeric tests.
- **report** — Markdown and self-contained HTML reports (rankings, grounding
  flags, dedup clusters, belief updates) with hostile-text escaping, embedded
  run metadata, and an honest limitations section; no network access required.

### Artifact stability

All JSONL/JSON artifacts are written through a single canonical serializer
(sorted keys, fixed separators, floats quantized at their production points), so
run outputs — and the golden digests pinned over them — are byte-identical
across platforms and interpreter versions. Checkpointed runs replay completed
stages from the ledger and produce artifacts byte-identical to straight runs.

## Examples and documentation

`examples/` contains three fully offline, runnable examples — planted-corpus
generation with grounding verification, a full generate→debate→rank tournament
showing Elo recovery of a planted skill order, and an evolution + dedup cycle
with measured novelty statistics. Each ships with its own README and real
captured output; see [docs/examples.md](docs/examples.md) for the overview.

Architecture, schemas, per-module notes, and the honesty conventions live under
`docs/`; start the module index at [docs/api.md](docs/api.md).

## License

MIT — see [LICENSE](LICENSE). Release history is in [CHANGELOG.md](CHANGELOG.md).
