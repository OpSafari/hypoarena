"""Command line interface for hypoarena.

Every subcommand drives the same offline pipeline (see :mod:`hypoarena.runner`)
over a synthetic corpus with planted ground truth, then prints a short human
summary. Failures raised as :class:`hypoarena.errors.HypoArenaError` are reported
on stderr and mapped to that error's ``exit_code`` so scripts can branch on the
category without parsing messages. Nothing here touches the network or torch.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from hypoarena._version import __version__
from hypoarena.artifacts import ArtifactStore
from hypoarena.config import STAGES, RunConfig
from hypoarena.errors import HypoArenaError
from hypoarena.grounding import summarize_reports
from hypoarena.runner import Pipeline, run_report
from hypoarena.synthetic import SyntheticConfig


@dataclass(frozen=True)
class CommandSpec:
    """One registered subcommand: a name, help text and handler."""

    name: str
    help: str
    handler: Callable[[argparse.Namespace], int]


# Commands register themselves in definition order, so ``--help`` lists them in
# the pipeline's own order. Each is appended right after its handler below.
COMMANDS: list[CommandSpec] = []


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the run-configuration flags shared by every subcommand."""
    parser.add_argument(
        "--seed", type=int, default=270106, help=" RNG seed for the synthetic run"
    )
    parser.add_argument(
        "--out", default="runs", help="artifact root directory (default: runs)"
    )
    parser.add_argument(
        "--run-id", default="run", help="run identifier; a single path segment"
    )
    parser.add_argument(
        "--chains", type=int, default=3, help="planted causal chains in the corpus"
    )
    parser.add_argument(
        "--chain-length",
        type=int,
        default=3,
        help="variables per planted chain (links = length - 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse existing checkpoints instead of re-running completed stages",
    )


def _stages_up_to(stage: str) -> tuple[str, ...]:
    """Return the pipeline prefix that ends at ``stage`` (inclusive)."""
    return STAGES[: STAGES.index(stage) + 1]


def _build_config(args: argparse.Namespace, up_to: str | None = None) -> RunConfig:
    """Build a :class:`RunConfig` from parsed CLI arguments."""
    stages = STAGES if up_to is None else _stages_up_to(up_to)
    return RunConfig(
        seed=args.seed,
        run_id=args.run_id,
        stages=stages,
        corpus=SyntheticConfig(
            seed=args.seed, chains=args.chains, chain_length=args.chain_length
        ),
    )


def _run(args: argparse.Namespace, up_to: str | None = None) -> Pipeline:
    """Run the pipeline through ``up_to`` and return it for summarising."""
    config = _build_config(args, up_to)
    pipeline = Pipeline(config, ArtifactStore(args.out, config.run_id))
    pipeline.run(resume=args.resume)
    return pipeline


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level parser with every registered subcommand."""
    parser = argparse.ArgumentParser(
        prog="hypoarena",
        description=(
            "Offline workbench for hypothesis-discovery pipelines: grounded "
            "claim graphs, synthetic literature, debate loops and Elo "
            "tournaments. Every command runs on synthetic data with planted "
            "ground truth and claims no real benchmark."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"hypoarena {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for spec in COMMANDS:
        sub = subparsers.add_parser(spec.name, help=spec.help, description=spec.help)
        _add_common_args(sub)
        sub.set_defaults(func=spec.handler)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv``, dispatch to a subcommand and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return int(handler(args))
    except HypoArenaError as error:
        print(f"hypoarena: {error}", file=sys.stderr)
        return error.exit_code


def _cmd_corpus(args: argparse.Namespace) -> int:
    """Generate the synthetic corpus and report its size and hash."""
    pipeline = _run(args, up_to="corpus")
    corpus = pipeline.state.corpus
    count = len(corpus) if corpus is not None else 0
    print(f"corpus: {count} documents (hash {pipeline.state.corpus_hash})")
    print(f"artifacts: {pipeline.store.root}")
    return 0


COMMANDS.append(
    CommandSpec(
        "corpus",
        "generate a synthetic corpus with planted ground truth",
        _cmd_corpus,
    )
)


def _cmd_generate(args: argparse.Namespace) -> int:
    """Propose candidate claims from the corpus and add them to the graph."""
    pipeline = _run(args, up_to="generate")
    print(f"generate: {len(pipeline.state.candidates)} candidate claims proposed")
    print(f"graph: {len(pipeline.state.graph.claims)} claims total")
    print(f"artifacts: {pipeline.store.root}")
    return 0


COMMANDS.append(
    CommandSpec(
        "generate",
        "propose candidate claims from the corpus",
        _cmd_generate,
    )
)


def _cmd_verify(args: argparse.Namespace) -> int:
    """Grade every claim against the corpus and summarise the flags."""
    pipeline = _run(args, up_to="verify")
    summary = summarize_reports(pipeline.state.reports)
    print(f"verify: {summary.total} claims graded")
    print(
        f"  grounded={summary.grounded} weak={summary.weakly_grounded} "
        f"ungrounded={summary.ungrounded} fabricated={summary.fabricated}"
    )
    print(
        f"  grounded_rate={summary.grounded_rate:.3f} "
        f"mean_score={summary.mean_score:.3f}"
    )
    return 0


COMMANDS.append(
    CommandSpec("verify", "grade every claim against the corpus", _cmd_verify)
)


def _cmd_dedup(args: argparse.Namespace) -> int:
    """Report the near-duplicate claim clusters found in the graph."""
    pipeline = _run(args, up_to="dedup")
    report = pipeline.state.dedup
    if report is None:
        print("dedup: no report produced")
        return 0
    print(f"dedup: {report.total} texts, {len(report.clusters)} clusters")
    print(
        f"  duplicated={report.duplicated} "
        f"duplicate_rate={report.duplicate_rate:.3f} method={report.config.method}"
    )
    return 0


COMMANDS.append(CommandSpec("dedup", "find near-duplicate claims", _cmd_dedup))


def _cmd_debate(args: argparse.Namespace) -> int:
    """Run the propose-critique-revise debate loop over proposed claims."""
    pipeline = _run(args, up_to="debate")
    print(f"debate: {len(pipeline.state.debates)} claims debated and revised")
    print(f"artifacts: {pipeline.store.root}")
    return 0


COMMANDS.append(
    CommandSpec("debate", "run the propose-critique-revise loop", _cmd_debate)
)


def _cmd_rank(args: argparse.Namespace) -> int:
    """Rank every claim with an Elo tournament and print the top standings."""
    pipeline = _run(args, up_to="rank")
    tournament = pipeline.state.tournament
    if tournament is None:
        print("rank: no tournament produced")
        return 0
    print(
        f"rank: {len(tournament.matches)} matches over "
        f"{len(tournament.subjects)} claims"
    )
    for row in tournament.standings()[:5]:
        print(f"  #{row['position']} {row['subject']} elo={row['elo']}")
    return 0


COMMANDS.append(CommandSpec("rank", "rank claims with an Elo tournament", _cmd_rank))


def _cmd_evolve(args: argparse.Namespace) -> int:
    """Expand the graph with the evolution operators and report acceptance."""
    pipeline = _run(args, up_to="evolve")
    steps = pipeline.state.evolution
    accepted = sum(step.accepted_count for step in steps)
    rejected = sum(len(step.rejected) for step in steps)
    print(f"evolve: {len(steps)} generations, {accepted} accepted, {rejected} rejected")
    print(f"graph: {len(pipeline.state.graph.claims)} claims after evolution")
    return 0


COMMANDS.append(
    CommandSpec("evolve", "expand the graph with evolution operators", _cmd_evolve)
)


def _cmd_accumulate(args: argparse.Namespace) -> int:
    """Update beliefs from the graded evidence and print the top posteriors."""
    pipeline = _run(args, up_to="accumulate")
    beliefs = pipeline.state.beliefs
    print(f"accumulate: {len(beliefs)} beliefs updated")
    top = sorted(beliefs, key=lambda item: (-item.posterior, item.claim_id))[:5]
    for belief in top:
        print(f"  {belief.claim_id} posterior={belief.posterior:.4f}")
    return 0


COMMANDS.append(
    CommandSpec("accumulate", "update beliefs from graded evidence", _cmd_accumulate)
)


def _cmd_report(args: argparse.Namespace) -> int:
    """Run the full pipeline and report where the rendered reports were written."""
    pipeline = _run(args)
    payload = run_report(pipeline)
    counts = payload["counts"]
    recovered = payload["recovered"]
    print(f"report: run {payload['run_id']}")
    print(
        f"  documents={counts['documents']} claims={counts['claims']} "
        f"evidence={counts['evidence']}"
    )
    print(
        f"  recovered {recovered['recovered']}/{recovered['planted']} planted "
        f"links (rate {recovered['rate']})"
    )
    print(f"  json: {pipeline.store.path('report.json')}")
    print(f"  markdown: {pipeline.store.path('report.md')}")
    print(f"  html: {pipeline.store.path('report.html')}")
    return 0


COMMANDS.append(
    CommandSpec("report", "run the full pipeline and write reports", _cmd_report)
)


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run a small offline demo and print recovered versus planted hypotheses.

    This is the one-shot, fully offline end-to-end path: it builds a synthetic
    corpus with planted causal chains, runs the whole pipeline and reports which
    planted links the run recovered. It demonstrates the mechanism only and makes
    no claim about real scientific discovery.
    """
    pipeline = _run(args)
    payload = run_report(pipeline)
    recovered = payload["recovered"]
    print("hypoarena demo - offline synthetic corpus with planted ground truth")
    print(
        f"planted links: {recovered['planted']}   "
        f"recovered: {recovered['recovered']}   rate: {recovered['rate']}"
    )
    for link in recovered["links"]:
        mark = "recovered" if link["recovered"] else "MISSING"
        print(f"  [{mark}] {link['statement']}")
    print("note: a synthetic demonstration only; no claim about real discovery.")
    return 0


COMMANDS.append(
    CommandSpec("demo", "run a small offline end-to-end demonstration", _cmd_demo)
)
