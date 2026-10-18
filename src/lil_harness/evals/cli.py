"""`lil-eval`: run the eval suite against a local model and print a report.

uv run lil-eval ollama --model qwen3.5:4b
uv run lil-eval vllm --repeats 3 --tag calculator
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import click

from lil_harness.backends.ollama import OllamaBackend
from lil_harness.backends.vllm import VLLMBackend
from lil_harness.evals.cases import load_cases
from lil_harness.evals.runner import RunRecord, run_suite, summarize


@click.command()
@click.argument("engine", type=click.Choice(["ollama", "vllm"]), default="ollama")
@click.option("--model", help="Model name. Defaults to qwen3:4b for Ollama; vLLM asks the server.")
@click.option("--think", is_flag=True, help="Turn on Qwen3's thinking mode.")
@click.option("--repeats", default=1, show_default=True, help="Runs per case.")
@click.option("--concurrency", default=4, show_default=True, help="Runs in flight at once.")
@click.option("--tag", "case_tags", multiple=True, help="Only cases with this tag (repeatable).")
@click.option("--case", "case_names", multiple=True, help="Only this case name (repeatable).")
@click.option(
    "--cases-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("evals/cases"), show_default=True,
)  # fmt: skip
@click.option(
    "--results-dir", type=click.Path(file_okay=False, path_type=Path),
    default=Path("evals/results"), show_default=True,
)  # fmt: skip
@click.option("-v", "--verbose", is_flag=True, help="Show every failed check.")
def main(
    engine: str,
    model: str | None,
    think: bool,
    repeats: int,
    concurrency: int,
    case_tags: tuple[str, ...],
    case_names: tuple[str, ...],
    cases_dir: Path,
    results_dir: Path,
    verbose: bool,
) -> None:
    """Run the eval cases against a local Ollama or vLLM model.

    \f
    (click shows the text above as --help; everything after the \\f is for readers.)

    Args:
        engine: "ollama" or "vllm".
        model: The model name, or None for the engine's default.
        think: Turn on thinking mode.
        repeats: Runs per case.
        concurrency: Runs in flight at once.
        case_tags: Keep only cases with one of these tags.
        case_names: Keep only these cases.
        cases_dir: Where the YAML case files are.
        results_dir: Where to write the JSONL results.
        verbose: Show every failed check, not just the first per run.

    Raises:
        click.ClickException: If no cases match, or every run failed (usually because
            the server isn't running).
    """
    cases = load_cases(cases_dir)
    if case_tags:
        cases = [case for case in cases if set(case_tags) & set(case.tags)]
    if case_names:
        cases = [case for case in cases if case.name in case_names]
    if not cases:
        raise click.ClickException("No cases match those filters.")

    backend_cls = OllamaBackend if engine == "ollama" else VLLMBackend
    backend = backend_cls(enable_thinking=think, **({"model": model} if model else {}))

    click.echo(f"Running {len(cases)} cases x {repeats} on {engine} ...")
    records = asyncio.run(run_suite(cases, backend, repeats=repeats, concurrency=concurrency))
    if all(r.error for r in records):
        raise click.ClickException(
            f"Every run failed, so the server is probably not reachable: {records[0].error}"
        )
    label = f"{engine} {backend.model or '?'}" + (" (thinking)" if think else "")
    print_report(records, label, verbose)

    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", label).strip("-")
    path = results_dir / f"{stamp}_{slug}.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.to_dict()) + "\n")
    click.echo(f"\nFull results: {path}")


def print_report(records: list[RunRecord], label: str, verbose: bool) -> None:
    """Print the per-case results, scores and summary.

    Args:
        records: The runs to report on.
        label: A heading, e.g. "ollama qwen3:4b".
        verbose: Show every failed check, not just the first per run.
    """
    summary = summarize(records)
    click.echo(f"\n{label}\n")
    width = max(len(k) for k in summary.by_case)
    for case_id, count in summary.by_case.items():
        ok = count.all_passed
        mark = click.style("PASS", fg="green") if ok else click.style("FAIL", fg="red")
        click.echo(f"  {mark}  {case_id:<{width}}  {count.passed}/{count.runs}")
        # Show each distinct reason once, with how many runs failed that way.
        reasons: Counter[str] = Counter()
        for record in records:
            if record.case_id != case_id or record.passed:
                continue
            if record.error:
                reasons[f"crashed: {record.error}"] += 1
                continue
            failed = [check for check in record.checks if not check.passed]
            for failed_check in failed if verbose else failed[:1]:
                reasons[f"{failed_check.name} ({failed_check.detail})"] += 1
        for reason, count in reasons.items():
            click.secho(f"          {count}x {reason}", dim=True)

    click.echo("\n  by tag:")
    for tag, count in summary.by_tag.items():
        click.echo(f"    {tag:<16} {count.passed}/{count.runs}")

    if summary.tool_scores:
        # Precision: when it called the tool, was that right? Recall: when it should
        # have called the tool, did it? Counted per run, from cases that say which
        # tools they expect.
        click.echo(
            f"\n  {'tool selection:':<20} {'precision':>9}  {'recall':>6}  {'f1':>5}  tp/fp/fn"
        )
        for name, tool_score in summary.tool_scores.items():
            click.echo(
                f"    {name:<18} {pct(tool_score.precision):>9}  {pct(tool_score.recall):>6}  {pct(tool_score.f1):>5}"
                f"  {tool_score.tp}/{tool_score.fp}/{tool_score.fn}"
            )
    error_rate = f"{summary.tool_errors}/{summary.tool_calls}" if summary.tool_calls else "0/0"
    click.echo(
        f"\n  passed {summary.passed}/{summary.runs}   tool errors {error_rate}   "
        f"avg steps {summary.avg_steps:.1f}   avg time {summary.avg_seconds:.1f}s"
        + (f"   crashed {summary.crashed}" if summary.crashed else "")
    )
    if any(count.runs > 1 for count in summary.by_case.values()):
        click.echo(f"  reliable {summary.reliable_cases}/{len(summary.by_case)} cases passed every repeat")


def pct(value: float | None) -> str:
    """Format a fraction as a whole percentage, or "-" when there is no value."""
    return "-" if value is None else f"{value:.0%}"


# Lets `python -m lil_harness.evals.cli` (and the debugger) run it without the lil-eval script.
if __name__ == "__main__":
    main()
