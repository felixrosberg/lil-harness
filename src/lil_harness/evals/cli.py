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
@click.option("--tag", "tags", multiple=True, help="Only cases with this tag (repeatable).")
@click.option("--case", "case_ids", multiple=True, help="Only this case id (repeatable).")
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
    tags: tuple[str, ...],
    case_ids: tuple[str, ...],
    cases_dir: Path,
    results_dir: Path,
    verbose: bool,
) -> None:
    """Run the eval cases against a local Ollama or vLLM model."""
    cases = load_cases(cases_dir)
    if tags:
        cases = [c for c in cases if set(tags) & set(c.tags)]
    if case_ids:
        cases = [c for c in cases if c.id in case_ids]
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
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
    click.echo(f"\nFull results: {path}")


def print_report(records: list[RunRecord], label: str, verbose: bool) -> None:
    s = summarize(records)
    click.echo(f"\n{label}\n")
    width = max(len(k) for k in s.by_case)
    for case_id, (ok, n) in s.by_case.items():
        mark = click.style("PASS", fg="green") if ok == n else click.style("FAIL", fg="red")
        click.echo(f"  {mark}  {case_id:<{width}}  {ok}/{n}")
        # Show each distinct reason once, with how many runs failed that way.
        reasons: Counter[str] = Counter()
        for r in records:
            if r.case_id != case_id or r.passed:
                continue
            if r.error:
                reasons[f"crashed: {r.error}"] += 1
                continue
            failed = [c for c in r.checks if not c.passed]
            for c in failed if verbose else failed[:1]:
                reasons[f"{c.name} ({c.detail})"] += 1
        for reason, count in reasons.items():
            click.secho(f"          {count}x {reason}", dim=True)

    click.echo("\n  by tag:")
    for tag, (ok, n) in s.by_tag.items():
        click.echo(f"    {tag:<16} {ok}/{n}")

    if s.tool_scores:
        # Precision: when it called the tool, was that right? Recall: when it should
        # have called the tool, did it? Counted per run, from cases that say which
        # tools they expect.
        click.echo(
            f"\n  {'tool selection:':<20} {'precision':>9}  {'recall':>6}  {'f1':>5}  tp/fp/fn"
        )
        for name, t in s.tool_scores.items():
            click.echo(
                f"    {name:<18} {pct(t.precision):>9}  {pct(t.recall):>6}  {pct(t.f1):>5}"
                f"  {t.tp}/{t.fp}/{t.fn}"
            )
    error_rate = f"{s.tool_errors}/{s.tool_calls}" if s.tool_calls else "0/0"
    click.echo(
        f"\n  passed {s.passed}/{s.runs}   tool errors {error_rate}   "
        f"avg steps {s.avg_steps:.1f}   avg time {s.avg_seconds:.1f}s"
        + (f"   crashed {s.crashed}" if s.crashed else "")
    )
    if any(n > 1 for _, n in s.by_case.values()):
        click.echo(f"  reliable {s.reliable_cases}/{len(s.by_case)} cases passed every repeat")


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"
