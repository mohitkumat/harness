"""``harness-eval`` command line: run a task subset against an agent and write a JSON report."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from harness_evals.agents import AGENTS, AgentOptions
from harness_evals.runner import Report, run_suite
from harness_evals.tasks import load_tasks


def _print_report(report: Report) -> None:
    width = max((len(r.task_id) for r in report.results), default=4)
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        reason = r.error or "; ".join(c.detail for c in r.checks if not c.passed)
        print(f"  {status}  {r.task_id:<{width}}  {reason}")
    print(
        f"\n{report.passed}/{len(report.results)} passed "
        f"({report.pass_rate:.0%}) — agent={report.agent}"
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Exits non-zero only on infrastructure errors or when below ``--fail-under``."""
    parser = argparse.ArgumentParser(prog="harness-eval", description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, default=Path("evals/tasks"))
    parser.add_argument("--agent", choices=sorted(AGENTS), default="null")
    parser.add_argument("--model", help="Model id for the harness agent (default: HARNESS_MODEL).")
    parser.add_argument(
        "--allow-host-shell",
        action="store_true",
        help="Let the harness agent run shell commands ON THIS MACHINE without approval "
        "(inside each task's temp workspace). Local use only; see docs/adr/0003.",
    )
    parser.add_argument("--tag", action="append", default=[], help="Only run tasks with this tag.")
    parser.add_argument("--out-dir", type=Path, default=Path("evals/results"))
    parser.add_argument(
        "--fail-under", type=float, default=0.0, help="Exit 1 if pass rate is below this (0-1)."
    )
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks_dir, set(args.tag) or None)
    if not tasks:
        print(f"no tasks found in {args.tasks_dir} (tags={args.tag})", file=sys.stderr)
        return 2

    options = AgentOptions(
        model=args.model, allow_host_shell=args.allow_host_shell, log_dir=args.out_dir / "sessions"
    )
    agent = AGENTS[args.agent](options)
    print(f"Running {len(tasks)} task(s) with agent={agent.name}\n")
    report = asyncio.run(run_suite(agent, tasks))
    _print_report(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    safe_agent = report.agent.replace(":", "_").replace("/", "_")
    out = args.out_dir / f"{report.started_at.replace(':', '')}-{safe_agent}.json"
    out.write_text(json.dumps(report.to_json_dict(), indent=2))
    print(f"report: {out}")

    return 1 if report.pass_rate < args.fail_under else 0


if __name__ == "__main__":
    raise SystemExit(main())
