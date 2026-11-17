"""Read the traces of a LangSmith project: latency, tokens and cost per question."""
import os
import sys
import warnings
from statistics import mean

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()
warnings.filterwarnings("ignore", category=DeprecationWarning)  # list_runs still works, but its successor is async


def fetch_runs(project: str, version: str, limit: int = 30) -> list:
    """Return the most recent traces of one version (newest first), one trace per question."""
    client = Client()
    return list(client.list_runs(project_name=project, is_root=True, filter='has(tags, "' + version + '")', limit=limit))


def latency(run) -> float:
    """Duration of a run in seconds."""
    return (run.end_time - run.start_time).total_seconds()


def summarize_runs(runs: list) -> dict:
    """Mean latency, tokens and cost per question, and the number of errors."""
    return {
        "traces": len(runs),
        "mean_latency_s": round(mean(latency(run) for run in runs), 1),
        "max_latency_s": round(max(latency(run) for run in runs), 1),
        "mean_tokens": round(mean(run.total_tokens or 0 for run in runs)),
        "mean_cost_usd": round(float(mean(run.total_cost or 0 for run in runs)), 5),
        "errors": sum(run.error is not None for run in runs),
    }


def find_run(runs: list, question_id: str):
    """Return the trace of one question of the reference set, or None."""
    for run in runs:
        if run.extra["metadata"].get("question_id") == question_id:
            return run
    return None


def get_step(run, project: str, name: str):
    """Return the first step called `name` inside a trace (for example generate_answer), or None."""
    for step in Client().list_runs(project_name=project, trace_id=str(run.trace_id)):
        if step.name == name:
            return step
    return None


def print_trace(run, project: str) -> None:
    """Print the tree of steps of a trace: indentation, type, name, tokens and duration."""
    steps = sorted(Client().list_runs(project_name=project, trace_id=str(run.trace_id)), key=lambda step: step.start_time)
    depth = {}
    for step in steps:
        depth[step.id] = 0 if step.parent_run_id is None else depth.get(step.parent_run_id, 0) + 1
        print("  " * depth[step.id] + step.run_type, step.name, "|", step.total_tokens or 0, "tokens |", round(latency(step), 2), "s")


if __name__ == "__main__":
    project = os.environ.get("LANGSMITH_PROJECT", "rag-fil-rouge")
    for version in sys.argv[1:] or ["baseline", "titled", "agent"]:
        runs = fetch_runs(project, version)
        print(version, summarize_runs(runs) if runs else "aucune trace")
