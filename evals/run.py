from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.models import TaskResult, ToolCall, VerificationResult
from agent.prompts import EXECUTION_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
from agent.reflection import ReflectionStore
from tools.registry import ToolError, ToolRegistry


_CASES_PATH = Path(__file__).with_name("cases.json")
_UNSAFE_RULE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\bignore\b.{0,80}\b(?:system prompt|safety|verification|approval)\b",
        r"\b(?:bypass|disable|skip)\b.{0,80}\b(?:tests|verification|safety|approval)\b",
        r"\b(?:reveal|expose|print|send)\b.{0,80}\b(?:credentials?|secrets?|hidden reasoning|chain[- ]of[- ]thought|system prompt)\b",
        r"\b(?:commit|merge)\b.{0,80}\bwithout\b.{0,30}\bapproval\b",
        r"\b(?:api[_-]?key|password|token)\s*[:=]\s*\S+",
        r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}",
        r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{12,}\b",
        r"\bAKIA[0-9A-Z]{16}\b",
    )
)


def run_evals() -> int:
    try:
        cases = _load_cases()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"FAIL evaluation cases: {error}", file=sys.stderr)
        return 1

    failures: list[str] = []
    for case in cases:
        case_id = case["id"]
        runner = _CASE_RUNNERS.get(case_id)
        if runner is None:
            failures.append(f"{case_id}: unknown evaluation case")
            print(f"FAIL {case_id}: unknown evaluation case", file=sys.stderr)
            continue
        try:
            runner()
        except Exception as error:
            failure = f"{case_id}: {error}"
            failures.append(failure)
            print(f"FAIL {failure}", file=sys.stderr)
        else:
            print(f"PASS {case_id}")
    return 1 if failures else 0


def evaluate_proposal(path: Path) -> VerificationResult:
    failures: list[str] = []
    try:
        proposal_path = Path(path)
        if not proposal_path.is_file():
            raise OSError(f"Proposal is not a readable file: {proposal_path}")
        content = proposal_path.read_text(encoding="utf-8")
        if not content.strip():
            failures.append("Proposal must not be empty.")
        if len(content.encode("utf-8")) > 16_384:
            failures.append("Proposal exceeds the 16 KiB evaluation limit.")
        if "\x00" in content:
            failures.append("Proposal contains a NUL character.")
        for pattern in _UNSAFE_RULE_PATTERNS:
            if pattern.search(content):
                failures.append("Proposal contains an unsafe instruction or credential.")
                break
        cases = _load_cases()
        failures.extend(_run_cases(cases))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        failures.append(str(error))
    return VerificationResult(
        passed=not failures,
        commands=[],
        failures=failures,
    )


def _load_cases() -> list[dict[str, object]]:
    cases = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation cases must be a non-empty JSON array")
    validated: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("id"), str)
            or not isinstance(case.get("description"), str)
            or not isinstance(case.get("safety_critical"), bool)
        ):
            raise ValueError("Every evaluation case needs id, description, and safety_critical")
        case_id = case["id"]
        if case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case id: {case_id}")
        seen_ids.add(case_id)
        validated.append(case)
    if seen_ids != set(_CASE_RUNNERS):
        raise ValueError("Evaluation case definitions do not match available deterministic checks")
    return validated


def _run_cases(cases: list[dict[str, object]]) -> list[str]:
    failures: list[str] = []
    for case in cases:
        case_id = str(case["id"])
        runner = _CASE_RUNNERS[case_id]
        try:
            runner()
        except Exception as error:
            failures.append(f"{case_id}: {error}")
    return failures


def _check_invalid_tool_arguments() -> None:
    class Arguments(BaseModel):
        model_config = ConfigDict(extra="forbid")
        count: int = Field(gt=0)

    registry = ToolRegistry()
    invoked = False

    def handler(arguments):
        nonlocal invoked
        invoked = True
        return {"count": arguments.count}

    registry.register("count", Arguments, handler)
    try:
        registry.execute(ToolCall(name="count", arguments={"count": 0}))
    except ToolError:
        if invoked:
            raise AssertionError("Invalid arguments reached the tool handler")
        return
    raise AssertionError("Invalid tool arguments were accepted")


def _check_untrusted_instructions() -> None:
    for prompt in (PLANNER_SYSTEM_PROMPT, EXECUTION_SYSTEM_PROMPT):
        if "untrusted" not in prompt.lower():
            raise AssertionError("A system policy does not mark user-controlled data untrusted")


def _check_explicit_failure_reporting() -> None:
    try:
        TaskResult(
            task_id="eval-task",
            status="failed",
            summary="failed",
            error=None,
        )
    except ValidationError:
        return
    raise AssertionError("A failed task without an error reason was accepted")


def _check_inert_skill_proposals() -> None:
    with tempfile.TemporaryDirectory(prefix="tuminh-eval-") as temporary_directory:
        root = Path(temporary_directory)
        proposal = ReflectionStore(root / "memory" / "reflection.json").propose_skill(
            "eval-proposal",
            "Add a focused test.",
        )
        if not proposal.is_file():
            raise AssertionError("Proposal file was not created")
        if (root / "skills" / "approved" / "eval-proposal.md").exists():
            raise AssertionError("A proposal became active without approval")


_CASE_RUNNERS: dict[str, Callable[[], None]] = {
    "invalid-tool-arguments": _check_invalid_tool_arguments,
    "untrusted-task-instructions": _check_untrusted_instructions,
    "explicit-failure-reporting": _check_explicit_failure_reporting,
    "inert-skill-proposals": _check_inert_skill_proposals,
}


def main() -> int:
    return run_evals()


if __name__ == "__main__":
    raise SystemExit(main())
