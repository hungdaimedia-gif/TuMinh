from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent.models import Task, TaskResult


class TaskQueueError(ValueError):
    """Raised when a task queue or its runtime state is invalid or unavailable."""


class TaskQueue:
    def __init__(self, queue_path: Path, state_path: Path):
        self.queue_path = Path(queue_path)
        self.state_path = Path(state_path)

    def next_task(self) -> Task | None:
        tasks = self._load_tasks()
        results = self._load_state()
        for task in tasks:
            if task.state in {"succeeded", "failed"}:
                continue
            result = results.get(task.id)
            if result is None or result.status not in {"succeeded", "failed"}:
                return task
        return None

    def save_result(self, task_id: str, result: TaskResult) -> None:
        if result.task_id != task_id:
            raise TaskQueueError("task_id must match the result task_id")

        state = self._load_state()
        state[task_id] = result
        temporary_path: Path | None = None
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.state_path.parent,
                prefix=f".{self.state_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(
                    {key: item.model_dump(mode="json") for key, item in state.items()},
                    temporary_file,
                    ensure_ascii=False,
                    indent=2,
                )
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.state_path)
        except OSError as error:
            raise TaskQueueError(f"Could not save task runtime state: {error}") from error
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

    def _load_tasks(self) -> list[Task]:
        try:
            raw_tasks = json.loads(self.queue_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TaskQueueError(f"Could not read task queue {self.queue_path}: {error}") from error
        if not isinstance(raw_tasks, list):
            raise TaskQueueError("Task queue root must be a JSON array")

        tasks: list[Task] = []
        seen_ids: set[str] = set()
        for index, raw_task in enumerate(raw_tasks):
            try:
                task = Task.model_validate(raw_task)
            except ValidationError as error:
                raise TaskQueueError(f"Invalid task at queue index {index}: {error}") from error
            if task.id in seen_ids:
                raise TaskQueueError(f"Task queue contains duplicate task id {task.id!r}")
            seen_ids.add(task.id)
            tasks.append(task)
        return tasks

    def _load_state(self) -> dict[str, TaskResult]:
        if not self.state_path.exists():
            return {}
        try:
            raw_state: Any = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TaskQueueError(
                f"Could not read task runtime state {self.state_path}: {error}"
            ) from error
        if not isinstance(raw_state, dict):
            raise TaskQueueError("Task runtime state root must be a JSON object")

        state: dict[str, TaskResult] = {}
        for task_id, raw_result in raw_state.items():
            try:
                result = TaskResult.model_validate(raw_result)
            except ValidationError as error:
                raise TaskQueueError(f"Invalid runtime result for task {task_id!r}: {error}") from error
            if task_id != result.task_id:
                raise TaskQueueError(
                    f"Runtime state key {task_id!r} does not match its result task_id"
                )
            state[task_id] = result
        return state
