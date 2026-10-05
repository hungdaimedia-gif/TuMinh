import json

import pytest

from agent.models import TaskResult
from agent.task_queue import TaskQueue, TaskQueueError


def write_queue(path, tasks):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks), encoding="utf-8")


def task_data(task_id="one", state="pending"):
    return {
        "id": task_id,
        "title": f"Task {task_id}",
        "description": "Make one safe change.",
        "acceptance_criteria": ["A focused test passes."],
        "source": "local",
        "state": state,
    }


def test_completed_task_is_skipped_without_mutating_queue(tmp_path):
    queue_path = tmp_path / "tasks.json"
    state_path = tmp_path / "memory" / "task_state.json"
    write_queue(queue_path, [task_data()])
    original = queue_path.read_bytes()
    queue = TaskQueue(queue_path, state_path)

    task = queue.next_task()
    assert task is not None
    queue.save_result(
        task.id,
        TaskResult(
            task_id=task.id,
            status="succeeded",
            summary="done",
            changed_files=[],
            error=None,
        ),
    )

    assert queue.next_task() is None
    assert queue_path.read_bytes() == original


@pytest.mark.parametrize("status", ["succeeded", "failed"])
def test_any_terminal_result_prevents_reselection(tmp_path, status):
    queue_path = tmp_path / "tasks.json"
    state_path = tmp_path / "memory" / "task_state.json"
    write_queue(queue_path, [task_data()])
    queue = TaskQueue(queue_path, state_path)
    result = TaskResult(
        task_id="one",
        status=status,
        summary="finished",
        error="failed" if status == "failed" else None,
    )

    queue.save_result("one", result)

    assert queue.next_task() is None


def test_failed_task_is_not_retried_automatically(tmp_path):
    queue_path = tmp_path / "tasks.json"
    state_path = tmp_path / "memory" / "task_state.json"
    write_queue(queue_path, [task_data()])
    queue = TaskQueue(queue_path, state_path)
    queue.save_result(
        "one",
        TaskResult(task_id="one", status="failed", summary="error", error="test failed"),
    )

    assert queue.next_task() is None
    assert queue.is_processed("one") is True


def test_unknown_task_is_not_processed(tmp_path):
    queue = TaskQueue(tmp_path / "tasks.json", tmp_path / "state.json")

    assert queue.is_processed("missing") is False


def test_next_pending_task_is_normalized_and_skips_queue_terminal_state(tmp_path):
    queue_path = tmp_path / "tasks.json"
    write_queue(queue_path, [task_data("done", "succeeded"), task_data("next")])

    task = TaskQueue(queue_path, tmp_path / "state.json").next_task()

    assert task is not None
    assert task.id == "next"
    assert task.source == "local"


def test_malformed_queue_json_raises_typed_error(tmp_path):
    queue_path = tmp_path / "tasks.json"
    queue_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(TaskQueueError, match="queue"):
        TaskQueue(queue_path, tmp_path / "state.json").next_task()


def test_invalid_task_and_duplicate_ids_raise_typed_error(tmp_path):
    queue_path = tmp_path / "tasks.json"
    write_queue(queue_path, [{"id": "incomplete"}])
    queue = TaskQueue(queue_path, tmp_path / "state.json")
    with pytest.raises(TaskQueueError, match="task"):
        queue.next_task()

    write_queue(queue_path, [task_data(), task_data()])
    with pytest.raises(TaskQueueError, match="duplicate"):
        queue.next_task()


def test_result_for_different_task_id_is_rejected(tmp_path):
    queue = TaskQueue(tmp_path / "tasks.json", tmp_path / "state.json")

    with pytest.raises(TaskQueueError, match="match"):
        queue.save_result(
            "one",
            TaskResult(task_id="two", status="succeeded", summary="wrong task"),
        )


def test_runtime_state_is_written_as_valid_json_atomically(tmp_path, monkeypatch):
    import agent.task_queue as task_queue_module

    queue = TaskQueue(tmp_path / "tasks.json", tmp_path / "memory" / "task_state.json")
    real_replace = task_queue_module.os.replace
    replacements = []

    def checked_replace(source, destination):
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(task_queue_module.os, "replace", checked_replace)
    queue.save_result(
        "one",
        TaskResult(task_id="one", status="succeeded", summary="done"),
    )

    assert len(replacements) == 1
    assert replacements[0][1] == queue.state_path
    state = json.loads(queue.state_path.read_text(encoding="utf-8"))
    assert state["one"]["status"] == "succeeded"
    assert list(queue.state_path.parent.iterdir()) == [queue.state_path]


def test_malformed_runtime_state_raises_typed_error(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("{broken", encoding="utf-8")
    write_queue(tmp_path / "tasks.json", [task_data()])

    with pytest.raises(TaskQueueError, match="state"):
        TaskQueue(tmp_path / "tasks.json", state_path).next_task()


def test_failed_atomic_replace_preserves_prior_state_and_surfaces_error(tmp_path, monkeypatch):
    import agent.task_queue as task_queue_module

    state_path = tmp_path / "state.json"
    original = '{"existing":{"task_id":"existing","status":"succeeded","summary":"done","changed_files":[],"error":null}}\n'
    state_path.write_text(original, encoding="utf-8")
    queue = TaskQueue(tmp_path / "tasks.json", state_path)

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(task_queue_module.os, "replace", fail_replace)
    with pytest.raises(TaskQueueError, match="simulated replace failure"):
        queue.save_result(
            "next",
            TaskResult(task_id="next", status="succeeded", summary="done"),
        )

    assert state_path.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [state_path]
