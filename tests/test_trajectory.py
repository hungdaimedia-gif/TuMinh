import json

import pytest

from agent.trajectory import TrajectoryError, TrajectoryJournal


def test_append_and_read_preserve_schema_and_sequence_order(tmp_path):
    journal = TrajectoryJournal(tmp_path / "memory" / "trajectories")

    journal.append("task-1", "task", {"title": "Example"})
    journal.append("task-1", "verification", {"passed": True})

    events = journal.read("task-1")
    assert [event.sequence_number for event in events] == [1, 2]
    assert [event.event_type for event in events] == ["task", "verification"]
    assert all(event.schema_version == 1 for event in events)


def test_nested_secret_values_are_redacted(tmp_path):
    journal = TrajectoryJournal(
        tmp_path / "trajectories",
        secret_values=["private-token"],
    )

    journal.append(
        "task-1",
        "tool_result",
        {"result": {"stdout": "received private-token", "items": ["private-token"]}},
    )

    serialized = journal.path_for_task("task-1").read_text(encoding="utf-8")
    assert "private-token" not in serialized
    assert serialized.count("[REDACTED]") == 2


@pytest.mark.parametrize("task_id", ["../escape", "..", "folder/task", "C:\\outside"])
def test_unsafe_task_id_is_rejected(tmp_path, task_id):
    journal = TrajectoryJournal(tmp_path / "trajectories")

    with pytest.raises(TrajectoryError, match="task id"):
        journal.path_for_task(task_id)


def test_unknown_event_type_is_rejected(tmp_path):
    journal = TrajectoryJournal(tmp_path / "trajectories")

    with pytest.raises(TrajectoryError, match="event type"):
        journal.append("task-1", "unknown", {})


@pytest.mark.parametrize(
    "prohibited",
    [
        "thought",
        "thinking",
        "analysis",
        "system_prompt",
        "developer_prompt",
        "credential",
        "environment",
        "api_key",
    ],
)
def test_nested_private_fields_are_rejected(tmp_path, prohibited):
    journal = TrajectoryJournal(tmp_path / "trajectories")

    with pytest.raises(TrajectoryError, match="prohibited"):
        journal.append(
            "task-1",
            "tool_result",
            {"outer": [{"nested": {prohibited: "must not be saved"}}]},
        )

    assert not journal.path_for_task("task-1").exists()


def test_patch_event_contains_only_redacted_diff_text(tmp_path):
    journal = TrajectoryJournal(tmp_path / "trajectories", ["private-token"])

    journal.append(
        "task-1",
        "patch",
        {"diff": "+password=private-token\n"},
    )

    event = journal.read("task-1")[0]
    assert set(event.data) == {"diff"}
    assert "private-token" not in event.data["diff"]


def test_patch_event_rejects_non_diff_fields(tmp_path):
    journal = TrajectoryJournal(tmp_path / "trajectories")

    with pytest.raises(TrajectoryError, match="diff"):
        journal.append(
            "task-1",
            "patch",
            {"diff": "+line\n", "changed_files": ["file.txt"]},
        )


def test_oversized_patch_is_rejected_without_partial_append(tmp_path):
    journal = TrajectoryJournal(tmp_path / "trajectories")

    with pytest.raises(TrajectoryError, match="size"):
        journal.append("task-1", "patch", {"diff": "x" * (40 * 1024)})

    assert not journal.path_for_task("task-1").exists()


def test_malformed_existing_jsonl_raises_visible_error(tmp_path):
    journal = TrajectoryJournal(tmp_path / "trajectories")
    path = journal.path_for_task("task-1")
    path.parent.mkdir(parents=True)
    path.write_text("{broken\n", encoding="utf-8")

    with pytest.raises(TrajectoryError, match="trajectory"):
        journal.read("task-1")

    with pytest.raises(TrajectoryError, match="trajectory"):
        journal.append("task-1", "task", {"title": "no overwrite"})


def test_existing_jsonl_with_prohibited_fields_is_rejected(tmp_path):
    journal = TrajectoryJournal(tmp_path / "trajectories")
    path = journal.path_for_task("task-1")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "task-1",
                "sequence_number": 1,
                "event_type": "tool_result",
                "timestamp": "2026-10-05T12:00:00Z",
                "data": {"thinking": "private reasoning"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TrajectoryError, match="prohibited"):
        journal.read("task-1")


def test_valid_jsonl_records_include_required_fields(tmp_path):
    journal = TrajectoryJournal(tmp_path / "trajectories")
    journal.append("task-1", "task", {"title": "Example"})

    record = json.loads(journal.path_for_task("task-1").read_text(encoding="utf-8"))
    assert {
        "schema_version",
        "task_id",
        "sequence_number",
        "event_type",
        "timestamp",
        "data",
    } == set(record)


def test_github_task_id_is_mapped_to_safe_unique_filename(tmp_path):
    journal = TrajectoryJournal(tmp_path / "trajectories")

    assert journal.path_for_task("github:123").parent == tmp_path / "trajectories"
    assert journal.path_for_task("github:123").name != journal.path_for_task("github-123").name
