"""Chuyển đổi export thô từ Orca Desktop sang khuôn dạng TADP/1 JSONL.

Kịch bản này KHÔNG tự động đoán nội dung nhạy cảm — việc lọc secret/token đã
được `distill.importers.ExternalImporter` thực hiện ở bước `import-external`.
Script này chỉ làm nhiệm vụ "quy đổi khuôn dạng" (format transcoding).

Giả định đầu vào (--orca-export) là JSON với cấu trúc tối thiểu:

{
  "task_id": "orca-task-0001",
  "outcome": "succeeded" | "failed",
  "verification_passed": true,
  "eval_passed": true,
  "events": [
    {"role": "user", "content": "...", "timestamp": "2026-10-05T12:00:01Z"},
    {"role": "assistant", "content": "...", "timestamp": "2026-10-05T12:00:02Z"},
    {"role": "tool", "tool_name": "pytest", "content": "passed",
     "timestamp": "2026-10-05T12:00:03Z"}
  ]
}

Điều chỉnh hàm `_load_orca_export` nếu khuôn dạng xuất thực tế của Orca
Desktop khác với giả định trên.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROLE_TO_EVENT_TYPE = {
    "user": "user_message",
    "assistant": "assistant_message",
    "tool": "tool_message",
}


def _load_orca_export(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _build_event_record(task_id: str, sequence_number: int, raw_event: dict[str, Any]) -> dict[str, Any]:
    role = raw_event["role"]
    if role not in _ROLE_TO_EVENT_TYPE:
        raise ValueError(f"Unsupported role in Orca export: {role!r}")
    message: dict[str, Any] = {"role": role, "content": raw_event["content"]}
    if role == "tool":
        tool_name = raw_event.get("tool_name")
        if not tool_name:
            raise ValueError("Tool events require a 'tool_name' field")
        message["tool_name"] = tool_name
    return {
        "schema_version": 1,
        "task_id": task_id,
        "sequence_number": sequence_number,
        "event_type": _ROLE_TO_EVENT_TYPE[role],
        "timestamp": raw_event["timestamp"],
        "data": {"message": message},
    }


def convert(orca_export: dict[str, Any]) -> list[dict[str, Any]]:
    task_id = orca_export["task_id"]
    outcome = orca_export.get("outcome", "succeeded")
    if outcome not in {"succeeded", "failed"}:
        raise ValueError(f"Unsupported task_outcome: {outcome!r}")
    verification_passed = bool(orca_export.get("verification_passed", False))
    eval_passed = orca_export.get("eval_passed")

    header = {
        "record_type": "header",
        "schema_version": 1,
        "trajectory_id": task_id,
        "task_id": task_id,
        "task_outcome": outcome,
        "verification_passed": verification_passed,
        "eval_passed": eval_passed,
    }

    records: list[dict[str, Any]] = [header]
    for index, raw_event in enumerate(orca_export.get("events", []), start=1):
        records.append(_build_event_record(task_id, index, raw_event))

    if orca_export.get("verification_detail") is not None:
        records.append(
            {
                "schema_version": 1,
                "task_id": task_id,
                "sequence_number": len(records),
                "event_type": "verification",
                "timestamp": orca_export["verification_detail"].get(
                    "timestamp", orca_export.get("events", [{}])[-1].get("timestamp")
                ),
                "data": {"passed": verification_passed},
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orca-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    orca_export = _load_orca_export(arguments.orca_export)
    records = convert(orca_export)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False))
            stream.write("\n")

    print(f"Wrote {len(records)} TADP/1 record(s) to {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
