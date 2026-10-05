from __future__ import annotations

from agent.model_client import ModelClient
from agent.models import Task, TaskPlan
from agent.prompts import PLANNER_SYSTEM_PROMPT


class Planner:
    def __init__(self, model: ModelClient):
        self.model = model

    def create_plan(self, task: Task) -> TaskPlan:
        plan = self.model.complete_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_payload=task.model_dump_json(indent=2),
            response_type=TaskPlan,
        )
        if plan.task_id != task.id:
            raise ValueError(
                f"Planner returned task_id {plan.task_id!r} for task {task.id!r}"
            )
        return plan
