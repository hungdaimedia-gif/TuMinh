from __future__ import annotations


import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


from pydantic import ValidationError


from agent.models import Reflection, VerificationResult




class ReflectionError(ValueError):
    """Raised when reflection data or a skill proposal cannot be handled safely."""




@dataclass(frozen=True, slots=True)
class HumanApproval:
    approved_by: str
    confirmed_at: datetime


    def __post_init__(self) -> None:
        if not self.approved_by.strip():
            raise ValueError("approved_by must not be empty")
        if self.confirmed_at.tzinfo is None or self.confirmed_at.utcoffset() is None:
            raise ValueError("confirmed_at must be timezone-aware")




class ReflectionStore:
    _SLUG_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
    # Matches one "stdout=<value>" / "stderr=<value>" span. The value runs up
    # to the next "stdout="/stderr=" marker or the end of the text, so quoted
    # values containing ";" are still captured correctly.
    _COMMAND_OUTPUT_PATTERN = re.compile(
        r"\b(?P<key>stdout|stderr)\s*=\s*(?P<value>.*?)"
        r"(?=(?:\s*;\s*(?:stdout|stderr)\s*=)|\s*$)",
        re.IGNORECASE | re.DOTALL,
    )


    def __init__(self, path: Path, secret_values: Sequence[str] = ()):
        self.path = Path(path)
        self.secret_values = tuple(
            sorted(
                {secret for secret in secret_values if secret},
                key=len,
                reverse=True,
            )
        )


    def append(self, reflection: Reflection) -> None:
        reflections = self.read_all()
        sanitized = Reflection.model_validate(
            {
                **reflection.model_dump(mode="python"),
                "task_id": self._sanitize_text(reflection.task_id),
                "failure_category": (
                    self._sanitize_text(reflection.failure_category)
                    if reflection.failure_category is not None
                    else None
                ),
                "lessons": [self._sanitize_text(lesson) for lesson in reflection.lessons],
            }
        )
        reflections.append(sanitized)
        serialized = [item.model_dump(mode="json") for item in reflections]
        self._write_atomic(self.path, serialized, replace=True)


    def read_all(self) -> list[Reflection]:
        if not self.path.exists():
            return []
        try:
            raw_reflections = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReflectionError(f"Could not read reflection file {self.path}: {error}") from error
        if not isinstance(raw_reflections, list):
            raise ReflectionError("Reflection file root must be a JSON array")


        reflections: list[Reflection] = []
        for index, raw_reflection in enumerate(raw_reflections):
            try:
                reflections.append(Reflection.model_validate(raw_reflection))
            except ValidationError as error:
                raise ReflectionError(
                    f"Invalid reflection at index {index}: {error}"
                ) from error
        return reflections


    def propose_skill(self, slug: str, content: str) -> Path:
        if not isinstance(slug, str) or not self._SLUG_PATTERN.fullmatch(slug):
            raise ReflectionError("Skill proposal slug must be a lowercase safe filename")
        if not isinstance(content, str) or not content.strip():
            raise ReflectionError("Skill proposal content must not be empty")
        sanitized_content = self._sanitize_text(content)
        root = self._repository_root()
        proposals_dir = self._contained_directory(root / "skills" / "proposals", root)
        proposal_path = proposals_dir / f"{slug}.md"
        if proposal_path.exists():
            raise ReflectionError(f"Skill proposal already exists: {slug}")
        self._write_atomic(proposal_path, sanitized_content, replace=False)
        return proposal_path


    def activate_proposal(
        self,
        path: Path,
        *,
        eval_result: VerificationResult,
        approval: HumanApproval,
    ) -> Path:
        if not isinstance(eval_result, VerificationResult) or not eval_result.passed:
            raise ReflectionError("A passing proposal evaluation is required")
        if not isinstance(approval, HumanApproval):
            raise ReflectionError("Explicit human approval is required")


        root = self._repository_root()
        proposals_dir = self._contained_directory(root / "skills" / "proposals", root)
        try:
            proposal_path = Path(path).resolve(strict=True)
            relative_path = proposal_path.relative_to(proposals_dir)
        except (OSError, ValueError) as error:
            raise ReflectionError("Skill path must identify a proposal under skills/proposals") from error
        if proposal_path.suffix != ".md" or not self._SLUG_PATTERN.fullmatch(proposal_path.stem):
            raise ReflectionError("Skill proposal filename is invalid")
        if len(relative_path.parts) != 1 or not proposal_path.is_file():
            raise ReflectionError("Skill proposal must be a file directly under skills/proposals")


        try:
            content = proposal_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ReflectionError(f"Could not read skill proposal: {error}") from error


        approved_dir = self._contained_directory(root / "skills" / "approved", root)
        active_path = approved_dir / proposal_path.name
        if active_path.exists():
            raise ReflectionError(f"Approved skill already exists: {proposal_path.stem}")
        self._write_atomic(active_path, content, replace=False)
        return active_path


    def _repository_root(self) -> Path:
        try:
            return self.path.parent.parent.resolve(strict=False)
        except OSError as error:
            raise ReflectionError(f"Could not resolve repository root: {error}") from error


    @staticmethod
    def _contained_directory(path: Path, root: Path) -> Path:
        try:
            path.mkdir(parents=True, exist_ok=True)
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise ReflectionError(f"Skill directory escapes the repository: {path}") from error
        if not resolved.is_dir():
            raise ReflectionError(f"Skill path is not a directory: {resolved}")
        return resolved


    def _sanitize_text(self, text: str) -> str:
        # Redact only the command-output spans themselves (e.g. the verifier's
        # "stdout='...'; stderr='...'" strings) instead of dropping the whole
        # lesson: the surrounding failure reason is the most useful part of a
        # reflection and must be kept.
        sanitized = self._COMMAND_OUTPUT_PATTERN.sub(
            lambda match: f"{match.group('key')}=[command output omitted]",
            text,
        )
        for secret in self.secret_values:
            sanitized = sanitized.replace(secret, "[REDACTED]")
        return sanitized


    @staticmethod
    def _write_atomic(path: Path, content: object, *, replace: bool) -> None:
        temporary_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                if isinstance(content, str):
                    temporary_file.write(content)
                else:
                    json.dump(content, temporary_file, ensure_ascii=False, indent=2)
                    temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            if replace:
                os.replace(temporary_path, path)
            else:
                if path.exists():
                    raise ReflectionError(f"Refusing to overwrite existing file: {path}")
                os.rename(temporary_path, path)
        except ReflectionError:
            raise
        except OSError as error:
            raise ReflectionError(f"Could not write {path}: {error}") from error
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass