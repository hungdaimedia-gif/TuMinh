from __future__ import annotations

from pathlib import Path


class WorkspaceError(ValueError):
    """Raised when a workspace file operation is invalid or unavailable."""


class Workspace:
    def __init__(self, root: Path):
        try:
            self.root = Path(root).expanduser().resolve(strict=True)
        except OSError as error:
            raise WorkspaceError(f"Workspace root is unavailable: {error}") from error
        if not self.root.is_dir():
            raise WorkspaceError(f"Workspace root is not a directory: {self.root}")

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"Cannot read non-file workspace path: {path}")
        try:
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise WorkspaceError(f"Could not read workspace file {path!r}: {error}") from error

    def write_file(self, path: str, content: str) -> None:
        target = self._resolve(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target = self._resolve(path)
            target.write_text(content, encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise WorkspaceError(f"Could not write workspace file {path!r}: {error}") from error

    def list_files(self, path: str = ".") -> list[str]:
        target = self._resolve(path)
        if not target.exists():
            raise WorkspaceError(f"Workspace path does not exist: {path}")
        if target.is_file():
            return [target.relative_to(self.root).as_posix()]
        if not target.is_dir():
            raise WorkspaceError(f"Workspace path is not a file or directory: {path}")

        files: list[str] = []
        try:
            for child in target.rglob("*"):
                resolved = self._resolve(str(child))
                if resolved.is_file():
                    files.append(resolved.relative_to(self.root).as_posix())
        except OSError as error:
            raise WorkspaceError(f"Could not list workspace path {path!r}: {error}") from error
        return sorted(set(files))

    def _resolve(self, path: str) -> Path:
        if not isinstance(path, str) or not path:
            raise WorkspaceError("Workspace path must be a non-empty string")
        candidate_path = Path(path)
        if ".." in candidate_path.parts:
            raise WorkspaceError(f"Parent traversal is not allowed: {path!r}")
        candidate = candidate_path if candidate_path.is_absolute() else self.root / candidate_path
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(self.root)
        except (OSError, ValueError) as error:
            raise WorkspaceError(f"Path escapes the workspace: {path!r}") from error
        return resolved
