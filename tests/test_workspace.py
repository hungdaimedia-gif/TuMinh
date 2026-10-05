import os

import pytest

from tools.workspace import Workspace, WorkspaceError


def test_workspace_reads_writes_and_lists_contained_files(tmp_path):
    workspace = Workspace(tmp_path)

    workspace.write_file("nested\\example.txt", "hello")

    assert workspace.read_file("nested/example.txt") == "hello"
    assert workspace.list_files() == ["nested/example.txt"]


def test_workspace_rejects_parent_traversal(tmp_path):
    workspace = Workspace(tmp_path)
    escape = tmp_path.parent / "escape.txt"

    with pytest.raises(WorkspaceError):
        workspace.write_file("..\\escape.txt", "no")

    assert not escape.exists()


def test_workspace_rejects_absolute_path_outside_root(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace(workspace_root)
    outside = tmp_path / "outside.txt"

    with pytest.raises(WorkspaceError):
        workspace.write_file(str(outside), "no")

    assert not outside.exists()


def test_workspace_rejects_symlink_escape(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "external"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")

    workspace = Workspace(root)
    with pytest.raises(WorkspaceError):
        workspace.write_file("external/escape.txt", "no")
    with pytest.raises(WorkspaceError):
        workspace.list_files("external")

    assert list(outside.iterdir()) == []


def test_workspace_rejects_symlinked_file_escape_on_read(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("private", encoding="utf-8")
    link = root / "secret.txt"
    try:
        os.symlink(outside, link)
    except OSError as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")

    with pytest.raises(WorkspaceError):
        Workspace(root).read_file("secret.txt")


def test_workspace_reports_missing_file(tmp_path):
    with pytest.raises(WorkspaceError, match="read"):
        Workspace(tmp_path).read_file("missing.txt")
