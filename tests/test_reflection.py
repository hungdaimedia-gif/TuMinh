from datetime import UTC, datetime

import pytest

from agent.models import Reflection, VerificationResult
from agent.reflection import HumanApproval, ReflectionError, ReflectionStore


def test_append_and_read_round_trip_multiple_reflections(tmp_path):
    store = ReflectionStore(tmp_path / "memory" / "reflection.json")
    first = Reflection(task_id="one", outcome="succeeded", lessons=["focused test passed"])
    second = Reflection(task_id="two", outcome="failed", lessons=["tool was unavailable"])

    store.append(first)
    store.append(second)

    assert store.read_all() == [first, second]


def test_reflection_storage_redacts_configured_secrets(tmp_path):
    path = tmp_path / "memory" / "reflection.json"
    store = ReflectionStore(path, secret_values=["private-token", "another-secret"])
    store.append(
        Reflection(
            task_id="one",
            outcome="failed",
            lessons=["Request failed with private-token and another-secret"],
        )
    )

    contents = path.read_text(encoding="utf-8")
    assert "private-token" not in contents
    assert "another-secret" not in contents
    assert contents.count("[REDACTED]") == 2


def test_reflection_omits_full_command_output(tmp_path):
    path = tmp_path / "memory" / "reflection.json"
    store = ReflectionStore(path)
    store.append(
        Reflection(
            task_id="one",
            outcome="failed",
            lessons=["pytest failed; stdout='sensitive command output'; stderr='details'"],
        )
    )

    contents = path.read_text(encoding="utf-8")
    assert "sensitive command output" not in contents
    assert "details" not in contents
    assert "command output omitted" in contents


def test_malformed_reflection_file_raises_visible_error(tmp_path):
    path = tmp_path / "memory" / "reflection.json"
    path.parent.mkdir()
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ReflectionError, match="reflection"):
        ReflectionStore(path).read_all()


def test_malformed_reflections_are_not_overwritten(tmp_path):
    path = tmp_path / "memory" / "reflection.json"
    path.parent.mkdir()
    path.write_text("{broken", encoding="utf-8")
    store = ReflectionStore(path)

    with pytest.raises(ReflectionError):
        store.append(Reflection(task_id="one", outcome="succeeded"))

    assert path.read_text(encoding="utf-8") == "{broken"


def test_atomic_append_failure_preserves_prior_reflections(tmp_path, monkeypatch):
    import agent.reflection as reflection_module

    path = tmp_path / "memory" / "reflection.json"
    store = ReflectionStore(path)
    store.append(Reflection(task_id="one", outcome="succeeded"))
    original = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(reflection_module.os, "replace", fail_replace)
    with pytest.raises(ReflectionError, match="simulated replacement failure"):
        store.append(Reflection(task_id="two", outcome="succeeded"))

    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


def test_skill_proposal_is_written_under_inert_proposals_directory(tmp_path):
    store = ReflectionStore(tmp_path / "memory" / "reflection.json")

    proposal = store.propose_skill("safe-testing", "Prefer focused regression tests.")

    assert proposal == tmp_path / "skills" / "proposals" / "safe-testing.md"
    assert proposal.read_text(encoding="utf-8") == "Prefer focused regression tests."
    assert not (tmp_path / "skills" / "approved" / "safe-testing.md").exists()


@pytest.mark.parametrize("slug", ["../escape", "UPPER", "has spaces", ""])
def test_invalid_skill_slug_is_rejected(tmp_path, slug):
    store = ReflectionStore(tmp_path / "memory" / "reflection.json")

    with pytest.raises(ReflectionError, match="slug"):
        store.propose_skill(slug, "content")


def test_failed_eval_does_not_activate_proposal(tmp_path):
    store = ReflectionStore(tmp_path / "memory" / "reflection.json")
    proposal = store.propose_skill("safe-test", "Prefer focused regression tests.")
    approval = HumanApproval(approved_by="operator", confirmed_at=datetime.now(UTC))
    failed_eval = VerificationResult(passed=False, commands=[], failures=["eval failed"])

    with pytest.raises(ReflectionError, match="evaluation"):
        store.activate_proposal(proposal, eval_result=failed_eval, approval=approval)
    assert not (tmp_path / "skills" / "approved" / "safe-test.md").exists()


def test_skill_activation_requires_explicit_approval(tmp_path):
    store = ReflectionStore(tmp_path / "memory" / "reflection.json")
    proposal = store.propose_skill("safe-test", "Content.")

    with pytest.raises(ReflectionError, match="approval"):
        store.activate_proposal(
            proposal,
            eval_result=VerificationResult(passed=True, commands=[], failures=[]),
            approval=None,
        )


def test_valid_eval_and_approval_activate_only_contained_proposal(tmp_path):
    store = ReflectionStore(tmp_path / "memory" / "reflection.json")
    proposal = store.propose_skill("safe-test", "Prefer focused regression tests.")
    approval = HumanApproval(approved_by="operator", confirmed_at=datetime.now(UTC))
    passed_eval = VerificationResult(passed=True, commands=[], failures=[])

    active = store.activate_proposal(
        proposal,
        eval_result=passed_eval,
        approval=approval,
    )

    assert active == tmp_path / "skills" / "approved" / "safe-test.md"
    assert active.read_text(encoding="utf-8") == "Prefer focused regression tests."


def test_proposal_outside_proposals_directory_cannot_be_activated(tmp_path):
    store = ReflectionStore(tmp_path / "memory" / "reflection.json")
    outside = tmp_path / "skills" / "other.md"
    outside.parent.mkdir(parents=True)
    outside.write_text("not a proposal", encoding="utf-8")

    with pytest.raises(ReflectionError, match="proposal"):
        store.activate_proposal(
            outside,
            eval_result=VerificationResult(passed=True, commands=[], failures=[]),
            approval=HumanApproval(
                approved_by="operator",
                confirmed_at=datetime.now(UTC),
            ),
        )
