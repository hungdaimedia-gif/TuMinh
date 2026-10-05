import pytest
from pydantic import BaseModel, ConfigDict, Field

from agent.models import ToolCall
from tools.registry import ToolError, ToolRegistry


class ReadArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)


def test_registry_validates_arguments_before_dispatch():
    registry = ToolRegistry()
    calls = []
    registry.register("read_file", ReadArguments, lambda args: calls.append(args.path) or {"text": "ok"})

    result = registry.execute(
        ToolCall(name="read_file", arguments={"path": "README.md"})
    )

    assert result == {"text": "ok"}
    assert calls == ["README.md"]


def test_registry_rejects_unknown_tool():
    with pytest.raises(ToolError, match="Unknown tool"):
        ToolRegistry().execute(ToolCall(name="run_shell", arguments={}))


def test_registry_rejects_invalid_arguments():
    registry = ToolRegistry()
    registry.register("read_file", ReadArguments, lambda args: {"text": args.path})

    with pytest.raises(ToolError, match="arguments"):
        registry.execute(ToolCall(name="read_file", arguments={"path": ""}))
    with pytest.raises(ToolError, match="arguments"):
        registry.execute(
            ToolCall(name="read_file", arguments={"path": "file", "extra": True})
        )


def test_registry_surfaces_handler_exception():
    registry = ToolRegistry()

    def broken_handler(args):
        raise RuntimeError("disk is unavailable")

    registry.register("read_file", ReadArguments, broken_handler)
    with pytest.raises(ToolError, match="disk is unavailable") as raised:
        registry.execute(ToolCall(name="read_file", arguments={"path": "file"}))

    assert isinstance(raised.value.__cause__, RuntimeError)


def test_registry_requires_dictionary_output():
    registry = ToolRegistry()
    registry.register("read_file", ReadArguments, lambda args: "bad result")

    with pytest.raises(ToolError, match="JSON object"):
        registry.execute(ToolCall(name="read_file", arguments={"path": "file"}))
