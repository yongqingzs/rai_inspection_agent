from pathlib import Path
import os
import importlib.util

from langchain_core.messages import AIMessage
from rai.messages import store_artifacts


def _load_tool_class():
    module_path = (
        Path(__file__).parents[1]
        / "rai_inspection_agent"
        / "tools"
        / "artifact_analysis.py"
    )
    spec = importlib.util.spec_from_file_location("artifact_analysis", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.AnalyzeArtifactImageTool


AnalyzeArtifactImageTool = _load_tool_class()


class _FakeVisionModel:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content="visible scene looks normal")


def test_analyze_artifact_image_uses_stored_image_and_returns_text(tmp_path: Path):
    artifact_root = tmp_path / "data" / "artifacts"
    store_artifacts(
        "call-1",
        [{"summary": "captured", "raw_images": ["iVBORw0KGgo="], "images": [], "audios": []}],
        db_path=artifact_root,
    )
    fake_model = _FakeVisionModel()
    tool = AnalyzeArtifactImageTool(
        artifact_root=str(artifact_root),
        llm=fake_model,
    )

    result = tool._run(tool_call_id="call-1", question="What is visible?")

    assert "tool_call_id=call-1" in result
    assert "visible scene looks normal" in result
    assert fake_model.messages is not None
    assert fake_model.messages[0].images == ["iVBORw0KGgo="]


def test_analyze_artifact_image_defaults_to_latest_artifact(tmp_path: Path):
    artifact_root = tmp_path / "data" / "artifacts"
    store_artifacts(
        "older-call",
        [{"summary": "captured", "raw_images": ["iVBORw0KGgo="], "images": [], "audios": []}],
        db_path=artifact_root,
    )
    store_artifacts(
        "newer-call",
        [{"summary": "captured", "raw_images": ["iVBORw0KGgo="], "images": [], "audios": []}],
        db_path=artifact_root,
    )
    os.utime(artifact_root / "newer-call" / "metadata.json", (2_000_000_000, 2_000_000_000))
    tool = AnalyzeArtifactImageTool(
        artifact_root=str(artifact_root),
        llm=_FakeVisionModel(),
    )

    result = tool._run()

    assert "tool_call_id=newer-call" in result
