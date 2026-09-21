"""vLLM is launched with native tool calling on: the goal-graph synthesizer's rounds are a
400 without it."""
from __future__ import annotations

from lara.serve import runtimes as R


def _cmd(vllm=None):
    return R.VllmBackend().command("some/model", 8000, {"vllm": vllm or {}})


def test_tool_calling_flags_are_on_by_default():
    cmd = _cmd()
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == R.DEFAULT_TOOL_CALL_PARSER


def test_the_parser_is_configurable():
    cmd = _cmd({"tool_call_parser": "hermes"})
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"


def test_null_parser_omits_both_flags():
    cmd = _cmd({"tool_call_parser": None})
    assert "--enable-auto-tool-choice" not in cmd and "--tool-call-parser" not in cmd


def test_an_extra_args_parser_wins_and_is_not_doubled():
    cmd = _cmd({"extra_args": ["--tool-call-parser", "hermes"]})
    assert cmd.count("--tool-call-parser") == 1
