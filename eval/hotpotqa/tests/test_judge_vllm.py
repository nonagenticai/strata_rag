"""The vLLM judge transport: same body as strata's vllm.rs, provenance from ``model``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import respx

from config import BenchConfig
from judge import Judge
from judge_vllm import VllmJudge, make_judge

VLLM = "https://vllm.test/vllm"


@pytest.fixture
def token_file(tmp_path: Path) -> Path:
    p = tmp_path / "licenses.env"
    p.write_text("VLLM_LICENSE_TOKEN=tok-secret\n")
    return p


def _cfg(monkeypatch: pytest.MonkeyPatch, token_file: Path, protocol: str | None) -> BenchConfig:
    monkeypatch.delenv("STRATA_BENCH_JUDGE_URL", raising=False)
    if protocol is None:
        monkeypatch.delenv("STRATA_BENCH_JUDGE_PROTOCOL", raising=False)
    else:
        monkeypatch.setenv("STRATA_BENCH_JUDGE_PROTOCOL", protocol)
    monkeypatch.setenv("STRATA_BENCH_JUDGE_TOKEN_FILE", str(token_file))
    return BenchConfig()


def test_router_stays_the_default(monkeypatch: pytest.MonkeyPatch, token_file: Path) -> None:
    cfg = _cfg(monkeypatch, token_file, None)
    assert cfg.judge_protocol == "router"
    assert cfg.judge_base_url == "https://api.nonagentic.ai/inference-router"
    judge, http = make_judge(cfg)
    assert type(judge) is Judge
    asyncio.run(http.aclose())


def test_vllm_protocol_defaults_to_the_vllm_edge(
    monkeypatch: pytest.MonkeyPatch, token_file: Path
) -> None:
    cfg = _cfg(monkeypatch, token_file, "vllm")
    assert cfg.judge_base_url == "https://api.nonagentic.ai/vllm"
    assert cfg.describe()["judge_protocol"] == "vllm"


def test_unknown_protocol_is_refused(monkeypatch: pytest.MonkeyPatch, token_file: Path) -> None:
    with pytest.raises(ValueError):
        _cfg(monkeypatch, token_file, "openai")


def test_vllm_request_shape_and_served_model(
    monkeypatch: pytest.MonkeyPatch, token_file: Path
) -> None:
    monkeypatch.setenv("STRATA_BENCH_JUDGE_URL", VLLM)
    monkeypatch.setenv("STRATA_BENCH_JUDGE_PROTOCOL", "vllm")
    monkeypatch.setenv("STRATA_BENCH_JUDGE_TOKEN_FILE", str(token_file))
    cfg = BenchConfig()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "local-llm",
                "choices": [{"message": {"content": '\n\n{"correct": true, "reason": "ok"}'}}],
            },
        )

    async def go() -> tuple[dict[str, object], Judge]:
        judge, http = make_judge(cfg)
        try:
            return await judge.judge_json_strict("SYS", "USER", require=("correct",)), judge
        finally:
            await http.aclose()

    with respx.mock(base_url=VLLM) as mock:
        mock.post("/v1/chat/completions").mock(side_effect=handler)
        verdict, judge = asyncio.run(go())

    assert isinstance(judge, VllmJudge)
    assert verdict == {"correct": True, "reason": "ok"}
    assert seen[0].headers["authorization"] == "Bearer tok-secret"
    assert json.loads(seen[0].content) == {
        "model": "local-llm",
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USER"},
        ],
        "temperature": 0.0,
        "max_tokens": 900,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert judge.served_models == {"local-llm"}
    assert judge.served_providers == {f"vllm:{VLLM}"}
    assert not judge.served_model_diverged


def test_vllm_403_is_permanent_and_not_retried(
    monkeypatch: pytest.MonkeyPatch, token_file: Path
) -> None:
    monkeypatch.setenv("STRATA_BENCH_JUDGE_URL", VLLM)
    monkeypatch.setenv("STRATA_BENCH_JUDGE_PROTOCOL", "vllm")
    monkeypatch.setenv("STRATA_BENCH_JUDGE_TOKEN_FILE", str(token_file))
    cfg = BenchConfig()

    async def go() -> Judge:
        judge, http = make_judge(cfg)
        try:
            assert await judge.judge_json("s", "u", require=("correct",)) is None
            return judge
        finally:
            await http.aclose()

    with respx.mock(base_url=VLLM) as mock:
        route = mock.post("/v1/chat/completions").respond(403, json={"error": "forbidden"})
        judge = asyncio.run(go())
    assert route.call_count == 1
    assert judge.failures == 1
    assert "HTTP 403" in (judge.last_error or "")
