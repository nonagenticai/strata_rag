"""OpenAI-vLLM transport for the copied KAG judge (``STRATA_BENCH_JUDGE_PROTOCOL=vllm``).

``judge.py`` stays a verbatim copy: this module subclasses :class:`judge.Judge` and overrides only
the wire call (:meth:`VllmJudge.complete`) and where provenance is read from. Retries, ``require``,
the JSON salvage and the failure counters are inherited unchanged.

Why this reaches the same instrument as KAG's baseline: KAG asked inference-router for an alias
that auto-pins to provider ``pgx-local`` (the platform's internal vLLM), model ``local-llm``.
That provider has ``force_chat_template_kwargs: true`` and the request default is
``enable_reasoning=False``, so the router forwards ``chat_template_kwargs: {enable_thinking:
false}``. Calling ``POST <vllm>/v1/chat/completions`` with ``model: local-llm`` and that same
``chat_template_kwargs`` (the shape strata's ``src/clients/vllm.rs`` uses), the same prompts,
``max_tokens`` and ``temperature``, is therefore the same model with the same template.

Provenance: a plain OpenAI/vLLM reply has no ``served_model`` / ``served_provider``. vLLM's
``model`` field is the served-model-name it answered under, and with no router in between there is
no alias layer, so it is recorded as the served model; the provider is recorded as
``vllm:<base_url>``. The caveat in ``judge.py`` (``model`` echoes the request) is about
inference-router's alias layer.
"""

from __future__ import annotations

import sys
from typing import Any

import httpx

from config import BenchConfig
from judge import UNREPORTED, Judge

CHAT_PATH = "/v1/chat/completions"


class VllmJudge(Judge):
    """:class:`Judge` speaking OpenAI-vLLM instead of inference-router."""

    @property
    def provider_label(self) -> str:
        return f"vllm:{self._cfg.judge_base_url}"

    def _record_provenance(self, data: dict[str, Any]) -> None:
        raw_model = data.get("model")
        served = raw_model if isinstance(raw_model, str) and raw_model else UNREPORTED
        provider = self.provider_label
        self.served_models.add(served)
        self.served_providers.add(provider)
        identity = (served, provider)
        if served != self._cfg.judge_model and identity not in self._mismatch_warned:
            self._mismatch_warned.add(identity)
            print(
                f"[judge-model-mismatch] requested {self._cfg.judge_model!r} but vLLM "
                f"reported model {served!r} (warned once per served identity).",
                file=sys.stderr,
                flush=True,
            )

    async def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        body = {
            "model": self._cfg.judge_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": self._cfg.judge_max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = await self._http.post(CHAT_PATH, json=body)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            self._record_provenance(data)
        return data["choices"][0]["message"]["content"] or ""


def make_judge(cfg: BenchConfig) -> tuple[Judge, httpx.AsyncClient]:
    """The judge for ``cfg.judge_protocol`` plus the http client the caller must close."""
    http = Judge.make_http(cfg)
    if cfg.judge_protocol == "vllm":
        return VllmJudge(cfg, http), http
    return Judge(cfg, http), http
