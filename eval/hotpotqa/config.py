# copied from kag@d653afba14d3f3c82ef84dea1a88f226d20e5726 eval/config.py, trimmed to
# the judge settings; env names renamed KAG_EVAL_JUDGE_* -> STRATA_BENCH_JUDGE_*.
"""Judge configuration for the strata HotpotQA benchmark.

Only the LLM-judge knobs of KAG's ``BenchConfig`` survive, with KAG's defaults, so the judge is
the same pinned instrument KAG's promoted baseline used (model ``local-llm``, ``max_tokens`` 900,
``temperature`` 0.0, 120 s per attempt, 2 retries, 300 s per-call budget). Every knob is
env-overridable with a ``STRATA_BENCH_JUDGE_*`` name.

The bearer licence token is read from a FILE, never from argv and never printed. The file uses
strata's own licences format (``KEY=value`` lines, ``#`` comments, optional ``export`` and quotes;
see ``src/config.rs`` ``parse_env_file``), so ``~/.config/strata-rag/licenses.env`` works as-is:

* ``STRATA_BENCH_JUDGE_TOKEN_FILE`` - the file (default: ``STRATA_LICENSES_FILE``, else
  ``~/.config/strata-rag/licenses.env``, the same lookup strata uses).
* ``STRATA_BENCH_JUDGE_TOKEN_KEY`` - which key to read (default ``VLLM_LICENSE_TOKEN``, the
  licence strata's generator already uses). A file holding one bare token line also works.

``STRATA_BENCH_JUDGE_PROTOCOL`` picks the transport: ``router`` (default, KAG's
``/api/chat/completions`` on inference-router) or ``vllm`` (OpenAI ``/v1/chat/completions`` straight
to vLLM, see ``judge_vllm.py``). ``STRATA_BENCH_JUDGE_URL`` defaults per protocol.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_JUDGE_URL = "https://api.nonagentic.ai/inference-router"
PUBLIC_JUDGE_URL = "https://api.nonagentic.ai/inference-router"
# The same vLLM the router's local provider serves (``local-llm``), via the public edge
# strata's own generator uses (``src/config.rs`` ``DEFAULT_VLLM_URL``). See ``judge_vllm.py``.
DEFAULT_VLLM_JUDGE_URL = "https://api.nonagentic.ai/vllm"
JUDGE_PROTOCOLS = ("router", "vllm")
DEFAULT_TOKEN_KEY = "VLLM_LICENSE_TOKEN"

_NO_TOKEN_MSG = (
    "no judge licence token found. Set STRATA_BENCH_JUDGE_TOKEN_FILE to a strata-format "
    "licences file (KEY=value lines) and STRATA_BENCH_JUDGE_TOKEN_KEY to the key to read "
    f"(default {DEFAULT_TOKEN_KEY}), or run with --no-judge."
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def parse_env_file(contents: str) -> dict[str, str]:
    """Python twin of strata's ``config::parse_env_file`` (same accepted syntax)."""
    out: dict[str, str] = {}
    for raw in contents.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            continue
        value = value.strip()
        for q in ('"', "'"):
            if len(value) >= 2 and value.startswith(q) and value.endswith(q):
                value = value[1:-1]
                break
        out[key] = value
    return out


def token_file_path() -> Path | None:
    for name in ("STRATA_BENCH_JUDGE_TOKEN_FILE", "STRATA_LICENSES_FILE"):
        val = os.environ.get(name, "").strip()
        if val:
            return Path(val).expanduser()
    home = os.environ.get("HOME", "").strip()
    return Path(home) / ".config/strata-rag/licenses.env" if home else None


def read_token(path: Path | None, key: str) -> str:
    """The token under ``key`` in ``path``; a single bare line is accepted as the token itself.

    Returns "" when nothing usable is found. Never logs or echoes the value.
    """
    if path is None or not path.is_file():
        return ""
    contents = path.read_text(encoding="utf-8")
    parsed = parse_env_file(contents)
    if parsed:
        return parsed.get(key, "").strip()
    lines = [ln.strip() for ln in contents.splitlines() if ln.strip() and not ln.startswith("#")]
    return lines[0] if len(lines) == 1 else ""


def _default_verify(url: str) -> bool:
    # A private host is normally served with a self-signed CA (KAG ran with verify off for the
    # same reason); the public edge has a real certificate and is verified.
    env = os.environ.get("STRATA_BENCH_JUDGE_VERIFY_TLS")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    host = urlparse(url).hostname or ""
    return not host.endswith(".internal")


def judge_protocol() -> str:
    proto = os.environ.get("STRATA_BENCH_JUDGE_PROTOCOL", "router").strip().lower() or "router"
    if proto not in JUDGE_PROTOCOLS:
        raise ValueError(
            f"STRATA_BENCH_JUDGE_PROTOCOL={proto!r}; expected one of {', '.join(JUDGE_PROTOCOLS)}"
        )
    return proto


def _default_judge_url(protocol: str) -> str:
    fallback = DEFAULT_VLLM_JUDGE_URL if protocol == "vllm" else DEFAULT_JUDGE_URL
    return os.environ.get("STRATA_BENCH_JUDGE_URL", "").strip() or fallback


@dataclass
class BenchConfig:
    judge_protocol: str = field(default_factory=judge_protocol)
    judge_base_url: str = ""
    judge_model: str = field(
        default_factory=lambda: os.environ.get("STRATA_BENCH_JUDGE_MODEL", "local-llm")
    )
    judge_provider: str = field(
        default_factory=lambda: os.environ.get("STRATA_BENCH_JUDGE_PROVIDER", "")
    )
    judge_max_tokens: int = field(
        default_factory=lambda: _env_int("STRATA_BENCH_JUDGE_MAX_TOKENS", 900)
    )
    judge_timeout_s: float = field(
        default_factory=lambda: _env_float("STRATA_BENCH_JUDGE_TIMEOUT_S", 120.0)
    )
    judge_retries: int = field(default_factory=lambda: _env_int("STRATA_BENCH_JUDGE_RETRIES", 2))
    judge_retry_backoff_s: float = field(
        default_factory=lambda: _env_float("STRATA_BENCH_JUDGE_RETRY_BACKOFF_S", 1.0)
    )
    judge_call_budget_s: float = field(
        default_factory=lambda: _env_float("STRATA_BENCH_JUDGE_CALL_BUDGET_S", 300.0)
    )
    token_key: str = field(
        default_factory=lambda: os.environ.get("STRATA_BENCH_JUDGE_TOKEN_KEY", DEFAULT_TOKEN_KEY)
    )
    token_path: Path | None = field(default_factory=token_file_path)
    verify_tls: bool = True

    def __post_init__(self) -> None:
        if self.judge_protocol not in JUDGE_PROTOCOLS:
            raise ValueError(f"judge_protocol {self.judge_protocol!r} not in {JUDGE_PROTOCOLS}")
        self.judge_base_url = (
            self.judge_base_url or _default_judge_url(self.judge_protocol)
        ).rstrip("/")
        self.verify_tls = _default_verify(self.judge_base_url)

    def require_license_token(self) -> str:
        """The judge bearer, or a loud failure. The value is never printed."""
        token = read_token(self.token_path, self.token_key)
        if not token:
            raise RuntimeError(_NO_TOKEN_MSG)
        return token

    def describe(self) -> dict[str, object]:
        """Provenance for meta.json: everything except the token."""
        return {
            "judge_protocol": self.judge_protocol,
            "judge_base_url": self.judge_base_url,
            "judge_model": self.judge_model,
            "judge_provider": self.judge_provider or None,
            "judge_max_tokens": self.judge_max_tokens,
            "judge_timeout_s": self.judge_timeout_s,
            "judge_retries": self.judge_retries,
            "judge_call_budget_s": self.judge_call_budget_s,
            "verify_tls": self.verify_tls,
            "token_file": str(self.token_path) if self.token_path else None,
            "token_key": self.token_key,
        }
