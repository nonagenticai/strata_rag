# copied from kag@d653afba14d3f3c82ef84dea1a88f226d20e5726
# eval/judge.py, unchanged except imports
"""LLM-judge transport for the eval harness.

The system-under-test (KAG) answers via its production WM path inside
``/query/ask``. The JUDGE is a separate *measuring instrument*: it calls
inference-router directly (OpenAI-compatible ``/api/chat/completions``) with a
PINNED model, so judge scores are reproducible and the harness runs from
anywhere (m1 dev + CI runner) — WM itself is not exposed at the edge.

⚠️ That pin is a REQUEST name, and a request name does not pin the SERVING
model. Measured live 2026-09-16: the harness pins
``KAG_EVAL_JUDGE_MODEL=deepseek-v4-flash`` (``Makefile:62``), but the live
the live router declares, under provider ``pgx-local``
(an internal vLLM endpoint, ``model: local-llm``)::

    model_aliases:
      local-llm: local-llm
      deepseek-v4-flash: local-llm

and an explicit alias makes the caller auto-pin to THAT provider
(``_auto_pin_by_alias``), not merely fall through via ``default_chain``. So the
judge behind that name is the local vLLM, not DeepSeek (which is out of
credit). ``judge_correct`` is a PAIRED_METRIC and the H1 headline, so who
actually judged is load-bearing for every claim this harness makes.

⚠️⚠️ Read the RIGHT field. Probed live inside pod
``inference-router-6894d4dddc-w6b69`` (``POST /api/chat/completions`` with
``{"model": "deepseek-v4-flash", ...}``), the reply's top-level keys are
``['choices', 'created', 'id', 'model', 'object', 'provider_request_id',
'served_model', 'served_provider', 'usage']`` with::

    model           = 'deepseek-v4-flash'   # ECHOES THE REQUEST
    served_model    = 'local-llm'
    served_provider = 'pgx-local'

``model`` is the OpenAI-schema echo of what we asked for, so a divergence check
written against it can NEVER fire — it is vacuous by construction. The truth
lives in ``served_model`` / ``served_provider``, which are this router's
non-standard extension: they are recorded on :attr:`Judge.served_models` /
:attr:`Judge.served_providers`, must be treated as OPTIONAL (a plain
OpenAI-shaped backend omits them), and a divergence from the pin is warned
about once per distinct served identity.

inference-router advertises ``has_json_schema: false`` for the default model,
so we do NOT rely on server-side grammar: every judge prompt asks for a bare
JSON object and we parse it defensively (fenced-block strip + outermost-brace
scan). A malformed judge reply yields ``None`` and is treated as a skipped
score by callers — one bad judge call never aborts a run.

⚠️ ``None`` is *not* a score. A caller that folds it into a neutral verdict
(e.g. QFS's "tie") makes a total instrument failure byte-identical to a
genuine dead heat. Use :meth:`Judge.judge_json_strict`, which raises
:class:`JudgeUnavailable`, wherever the difference is scientifically
load-bearing; every failure is also counted on ``Judge.failures`` /
``Judge.last_error`` so a run has a cheap positive control on its own judge.

A lost verdict is expensive out of proportion to its rate. ``n_scored`` in
``eval/summary.py`` counts rows whose primary metric is numeric, and
``make bench-promote`` requires EVERY ladder rung at ``n_scored >= 50`` -- so
ONE dropped ``judge_correct`` is one row off a gate denominator. Measured on
two real runs: 1/250 and 3/250 judge rows lost, with the short rungs MOVING
between runs. :meth:`Judge.judge_json_strict` therefore retries
(``judge_retries``, default 2 == 3 attempts, exponential
``judge_retry_backoff_s`` between attempts and never before the first) and is
the single chokepoint: :meth:`judge_json` delegates to it, so all six call
sites inherit the retry and none of them implements its own.

Three things can cost a verdict, and the third was invisible until the
``require`` contract existed: (A) transport, (B) a reply no JSON could be
extracted from, and (C) a reply that parses into a dict LACKING the key the
caller needs -- ``eval/scorers/qa.py`` guards on ``"correct" in verdict``, so
before ``require`` a keyless dict silently produced an unscored row while
``failures`` stayed 0. Pass ``require=("correct",)`` and C is counted and
retried exactly like B.

MEASURED over 520 live judge calls (verbatim qa prompts, ``max_tokens=900``,
``temperature=0.0``): A occurred 0 times, C occurred 0 times, and B occurred 7
times (1.3%) -- every one of them a complete verdict missing only its closing
brace, concentrated 100% on ABSTENTION gradings (7/240 = 2.9%, 0/280
otherwise). Re-sending an exact failing input 30x returned 30/30 OK with 7
distinct ``reason`` strings despite ``temperature=0.0`` (vLLM batching is not
deterministic), so those failures are INDEPENDENT and retry alone would fix
them. :func:`extract_json` nevertheless repairs that brace OFFLINE, because a
repair is deterministic and costs no second LLM call; retry is the backstop for
everything the repair cannot reach. A and C keep their retry despite never
having been observed: 0/520 bounds them below ~0.6%, which is not the same as
proving them impossible, and the guard is free on a path that never fires.

⚠️ Retrying must never manufacture a verdict. After the last attempt the method
raises :class:`JudgeUnavailable` exactly as it always did; the fix lowers the
PROBABILITY of a loss and an unrecoverable row stays honestly unscored.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from typing import Any

import httpx

from config import BenchConfig

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

#: Recorded on :attr:`Judge.served_models` / :attr:`Judge.served_providers`
#: when a 200 reply carries no usable ``served_model`` / ``served_provider``.
#: Recording a SENTINEL rather than nothing is deliberate: "the backend did not
#: say who served this" is a distinct, reportable state from "the judge was
#: never called" (an EMPTY set), and the empty set must keep meaning only the
#: latter — otherwise an un-attributable run reads as a run that never ran.
UNREPORTED = "<unreported>"


#: How many closing braces :func:`extract_json` may append as a last-resort
#: repair. MEASURED over 520 live judge calls: 7 replies (1.3%) were
#: unparseable and ALL 7 were one `}` short. 2 is one more than every observed
#: case and keeps the repair a bounded, auditable step rather than an
#: append-until-it-parses loop.
_MAX_BRACE_REPAIR = 2


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a JSON object from an LLM reply."""
    if not text:
        return None
    # 1) fenced ```json ... ``` block, if present
    m = _FENCE.search(text)
    candidates = [m.group(1)] if m else []
    candidates.append(text)
    # 2) outermost {...} span
    for cand in candidates:
        start = cand.find("{")
        end = cand.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(cand[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                continue

    # 3) LAST RESORT: unbalanced-brace repair. Reached only after (1) and (2)
    # have both failed, so the 98.7% path is untouched.
    #
    # MEASURED over 520 live judge calls: 7 replies (1.3%) yielded no verdict,
    # and all 7 were the SAME shape - a complete, valid JSON object missing only
    # its final `}`, at finish_reason="stop" and 27-33 completion tokens (so NOT
    # a max_tokens truncation; the model just drops the brace). All 7 were
    # abstention gradings. One lost verdict is one row off `n_scored`, which
    # `make bench-promote` requires at >= 50 for EVERY ladder rung.
    #
    # Why step (2) cannot catch them: with no closing brace anywhere,
    # ``rfind("}")`` is -1, so the ``end > start`` guard is false and the
    # candidate is never even handed to ``json.loads``. But that -1 is a
    # SYMPTOM, not the defect - a reply like ``{"a": {"b": 1}`` has a real
    # ``rfind`` index and still fails, because the span it slices is unbalanced.
    # This repair is therefore written against unbalanced braces in general.
    #
    # WARNING / why this is safe by construction: ``json.loads`` is the
    # validator. A genuinely mangled reply (``{"correct": true, "reas``) still
    # fails to parse with any number of appended braces and still returns
    # ``None``. The repair can never manufacture a dict out of garbage - it can
    # only finish an object the model had otherwise already written. (A reply
    # cut cleanly at a key boundary could yield a SHORTER dict than intended;
    # the values it does carry are still verbatim what the model emitted, and
    # ``require`` in :meth:`Judge.judge_json_strict` is what checks that the
    # keys a caller actually needs are present.)
    for cand in candidates:
        start = cand.find("{")
        if start == -1:
            continue
        trimmed = cand[start:].rstrip()
        for extra in range(1, _MAX_BRACE_REPAIR + 1):
            try:
                obj = json.loads(trimmed + "}" * extra)
            except ValueError:
                continue
            # NON-EMPTY, unlike step (2). A bare ``{`` repairs to a perfectly
            # valid ``{}``, and that is precisely the "manufactured verdict"
            # this repair must not produce: qfs's original defect was
            # ``judge_json(...) or {}`` rendering a dead judge as a unanimous
            # tie, so handing an empty dict BACK from the parser would rebuild
            # that failure one layer down. A genuinely empty ``{}`` on the wire
            # still parses at step (2), which is unchanged.
            if isinstance(obj, dict) and obj:
                return obj
    return None


class JudgeUnavailable(RuntimeError):
    """The judge did not return a usable verdict.

    Raised for a transport/HTTP failure, a malformed response envelope, or a
    reply no JSON object could be extracted from. Distinct by construction from
    any verdict the judge actually expressed — including a genuine tie.
    """


# HTTP statuses worth another attempt. Everything else in 4xx is the peer
# telling us the REQUEST is wrong (bad token, bad body, wrong path) — retrying
# cannot change the outcome and costs `judge_retries` extra timeouts plus
# backoff on every one of ~250 calls. 408 and 429 are the two 4xx that mean
# "later", not "never". Mirrors `eval/kag_api.py`'s `KagError.retryable`, which
# already modelled this correctly on the ask side.
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _is_permanent(exc: BaseException) -> int | None:
    """The HTTP status if this failure can never succeed on retry, else None.

    A transport failure (connect reset, read timeout) carries NO status and is
    always retryable — `httpx.HTTPStatusError` is the only branch that can be
    permanent, and it is a SUBCLASS of `httpx.HTTPError`, which is why a
    wholesale `except httpx.HTTPError` retried 401s for as long as it did.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code not in _RETRYABLE_STATUS:
            return code
    return None


class Judge:
    """Thin OpenAI-compatible client against inference-router."""

    def __init__(self, cfg: BenchConfig, http: httpx.AsyncClient) -> None:
        self._cfg = cfg
        self._http = http
        # Run-level positive control: >0 means the instrument itself failed at
        # least once, whatever the scores downstream of it look like.
        #
        # ⚠️ ITS CONTRACT IS PER-CALL AND RETRIES DO NOT CHANGE IT. One judge
        # call that ultimately yields no verdict adds exactly 1, however many
        # attempts were spent inside it. That contract predates retries and is
        # pinned by name -- `test_the_lenient_judge_still_returns_none_but_
        # counts_the_failure` and `test_the_strict_judge_raises_instead_of_
        # returning_none` both assert `failures == 1` for ONE dead call, and
        # `test_a_healthy_judge_records_no_failures` asserts 0.
        #
        # An earlier draft of the retry redefined this to count ATTEMPTS, which
        # would have made those three tests read 3 and silently changed what
        # every historical reading of the number meant. Redefining a measurement
        # to fit a new feature is the defect this harness exists to catch; the
        # attempt count is genuinely useful, so it gets its OWN name below
        # rather than taking this one's.
        self.failures = 0
        # Every failed ATTEMPT, including ones a later attempt recovered from.
        # `failures` answers "how many rows did this run lose?"; this answers
        # "is the judge flaky?" -- a run with failed_attempts=7, failures=0 lost
        # nothing but was one bad draw away from losing a baseline. The two are
        # equal only when judge_retries == 0.
        #
        # NB nothing outside this module reads either counter today (grepped:
        # zero consumers in eval/report.py, eval/summary.py, eval/run_bench.py).
        # Surfacing them in the artifact is a separate change -- do not wire
        # them into the report from here.
        self.failed_attempts = 0
        self.last_error: str | None = None
        # Run-level PROVENANCE control, same idiom: the distinct identities that
        # actually answered. Sets, not scalars — the router can fail over
        # mid-run, and a run judged by two models is a fact a scalar cannot
        # hold. Empty until the first response envelope is seen.
        self.served_models: set[str] = set()
        self.served_providers: set[str] = set()
        # Identities already warned about. 250+ judge calls would flood the log
        # if we warned per call; warning per DISTINCT (model, provider) is
        # bounded by the one or two backends a run is actually routed to, so a
        # mid-run failover stays audible without becoming noise.
        self._mismatch_warned: set[tuple[str, str]] = set()

    @property
    def requested_model(self) -> str:
        """The model NAME this judge asks for. Not necessarily who answers."""
        return self._cfg.judge_model

    @property
    def served_model_diverged(self) -> bool:
        """True once any reply was SERVED by something other than the pin."""
        return any(m != self._cfg.judge_model for m in self.served_models)

    @staticmethod
    def make_http(cfg: BenchConfig) -> httpx.AsyncClient:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.require_license_token()}",
        }
        if cfg.judge_provider:
            headers["X-Provider"] = cfg.judge_provider
        return httpx.AsyncClient(
            base_url=cfg.judge_base_url,
            headers=headers,
            verify=cfg.verify_tls,
            timeout=cfg.judge_timeout_s,
        )

    def _record_provenance(self, data: dict[str, Any]) -> None:
        """Record who actually served this reply; warn once per identity.

        NB ``data["model"]`` is deliberately NOT consulted: it echoes the
        request (measured, see module docstring), so any check against it is
        vacuous.
        """
        raw_model = data.get("served_model")
        raw_provider = data.get("served_provider")
        served = raw_model if isinstance(raw_model, str) and raw_model else UNREPORTED
        provider = raw_provider if isinstance(raw_provider, str) and raw_provider else UNREPORTED
        self.served_models.add(served)
        self.served_providers.add(provider)

        identity = (served, provider)
        if served != self._cfg.judge_model and identity not in self._mismatch_warned:
            self._mismatch_warned.add(identity)
            print(
                f"[judge-model-mismatch] requested {self._cfg.judge_model!r} but "
                f"inference-router served {served!r} via provider {provider!r} — "
                "judge scores come from the SERVED model, not the pinned name "
                "(warned once per served identity).",
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
        }
        resp = await self._http.post("/api/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        # BEFORE reading the content: an envelope we can attribute but not parse
        # still tells us who served it, and that is exactly the case a
        # provenance record must not lose.
        if isinstance(data, dict):
            self._record_provenance(data)
        return data["choices"][0]["message"]["content"] or ""

    async def judge_json_strict(
        self, system: str, user: str, *, require: tuple[str, ...] | None = None
    ) -> dict[str, Any]:
        """Complete + parse a JSON verdict, or raise :class:`JudgeUnavailable`.

        The swallowing variant below cannot tell "the judge said nothing" from
        "the judge said something neutral"; a scorer that must not launder a
        401 into a scientific result calls this one.

        Retried up to ``cfg.judge_retries`` times (see the module docstring for
        why one lost verdict costs a promote). This is the ONE place retry
        lives: ``judge_json`` delegates here, so adding it at a call site would
        multiply, not add.

        ``require`` names the keys that make a parsed dict a USABLE verdict for
        this caller. A dict missing any of them is counted and retried exactly
        like an unparseable reply. It defaults to ``None`` — meaning "any dict
        will do" — precisely so the five callers that read their own keys
        (crag, trace x2, qfs x2, abstention_replay) are byte-for-byte
        unaffected; only ``eval/scorers/qa.py``, whose verdict feeds the
        ``n_scored`` the promote gate counts, declares a contract.

        Raises :class:`JudgeUnavailable` once every attempt is spent, so an
        unrecoverable row is still honestly lost rather than invented.
        """
        attempts = max(0, self._cfg.judge_retries) + 1
        detail = "judge unavailable: no attempt was made"
        cause: BaseException | None = None
        started = time.monotonic()
        budget = self._cfg.judge_call_budget_s

        for attempt in range(attempts):
            if attempt:
                # Stop BEFORE starting an attempt that cannot finish inside the
                # budget, rather than after overshooting it: a retry that is
                # going to be cut off mid-flight has already cost the full
                # timeout. See `judge_call_budget_s` for the arithmetic this
                # protects. A budget <= 0 disables the ceiling.
                elapsed = time.monotonic() - started
                projected = elapsed + self._cfg.judge_retry_backoff_s * (2 ** (attempt - 1))
                if budget > 0 and projected + self._cfg.judge_timeout_s > budget:
                    detail = (
                        f"judge call budget exhausted after {elapsed:.0f}s of "
                        f"{budget:.0f}s ({attempt} of {attempts} attempt(s) spent); "
                        f"last error: {self.last_error}"
                    )
                    break
                # Between attempts only — never before the first. A pre-sleep
                # would tax all ~250 calls of a healthy run for the ~1% that
                # fail. Exponential, mirroring ingest_submit_backoff_s, so a
                # router mid-rollover is not re-hit instantly.
                await asyncio.sleep(self._cfg.judge_retry_backoff_s * (2 ** (attempt - 1)))

            try:
                text = await self.complete(system, user)
            except (httpx.HTTPError, KeyError, IndexError) as exc:  # A. transport
                self.failed_attempts += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                permanent = _is_permanent(exc)
                if permanent is not None:
                    # Spending the remaining attempts here buys nothing and, at
                    # the deployed config, is what pushes a failing run past its
                    # own job timeout (see the module docstring arithmetic). The
                    # status goes in the message because "judge unavailable"
                    # alone sent a previous round hunting a transport fault.
                    detail = f"judge refused with HTTP {permanent} (permanent, not retried): {exc}"
                    self.last_error = detail
                    cause = exc
                    break
                detail = f"judge transport error: {exc}"
                cause = exc
                continue

            verdict = extract_json(text)
            if verdict is None:  # B. nothing parseable in the reply
                self.failed_attempts += 1
                self.last_error = f"unparseable judge reply: {text[:200]!r}"
                detail = self.last_error
                cause = None
                continue

            if require is not None:  # C. parsed, but not the verdict asked for
                missing = [key for key in require if key not in verdict]
                if missing:
                    self.failed_attempts += 1
                    self.last_error = (
                        f"judge reply missing required key(s) {missing}: {str(verdict)[:200]!r}"
                    )
                    detail = self.last_error
                    cause = None
                    continue

            return verdict

        # Every attempt spent. ONE call gave up => ONE row will carry no score,
        # which is exactly the per-CALL contract `failures` has always had.
        self.failures += 1
        if cause is not None:
            raise JudgeUnavailable(detail) from cause
        raise JudgeUnavailable(detail)

    async def judge_json(
        self, system: str, user: str, *, require: tuple[str, ...] | None = None
    ) -> dict[str, Any] | None:
        """Complete + parse a JSON verdict; ``None`` on any failure.

        Kept for scorers where a skipped score is the honest outcome. The
        failure is still COUNTED on ``self.failures`` / ``self.failed_attempts`` — a
        swallowed error that leaves no trace anywhere is how a dead judge
        becomes a published result.
        """
        try:
            return await self.judge_json_strict(system, user, require=require)
        except JudgeUnavailable:
            return None
