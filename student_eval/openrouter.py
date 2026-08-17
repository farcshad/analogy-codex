"""Minimal OpenRouter client with strict endpoint routing and robust parsing."""

from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempts: list[dict] | None = None,
        status_code: int | None = None,
        failure_type: str | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.attempts = attempts or []
        self.status_code = status_code
        self.failure_type = failure_type
        self.retryable = retryable


def _classify_http_error(status_code: int, error_body: str) -> str:
    lowered = error_body.lower()
    if status_code == 429:
        return "rate_limit"
    if (
        400 <= status_code < 500
        and status_code not in {401, 403, 429}
        and any(
        marker in lowered
        for marker in ("response_format", "json_schema", "structured output")
        )
    ):
        return "unsupported_response_format"
    if status_code == 404 and "no endpoints found that can handle" in lowered:
        return "endpoint_unavailable"
    if status_code in {500, 502, 503, 504}:
        return "temporary_provider_error"
    return "http_error"


def _content_as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content or "")


def parse_student_answer(text: str) -> dict[str, str]:
    """Parse an answer, repairing common provider-side JSON defects safely."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    candidates = [cleaned]
    object_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if object_match and object_match.group(0) != cleaned:
        candidates.append(object_match.group(0))

    for candidate in candidates:
        for method, strict in (("strict_json", True), ("control_character_repair", False)):
            try:
                payload = json.loads(candidate, strict=strict)
            except json.JSONDecodeError:
                continue
            reason = str(payload.get("reason") or payload.get("Reason") or "").strip()
            choice = str(payload.get("choice") or payload.get("Choice") or "").strip().upper()
            if re.fullmatch(r"[ABCD]", choice):
                return {
                    "reason": reason,
                    "choice": choice,
                    "parse_method": method,
                    "parse_repaired": method != "strict_json",
                }

    # Conservative fallback: recover only an explicit JSON-style choice field.
    choice_matches = list(
        re.finditer(r'(?i)["\']choice["\']\s*:\s*["\']([ABCD])["\']', cleaned)
    )
    if choice_matches:
        choice_match = choice_matches[-1]
        reason_match = re.match(
            r'(?is)^\s*\{\s*["\']reason["\']\s*:\s*["\'](.*)["\']\s*,\s*'
            r'["\']choice["\']\s*:',
            cleaned[: choice_match.start()] + cleaned[choice_match.start() :],
        )
        reason = reason_match.group(1).strip() if reason_match else ""
        return {
            "reason": reason,
            "choice": choice_match.group(1).upper(),
            "parse_method": "explicit_choice_field_repair",
            "parse_repaired": True,
        }

    raise ValueError("Response does not contain a complete explicit choice field")


def parse_cot_answer(text: str) -> dict[str, str]:
    """Parse the free-form answer produced by the paper's CoT baseline prompt."""
    try:
        return parse_student_answer(text)
    except ValueError:
        pass

    cleaned = text.strip()
    explicit = list(
        re.finditer(
            r"(?i)\b(?:final\s+answer|answer|choice|option)"
            r"(?:\s+is)?\s*[:=\-]?\s*(?:\*\*)?\(?([ABCD])\)?(?:\*\*)?\b",
            cleaned,
        )
    )
    if explicit:
        match = explicit[-1]
        reason = cleaned[: match.start()].strip()
        return {
            "reason": reason,
            "choice": match.group(1).upper(),
            "parse_method": "explicit_cot_answer",
            "parse_repaired": False,
        }

    terminal = re.search(
        r"(?i)(?:^|\n)\s*(?:\*\*)?\(?([ABCD])\)?(?:\*\*)?[\s.!]*$", cleaned
    )
    if terminal:
        return {
            "reason": cleaned[: terminal.start()].strip(),
            "choice": terminal.group(1).upper(),
            "parse_method": "terminal_cot_choice",
            "parse_repaired": False,
        }
    raise ValueError("CoT response does not contain an explicit final answer")


def parse_task_answer(text: str, condition_id: int | None) -> dict[str, str]:
    """Use the parser appropriate for a condition's required output format."""
    if condition_id == 20:
        return parse_cot_answer(text)
    return parse_student_answer(text)


def chat_completion(
    *,
    api_key: str,
    model: str,
    provider: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    reasoning_enabled: bool,
    reasoning_effort: str,
    timeout_seconds: float,
    max_retries: int,
    final_answer_retries: int,
    max_recovery_tokens: int,
    retry_rate_limits: bool = True,
    require_provider_parameters: bool = True,
    response_format_mode: str = "json_schema",
    condition_id: int | None = None,
) -> dict:
    """Call one endpoint and recover when reasoning consumes the output budget."""
    if max_recovery_tokens < max_tokens:
        raise ValueError("max_recovery_tokens must be >= max_tokens")

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "student_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "maxLength": 1500},
                    "choice": {"type": "string", "enum": ["A", "B", "C", "D"]},
                },
                "required": ["reason", "choice"],
                "additionalProperties": False,
            },
        },
    }
    if response_format_mode not in {"json_schema", "json_object", "text"}:
        raise ValueError(
            "response_format_mode must be 'json_schema', 'json_object', or 'text'"
        )
    budgets: list[int] = []
    for recovery_index in range(final_answer_retries + 1):
        budget = min(max_tokens * (2**recovery_index), max_recovery_tokens)
        if budget not in budgets:
            budgets.append(budget)

    generation_attempts: list[dict] = []
    for budget in budgets:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": budget,
            "reasoning": (
                {"effort": reasoning_effort, "exclude": True}
                if reasoning_enabled
                else {"enabled": False}
            ),
            "provider": {
                "only": [provider],
                "allow_fallbacks": False,
                "require_parameters": require_provider_parameters,
            },
        }
        # Appendix C.4's CoT baseline is deliberately free-form. Do not impose
        # provider-side JSON formatting, which would change the paper prompt.
        if condition_id == 20:
            pass
        elif response_format_mode == "json_schema":
            body["response_format"] = schema
        elif response_format_mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://local.scua-evaluation",
                "X-Title": "SCUA Student Model Evaluation",
            },
            method="POST",
        )

        payload = None
        latency = None
        for network_attempt in range(max_retries + 1):
            try:
                started = time.perf_counter()
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                latency = time.perf_counter() - started
                break
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", "replace")[:2000]
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
                failure_type = _classify_http_error(exc.code, error_body)
                # A caller with an ordered provider list should move to the
                # next provider immediately instead of retrying the same
                # rate-limited endpoint several times.
                if exc.code == 429 and not retry_rate_limits:
                    raise OpenRouterError(
                        f"HTTP {exc.code}: {error_body}",
                        attempts=generation_attempts,
                        status_code=exc.code,
                        failure_type=failure_type,
                        retryable=True,
                    ) from exc
                if not retryable or network_attempt == max_retries:
                    raise OpenRouterError(
                        f"HTTP {exc.code}: {error_body}",
                        attempts=generation_attempts,
                        status_code=exc.code,
                        failure_type=failure_type,
                        retryable=retryable,
                    ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if network_attempt == max_retries:
                    raise OpenRouterError(str(exc), attempts=generation_attempts) from exc
            time.sleep(min(30.0, (2**network_attempt) + random.random()))

        raw_text = ""
        try:
            message = payload["choices"][0]["message"]
            raw_text = _content_as_text(message.get("content"))
            parsed = parse_task_answer(raw_text, condition_id)
        except (ValueError, KeyError, TypeError) as exc:
            usage = (payload or {}).get("usage", {})
            generation_attempts.append(
                {
                    "status": "unparseable_final_answer",
                    "max_tokens": budget,
                    "reasoning_enabled": reasoning_enabled,
                    "raw_response": raw_text,
                    "provider": (payload or {}).get("provider"),
                    "response_id": (payload or {}).get("id"),
                    "usage": usage,
                    "latency_seconds": round(latency or 0.0, 3),
                }
            )
            if budget != budgets[-1]:
                continue
            raise OpenRouterError(
                "No parseable final answer after token-budget recovery",
                attempts=generation_attempts,
            ) from exc

        generation_attempts.append(
            {
                "status": "success",
                "max_tokens": budget,
                "reasoning_enabled": reasoning_enabled,
                "provider": payload.get("provider"),
                "response_id": payload.get("id"),
                "usage": payload.get("usage", {}),
                "latency_seconds": round(latency or 0.0, 3),
            }
        )
        return {
            "parsed": parsed,
            "raw_response": raw_text,
            "reasoning": message.get("reasoning"),
            "response_id": payload.get("id"),
            "response_model": payload.get("model"),
            "provider": payload.get("provider"),
            "usage": payload.get("usage", {}),
            "latency_seconds": round(latency or 0.0, 3),
            "attempts": generation_attempts,
            "recovered_after_retry": len(generation_attempts) > 1,
        }

    raise AssertionError("unreachable")
