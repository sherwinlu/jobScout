from __future__ import annotations

import http.client
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from jobscout.security import is_allowed_api_key_env
from jobscout.storage import log

_MAX_LLM_RESPONSE_BYTES = 5 * 1024 * 1024


class LocalLLMError(RuntimeError):
    """Raised when the configured LLM(s) are unavailable or fail."""


LLMError = LocalLLMError


class _RetryableLLMError(LocalLLMError):
    """Transient connection or rate-limit/overload error where retrying the same LLM makes sense."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class _NonRetryableLLMError(LocalLLMError):
    """Permanent/unavailable error (e.g., server down, bad auth/model) that triggers backup failover."""


def extract_json_payload(text: str) -> Any:
    """Parse JSON from raw text or a markdown-fenced block."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty LLM response")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.I)
    if fence_match:
        return json.loads(fence_match.group(1).strip())

    starts = [i for i in (raw.find("{"), raw.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"No JSON found in response: {raw[:200]}")
    start = min(starts)
    closer = "}" if raw[start] == "{" else "]"
    end = raw.rfind(closer)
    if end <= start:
        raise ValueError(f"Unterminated JSON in response: {raw[:200]}")
    return json.loads(raw[start : end + 1])


_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "ollama": {"base_url": "http://localhost:11434", "model": "qwen2.5:7b", "api_key_env": ""},
    "llama": {"base_url": "http://localhost:11434", "model": "llama3.2", "api_key_env": ""},
    "openai_compat": {"base_url": "http://localhost:8080", "model": "default", "api_key_env": ""},
    "mlx": {"base_url": "http://localhost:8080", "model": "mlx-community/Qwen2.5-7B-Instruct-4bit", "api_key_env": ""},
    "mlx_lm": {"base_url": "http://localhost:8080", "model": "mlx-community/Qwen2.5-7B-Instruct-4bit", "api_key_env": ""},
    "lmstudio": {"base_url": "http://localhost:1234", "model": "default", "api_key_env": ""},
    "vllm": {"base_url": "http://localhost:8000", "model": "default", "api_key_env": ""},
    "llama_cpp": {"base_url": "http://localhost:8080", "model": "default", "api_key_env": ""},
    "openai": {"base_url": "https://api.openai.com", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
    "chatgpt": {"base_url": "https://api.openai.com", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com", "model": "gemini-2.5-flash", "api_key_env": "GEMINI_API_KEY"},
    "google": {"base_url": "https://generativelanguage.googleapis.com", "model": "gemini-2.5-flash", "api_key_env": "GEMINI_API_KEY"},
    "anthropic": {"base_url": "https://api.anthropic.com", "model": "claude-sonnet-4-20250514", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude": {"base_url": "https://api.anthropic.com", "model": "claude-sonnet-4-20250514", "api_key_env": "ANTHROPIC_API_KEY"},
}

_RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504, 529}


def _parse_retry_after(headers: Any) -> float | None:
    if not headers:
        return None
    raw = headers.get("Retry-After") if hasattr(headers, "get") else None
    if not raw:
        return None
    try:
        return max(0.0, min(30.0, float(str(raw).strip())))
    except ValueError:
        return None


class LocalLLMClient:
    """Zero-dependency HTTP client supporting local/cloud LLMs, retries, and ordered backup failover."""

    _OLLAMA_PROVIDERS = {"ollama", "llama"}
    _OPENAI_COMPAT_PROVIDERS = {"openai_compat", "mlx", "mlx_lm", "lmstudio", "vllm", "llama_cpp"}
    _OPENAI_CLOUD_PROVIDERS = {"openai", "chatgpt"}
    _GEMINI_PROVIDERS = {"gemini", "google"}
    _ANTHROPIC_PROVIDERS = {"anthropic", "claude"}

    def __init__(
        self,
        provider: str = "ollama",
        base_url: str = "",
        model: str = "",
        timeout: int = 60,
        api_key_env: str = "",
        backups: list["LocalLLMClient"] | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = (provider or "ollama").strip().lower()
        defaults = _PROVIDER_DEFAULTS.get(self.provider, _PROVIDER_DEFAULTS["ollama"])
        self.base_url = (base_url or defaults["base_url"]).rstrip("/")
        self.model = (model or defaults["model"]).strip()
        self.timeout = int(timeout or 60)
        self.api_key_env = (api_key_env or defaults["api_key_env"]).strip()
        self.backups: list[LocalLLMClient] = list(backups or [])
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.sleep_fn = sleep_fn
        self._available_cache: bool | None = None
        self.unavailable_reason: str = ""
        self._active_index: int = 0
        self._json_format_unsupported = False

    @classmethod
    def from_config(
        cls,
        llm_cfg: Any,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> "LocalLLMClient":
        def _get(obj: Any, key: str, default: Any = "") -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        timeout = int(_get(llm_cfg, "timeout_seconds", 60) or 60)
        max_retries = int(_get(llm_cfg, "max_retries", 2) if _get(llm_cfg, "max_retries", None) is not None else 2)
        backoff = float(
            _get(llm_cfg, "retry_backoff_seconds", 1.0)
            if _get(llm_cfg, "retry_backoff_seconds", None) is not None
            else 1.0
        )

        backup_clients: list[LocalLLMClient] = []
        for b in (_get(llm_cfg, "backups", None) or []):
            if isinstance(b, LocalLLMClient):
                backup_clients.append(b)
            elif isinstance(b, dict):
                backup_clients.append(
                    cls(
                        provider=b.get("provider", "ollama"),
                        base_url=b.get("base_url", ""),
                        model=b.get("model", ""),
                        timeout=int(b.get("timeout_seconds", timeout) or timeout),
                        api_key_env=b.get("api_key_env", ""),
                        max_retries=int(b.get("max_retries", max_retries) if b.get("max_retries") is not None else max_retries),
                        retry_backoff_seconds=float(
                            b.get("retry_backoff_seconds", backoff)
                            if b.get("retry_backoff_seconds") is not None
                            else backoff
                        ),
                        sleep_fn=sleep_fn,
                    )
                )

        return cls(
            provider=str(_get(llm_cfg, "provider", "ollama") or "ollama"),
            base_url=str(_get(llm_cfg, "base_url", "") or ""),
            model=str(_get(llm_cfg, "model", "") or ""),
            timeout=timeout,
            api_key_env=str(_get(llm_cfg, "api_key_env", "") or ""),
            backups=backup_clients,
            max_retries=max_retries,
            retry_backoff_seconds=backoff,
            sleep_fn=sleep_fn,
        )

    def _chain(self) -> list["LocalLLMClient"]:
        return [self, *self.backups]

    @property
    def active_client(self) -> "LocalLLMClient":
        chain = self._chain()
        if 0 <= self._active_index < len(chain):
            return chain[self._active_index]
        return self

    @property
    def active_model(self) -> str:
        return self.active_client.model

    @property
    def active_provider(self) -> str:
        return self.active_client.provider

    def _validate_base_url(self) -> str | None:
        try:
            parsed = urllib.parse.urlsplit(self.base_url)
        except Exception:
            return "malformed base_url"
        if parsed.username or parsed.password:
            return "base_url must not contain embedded credentials"
        scheme = parsed.scheme.lower()
        if self.is_cloud_provider and scheme != "https":
            return f"cloud provider '{self.provider}' requires an https:// base_url"
        if scheme not in {"http", "https"}:
            return f"unsupported base_url scheme '{scheme}'"
        if not parsed.hostname:
            return "missing host in base_url"
        return None

    def _resolve_api_key(self) -> str:
        if self.api_key_env and is_allowed_api_key_env(self.api_key_env):
            val = os.environ.get(self.api_key_env, "").strip()
            if val:
                return val
        if self.provider in self._GEMINI_PROVIDERS:
            return os.environ.get("GOOGLE_API_KEY", "").strip()
        return ""

    @property
    def is_cloud_provider(self) -> bool:
        return (
            self.provider in self._OPENAI_CLOUD_PROVIDERS
            or self.provider in self._GEMINI_PROVIDERS
            or self.provider in self._ANTHROPIC_PROVIDERS
        )

    def _is_single_available(self) -> bool:
        if self._available_cache is not None:
            return self._available_cache

        if self.provider == "none" or self.provider not in _PROVIDER_DEFAULTS:
            self.unavailable_reason = (
                "LLM provider is disabled"
                if self.provider == "none"
                else f"unsupported provider '{self.provider}'"
            )
            self._available_cache = False
            return False

        url_err = self._validate_base_url()
        if url_err:
            self.unavailable_reason = url_err
            self._available_cache = False
            return False

        if self.is_cloud_provider:
            if self.api_key_env and not is_allowed_api_key_env(self.api_key_env):
                self.unavailable_reason = (
                    f"disallowed api_key_env '{self.api_key_env}' "
                    "(must use an LLM/JobScout-namespaced key variable)"
                )
                self._available_cache = False
                return False
            if not self._resolve_api_key():
                env_hint = self.api_key_env or "API key env var"
                self.unavailable_reason = f"missing API key in environment variable {env_hint}"
                self._available_cache = False
                return False
            self._available_cache = True
            return True

        # Local server health & exact model check (never silently substitute models).
        path, list_key, name_key = (
            ("/api/tags", "models", "name")
            if self.provider in self._OLLAMA_PROVIDERS
            else ("/v1/models", "data", "id")
        )
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{self.base_url}{path}"), timeout=3) as resp:
                raw_bytes = self._read_bounded(resp)
                data = json.loads(raw_bytes.decode("utf-8"))
            names = [m.get(name_key, "") for m in data.get(list_key, []) if m.get(name_key)]
            if self.model not in names:
                avail = ", ".join(names[:5]) or "none"
                self.unavailable_reason = (
                    f"configured model '{self.model}' not found on {self.provider} (available: {avail})"
                )
                self._available_cache = False
                return False
            self._available_cache = True
            return True
        except urllib.error.HTTPError as exc:
            exc.close()
            self.unavailable_reason = f"HTTP {exc.code}"
        except Exception as exc:
            self.unavailable_reason = f"not reachable at {self.base_url} ({exc})"

        self._available_cache = False
        return False

    def is_available(self) -> bool:
        chain = self._chain()
        for idx in range(self._active_index, len(chain)):
            if chain[idx]._is_single_available():
                self._active_index = idx
                return True
        reasons = [f"{c.provider}({c.model}): {c.unavailable_reason or 'unavailable'}" for c in chain]
        self.unavailable_reason = "; ".join(reasons)
        return False

    def ensure_available(self) -> None:
        """Fail fast if all configured LLMs are offline, missing models, or unconfigured."""
        if not self.is_available():
            raise LocalLLMError(
                f"No configured LLM is available ({self.unavailable_reason}). "
                "Job aborted because LLM access is required."
            )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        expect_json: bool = False,
        temperature: float = 0.2,
    ) -> str:
        self.ensure_available()
        chain = self._chain()
        errors: list[str] = []

        idx = self._active_index
        while idx < len(chain):
            client = chain[idx]
            if not client._is_single_available():
                errors.append(f"{client.provider}: {client.unavailable_reason}")
                idx += 1
                continue

            try:
                result = client._chat_with_retries(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    expect_json=expect_json,
                    temperature=temperature,
                )
                self._active_index = idx
                return result
            except _RetryableLLMError as exc:
                # Transient error after exhausting retries on this LLM: do NOT fail over to backup.
                raise LocalLLMError(str(exc)) from exc
            except _NonRetryableLLMError as exc:
                client._available_cache = False
                client.unavailable_reason = str(exc)
                errors.append(f"{client.provider}: {exc}")
                idx += 1

        self.unavailable_reason = "; ".join(errors)
        raise LocalLLMError(f"All configured LLMs failed: {self.unavailable_reason}")

    def _chat_with_retries(
        self,
        system_prompt: str,
        user_prompt: str,
        expect_json: bool,
        temperature: float,
    ) -> str:
        last_exc: _RetryableLLMError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._chat_single(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    expect_json=expect_json,
                    temperature=temperature,
                )
            except _RetryableLLMError as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                delay = (
                    exc.retry_after
                    if exc.retry_after is not None
                    else self.retry_backoff_seconds * (2**attempt)
                )
                if delay > 0:
                    self.sleep_fn(delay)
        assert last_exc is not None
        raise last_exc

    def _chat_single(
        self,
        system_prompt: str,
        user_prompt: str,
        expect_json: bool,
        temperature: float,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.provider in self._OLLAMA_PROVIDERS:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            }
            if expect_json and not self._json_format_unsupported:
                payload["format"] = "json"
            try:
                raw = self._post_json(f"{self.base_url}/api/chat", payload)
            except _NonRetryableLLMError as exc:
                # Some builds (e.g. MLX-backed models) reject `format`. Retry without it;
                # JSON is now prompt-only (extract_json_payload parses bare/fenced JSON).
                if "format" not in payload or "structured output is unavailable" not in str(exc).lower():
                    raise
                if not self._json_format_unsupported:
                    log(
                        f"llm: {self.provider}/{self.model} rejects format=json; "
                        "retrying without it (JSON is prompt-enforced)"
                    )
                self._json_format_unsupported = True
                del payload["format"]
                raw = self._post_json(f"{self.base_url}/api/chat", payload)
            return ((raw.get("message") or {}).get("content") or "").strip()

        if self.provider in self._OPENAI_COMPAT_PROVIDERS or self.provider in self._OPENAI_CLOUD_PROVIDERS:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            headers: dict[str, str] = {}
            if self.provider in self._OPENAI_CLOUD_PROVIDERS:
                headers["Authorization"] = f"Bearer {self._resolve_api_key()}"
                if expect_json:
                    payload["response_format"] = {"type": "json_object"}
            raw = self._post_json(f"{self.base_url}/v1/chat/completions", payload, extra_headers=headers)
            choices = raw.get("choices") or []
            if not choices:
                raise _NonRetryableLLMError("Empty choices from LLM server")
            return ((choices[0].get("message") or {}).get("content") or "").strip()

        if self.provider in self._GEMINI_PROVIDERS:
            gen_cfg: dict[str, Any] = {"temperature": temperature}
            if expect_json:
                gen_cfg["responseMimeType"] = "application/json"
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": gen_cfg,
            }
            url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
            raw = self._post_json(url, payload, extra_headers={"x-goog-api-key": self._resolve_api_key()})
            candidates = raw.get("candidates") or []
            if not candidates:
                raise _NonRetryableLLMError("Empty candidates from Gemini API")
            parts = (candidates[0].get("content") or {}).get("parts") or []
            return "".join(str(p.get("text") or "") for p in parts).strip()

        if self.provider in self._ANTHROPIC_PROVIDERS:
            payload = {
                "model": self.model,
                "max_tokens": 2048,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            headers = {
                "x-api-key": self._resolve_api_key(),
                "anthropic-version": "2023-06-01",
            }
            raw = self._post_json(f"{self.base_url}/v1/messages", payload, extra_headers=headers)
            blocks = raw.get("content") or []
            return "".join(str(b.get("text") or "") for b in blocks if b.get("type") == "text").strip()

        raise _NonRetryableLLMError(f"Unsupported LLM provider: {self.provider}")

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> Any:
        text = self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expect_json=True,
            temperature=temperature,
        )
        try:
            return extract_json_payload(text)
        except Exception as exc:
            raise LocalLLMError(f"Failed to parse JSON from LLM: {exc}") from exc

    @staticmethod
    def _read_bounded(resp: Any, max_bytes: int = _MAX_LLM_RESPONSE_BYTES) -> bytes:
        try:
            raw = resp.read(max_bytes + 1)
        except TypeError:
            raw = resp.read()
        if len(raw) > max_bytes:
            raise _NonRetryableLLMError(f"LLM response exceeded {max_bytes} byte limit")
        return raw

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url_err = self._validate_base_url()
        if url_err:
            raise _NonRetryableLLMError(url_err)
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(self._read_bounded(resp).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="ignore")[:300]
            except Exception:
                pass
            retry_after = _parse_retry_after(exc.headers)
            code = exc.code
            exc.close()
            msg = f"Request to {url} failed (HTTP {code}): {detail}"
            if code in _RETRYABLE_HTTP_CODES:
                raise _RetryableLLMError(msg, retry_after=retry_after) from exc
            raise _NonRetryableLLMError(msg) from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            reason_str = str(reason).lower()
            if isinstance(reason, ConnectionRefusedError) or "connection refused" in reason_str:
                raise _NonRetryableLLMError(f"LLM server down at {url}: {reason}") from exc
            if isinstance(reason, (TimeoutError, socket.timeout, ConnectionResetError)) or any(
                tok in reason_str for tok in ("timed out", "reset", "temporarily unavailable", "eof")
            ):
                raise _RetryableLLMError(f"Transient connection error to {url}: {reason}") from exc
            raise _NonRetryableLLMError(f"Request to {url} failed: {exc}") from exc
        except (TimeoutError, socket.timeout, ConnectionResetError, http.client.RemoteDisconnected) as exc:
            raise _RetryableLLMError(f"Transient connection error to {url}: {exc}") from exc
        except ConnectionRefusedError as exc:
            raise _NonRetryableLLMError(f"LLM server down at {url}: {exc}") from exc


LLMClient = LocalLLMClient
