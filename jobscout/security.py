from __future__ import annotations

import os
import re
import secrets
from pathlib import Path
from typing import Any

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.\-])?(?:\(\d{3}\)|\d{3})[\s.\-]\d{3}[\s.\-]\d{4}(?!\w)"
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_XML_INVALID_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_CRLF_RE = re.compile(r"[\r\n\x00]+")

_ATS_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,100}$")
_SAFE_API_KEY_ENV_RE = re.compile(
    r"^(?:(?:OPENAI|GEMINI|GOOGLE|ANTHROPIC|CLAUDE|JOBSCOUT|LLM)_[A-Z0-9_]+|[A-Z0-9_]+_LLM_(?:API_)?KEY)$"
)


def strip_header_crlf(text: str) -> str:
    """Remove CR/LF/NUL characters to prevent HTTP header injection."""
    return _CRLF_RE.sub(" ", str(text or "")).strip()


def strip_xml_control_chars(text: str) -> str:
    """Remove characters disallowed by XML 1.0 before serializing OOXML."""
    return _XML_INVALID_RE.sub("", str(text or ""))


def sanitize_untrusted_text(text: str, max_len: int | None = None) -> str:
    """Strip unsafe ASCII control characters from untrusted external/DOM text."""
    cleaned = _CONTROL_RE.sub(" ", str(text or "")).strip()
    if max_len is not None and len(cleaned) > max_len:
        return cleaned[:max_len]
    return cleaned


def wrap_untrusted(tag: str, text: str, max_len: int | None = None) -> str:
    """Wrap untrusted text in explicit XML-like delimiters to harden LLM prompts."""
    safe_tag = re.sub(r"[^a-zA-Z0-9_]", "_", tag)
    cleaned = sanitize_untrusted_text(text, max_len=max_len)
    cleaned = re.sub(rf"</\s*{re.escape(safe_tag)}\s*>", f"< /{safe_tag}>", cleaned, flags=re.I)
    return f"<{safe_tag}>\n{cleaned}\n</{safe_tag}>"


def _cand_field(candidate: Any, key: str) -> str:
    if candidate is None:
        return ""
    if isinstance(candidate, dict):
        return str(candidate.get(key) or "").strip()
    return str(getattr(candidate, key, "") or "").strip()


def redact_pii_for_llm(text: str, candidate: Any = None) -> str:
    """Scrub contact details and sensitive PII before sending text to any LLM."""
    out = str(text or "")
    for key, label in (
        ("email", "[EMAIL_REDACTED]"),
        ("phone", "[PHONE_REDACTED]"),
        ("desired_compensation", "[COMPENSATION_REDACTED]"),
        ("work_authorization", "[WORK_AUTH_REDACTED]"),
        ("require_sponsorship", "[SPONSORSHIP_REDACTED]"),
        ("notice_period", "[NOTICE_REDACTED]"),
    ):
        val = _cand_field(candidate, key)
        if len(val) >= 4:
            out = re.sub(re.escape(val), label, out, flags=re.I)

    out = _EMAIL_RE.sub("[EMAIL_REDACTED]", out)
    out = _PHONE_RE.sub("[PHONE_REDACTED]", out)
    out = _SSN_RE.sub("[ID_REDACTED]", out)
    return out


def is_valid_ats_slug(slug: str) -> bool:
    """Validate ATS board slug to prevent path traversal or query/fragment injection."""
    s = (slug or "").strip()
    return bool(s and ".." not in s and _ATS_SLUG_RE.match(s))


def is_allowed_api_key_env(env_name: str) -> bool:
    """Allow only LLM/JobScout-namespaced env var names so configs cannot read arbitrary secrets."""
    name = (env_name or "").strip()
    return bool(name and _SAFE_API_KEY_ENV_RE.match(name))


def ensure_private_dir(path: Path) -> Path:
    """Create directory with owner-only permissions (0700) on POSIX systems."""
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass
    return path


def harden_file_permissions(path: Path) -> None:
    """Restrict existing file to owner read/write (0600) on POSIX systems."""
    if os.name == "posix" and path.exists():
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def write_private_bytes(path: Path, data: bytes) -> Path:
    """Atomically write bytes with owner-only permissions (0600) from creation on POSIX."""
    ensure_private_dir(path.parent)
    if os.name != "posix":
        path.write_bytes(data)
        return path

    tmp_path = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(tmp_path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def write_private_text(path: Path, content: str, encoding: str = "utf-8") -> Path:
    """Write text to disk atomically with restrictive owner-only file and directory permissions."""
    return write_private_bytes(path, content.encode(encoding))
