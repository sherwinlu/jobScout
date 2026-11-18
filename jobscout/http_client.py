from __future__ import annotations

import ipaddress
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import zlib
from dataclasses import dataclass
from typing import Any, Callable

from jobscout.security import strip_header_crlf

_DEFAULT_MAX_RESPONSE_BYTES = 10 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024


class RobotsDisallowedError(RuntimeError):
    """Raised when a URL is blocked by robots.txt, crawl-delay, or URL safety policy."""

    def __init__(self, url: str, reason: str = "disallowed by robots.txt") -> None:
        super().__init__(f"{reason}: {url}")
        self.url = url
        self.reason = reason


@dataclass
class _RobotsPolicy:
    parser: urllib.robotparser.RobotFileParser | None = None
    allow_all: bool = False
    disallow_all: bool = False
    crawl_delay: float = 0.0
    rate_delay: float = 0.0
    reason: str = ""


def _is_disallowed_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        not ip.is_global
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def validate_public_url(url: str, allow_private_hosts: bool = False) -> str | None:
    """Return an error reason if URL uses an unsafe scheme or targets a private/local host."""
    try:
        parsed = urllib.parse.urlsplit(url)
    except Exception:
        return "malformed URL"

    if parsed.scheme.lower() not in {"http", "https"}:
        return f"disallowed URL scheme '{parsed.scheme}'"

    if parsed.username or parsed.password:
        return "URL must not contain embedded credentials"

    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        return "missing URL host"

    if allow_private_hosts:
        return None

    if host in {"localhost", "localhost.localdomain"} or host.endswith((".local", ".internal")):
        return f"disallowed private/local host '{host}'"

    try:
        ip = ipaddress.ip_address(host)
        if _is_disallowed_ip(ip):
            return f"disallowed non-public IP '{host}'"
        return None
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        for info in infos:
            addr = info[4][0].split("%")[0]
            ip = ipaddress.ip_address(addr)
            if _is_disallowed_ip(ip):
                return f"host '{host}' resolves to non-public IP '{addr}'"
    except socket.gaierror:
        pass

    return None


class PoliteHttpClient:
    """HTTP client enforcing URL safety, robots.txt, Crawl-delay, per-host rate limits, and 429 backoff."""

    def __init__(
        self,
        user_agent: str,
        timeout: int = 25,
        respect_robots: bool = True,
        min_delay_seconds: float = 0.5,
        max_crawl_delay_seconds: float = 10.0,
        max_retries: int = 1,
        contact_email: str = "",
        allow_private_hosts: bool = False,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
        opener: Callable[[urllib.request.Request, int], Any] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        clean_ua = strip_header_crlf(user_agent) or "job-scout/2.0"
        self.contact_email = strip_header_crlf(contact_email)
        if self.contact_email and self.contact_email not in clean_ua:
            clean_ua = (
                f"{clean_ua[:-1]}; mailto:{self.contact_email})"
                if clean_ua.endswith(")")
                else f"{clean_ua} (mailto:{self.contact_email})"
            )
        self.user_agent = clean_ua
        self.timeout = int(timeout)
        self.respect_robots = bool(respect_robots)
        self.min_delay_seconds = max(0.0, float(min_delay_seconds))
        self.max_crawl_delay_seconds = max(0.0, float(max_crawl_delay_seconds))
        self.max_retries = max(0, int(max_retries))
        self.allow_private_hosts = bool(allow_private_hosts)
        self.max_response_bytes = max(1024, int(max_response_bytes))
        self._opener = opener or self._build_safe_opener(self.allow_private_hosts)
        self.sleep_fn = sleep_fn
        self.clock_fn = clock_fn

        self._meta_lock = threading.Lock()
        self._robots_cache: dict[str, _RobotsPolicy] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._last_request: dict[str, float] = {}
        self._host_cooldown: dict[str, float] = {}

    @staticmethod
    def _build_safe_opener(allow_private_hosts: bool) -> Callable[[urllib.request.Request, int], Any]:
        class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                err = validate_public_url(newurl, allow_private_hosts=allow_private_hosts)
                if err:
                    raise RobotsDisallowedError(newurl, f"unsafe redirect target ({err})")
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        url_opener = urllib.request.build_opener(_SafeRedirectHandler())
        return lambda req, timeout: url_opener.open(req, timeout=timeout)

    def _lock_for(self, key: str) -> threading.Lock:
        with self._meta_lock:
            return self._locks.setdefault(key, threading.Lock())

    @staticmethod
    def _origin_and_host(url: str) -> tuple[str, str]:
        parsed = urllib.parse.urlsplit(url)
        return f"{parsed.scheme}://{parsed.netloc}", parsed.netloc.lower()

    def _build_headers(self, accept: str) -> dict[str, str]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": accept,
            "Accept-Encoding": "gzip, identity",
        }
        if self.contact_email:
            headers["From"] = self.contact_email
        return headers

    @staticmethod
    def _read_bounded_raw(resp: Any, max_bytes: int) -> bytes:
        chunks: list[bytes] = []
        total = 0
        try:
            first = resp.read(_READ_CHUNK_BYTES)
        except TypeError:
            # Handle minimal test mocks whose .read() accepts no size argument.
            raw = resp.read()
            if len(raw) > max_bytes:
                raise ValueError(f"HTTP response exceeded {max_bytes} byte limit")
            return raw

        while first:
            total += len(first)
            if total > max_bytes:
                raise ValueError(f"HTTP response exceeded {max_bytes} byte limit")
            chunks.append(first)
            first = resp.read(_READ_CHUNK_BYTES)
        return b"".join(chunks)

    @classmethod
    def _read_response_bytes(cls, resp: Any, max_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES) -> bytes:
        raw = cls._read_bounded_raw(resp, max_bytes)
        headers = getattr(resp, "headers", None)
        if headers and (headers.get("Content-Encoding") or "").lower() == "gzip":
            dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
            out = dec.decompress(raw, max_bytes + 1)
            if len(out) > max_bytes or dec.unconsumed_tail:
                raise ValueError(f"Decompressed HTTP response exceeded {max_bytes} byte limit")
            tail = dec.flush()
            if len(out) + len(tail) > max_bytes:
                raise ValueError(f"Decompressed HTTP response exceeded {max_bytes} byte limit")
            return out + tail
        return raw

    @staticmethod
    def _policy_for_status(status: int) -> _RobotsPolicy | None:
        if status in {404, 410}:
            return _RobotsPolicy(allow_all=True)
        if status in {401, 403, 429} or status >= 500:
            return _RobotsPolicy(disallow_all=True, reason=f"robots.txt returned HTTP {status}")
        return None

    def _fetch_robots_policy(self, origin: str, host: str) -> _RobotsPolicy:
        robots_url = f"{origin}/robots.txt"
        req = urllib.request.Request(
            robots_url, headers=self._build_headers("text/plain,*/*;q=0.5")
        )
        with self._lock_for(f"host:{host}"):
            self._enforce_host_delay_locked(host, min(self.min_delay_seconds, 0.25))
            try:
                with self._opener(req, min(self.timeout, 10)) as resp:
                    status = getattr(resp, "status", 200)
                    err_policy = self._policy_for_status(status)
                    if err_policy is not None:
                        return err_policy
                    body = self._read_response_bytes(resp, min(self.max_response_bytes, 1024 * 1024)).decode(
                        "utf-8", errors="replace"
                    )
            except urllib.error.HTTPError as exc:
                status = exc.code
                exc.close()
                return self._policy_for_status(status) or _RobotsPolicy(
                    disallow_all=True, reason=f"robots.txt returned HTTP {status}"
                )
            except Exception as exc:
                return _RobotsPolicy(disallow_all=True, reason=f"robots.txt unreachable ({exc})")

        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.parse(body.splitlines())

        crawl_delay = 0.0
        rate_delay = 0.0
        try:
            cd = rp.crawl_delay(self.user_agent)
            if cd is not None:
                crawl_delay = max(0.0, float(cd))
            rr = rp.request_rate(self.user_agent)
            if rr is not None and rr.requests > 0 and rr.seconds > 0:
                rate_delay = float(rr.seconds) / float(rr.requests)
        except Exception:
            pass

        return _RobotsPolicy(parser=rp, crawl_delay=crawl_delay, rate_delay=rate_delay)

    def _get_robots_policy(self, origin: str, host: str) -> _RobotsPolicy:
        with self._meta_lock:
            cached = self._robots_cache.get(origin)
        if cached is not None:
            return cached

        with self._lock_for(f"origin:{origin}"):
            with self._meta_lock:
                cached = self._robots_cache.get(origin)
            if cached is None:
                cached = self._fetch_robots_policy(origin, host)
                with self._meta_lock:
                    self._robots_cache[origin] = cached
            return cached

    def can_fetch(self, url: str) -> tuple[bool, float, str]:
        url_err = validate_public_url(url, allow_private_hosts=self.allow_private_hosts)
        if url_err:
            return False, 0.0, url_err

        if not self.respect_robots:
            return True, self.min_delay_seconds, ""

        origin, host = self._origin_and_host(url)
        policy = self._get_robots_policy(origin, host)
        if policy.disallow_all:
            return False, 0.0, policy.reason or "disallowed by robots.txt"
        if not policy.allow_all and policy.parser and not policy.parser.can_fetch(self.user_agent, url):
            return False, 0.0, "disallowed by robots.txt"

        delay = max(self.min_delay_seconds, policy.crawl_delay, policy.rate_delay)
        if delay > self.max_crawl_delay_seconds:
            return False, delay, f"Crawl-delay ({delay:.1f}s) exceeds cap ({self.max_crawl_delay_seconds:.1f}s)"
        return True, delay, ""

    def _enforce_host_delay_locked(self, host: str, delay: float) -> None:
        now = self.clock_fn()
        target_time = self._host_cooldown.get(host, 0.0)
        last = self._last_request.get(host)
        if last is not None:
            target_time = max(target_time, last + delay)
        if target_time > now:
            self.sleep_fn(target_time - now)
        self._last_request[host] = self.clock_fn()

    @staticmethod
    def _parse_retry_after(exc: urllib.error.HTTPError, default_delay: float) -> float:
        hdr = exc.headers.get("Retry-After") if exc.headers else None
        if hdr:
            try:
                return max(default_delay, float(hdr.strip()))
            except ValueError:
                pass
        return max(default_delay, 2.0)

    def fetch_bytes(self, url: str, accept: str = "*/*") -> bytes:
        allowed, delay, reason = self.can_fetch(url)
        if not allowed:
            raise RobotsDisallowedError(url, reason)

        _, host = self._origin_and_host(url)
        host_lock = self._lock_for(f"host:{host}")

        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, headers=self._build_headers(accept))
            with host_lock:
                self._enforce_host_delay_locked(host, delay)
                try:
                    with self._opener(req, self.timeout) as resp:
                        return self._read_response_bytes(resp, self.max_response_bytes)
                except urllib.error.HTTPError as exc:
                    if exc.code in {429, 503} and attempt < self.max_retries:
                        wait_s = self._parse_retry_after(exc, delay)
                        exc.close()
                        if wait_s <= self.max_crawl_delay_seconds:
                            self._host_cooldown[host] = self.clock_fn() + wait_s
                            continue
                    raise

        raise RuntimeError(f"Unreachable retry loop for {url}")
