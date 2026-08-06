from __future__ import annotations

import email.utils
import json
import mimetypes
import random
import re
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

RETRYABLE_HTTP_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
DEFAULT_MAX_ATTEMPTS = 6
DEFAULT_RETRY_DELAY_SECONDS = 1.0
MAX_RETRY_DELAY_SECONDS = 60.0
SENSITIVE_QUERY_KEYS = {"key", "api_key", "access_token", "token", "signature", "sig"}
ERROR_BODY_SNIPPET_CHARS = 500


class HttpRequestError(RuntimeError):
    """HTTP failure with a machine-readable status code."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def safe_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    redacted = [(key, "REDACTED" if key.lower() in SENSITIVE_QUERY_KEYS else value) for key, value in query]
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(redacted)))


def safe_error_body(raw_body: str) -> str:
    text = raw_body.replace("\r", "\\r").replace("\n", "\\n")
    for key in SENSITIVE_QUERY_KEYS:
        text = re.sub(
            rf'("{re.escape(key)}"\s*:\s*")[^"]+(")',
            rf"\1REDACTED\2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            rf"({re.escape(key)}=)[^&\s]+",
            rf"\1REDACTED",
            text,
            flags=re.IGNORECASE,
        )
    if len(text) > ERROR_BODY_SNIPPET_CHARS:
        return text[:ERROR_BODY_SNIPPET_CHARS] + "..."
    return text


def request_json(
    method: str,
    url: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retryable_empty_error_status_codes: set[int] | None = None,
) -> Any:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Accept": "application/json",
            "User-Agent": "voice-code-bench/1.0",
            **(headers or {}),
        },
    )
    raw = ""
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            retryable_empty_statuses = retryable_empty_error_status_codes or set()
            is_retryable = exc.code in RETRYABLE_HTTP_STATUS_CODES or (
                exc.code in retryable_empty_statuses and not error_body.strip()
            )
            if is_retryable and attempt < max_attempts:
                sleep_before_retry(method, url, attempt, max_attempts, exc.code, exc.headers)
                continue
            snippet = safe_error_body(error_body)
            body_detail = f": {snippet}" if snippet else ""
            raise HttpRequestError(
                f"{method} {safe_url(url)} failed HTTP {exc.code}{body_detail}",
                status_code=exc.code,
                body=error_body,
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt < max_attempts:
                sleep_before_retry(method, url, attempt, max_attempts, None, None)
                continue
            raise HttpRequestError(f"{method} {safe_url(url)} failed: {exc}") from exc
    if not raw:
        return {}
    return json.loads(raw)


def request_bytes(
    method: str,
    url: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> bytes:
    """Issue an HTTP request and return the response without JSON decoding."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"User-Agent": "voice-code-bench/1.0", **(headers or {})},
    )
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code in RETRYABLE_HTTP_STATUS_CODES and attempt < max_attempts:
                sleep_before_retry(method, url, attempt, max_attempts, exc.code, exc.headers)
                continue
            snippet = safe_error_body(error_body)
            detail = f": {snippet}" if snippet else ""
            raise HttpRequestError(
                f"{method} {safe_url(url)} failed HTTP {exc.code}{detail}",
                status_code=exc.code,
                body=error_body,
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt < max_attempts:
                sleep_before_retry(method, url, attempt, max_attempts, None, None)
                continue
            raise HttpRequestError(f"{method} {safe_url(url)} failed: {exc}") from exc
    raise AssertionError("unreachable")


def get_json(url: str, headers: dict[str, str] | None = None, **kwargs: Any) -> Any:
    return request_json("GET", url, headers=headers, **kwargs)


def get_bytes(url: str, headers: dict[str, str] | None = None, **kwargs: Any) -> bytes:
    return request_bytes("GET", url, headers=headers, **kwargs)


def post_json(
    url: str,
    payload: Any,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retryable_empty_error_status_codes: set[int] | None = None,
) -> Any:
    return request_json(
        "POST",
        url,
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json", **(headers or {})},
        timeout=timeout,
        max_attempts=max_attempts,
        retryable_empty_error_status_codes=retryable_empty_error_status_codes,
    )


def post_multipart_json(
    url: str,
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
    headers: dict[str, str] | None = None,
    *,
    file_content_type: str | None = None,
    timeout: float = 120.0,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Any:
    body, content_type = multipart_body(fields, file_field, file_path, file_content_type)
    return request_json(
        "POST",
        url,
        body,
        {"Content-Type": content_type, **(headers or {})},
        timeout=timeout,
        max_attempts=max_attempts,
    )


def sleep_before_retry(
    method: str,
    url: str,
    attempt: int,
    max_attempts: int,
    status_code: int | None,
    response_headers: Any,
) -> None:
    delay = retry_delay_seconds(attempt, response_headers)
    status = f" HTTP {status_code}" if status_code is not None else ""
    print(
        f"{method} {safe_url(url)} failed{status}; retrying in {delay:.1f}s "
        f"(attempt {attempt + 1}/{max_attempts})",
        file=sys.stderr,
        flush=True,
    )
    time.sleep(delay)


def retry_delay_seconds(attempt: int, response_headers: Any) -> float:
    retry_after = parse_retry_after(response_headers)
    if retry_after is not None:
        return min(retry_after, MAX_RETRY_DELAY_SECONDS)
    exponential_delay = DEFAULT_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, DEFAULT_RETRY_DELAY_SECONDS)
    return min(exponential_delay + jitter, MAX_RETRY_DELAY_SECONDS)


def parse_retry_after(response_headers: Any) -> float | None:
    if response_headers is None:
        return None
    raw_value = response_headers.get("Retry-After")
    if raw_value is None:
        return None
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        try:
            parsed_date = email.utils.parsedate_to_datetime(raw_value)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if parsed_date is None:
            return None
        return max(0.0, parsed_date.timestamp() - time.time())


def multipart_body(fields: dict[str, str], file_field: str, file_path: Path, content_type: str | None = None) -> tuple[bytes, str]:
    boundary = "----voice-code-bench-" + secrets.token_hex(16)
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    guessed_type = content_type or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode("utf-8"),
            f"Content-Type: {guessed_type}\r\n\r\n".encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
