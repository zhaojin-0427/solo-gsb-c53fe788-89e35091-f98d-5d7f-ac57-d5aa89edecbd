"""Signed cursors for object-index listings.

The first ``GET /object-index`` request freezes one listing as a
ten-minute *snapshot*: the current revisions matching its prefix, in
ascending path order, with revision number, strong ETag, media type and
byte length.  Every subsequent page is addressed by an opaque *cursor*:
a tamper-evident token binding the snapshot id, the prefix, the page size
and the offset to resume from.

Integrity is provided by HMAC-SHA-256 keyed with a server-side secret
stored in the ``service_meta`` table (so tokens survive restarts):

* re-issuing a cursor for the same position always yields the same token;
* editing the offset, prefix, limit or snapshot id invalidates the MAC,
  so a tampered cursor is answered 400 -- the page it claims can never be
  served from a different snapshot;
* a validly signed cursor whose snapshot has expired (or was swept) is
  answered 410: the snapshot, not the signature, is gone.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass

CURSOR_VERSION = 1
_MAC_LEN = 18  # bytes of HMAC-SHA-256 kept in the token


@dataclass(frozen=True, slots=True)
class IndexCursor:
    """Decoded object-index cursor payload.

    ``offset`` is the ordinal of the first entry the next page must
    return (0 for the cursor handed back with the first page).
    """

    snapshot_id: str
    prefix: str
    limit: int
    offset: int


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(token: str) -> bytes | None:
    try:
        padded = token + "=" * (-len(token) % 4)
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except Exception:
        return None


def encode_index_cursor(secret: str, cursor: IndexCursor) -> str:
    """Serialise and sign a cursor; deterministic for the same input."""
    payload = json.dumps(
        {
            "v": CURSOR_VERSION,
            "snap": cursor.snapshot_id,
            "pre": cursor.prefix,
            "lim": cursor.limit,
            "off": cursor.offset,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    mac = hmac.new(secret.encode("ascii"), payload, hashlib.sha256).digest()
    return _b64encode(payload) + "." + _b64encode(mac[:_MAC_LEN])


def decode_index_cursor(secret: str, token: str) -> IndexCursor | None:
    """Verify and decode a cursor token; ``None`` when invalid."""
    if not token or "." not in token:
        return None
    payload_b64, mac_b64 = token.rsplit(".", 1)
    payload = _b64decode(payload_b64)
    mac = _b64decode(mac_b64)
    if payload is None or mac is None:
        return None
    expected = hmac.new(
        secret.encode("ascii"), payload, hashlib.sha256
    ).digest()[:_MAC_LEN]
    if not hmac.compare_digest(mac, expected):
        return None
    try:
        data = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("v") != CURSOR_VERSION:
        return None
    try:
        limit = data["lim"]
        offset = data["off"]
        if not _is_positive_int(limit) or not _is_nonneg_int(offset):
            return None
        return IndexCursor(
            snapshot_id=_checked_str(data["snap"]),
            prefix=_checked_str(data["pre"]),
            limit=limit,
            offset=offset,
        )
    except (KeyError, TypeError):
        return None


def _is_positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _is_nonneg_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _checked_str(value) -> str:
    if not isinstance(value, str):
        raise TypeError("cursor field must be a string")
    return value


def prefix_upper_bound(prefix: str) -> str | None:
    """Smallest encodable string strictly greater than every string
    starting with ``prefix`` (for a half-open index range scan), or
    ``None`` when no such finite string exists (``prefix`` is all
    U+10FFFF).

    Stored paths are valid UTF-8 ordered bytewise with ``COLLATE "C"``,
    which matches code-point order.  The rightmost incrementable code
    point is advanced by one; landing in the surrogate gap (unencodable)
    jumps over the whole gap to U+E000, and U+10FFFF carries one
    position left.
    """
    chars = list(prefix)
    for i in range(len(chars) - 1, -1, -1):
        code = ord(chars[i])
        if code >= 0x10FFFF:
            continue
        nxt = code + 1
        if 0xD800 <= nxt <= 0xDFFF:
            nxt = 0xE000
        chars[i] = chr(nxt)
        return "".join(chars[: i + 1])
    return None
