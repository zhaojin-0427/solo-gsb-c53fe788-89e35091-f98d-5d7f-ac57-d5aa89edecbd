"""Unit tests for object-index cursors and prefix bounds."""
from dataclasses import replace

from app.object_index import (
    IndexCursor,
    decode_index_cursor,
    encode_index_cursor,
    prefix_upper_bound,
)

SECRET = "00" * 32
CURSOR = IndexCursor(
    snapshot_id="s" * 32,
    prefix="dir/",
    limit=50,
    offset=100,
)


def test_cursor_roundtrip():
    token = encode_index_cursor(SECRET, CURSOR)
    assert decode_index_cursor(SECRET, token) == CURSOR


def test_cursor_roundtrip_empty_prefix():
    cur = replace(CURSOR, prefix="", offset=0)
    assert decode_index_cursor(SECRET, encode_index_cursor(SECRET, cur)) == cur


def test_cursor_is_deterministic():
    assert encode_index_cursor(SECRET, CURSOR) == encode_index_cursor(
        SECRET, CURSOR
    )


def test_cursor_offsets_differ():
    assert encode_index_cursor(
        SECRET, replace(CURSOR, offset=150)
    ) != encode_index_cursor(SECRET, CURSOR)


def test_cursor_rejects_wrong_secret():
    token = encode_index_cursor(SECRET, CURSOR)
    assert decode_index_cursor("ff" * 32, token) is None


def test_cursor_rejects_tampering():
    token = encode_index_cursor(SECRET, CURSOR)
    payload, mac = token.rsplit(".", 1)
    flip = "A" if payload[5] != "A" else "B"
    assert (
        decode_index_cursor(SECRET, payload[:5] + flip + payload[6:] + "." + mac)
        is None
    )
    flip = "A" if mac[2] != "A" else "B"
    assert (
        decode_index_cursor(SECRET, payload + "." + mac[:2] + flip + mac[3:])
        is None
    )


def test_cursor_rejects_garbage():
    for token in ("", "not-a-cursor", "!!!.@@@", "aGVsbG8.d29ybGQ", ".", "a."):
        assert decode_index_cursor(SECRET, token) is None


def test_cursor_rejects_non_integer_fields():
    base = {
        "v": 1,
        "snap": "x",
        "pre": "p/",
        "lim": 50,
        "off": 0,
    }
    import base64
    import hashlib
    import hmac
    import json

    def sign(data):
        payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        m = hmac.new(SECRET.encode(), payload, hashlib.sha256).digest()[:18]
        b = lambda x: base64.urlsafe_b64encode(x).decode().rstrip("=")
        return b(payload) + "." + b(m)

    assert decode_index_cursor(SECRET, sign(base)) is not None
    for patch in (
        {"lim": 0},
        {"lim": -1},
        {"lim": 1.5},
        {"lim": True},
        {"lim": "50"},
        {"off": -1},
        {"off": 1.0},
        {"off": False},
        {"v": 2},
        {"snap": 9},
        {"pre": None},
    ):
        data = dict(base)
        data.update(patch)
        assert decode_index_cursor(SECRET, sign(data)) is None, patch


# ---- prefix upper bound -------------------------------------------------------


def test_prefix_upper_bound_ascii():
    assert prefix_upper_bound("dir/") == "dir0"
    assert prefix_upper_bound("abc") == "abd"
    assert prefix_upper_bound("z") == "{"


def test_prefix_upper_bound_empty():
    assert prefix_upper_bound("") is None


def test_prefix_upper_bound_multibyte_bytewise():
    # U+00E9 -> next code point U+00EA: same lead byte, so the bound
    # keeps every string starting with the prefix below it.
    assert prefix_upper_bound("café") == "cafê"
    # U+00FF carries to U+0100; the gap before U+E000 is unencodable, so
    # U+D7FF jumps straight to U+E000 without excluding any valid path.
    assert prefix_upper_bound("ÿ") == "Ā"
    assert prefix_upper_bound("퟿") == chr(0xE000)


def test_prefix_upper_bound_max_only():
    assert prefix_upper_bound(chr(0x10FFFF)) is None
    assert prefix_upper_bound("a" + chr(0x10FFFF)) == "b"
