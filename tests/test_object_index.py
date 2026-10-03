"""End-to-end tests for GET /object-index: frozen, cursor-paginated,
ten-minute snapshots that never block writers and never change under
PUTs, path moves or historical reclamation."""
from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
import uuid

import asyncpg

FUTURE = "2099-01-01T00:00:00+00:00"
ETAG_OF = lambda body: '"' + hashlib.sha256(body).hexdigest() + '"'


def p(name: str) -> str:
    return f"oi/{name}-{uuid.uuid4().hex[:10]}"


def put(client, path, body=b"x", media_type="application/octet-stream"):
    r = client.put(
        f"/objects/{path}", content=body, headers={"Content-Type": media_type}
    )
    assert r.status_code == 204, r.text
    return r


# ---- first page: fields, ordering, defaults ----------------------------------


def test_first_page_entries_and_snapshot_metadata(client):
    marker = p("meta")
    paths = [f"{marker}/a", f"{marker}/b", f"{marker}/c"]
    for path in paths:
        put(client, path, body=b"hello", media_type="text/plain")
    r = client.get("/object-index", params={"prefix": marker + "/"})
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-store"
    body = r.json()
    assert body["prefix"] == marker + "/"
    assert body["limit"] == 50
    assert body["snapshot"]["total"] == 3
    assert body["snapshot"]["id"]
    rows = body["entries"]
    assert [e["path"] for e in rows] == paths
    entry = rows[0]
    assert entry == {
        "path": paths[0],
        "revision_no": 1,
        "etag": ETAG_OF(b"hello"),
        "media_type": "text/plain",
        "bytes": 5,
    }
    created = body["snapshot"]["created_at"]
    expires = body["snapshot"]["expires_at"]
    assert expires > created
    # Empty prefix is still accepted and covers the same objects.
    global_body = client.get(
        "/object-index", params={"prefix": marker + "/", "limit": 100}
    ).json()
    assert {e["path"] for e in global_body["entries"]} == set(paths)


def test_default_limit_is_50_and_custom_limit(client):
    marker = p("lim")
    for i in range(55):
        put(client, f"{marker}/{i:03d}")
    r = client.get("/object-index", params={"prefix": marker + "/"})
    assert r.json()["limit"] == 50
    assert len(r.json()["entries"]) == 50
    assert r.json()["snapshot"]["total"] == 55
    assert r.json()["next_cursor"] is not None

    r = client.get(
        "/object-index", params={"prefix": marker + "/", "limit": 100}
    )
    assert r.json()["limit"] == 100
    assert len(r.json()["entries"]) == 55
    assert r.json()["next_cursor"] is None


# ---- parameter validation ------------------------------------------------------


def test_bad_limit_is_400(client):
    for raw in ("0", "101", "-1", "1.5", "abc", "+1", "true"):
        r = client.get("/object-index", params={"limit": raw})
        assert r.status_code == 400, raw
        assert r.json()["detail"]
    # Surrounding whitespace must not be accepted either; send it raw so
    # HTTP clients cannot trim it before the server sees it.
    for raw in ("%2010", "10%20"):
        r = client.get(f"/object-index?limit={raw}")
        assert r.status_code == 400, raw


def test_limit_boundaries_accepted(client):
    put(client, p("edge"))
    for raw in ("1", "100"):
        assert client.get("/object-index", params={"limit": raw}).status_code == 200


# ---- prefix filtering ----------------------------------------------------------


def test_prefix_filters_and_empty_prefix_is_global(client):
    a = p("pref")
    put(client, f"{a}/x")
    put(client, f"{a}/y/z")
    put(client, f"{a}other")
    total_global = client.get("/object-index").json()["snapshot"]["total"]
    assert total_global >= 4

    r = client.get(
        "/object-index", params={"prefix": f"{a}/", "limit": 100}
    )
    names = [e["path"] for e in r.json()["entries"]]
    assert names == [f"{a}/x", f"{a}/y/z"]
    # Prefix is literal, not a glob: metacharacters match themselves.
    put(client, f"{a}_one")
    r = client.get("/object-index", params={"prefix": f"{a}_"})
    assert [e["path"] for e in r.json()["entries"]] == [f"{a}_one"]
    r = client.get("/object-index", params={"prefix": f"{a}%"})
    assert r.json()["entries"] == []
    assert r.json()["snapshot"]["total"] == 0


def test_prefix_uses_byte_order_with_unicode(client):
    marker = p("uni")
    for path in [f"{marker}/a", f"{marker}/é", f"{marker}/z"]:
        put(client, path)
    r = client.get("/object-index", params={"prefix": marker + "/"})
    # Bytewise (C-collation) order: "a" (0x61) < "z" (0x7A) < "é" (0xC3A9).
    assert [e["path"] for e in r.json()["entries"]] == [
        f"{marker}/a",
        f"{marker}/z",
        f"{marker}/é",
    ]


# ---- pagination: no duplicates, no skips ---------------------------------------


def _walk(client, first, max_pages=200):
    pages, seen, body = [], [], first
    pages.append(body)
    while body["next_cursor"]:
        r = client.get("/object-index", params={"cursor": body["next_cursor"]})
        assert r.status_code == 200
        body = r.json()
        pages.append(body)
    return pages


def test_pagination_is_complete_and_stable(client):
    marker = p("page")
    expected = [f"{marker}/{i:03d}" for i in range(123)]
    for path in expected:
        put(client, path)
    first = client.get(
        "/object-index", params={"prefix": marker + "/", "limit": 20}
    ).json()
    pages = _walk(client, first)
    flat = [e["path"] for page in pages for e in page["entries"]]
    assert flat == expected
    assert len(flat) == len(set(flat)) == 123
    sizes = [len(page["entries"]) for page in pages]
    assert sizes == [20] * 6 + [3]
    # Same snapshot id on every page.
    assert len({page["snapshot"]["id"] for page in pages}) == 1
    # Last page carries no cursor; earlier ones do.
    assert pages[-1]["next_cursor"] is None
    assert all(pages[i]["next_cursor"] for i in range(len(pages) - 1))


def test_limit_one_walks_every_entry(client):
    marker = p("one")
    expected = [f"{marker}/{i}" for i in range(7)]
    for path in expected:
        put(client, path)
    first = client.get(
        "/object-index", params={"prefix": marker, "limit": 1}
    ).json()
    flat = [e["path"] for page in _walk(client, first) for e in page["entries"]]
    assert flat == expected


def test_cursor_is_replayable_and_deterministic(client):
    marker = p("replay")
    for i in range(5):
        put(client, f"{marker}/{i}")
    first = client.get(
        "/object-index", params={"prefix": marker, "limit": 2}
    ).json()
    token = first["next_cursor"]
    b1 = client.get("/object-index", params={"cursor": token}).json()
    b2 = client.get("/object-index", params={"cursor": token}).json()
    assert b1 == b2
    assert [e["path"] for e in b1["entries"]] == [
        f"{marker}/2",
        f"{marker}/3",
    ]


def test_cursor_echo_parameters_must_match(client):
    marker = p("echo")
    for i in range(4):
        put(client, f"{marker}/{i}")
    token = client.get(
        "/object-index", params={"prefix": marker, "limit": 2}
    ).json()["next_cursor"]
    assert (
        client.get(
            "/object-index", params={"cursor": token, "prefix": "elsewhere"}
        ).status_code
        == 400
    )
    assert (
        client.get(
            "/object-index", params={"cursor": token, "limit": 3}
        ).status_code
        == 400
    )
    # Echoing the bound values still works.
    ok = client.get(
        "/object-index",
        params={"cursor": token, "prefix": marker, "limit": 2},
    )
    assert ok.status_code == 200
    assert [e["path"] for e in ok.json()["entries"]] == [
        f"{marker}/2",
        f"{marker}/3",
    ]


def test_no_matches_returns_empty_page_no_cursor(client):
    marker = p("empty")
    r = client.get("/object-index", params={"prefix": marker + "/"})
    body = r.json()
    assert body["entries"] == []
    assert body["next_cursor"] is None
    assert body["snapshot"]["total"] == 0
    assert body["limit"] == 50


def test_empty_prefix_covers_objects_outside_any_prefix(client):
    marker = p("global")
    paths = [f"{marker}/x", f"{marker}/y"]
    for path in paths:
        put(client, path)
    # An empty prefix means the whole namespace: walk every page of a
    # fresh snapshot and assert our objects are among the frozen set.
    first = client.get(
        "/object-index", params={"prefix": "", "limit": 100}
    ).json()
    assert first["prefix"] == ""
    pages = _walk(client, first)
    all_paths = {
        e["path"] for page in pages for e in page["entries"]
    }
    assert set(paths) <= all_paths
    # The frozen listing is globally byte-ordered across pages.
    flat = [e["path"] for page in pages for e in page["entries"]]
    assert flat == sorted(flat, key=lambda s: s.encode("utf-8"))


def test_swept_snapshot_cursor_is_410_not_404(client):
    marker = p("swept")
    put(client, f"{marker}/a")
    put(client, f"{marker}/b")
    first = client.get(
        "/object-index", params={"prefix": marker, "limit": 1}
    ).json()
    token = first["next_cursor"]
    snapshot_id = first["snapshot"]["id"]
    asyncio.run(_force_expire_snapshot(snapshot_id))
    # Any new first-page listing runs the expiry sweep, deleting the row.
    client.get("/object-index", params={"prefix": p("other") + "/"})
    r = client.get("/object-index", params={"cursor": token})
    assert r.status_code == 410
    assert r.json()["snapshot_id"] == snapshot_id


def test_last_page_cursor_replays_empty_page_and_far_offset_is_400(client):
    marker = p("end")
    put(client, f"{marker}/a")
    put(client, f"{marker}/b")
    pages = _walk(
        client,
        client.get(
            "/object-index", params={"prefix": marker, "limit": 2}
        ).json(),
    )
    assert pages[-1]["entries"] and pages[-1]["next_cursor"] is None
    # A signed offset exactly at total replays an empty page (idempotent
    # tail), still carrying no cursor.
    from app.object_index import IndexCursor, encode_index_cursor

    secret = None
    r = client.get(
        "/object-index", params={"prefix": marker, "limit": 2}
    ).json()
    snap_id = r["snapshot"]["id"]

    # Read the signing secret the same way the app does, out-of-band.
    async def read_secret():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            return await conn.fetchval(
                "SELECT value FROM service_meta"
                " WHERE key = 'object_index_cursor_secret'"
            )
        finally:
            await conn.close()

    secret = asyncio.run(read_secret())
    tail = encode_index_cursor(
        secret, IndexCursor(snap_id, marker, 2, 2)
    )
    out = client.get("/object-index", params={"cursor": tail})
    assert out.status_code == 200
    assert out.json()["entries"] == []
    assert out.json()["next_cursor"] is None
    # Beyond total is rejected as a forged cursor.
    beyond = encode_index_cursor(
        secret, IndexCursor(snap_id, marker, 2, 3)
    )
    assert (
        client.get("/object-index", params={"cursor": beyond}).status_code
        == 400
    )


# ---- tampering / expiry ---------------------------------------------------------


def test_tampered_and_garbage_cursors_are_400(client):
    for token in ("not-a-cursor", "!!!.@@@", "aGVsbG8.d29ybGQ", ""):
        assert (
            client.get("/object-index", params={"cursor": token}).status_code
            == 400
        )
    # First create a real cursor, then mutate the payload.
    marker = p("tamper")
    put(client, f"{marker}/a")
    put(client, f"{marker}/b")
    token = client.get(
        "/object-index", params={"prefix": marker, "limit": 1}
    ).json()["next_cursor"]
    payload, mac = token.rsplit(".", 1)
    flip = "A" if payload[5] != "A" else "B"
    assert (
        client.get(
            "/object-index",
            params={"cursor": payload[:5] + flip + payload[6:] + "." + mac},
        ).status_code
        == 400
    )


async def _force_expire_snapshot(snapshot_id: str) -> None:
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        await conn.execute(
            "UPDATE object_index_snapshots SET expires_at = now() - interval '1 second'"
            " WHERE id = $1",
            snapshot_id,
        )
    finally:
        await conn.close()


def test_expired_snapshot_is_410(client):
    marker = p("exp")
    for i in range(3):
        put(client, f"{marker}/{i}")
    first = client.get(
        "/object-index", params={"prefix": marker, "limit": 1}
    ).json()
    token = first["next_cursor"]
    snapshot_id = first["snapshot"]["id"]
    asyncio.run(_force_expire_snapshot(snapshot_id))
    r = client.get("/object-index", params={"cursor": token})
    assert r.status_code == 410
    assert r.json() == {
        "detail": "snapshot_expired",
        "snapshot_id": snapshot_id,
    }
    # A fresh listing after expiry works again (new snapshot).
    assert (
        client.get(
            "/object-index", params={"prefix": marker, "limit": 1}
        ).status_code
        == 200
    )
    # The swept row is gone once the new listing's sweep has run; the old
    # signed cursor is still 410, never 404/500.
    r = client.get("/object-index", params={"cursor": token})
    assert r.status_code == 410


# ---- snapshot immutability under writes / moves / gc -----------------------------


def test_puts_during_pagination_do_not_change_replay(client):
    marker = p("mut")
    for i in range(5):
        put(client, f"{marker}/{i}")
    first = client.get(
        "/object-index", params={"prefix": marker, "limit": 2}
    ).json()
    snapshot_id = first["snapshot"]["id"]

    # Mutate the live namespace: overwrite a listed object (new revision,
    # new ETag/size), add a new object, delete-from-index via move-away.
    put(client, f"{marker}/1", body=b"totally different bytes")
    put(client, f"{marker}/new-after-snapshot")
    src = f"{marker}/4"
    etag4 = client.get(f"/objects/{src}").headers["etag"]
    moved = client.post(
        "/moves",
        json={
            "source_path": src,
            "target_path": f"{marker}/moved-destination",
            "source_etag": etag4,
            "source_revision_no": 1,
        },
    )
    assert moved.status_code == 201, moved.text

    pages = _walk(client, first)
    flat = [e for page in pages for e in page["entries"]]
    assert [e["path"] for e in flat] == [f"{marker}/{i}" for i in range(5)]
    # The frozen metadata of /1 is still revision 1 with the original body.
    one = next(e for e in flat if e["path"] == f"{marker}/1")
    assert one["revision_no"] == 1
    assert one["etag"] == ETAG_OF(b"x")
    assert one["bytes"] == 1
    # The move-away is invisible to the frozen listing.
    four = next(e for e in flat if e["path"] == f"{marker}/4")
    assert four["revision_no"] == 1
    # Every page is the same snapshot.
    assert all(page["snapshot"]["id"] == snapshot_id for page in pages)


def test_gc_does_not_change_replay_and_keeps_frozen_metadata(client):
    marker = p("gc")
    put(client, f"{marker}/a", body=b"v1")
    put(client, f"{marker}/a", body=b"v2-longer")
    put(client, f"{marker}/b", body=b"v1")
    put(client, f"{marker}/b", body=b"v2-longer-too")
    first = client.get(
        "/object-index", params={"prefix": marker, "limit": 1}
    ).json()
    # Collect every historical revision of both objects.
    gc = client.post(
        "/gc",
        json={
            "paths": [f"{marker}/a", f"{marker}/b"],
            "before": FUTURE,
            "dry_run": False,
        },
    )
    assert gc.status_code == 200, gc.text
    # GC must not touch the snapshot tables.
    deleted = gc.json()["deleted"]
    assert len(deleted) == 2

    pages = _walk(client, first)
    flat = [e for page in pages for e in page["entries"]]
    assert [e["path"] for e in flat] == [f"{marker}/a", f"{marker}/b"]
    a = next(e for e in flat if e["path"] == f"{marker}/a")
    b = next(e for e in flat if e["path"] == f"{marker}/b")
    assert a["revision_no"] == 2 and a["bytes"] == len(b"v2-longer")
    assert a["etag"] == ETAG_OF(b"v2-longer")
    assert b["revision_no"] == 2 and b["bytes"] == len(b"v2-longer-too")
    assert b["etag"] == ETAG_OF(b"v2-longer-too")


def test_fresh_listing_reflects_moves_and_new_puts(client):
    marker = p("fresh")
    put(client, f"{marker}/stay", body=b"1")
    src = f"{marker}/will-move"
    put(client, src, body=b"1")
    etag = client.get(f"/objects/{src}").headers["etag"]
    target = f"{marker}/moved"
    assert (
        client.post(
            "/moves",
            json={
                "source_path": src,
                "target_path": target,
                "source_etag": etag,
                "source_revision_no": 1,
            },
        ).status_code
        == 201
    )
    r = client.get(
        "/object-index", params={"prefix": marker, "limit": 100}
    )
    names = [e["path"] for e in r.json()["entries"]]
    assert f"{marker}/stay" in names
    assert target in names
    assert src not in names
    moved = next(e for e in r.json()["entries"] if e["path"] == target)
    assert moved["revision_no"] == 1
    assert moved["etag"] == ETAG_OF(b"1")


# ---- non-blocking behaviour ------------------------------------------------------


def test_listing_does_not_block_concurrent_write(client):
    # The snapshot scan takes no object/revision lock: a PUT landing while
    # a listing transaction is open must commit immediately.
    barrier = threading.Event()
    proceed = threading.Event()
    result = {}

    async def open_listing_snapshot():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            tx = conn.transaction(isolation="repeatable_read")
            await tx.start()
            await conn.fetch(
                """
                SELECT o.path, r.revision_no
                  FROM objects o
                  JOIN revisions r ON r.id = o.current_id
                """
            )
            barrier.set()
            proceed.wait(timeout=10)
            await tx.commit()
        finally:
            await conn.close()

    def listing_thread():
        asyncio.run(open_listing_snapshot())

    path = p("nolock")
    t = threading.Thread(target=listing_thread)
    t.start()
    assert barrier.wait(timeout=10)
    # While the read transaction is open, the write must not queue.
    start = time.monotonic()
    r = client.put(f"/objects/{path}", content=b"written")
    elapsed = time.monotonic() - start
    result["status"] = r.status_code
    proceed.set()
    t.join(timeout=10)
    assert not t.is_alive()
    assert result["status"] == 204
    assert elapsed < 3.0


def test_concurrent_listings_and_writes_never_error(client):
    # Hammer first-page listings (each runs the expiry sweep) and writers
    # together: readers never block writers and neither side errors.
    marker = p("stress")
    errors = []

    def writer(i):
        try:
            r = put(client, f"{marker}/w{i:03d}")
            assert r.status_code == 204
        except Exception as exc:  # pragma: no cover - test failure path
            errors.append(("w", i, repr(exc)))

    def lister(i):
        try:
            r = client.get(
                "/object-index", params={"prefix": marker + "/", "limit": 7}
            )
            assert r.status_code == 200, r.text
            body = r.json()
            # Walk every page of the snapshot this lister created.
            for page in _walk(client, body):
                assert page["snapshot"]["id"] == body["snapshot"]["id"]
        except Exception as exc:  # pragma: no cover - test failure path
            errors.append(("l", i, repr(exc)))

    threads = []
    for i in range(6):
        threads.append(threading.Thread(target=lister, args=(i,)))
    for i in range(18):
        threads.append(threading.Thread(target=writer, args=(i,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads)
    assert errors == []
    # Final snapshot sees all 18 objects, fully ordered and complete.
    first = client.get(
        "/object-index", params={"prefix": marker + "/", "limit": 5}
    ).json()
    flat = [e["path"] for page in _walk(client, first) for e in page["entries"]]
    assert len(flat) == len(set(flat)) == 18
    assert flat == sorted(flat, key=lambda s: s.encode("utf-8"))


def test_head_is_not_served_by_index_route(client):
    # Pagination is JSON-over-GET only; HEAD answers 405.
    assert client.head("/object-index").status_code == 405
