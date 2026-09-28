"""
xau_bg.py — stale-while-revalidate cache with background refresh.

swr(key, fn, ttl) returns the last good value immediately. When it is older
than `ttl`, a daemon thread calls fn() to refresh it while the page carries on
with the old value — the page never waits on a download except the very first
time a key is requested (up to `wait` seconds).

Failures keep the last good value (stale-while-error) and are not retried for
`neg_ttl` seconds, so an unreachable source can't slow every refresh.

The store is module-level and shared by every session in the server process.
This module is deliberately NOT in the app's auto-reload list so the store
survives pushes of other files.
"""
import threading
import time
from typing import Any, Callable, Dict, Optional

_STORE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def _entry(key: str) -> Dict[str, Any]:
    e = _STORE.get(key)
    if e is None:
        e = _STORE[key] = {"val": None, "t": 0.0, "err_t": 0.0, "err": "", "busy": False,
                           "ev": threading.Event(), "runs": 0}
        e["ev"].set()
    return e


def _run(e: Dict[str, Any], fn: Callable[[], Any]) -> None:
    try:
        v = fn()
        if v is None:
            raise ValueError("source returned nothing")
        e.update(val=v, t=time.time(), err="")
    except Exception as ex:  # noqa: BLE001
        e.update(err_t=time.time(), err=f"{type(ex).__name__}: {ex}"[:200])
    finally:
        e["runs"] += 1
        e["busy"] = False
        e["ev"].set()


def swr(key: str, fn: Callable[[], Any], ttl: float, neg_ttl: float = 300.0,
        wait: float = 0.0) -> Optional[Any]:
    """Cached value for `key`, refreshing in the background when stale."""
    now = time.time()
    with _LOCK:
        e = _entry(key)
        stale = e["val"] is None or now - e["t"] > ttl
        backoff = now - e["err_t"] < neg_ttl
        if stale and not e["busy"] and not backoff:
            e["busy"] = True
            e["ev"].clear()
            threading.Thread(target=_run, args=(e, fn), daemon=True,
                             name=f"xau-bg-{key}").start()
    if e["val"] is None and wait > 0 and e["busy"]:
        e["ev"].wait(wait)
    return e["val"]


def peek(key: str) -> Optional[Any]:
    e = _STORE.get(key)
    return None if e is None else e["val"]


def status() -> Dict[str, Dict[str, Any]]:
    now = time.time()
    return {k: {"age_s": None if not e["t"] else round(now - e["t"]),
                "busy": e["busy"], "error": e["err"],
                "retry_in_s": max(0, round(e["err_t"] - now)) if e["err"] else 0}
            for k, e in _STORE.items()}


def reset(key: Optional[str] = None) -> None:
    with _LOCK:
        if key is None:
            _STORE.clear()
        else:
            _STORE.pop(key, None)


def wait_idle(timeout: float = 10.0) -> None:
    """Block until no refresh is running (tests)."""
    end = time.time() + timeout
    for e in list(_STORE.values()):
        e["ev"].wait(max(0.0, end - time.time()))
