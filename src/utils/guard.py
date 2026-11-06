"""
Protection for a service that can spend GPU money on someone's account.

  - DailyCap: a per-process limit on GPU calls per UTC day (DOCINTEL_MAX_GPU_CALLS_PER_DAY, default 200).
  - check_access: an API key (DOCINTEL_API_KEY, sent as the X-API-Key header). With no key configured, only a client
    on the same machine is let in, so exposing the port or putting a proxy in front of it never opens the GPU by accident.
"""

import hmac
import os
import threading
from datetime import datetime, timezone

# "testclient" is Starlette's TestClient, never a real socket.
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


class DailyCap:
    """At most `limit` GPU calls per UTC day, per process; thread-safe."""

    def __init__(self, limit: int):
        self.limit, self._day, self._n, self._lock = limit, None, 0, threading.Lock()

    def take(self) -> bool:
        """Count one GPU call; False when today's limit is used up."""
        today = datetime.now(timezone.utc).date()
        with self._lock:
            if today != self._day:
                self._day, self._n = today, 0
            if self._n >= self.limit:
                return False
            self._n += 1
            return True


def cap_from_env() -> DailyCap:
    return DailyCap(int(os.environ.get("DOCINTEL_MAX_GPU_CALLS_PER_DAY", "200")))


def check_access(api_key: str | None, client_host: str | None, forwarded_for: str | None) -> str | None:
    """None when the request may proceed, otherwise the reason it must be refused (HTTP 401)."""
    configured = os.environ.get("DOCINTEL_API_KEY")
    if configured:
        if api_key and hmac.compare_digest(api_key.encode(), configured.encode()):
            return None
        return "missing or wrong X-API-Key header"
    if client_host in LOCAL_HOSTS and not forwarded_for:
        return None
    return (
        "this service is only open to the local machine until DOCINTEL_API_KEY is set; send it as the X-API-Key header"
    )
