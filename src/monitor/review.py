"""
Human review queue that records corrections.

Fields the cascade is not sure about (status "review") wait here. A reviewer confirms or corrects each one; every
decision is appended to a JSONL file: the corrected values for exactly the cases the model finds hard.

It is a starting point, not yet a training set: it keeps no image, nothing about auto-accepted fields, and only the
calibrated probability (not the raw token score), so it cannot retrain the model or refit the calibration on its own.

Append-only JSONL keeps it dependency-free and crash-safe (a torn last line is skipped on read).
"""

import json
import threading
import time
import uuid
from pathlib import Path


class ReviewStore:
    """Append-only JSONL log of documents queued for review and the corrections reviewers make.
    Single process: a lock guards writes, nothing guards two processes sharing one file.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _append(self, rec: dict) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # a torn write at the end of the file
                continue
        return out

    def enqueue(self, doc_id: str | None, doc_type: str, fields: dict) -> str:
        """Queue every field with status 'review'. Returns the document id."""
        doc_id = doc_id or uuid.uuid4().hex[:12]
        pending = {
            f: {"value": v["value"], "probability": v["probability"], "source": v["source"]}
            for f, v in fields.items()
            if v["status"] == "review"
        }
        self._append({"kind": "queued", "doc_id": doc_id, "doc_type": doc_type, "fields": pending, "t": time.time()})
        return doc_id

    def resolve(self, doc_id: str, field: str, value: str | None, reviewer: str = "anonymous") -> None:
        """Record the reviewer's final value for one field (the same value as proposed means 'confirmed')."""
        queued = self.get(doc_id)
        if queued is None or field not in queued["fields"]:
            raise KeyError(f"{doc_id}/{field} is not in the queue")
        proposed = queued["fields"][field]["value"]
        self._append(
            {
                "kind": "resolved",
                "doc_id": doc_id,
                "field": field,
                "value": value,
                "proposed": proposed,
                "confirmed": value == proposed,
                "reviewer": reviewer,
                "t": time.time(),
            }
        )

    def get(self, doc_id: str) -> dict | None:
        return next((r for r in reversed(self._read()) if r["kind"] == "queued" and r["doc_id"] == doc_id), None)

    def pending(self) -> list[dict]:
        """Queued documents with only the fields nobody has resolved yet."""
        recs = self._read()
        done = {(r["doc_id"], r["field"]) for r in recs if r["kind"] == "resolved"}
        out = []
        for r in recs:
            if r["kind"] != "queued":
                continue
            left = {f: v for f, v in r["fields"].items() if (r["doc_id"], f) not in done}
            if left:
                out.append({"doc_id": r["doc_id"], "doc_type": r["doc_type"], "fields": left})
        return out

    def corrections(self) -> list[dict]:
        """Resolved decisions: the corrected values the review process produces."""
        return [r for r in self._read() if r["kind"] == "resolved"]

    def stats(self) -> dict:
        """Queue size and how often the model turned out to be right when a person checked."""
        res = self.corrections()
        return {
            "pending_docs": len(self.pending()),
            "resolved_fields": len(res),
            "model_was_right": sum(r["confirmed"] for r in res),
            "model_was_wrong": sum(not r["confirmed"] for r in res),
        }
