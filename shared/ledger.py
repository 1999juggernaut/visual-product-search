"""The catalogue ledger: which version of each catalogue image is current.

WHAT: one JSON file (data_dir/ledger.json) with one row per catalogue IMAGE:
    image_id = "<tenant>:<sku>:<sha256 of the image bytes>"   <- the VERSION is named by its bytes
    status   = indexed | retired | withdrawn
  - ``register()``  : a new image becomes "indexed". The same bytes twice = one version (a duplicate).
  - ``retire()``    : an image is replaced or a SKU discontinued; the row is KEPT (an undo only flips status).
  - ``withdraw()``  : taken down by hand: a tombstone. Nothing automatic brings it back.
  - ``restore()``   : undoes retire or withdraw; the image becomes indexed again.
  - ``status()``    : holds the vector index against the ledger: MISSING (current but not in the index) and
                      ORPHANS (in the index but not current). "The check is not the writer": it repairs nothing.
WHY: a wrong answer is often a version problem, not a model problem.
With the version inside every candidate's image_id, the trail shows exactly which bytes answered.
"""

import json
from pathlib import Path

from shared.events import now_iso, sha256_bytes

STATUSES = ["indexed", "retired", "withdrawn"]


def make_image_id(tenant: str, sku: str, data: bytes) -> str:
    return f"{tenant}:{sku}:{sha256_bytes(data)}"


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        # rows = {image_id: {"tenant": ..., "sku": ..., "path": ..., "status": ...}}
        if path.exists():
            self.rows = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.rows = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.rows, indent=1), encoding="utf-8")

    def register(self, tenant: str, sku: str, path: Path) -> tuple[str, str]:
        """Add one catalogue image. Returns (image_id, what happened): indexed | duplicate | retired | withdrawn.
        A retired or withdrawn image whose file is still on disk STAYS down: a folder scan must never undo
        an operator's decision. Only restore() brings an image back."""
        image_id = make_image_id(tenant, sku, path.read_bytes())

        if image_id in self.rows:
            status = self.rows[image_id]["status"]
            if status == "indexed":
                return image_id, "duplicate"     # same bytes again: nothing to do
            return image_id, status              # "retired" or "withdrawn": leave it down

        self.rows[image_id] = {"tenant": tenant, "sku": sku, "path": str(path), "status": "indexed",
                               "indexed_at": now_iso()}
        return image_id, "indexed"

    def set_status(self, image_ids: list[str], new_status: str) -> list[str]:
        """Change the status of these images. Returns the ids that actually changed."""
        changed = []
        for image_id in image_ids:
            if image_id not in self.rows:
                continue
            row = self.rows[image_id]
            if row["status"] != new_status:
                row["status"] = new_status
                row[new_status + "_at"] = now_iso()     # e.g. "retired_at": when it happened
                changed.append(image_id)
        return changed

    def retire(self, image_ids: list[str]) -> list[str]:
        return self.set_status(image_ids, "retired")

    def withdraw(self, image_ids: list[str]) -> list[str]:
        return self.set_status(image_ids, "withdrawn")

    def restore(self, image_ids: list[str]) -> list[str]:
        """Bring retired or withdrawn images back (the undo is a status flip: no re-upload, no re-embed)."""
        can_restore = []
        for image_id in image_ids:
            if image_id in self.rows and self.rows[image_id]["status"] in ("retired", "withdrawn"):
                can_restore.append(image_id)
        return self.set_status(can_restore, "indexed")

    def ids_for(self, tenant: str, sku=None) -> list[str]:
        """Every image id of this tenant (and this SKU, if given), whatever its status."""
        found = []
        for image_id, row in self.rows.items():
            if row["tenant"] != tenant:
                continue
            if sku is not None and row["sku"] != sku:
                continue
            found.append(image_id)
        return found

    def current(self, tenant=None) -> dict:
        """The images that should be in the index: status "indexed" (for one tenant, or all if tenant is None)."""
        current_rows = {}
        for image_id, row in self.rows.items():
            if row["status"] != "indexed":
                continue
            if tenant is not None and row["tenant"] != tenant:
                continue
            current_rows[image_id] = row
        return current_rows

    def status(self, index_ids: set, tenant=None) -> dict:
        """Hold the index against the ledger. "in sync" = nothing missing, nothing orphaned."""
        wanted = set(self.current(tenant).keys())

        held = set()
        for image_id in index_ids:
            if tenant is None or image_id.startswith(tenant + ":"):
                held.add(image_id)

        missing = sorted(wanted - held)          # set difference: in "wanted" but not in "held"
        orphans = sorted(held - wanted)          # in the index but no longer current
        if len(missing) == 0 and len(orphans) == 0:
            overall = "in sync"
        else:
            overall = "drift"
        return {"ledger_current": len(wanted), "held": len(held), "missing": missing, "orphans": orphans,
                "status": overall}
