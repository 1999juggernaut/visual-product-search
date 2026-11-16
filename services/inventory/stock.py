"""Inventory: a plain table lookup. No model is ever involved (requirement: deterministic inventory join).

WHAT: ``InventoryTable(path).lookup(tenant, sku)`` reads data_dir/inventory.csv
      (columns: tenant, sku, qty, locations, as_of) and returns a Stock, or None when the SKU is unknown.
WHY NEVER CACHED IN ANSWERS: stock changes by the hour. The re-rank cache stores which product matched;
      stock is looked up fresh on every request, and ``as_of`` tells the reader how old the number is.
LATER: Phase 2 replaces this file's lookup with the system of record. The rest of
      the code only calls lookup(), so nothing else changes.
"""

import csv
from pathlib import Path

from shared.schemas import Stock


class InventoryTable:
    def __init__(self, path: Path):
        self.path = path
        self.file_time = None        # when the file was last changed, at our last read
        self.rows = {}               # {(tenant, sku): Stock}

    def refresh(self) -> None:
        """Re-read the file only when it changed on disk (cheap, and edits show up without a restart)."""
        if self.path.exists():
            file_time = self.path.stat().st_mtime
        else:
            file_time = None
        if file_time == self.file_time:
            return                                   # nothing changed since the last read

        self.rows = {}
        self.file_time = file_time
        if file_time is None:
            return

        with open(self.path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):            # each row is a dict: {"tenant": ..., "sku": ..., ...}
                qty = int(row.get("qty") or 0)
                locations = []
                for location in (row.get("locations") or "").split(";"):
                    if location != "":
                        locations.append(location)
                stock = Stock(sku=row["sku"], qty=qty, available=qty > 0, locations=locations,
                              as_of=row.get("as_of") or None)
                self.rows[(row["tenant"], row["sku"])] = stock

    def lookup(self, tenant: str, sku: str):
        """The Stock for this SKU, or None if this tenant has no such SKU."""
        self.refresh()
        return self.rows.get((tenant, sku))
