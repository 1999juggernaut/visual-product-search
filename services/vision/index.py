"""The vector index: stores one vector per catalogue image and finds the nearest ones to a photo.

WHAT: ``VectorIndex`` keeps a numpy matrix (one row per image) plus a payload per row
      (tenant, sku, image path). ``search(vec, k, tenant)`` returns the k most similar rows OF THAT TENANT.
      ``save()``/``load()`` write it to data_dir/index/<encoder stamp>.npz + .json.
WHY NUMPY AND NOT QDRANT HERE: an exact search over 10,000 vectors is one matrix multiply (a few
      milliseconds) and needs no server or extra install. A vector database earns its place when the catalogue
      reaches hundreds of thousands of images or several services share it: that is Phase 2, and only this
      file changes (the rest of the code calls search() and doesn't know what is behind it).
WHY THE TENANT FILTER IS INSIDE search(): one tenant's photo must never match another tenant's catalogue.
      Filtering inside the one search function means no caller can forget it.
WHY THE STAMP: the file name and the header carry the encoder stamp; identify() refuses to search with a
      vector from a different encoder (index_stamp_mismatch) instead of returning nonsense.
"""

import json
from pathlib import Path

import numpy as np


def stamp_filename(stamp: str) -> str:
    """A stamp like "dinov2_small@main|flip" -> a safe file name "dinov2_small_at_main__flip"."""
    name = stamp.replace("|", "__")
    name = name.replace("@", "_at_")
    name = name.replace("/", "-")
    return name


class VectorIndex:
    def __init__(self, stamp: str):
        self.stamp = stamp
        self.vectors = np.zeros((0, 0), dtype=np.float32)   # one row per image
        self.ids = []                                        # image_id per row: tenant:sku:<sha256>
        self.payloads = []                                   # {"tenant", "sku", "path"} per row

    def __len__(self) -> int:
        # Lets you write len(index) to get the number of images.
        return len(self.ids)

    def add(self, ids: list, vectors: np.ndarray, payloads: list) -> None:
        """Add images. An image already in the index is skipped (the same image twice = once)."""
        already_here = set(self.ids)
        for row_number in range(len(ids)):
            image_id = ids[row_number]
            if image_id in already_here:
                continue
            new_row = vectors[row_number:row_number + 1].astype(np.float32)    # keep it 2-D: 1 x dim
            if len(self.ids) == 0:
                self.vectors = new_row
            else:
                self.vectors = np.vstack([self.vectors, new_row])              # stack under the old rows
            self.ids.append(image_id)
            self.payloads.append(payloads[row_number])
            already_here.add(image_id)

    def remove(self, ids_to_remove: set) -> int:
        """Remove these images. Returns how many were removed."""
        keep_rows = []
        for row_number in range(len(self.ids)):
            if self.ids[row_number] not in ids_to_remove:
                keep_rows.append(row_number)

        removed = len(self.ids) - len(keep_rows)
        if len(keep_rows) == 0:
            self.vectors = np.zeros((0, 0), dtype=np.float32)
        else:
            self.vectors = self.vectors[keep_rows]                 # numpy: pick these rows only
        new_ids = []
        new_payloads = []
        for row_number in keep_rows:
            new_ids.append(self.ids[row_number])
            new_payloads.append(self.payloads[row_number])
        self.ids = new_ids
        self.payloads = new_payloads
        return removed

    def search(self, query_vector: np.ndarray, k: int, tenant: str) -> list:
        """The k closest images of THIS tenant, best first, as (image_id, cosine score, payload) tuples."""
        tenant_rows = []
        for row_number in range(len(self.payloads)):
            if self.payloads[row_number]["tenant"] == tenant:
                tenant_rows.append(row_number)
        if len(tenant_rows) == 0:
            return []

        # One matrix multiply gives the similarity of the query to every image of this tenant.
        scores = self.vectors[tenant_rows] @ query_vector.astype(np.float32)
        best_first = np.argsort(-scores)[:k]           # positions sorted from highest score to lowest

        results = []
        for position in best_first:
            row_number = tenant_rows[position]
            results.append((self.ids[row_number], float(scores[position]), self.payloads[row_number]))
        return results

    # ----- files -----
    @staticmethod
    def file_for(folder: Path, stamp: str) -> Path:
        # "@staticmethod": a function kept inside the class that doesn't need an index object (no "self").
        return folder / (stamp_filename(stamp) + ".npz")

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = VectorIndex.file_for(folder, self.stamp)
        np.savez(path, vectors=self.vectors)                   # the numbers, in numpy's own format
        details = {"stamp": self.stamp, "ids": self.ids, "payloads": self.payloads}
        path.with_suffix(".json").write_text(json.dumps(details), encoding="utf-8")
        return path

    @staticmethod
    def load(folder: Path, stamp: str):
        """The saved index for this stamp, or None if there isn't one."""
        path = VectorIndex.file_for(folder, stamp)
        if not path.exists():
            return None
        details = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        index = VectorIndex(details["stamp"])
        index.vectors = np.load(path)["vectors"]
        index.ids = details["ids"]
        index.payloads = details["payloads"]
        return index


def fuse_by_sku(hits: list) -> list:
    """Many images per SKU -> one row per SKU, scored by its BEST image ("max fusion": the notebook found it
    beats the average when one catalogue view matches the photo's angle and the others don't).
    Returns a list of dicts, best SKU first."""
    best_per_sku = {}
    for image_id, score, payload in hits:
        sku = payload["sku"]
        if sku not in best_per_sku or score > best_per_sku[sku]["score"]:
            best_per_sku[sku] = {"sku": sku, "score": score, "image_id": image_id,
                                 "image_path": payload.get("path", "")}

    candidates = list(best_per_sku.values())
    candidates.sort(key=get_score, reverse=True)       # highest score first
    return candidates


def get_score(candidate: dict) -> float:
    """Used by sort(): tells it to sort the candidates by their "score"."""
    return candidate["score"]
