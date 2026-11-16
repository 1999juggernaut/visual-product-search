"""Catalogue commands: build the index, and retire / withdraw / restore / status.

    python -m commands.catalog build                    register every catalogue image + embed what's missing
    python -m commands.catalog status                   ledger vs index: missing / orphans (READ-ONLY)
    python -m commands.catalog retire   --tenant practice --sku PRD-003
    python -m commands.catalog withdraw --tenant practice --sku PRD-003
    python -m commands.catalog restore  --tenant practice --sku PRD-003

WHAT "build" does: walks data_dir/catalog_images/<tenant>/<sku>/*.jpg, registers each image in the ledger
  (its id names its bytes), embeds ONLY images the index doesn't hold yet, and removes index rows the ledger
  no longer calls current. Re-running it is safe: the second run embeds nothing (idempotent).
WHY "status" REPAIRS NOTHING: "the check is not the writer". It tells you what is wrong;
  ``build`` is the one thing that changes the index. A read-only check can be run any time, by anyone.
WHY RETIRE KEEPS THE ROW: undo is a status flip, not a re-upload.
"""

import argparse
import json

from services.vision.encoders import load_encoder, stamp_of
from services.vision.index import VectorIndex
from shared.events import log_event
from shared.images import load_path
from shared.ledger import Ledger
from shared.settings import get_settings


def sub_folders(folder) -> list:
    """The folders directly inside ``folder``, sorted by name (empty list if it doesn't exist)."""
    found = []
    if not folder.exists():
        return found
    for item in folder.iterdir():
        if item.is_dir():
            found.append(item)
    return sorted(found)


def register_all_images(ledger: Ledger, images_root) -> dict:
    """Add every catalogue image on disk to the ledger. Returns how many of each outcome."""
    counts = {"indexed": 0, "duplicate": 0, "retired": 0, "withdrawn": 0}
    for tenant_folder in sub_folders(images_root):
        for sku_folder in sub_folders(tenant_folder):
            image_files = sorted(sku_folder.glob("*.jpg")) + sorted(sku_folder.glob("*.png"))
            for image_file in image_files:
                _image_id, outcome = ledger.register(tenant_folder.name, sku_folder.name, image_file)
                counts[outcome] += 1
    return counts


def build(encoder=None) -> dict:
    settings = get_settings()
    ledger = Ledger(settings.path("ledger.json"))
    counts = register_all_images(ledger, settings.path("catalog_images"))
    ledger.save()

    if encoder is None:
        encoder = load_encoder(settings.encoder, settings.use_flip_tta, settings.batch_size)
    index = VectorIndex.load(settings.path("index"), encoder.stamp)
    if index is None:
        index = VectorIndex(encoder.stamp)

    current = ledger.current()

    # Orphans out: in the index, but no longer current in the ledger.
    orphans = set(index.ids) - set(current.keys())
    removed = index.remove(orphans)

    # Missing in: current in the ledger, but not in the index yet.
    in_index = set(index.ids)
    missing = []
    for image_id in current:
        if image_id not in in_index:
            missing.append(image_id)

    if len(missing) > 0:
        images = []
        payloads = []
        for image_id in missing:
            row = current[image_id]
            images.append(load_path(row["path"]))
            payloads.append({"tenant": row["tenant"], "sku": row["sku"], "path": row["path"]})
        vectors = encoder.embed(images)
        index.add(missing, vectors, payloads)
    index.save(settings.path("index"))

    summary = {"stamp": encoder.stamp, "ledger": counts, "embedded": len(missing), "removed": removed,
               "index_images": len(index)}
    log_event("catalog_build", **summary)
    return summary


def status(stamp=None) -> dict:
    settings = get_settings()
    if stamp is None:
        stamp = stamp_of(settings.encoder, settings.use_flip_tta)   # worked out, so no model is loaded
    index = VectorIndex.load(settings.path("index"), stamp)
    if index is None:
        index_ids = set()
    else:
        index_ids = set(index.ids)
    result = {"stamp": stamp}
    result.update(Ledger(settings.path("ledger.json")).status(index_ids))
    return result


def change(action: str, tenant: str, sku: str) -> list:
    """Retire, withdraw or restore every image of one SKU. Returns the image ids that changed."""
    settings = get_settings()
    ledger = Ledger(settings.path("ledger.json"))
    image_ids = ledger.ids_for(tenant, sku)
    if action == "retire":
        changed = ledger.retire(image_ids)
    elif action == "withdraw":
        changed = ledger.withdraw(image_ids)
    elif action == "restore":
        changed = ledger.restore(image_ids)
    else:
        raise ValueError(f"unknown action {action!r}")
    ledger.save()
    log_event(f"catalog_{action}", tenant=tenant, sku=sku, changed=len(changed))
    return changed


def main() -> None:
    # argparse reads the words typed after the command, e.g. "retire --tenant practice --sku PRD-003".
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["build", "status", "retire", "withdraw", "restore"])
    parser.add_argument("--tenant")
    parser.add_argument("--sku")
    args = parser.parse_args()

    if args.action == "build":
        print(json.dumps(build(), indent=1))
    elif args.action == "status":
        print(json.dumps(status(), indent=1))
    else:
        if not args.tenant or not args.sku:
            parser.error(f"{args.action} needs --tenant and --sku")
        changed = change(args.action, args.tenant, args.sku)
        print(f"{args.action}: {len(changed)} image(s) changed. Run 'python tasks.py build' to apply to the index.")


if __name__ == "__main__":
    main()
