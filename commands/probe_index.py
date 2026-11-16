"""A READ-ONLY health check of the index (probes look, they never write).

    python -m commands.probe_index

WHAT it prints:
  1. which index files exist (one per encoder stamp) and how many images each holds per tenant
  2. ledger vs index for the active encoder: missing / orphans (same check as `catalog status`)
  3. the SELF-MATCH rate: for every catalogue image, is its nearest OTHER image the same SKU?
     A low number means the encoder can't even tell catalogue products apart, so customer photos will be worse.
     Needs only the stored vectors: no model is loaded, no photo is embedded.
"""

import json

import numpy as np

from commands.catalog import status
from services.vision.encoders import stamp_of
from services.vision.index import VectorIndex
from shared.settings import get_settings


def self_match(index: VectorIndex) -> dict:
    """{tenant: share of catalogue images whose nearest other image has the same SKU}."""
    tenants = []
    for payload in index.payloads:
        if payload["tenant"] not in tenants:
            tenants.append(payload["tenant"])

    rates = {}
    for tenant in tenants:
        rows = []
        for row_number in range(len(index.payloads)):
            if index.payloads[row_number]["tenant"] == tenant:
                rows.append(row_number)
        if len(rows) < 2:
            continue                                  # one image can't have a "nearest other"

        vectors = index.vectors[rows]
        similarities = vectors @ vectors.T            # every image against every image
        np.fill_diagonal(similarities, -2)            # ignore "an image is nearest to itself"
        nearest = similarities.argmax(axis=1)         # for each image: the position of its nearest other

        same_sku = 0
        for position in range(len(rows)):
            my_sku = index.payloads[rows[position]]["sku"]
            nearest_sku = index.payloads[rows[nearest[position]]]["sku"]
            if my_sku == nearest_sku:
                same_sku += 1
        rates[tenant] = round(same_sku / len(rows), 3)
    return rates


def main() -> None:
    settings = get_settings()
    folder = settings.path("index")

    print("index files:")
    if folder.exists():
        for details_file in sorted(folder.glob("*.json")):
            details = json.loads(details_file.read_text(encoding="utf-8"))
            per_tenant = {}
            for payload in details["payloads"]:
                tenant = payload["tenant"]
                per_tenant[tenant] = per_tenant.get(tenant, 0) + 1
            print(f"  {details['stamp']:<32} {per_tenant}")

    active = stamp_of(settings.encoder, settings.use_flip_tta)
    check = status(active)
    print(f"\nactive encoder {active}: ledger {check['ledger_current']} current, "
          f"index holds {check['held']} -> {check['status']}")
    if check["missing"] or check["orphans"]:
        print(f"  missing {len(check['missing'])}, orphans {len(check['orphans'])}  (fix: python tasks.py build)")

    index = VectorIndex.load(folder, active)
    if index is not None and len(index) > 0:
        print("self-match rate per tenant:", self_match(index))


if __name__ == "__main__":
    main()
