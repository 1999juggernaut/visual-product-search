"""Everything identify() needs, loaded ONCE per process (models are slow to load, fast to use).

WHAT: ``get_runtime()`` returns one Runtime: settings, encoder, cropper, vector index, catalogue rows,
      inventory table, re-ranker and budget breaker. The API, the MCP server and the eval scripts all call
      get_runtime(), so they all use the very same objects (one implementation behind every surface).
WHY a function and not variables made at import: importing a file must never download a model. Tests build
      their own Runtime with the offline encoder and pass it in.
"""

import csv
import json

from services.inventory.stock import InventoryTable
from services.rerank.budget import Breaker
from services.rerank.reranker import Reranker
from services.vision.detector import Cropper
from services.vision.encoders import load_encoder
from services.vision.index import VectorIndex
from shared.settings import get_settings
from shared.usage import read_usage


class Runtime:
    """A box holding everything the pipeline uses."""

    def __init__(self, settings, encoder, cropper, index, other_index_stamps, catalog, inventory, reranker,
                 breaker):
        self.settings = settings
        self.encoder = encoder
        self.cropper = cropper
        self.index = index                            # None = not built yet for this encoder
        self.other_index_stamps = other_index_stamps  # indexes on disk for OTHER encoders (a mismatch)
        self.catalog = catalog                        # {(tenant, sku): catalogue row}
        self.inventory = inventory
        self.reranker = reranker
        self.breaker = breaker


def load_catalog(settings) -> dict:
    """catalog.csv -> {(tenant, sku): row}. The ONLY source of a candidate's family/finish/lifecycle."""
    path = settings.path("catalog.csv")
    catalog = {}
    if not path.exists():
        return catalog
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            catalog[(row["tenant"], row["sku"])] = row
    return catalog


def find_other_index_stamps(folder) -> list:
    """The stamps of every index saved in the folder (each index has a .json file holding its stamp)."""
    stamps = []
    if not folder.exists():
        return stamps
    for details_file in folder.glob("*.json"):
        details = json.loads(details_file.read_text(encoding="utf-8"))
        stamps.append(details["stamp"])
    return stamps


def build_runtime(settings, encoder=None) -> Runtime:
    if encoder is None:
        encoder = load_encoder(settings.encoder, settings.use_flip_tta, settings.batch_size)

    index_folder = settings.path("index")
    index = VectorIndex.load(index_folder, encoder.stamp)
    other_stamps = []
    if index is None:
        other_stamps = find_other_index_stamps(index_folder)

    return Runtime(
        settings=settings,
        encoder=encoder,
        cropper=Cropper(settings.detector, settings.crop_prompts),
        index=index,
        other_index_stamps=other_stamps,
        catalog=load_catalog(settings),
        inventory=InventoryTable(settings.path("inventory.csv")),
        reranker=Reranker(settings),
        breaker=Breaker(settings, read_usage()),
    )


# The one Runtime of this program. None = not built yet.
_runtime = None


def get_runtime() -> Runtime:
    """Build the Runtime the first time it is needed, then return the same one every time."""
    global _runtime                      # change the variable above, not a new local one
    if _runtime is None:
        _runtime = build_runtime(get_settings())
    return _runtime
