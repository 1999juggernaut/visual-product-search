"""Make a small PRACTICE dataset on disk, with no download: drawn "products" in several shapes and finishes.

WHAT it writes into data_dir (default data/synthetic/):
    catalog_images/<tenant>/<sku>/view1.jpg, view2.jpg   clean catalogue photos (2 views per SKU)
    queries/qNNN.jpg                                     "customer photos": busy background, rotated, blurred
    catalog.csv      tenant, sku, family, finish, lifecycle
    inventory.csv    tenant, sku, qty, locations, as_of
    golden.jsonl     one line per query: the right answer (or null = not in the catalogue) and "required"
    tenants.json     two tenants: "practice" (data_region any) and "acme" (data_region in)
    dev_tokens.json  local test tokens (see shared/auth.py)
WHY SYNTHETIC: so you can read and run every step today, on this laptop, before any real catalogue data or model
    download. The numbers it gives are about the CODE working, not about accuracy. Accuracy is measured on
    Colab with real photos (the Stanford Cars practice set, then the target catalogue).
WHY "OUT OF CATALOGUE" QUERIES: a product we don't sell must get "not sure", not a confident wrong SKU.
    Those rows have expected_sku = null, and the gate counts how often we wrongly answer them.

Run:  python tasks.py data
"""

import json
import random
import shutil

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from shared.settings import get_settings

FINISHES = {                       # name: (body colour, highlight colour) as (red, green, blue)
    "chrome": ((178, 182, 190), (235, 238, 242)),
    "matte_black": ((38, 38, 42), (80, 80, 86)),
    "brushed_brass": ((181, 148, 82), (222, 196, 130)),
    "white": ((228, 228, 224), (250, 250, 248)),
    "gunmetal": ((88, 92, 100), (130, 134, 142)),
    "copper": ((168, 94, 60), (210, 140, 100)),        # used ONLY for out-of-catalogue queries
}

# The practice tenant sells 4 shapes x 3 finishes = 12 SKUs.
PRACTICE = []
for family in ["bar", "arch", "disc", "knob"]:
    for finish in ["chrome", "matte_black", "brushed_brass"]:
        PRACTICE.append((family, finish))
ACME = [("bar", "white"), ("arch", "gunmetal"), ("knob", "white")]
OUT_OF_CATALOGUE = [("ring", "chrome"), ("bar", "copper"), ("disc", "copper"), ("ring", "matte_black")]


def jitter(rng: random.Random, value: int) -> int:
    """Move a coordinate by up to 6 pixels either way, so two drawings of a product are not identical."""
    return value + rng.randint(-6, 6)


def draw_product(family: str, finish: str, rng: random.Random, size: int = 320) -> Image.Image:
    """Draw one product on a white background."""
    body, light = FINISHES[finish]
    image = Image.new("RGB", (size, size), (255, 255, 255))
    pen = ImageDraw.Draw(image)
    j = jitter

    if family == "bar":
        pen.rectangle([j(rng, 130), j(rng, 150), j(rng, 190), j(rng, 280)], fill=body)       # body
        pen.rectangle([j(rng, 110), j(rng, 120), j(rng, 260), j(rng, 150)], fill=light)      # flat bar handle
        pen.rectangle([j(rng, 60), j(rng, 190), j(rng, 130), j(rng, 215)], fill=body)        # short arm
    elif family == "arch":
        pen.rectangle([j(rng, 200), j(rng, 130), j(rng, 225), j(rng, 285)], fill=body)       # tall pillar
        pen.arc([j(rng, 80), j(rng, 40), j(rng, 225), j(rng, 200)], 180, 360, fill=body, width=22)  # the arch
        pen.rectangle([j(rng, 78), j(rng, 115), j(rng, 102), j(rng, 160)], fill=light)       # outlet
    elif family == "disc":
        pen.rectangle([j(rng, 40), j(rng, 60), j(rng, 170), j(rng, 80)], fill=body)          # arm
        pen.ellipse([j(rng, 150), j(rng, 40), j(rng, 290), j(rng, 180)], fill=body)          # round head
        for column in range(5):                                                              # nozzles
            for row in range(5):
                x = 175 + column * 22
                y = 65 + row * 22
                pen.ellipse([x, y, x + 8, y + 8], fill=light)
    elif family == "knob":
        pen.ellipse([j(rng, 90), j(rng, 90), j(rng, 230), j(rng, 230)], fill=body)           # trim plate
        pen.ellipse([j(rng, 130), j(rng, 130), j(rng, 190), j(rng, 190)], fill=light)        # knob
        pen.line([160, 120, 160, 200], fill=body, width=8)
    elif family == "ring":
        pen.ellipse([j(rng, 70), j(rng, 70), j(rng, 250), j(rng, 250)], fill=body)
        for bar in range(7):                                                                  # grate bars
            x = 95 + bar * 22
            pen.line([x, 110, x, 210], fill=light, width=6)
    return image


def product_mask(product: Image.Image) -> Image.Image:
    """A black-and-white picture: white where the product is, black where the white background is."""
    # A lookup table: for each grey level 0..255, what the mask should be. point() then applies it to every
    # pixel at once (much faster than visiting the pixels one by one in Python).
    table = []
    for grey_level in range(256):
        if grey_level < 250:             # not (almost) white = part of the product
            table.append(255)
        else:
            table.append(0)
    return product.convert("L").point(table)


def customer_photo(product: Image.Image, rng: random.Random) -> Image.Image:
    """Turn a clean picture into a 'phone photo': coloured wall, turned, smaller, blurred, darker or lighter."""
    wall = (rng.randint(90, 220), rng.randint(90, 220), rng.randint(90, 220))
    canvas = Image.new("RGB", product.size, wall)
    canvas.paste(product, (0, 0), product_mask(product))        # the product on the coloured wall
    canvas = canvas.rotate(rng.uniform(-15, 15), fillcolor=wall)

    scale = rng.uniform(0.75, 1.0)
    side = int(product.width * scale)
    canvas = canvas.resize((side, side))
    photo = Image.new("RGB", product.size, wall)
    left = rng.randint(0, product.width - side)
    top = rng.randint(0, product.height - side)
    photo.paste(canvas, (left, top))

    photo = photo.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.5)))
    return ImageEnhance.Brightness(photo).enhance(rng.uniform(0.75, 1.2))


def write_csv(path, header: list, rows: list) -> None:
    lines = [",".join(header)]
    for row in rows:
        texts = []
        for value in row:
            texts.append(str(value))
        lines.append(",".join(texts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(queries_per_sku: int = 3, seed: int = 7) -> None:
    settings = get_settings()
    root = settings.data_dir
    rng = random.Random(seed)                              # same seed = same dataset every time

    # A fresh start: delete what an earlier run made (only inside the data folder).
    for folder_name in ["catalog_images", "queries", "index", "rerank_cache"]:
        shutil.rmtree(root / folder_name, ignore_errors=True)
    for file_name in ["ledger.json", "events.jsonl", "usage.jsonl"]:
        (root / file_name).unlink(missing_ok=True)
    (root / "queries").mkdir(parents=True, exist_ok=True)

    catalog = []
    inventory = []
    golden = []
    query_number = 0

    for tenant, items, prefix in [("practice", PRACTICE, "PRD"), ("acme", ACME, "ACM")]:
        sku_number = 0
        for family, finish in items:
            sku_number += 1
            sku = f"{prefix}-{sku_number:03d}"           # e.g. PRD-001 (03d = three digits, zero-padded)

            # Two catalogue views.
            folder = root / "catalog_images" / tenant / sku
            folder.mkdir(parents=True, exist_ok=True)
            draw_product(family, finish, rng).save(folder / "view1.jpg", quality=92)
            draw_product(family, finish, rng).save(folder / "view2.jpg", quality=92)

            if sku_number % 6 == 0:                      # every 6th SKU is discontinued, for realism
                lifecycle = "discontinued"
            else:
                lifecycle = "active"
            catalog.append([tenant, sku, family, finish, lifecycle])

            qty = rng.choice([0, 3, 12, 40])
            if qty > 0:
                locations = "WH-MUM;WH-DEL"
            else:
                locations = ""
            inventory.append([tenant, sku, qty, locations, "2026-10-18T09:00:00+00:00"])

            # Customer photos of this SKU. The first photo of the first 4 SKUs is a "must pass" row.
            for photo_number in range(queries_per_sku):
                query_number += 1
                path = root / "queries" / f"q{query_number:03d}.jpg"
                customer_photo(draw_product(family, finish, rng), rng).save(path, quality=85)
                required = photo_number == 0 and sku_number <= 4
                golden.append({"query": f"queries/{path.name}", "tenant": tenant, "expected_sku": sku,
                               "required": required})

    # Products we don't sell: the right answer is "not sure" (expected_sku = None, written as null).
    for family, finish in OUT_OF_CATALOGUE:
        for _ in range(2):                               # "_" = a loop variable we don't use
            query_number += 1
            path = root / "queries" / f"q{query_number:03d}.jpg"
            customer_photo(draw_product(family, finish, rng), rng).save(path, quality=85)
            golden.append({"query": f"queries/{path.name}", "tenant": "practice", "expected_sku": None,
                           "required": False})

    write_csv(root / "catalog.csv", ["tenant", "sku", "family", "finish", "lifecycle"], catalog)
    write_csv(root / "inventory.csv", ["tenant", "sku", "qty", "locations", "as_of"], inventory)

    golden_lines = []
    for row in golden:
        golden_lines.append(json.dumps(row))
    (root / "golden.jsonl").write_text("\n".join(golden_lines) + "\n", encoding="utf-8")

    tenants = {"tenants": {
        "practice": {"members": ["dev@local", "ui@local", "eval@local"], "data_region": "any"},
        "acme": {"members": ["buyer@acme.local"], "data_region": "in"},
    }}
    (root / "tenants.json").write_text(json.dumps(tenants, indent=1), encoding="utf-8")

    api_address = settings.self_url
    mcp_address = settings.mcp_url
    tokens = {"tokens": {
        "dev-token": {"email": "dev@local", "audience": api_address},
        "acme-token": {"email": "buyer@acme.local", "audience": api_address},
        "dev-mcp-token": {"email": "dev@local", "audience": mcp_address},
        "no-email-token": {"audience": api_address},                       # no email: must be refused
        "other-service-token": {"email": "dev@local", "audience": "http://localhost:7777"},   # wrong audience
    }}
    (root / "dev_tokens.json").write_text(json.dumps(tokens, indent=1), encoding="utf-8")

    print(f"wrote {len(catalog)} SKUs ({len(catalog) * 2} catalogue images), {query_number} queries "
          f"({len(OUT_OF_CATALOGUE) * 2} out-of-catalogue) to {root}")
    print("next: python tasks.py build")


if __name__ == "__main__":
    main()
