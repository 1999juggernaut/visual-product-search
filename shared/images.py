"""Opening and checking photos: the FIRST door, the arguments (arguments are checked first).

WHAT: ``decode_photo(data)`` refuses anything that is not a real JPEG/PNG/WEBP of a sensible size, applies
      the phone's rotation flag (EXIF) and returns an RGB picture. ``shrink()`` makes a smaller copy.
WHY first: a bad file should be refused before we spend anything ("check before you spend").
"""

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from shared.schemas import Refused

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_BYTES = 10 * 1024 * 1024            # 10 MB: a phone photo is 2-6 MB
MIN_SIDE = 64                           # smaller than this cannot show a product's details


def decode_photo(data: bytes) -> Image.Image:
    if not data:
        raise Refused("empty_image", "the image is empty", "send the photo's bytes")
    if len(data) > MAX_BYTES:
        raise Refused("image_too_large", f"{len(data):,} bytes is over {MAX_BYTES:,}", "send a smaller photo")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img = ImageOps.exif_transpose(img).convert("RGB")     # phones store "rotate me" as a flag
    except (UnidentifiedImageError, OSError):
        raise Refused("not_an_image", "the bytes are not an image", "send a JPEG, PNG or WEBP") from None
    if fmt not in ALLOWED_FORMATS:
        raise Refused("unsupported_format", f"{fmt} is not accepted", f"use one of {sorted(ALLOWED_FORMATS)}")
    if min(img.size) < MIN_SIDE:
        raise Refused("image_too_small", f"{img.size} is under {MIN_SIDE}px on a side", "send a larger photo")
    return img


def load_path(path) -> Image.Image:
    """Open a catalogue image from disk (trusted files, so fewer checks)."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def shrink(img: Image.Image, max_side: int) -> Image.Image:
    copy = img.copy()
    copy.thumbnail((max_side, max_side))         # in place, keeps proportions
    return copy


def to_jpeg(img: Image.Image, max_side: int, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    shrink(img, max_side).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
