"""Image -> vector ("embedding"). Similar-looking products get vectors that point the same way.

WHAT: ``load_encoder(name)`` returns an Encoder with:
  - ``.embed(list_of_images)`` -> a numpy array, one row per image, each row of length 1 ("normalised"),
    so the dot product of two rows is their COSINE SIMILARITY (1.0 = same direction, 0 = unrelated).
  - ``.stamp``  = "name@revision|tta". The index stores the stamp it was built with; search refuses a photo
    embedded by a different stamp (version mismatch is a top cause of silent wrong answers).

WHICH MODELS AND WHY (the same families the Colab bake-off measures):
  local profile (CPU, this laptop):
    color_hist     no download. Colour + edges + silhouette. NOT accurate: it exists so you can run every step
                   offline while learning, and so the tests run in seconds.
    dinov2_small   facebook/dinov2-small, 22M parameters, ~90 MB. Self-supervised: learned shapes and
                   textures with no captions, which is what "same product, different photo" needs.
    siglip2_base   google/siglip2-base-patch16-224, ~375 MB. Learned from image-caption pairs: stronger on
                   "kind of thing", weaker on fine detail. Trying both tells us which matters for look-alike products.
  colab profile (T4 GPU): the bigger relatives that won in the notebook (siglip2_so400m, dinov2_large).

Python lesson: ``ENCODERS`` is a dict of dicts: short name -> how to load it. Adding a model = adding one line.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageOps

ENCODERS = {
    # name:            kind,              hugging face id,                          revision pin
    "color_hist":     {"kind": "hist",   "hf_id": "",                                 "revision": "v1"},
    "dinov2_small":   {"kind": "dino",   "hf_id": "facebook/dinov2-small",            "revision": "main"},
    "siglip2_base":   {"kind": "siglip", "hf_id": "google/siglip2-base-patch16-224",  "revision": "main"},
    # ---- colab profile (needs a GPU and several GB; do not load these on the laptop) ----
    "dinov2_large":   {"kind": "dino",   "hf_id": "facebook/dinov2-large",            "revision": "main"},
    "siglip2_so400m": {"kind": "siglip", "hf_id": "google/siglip2-so400m-patch14-384", "revision": "main"},
}
# "main" means "latest". Before a demo, replace it with the exact commit hash shown on the model's
# Hugging Face page, so the model can't change under you (the stamp then names the exact weights).


def stamp_of(name: str, use_flip_tta: bool) -> str:
    """The stamp, worked out from the settings alone (no model is loaded): "dinov2_small@main|flip"."""
    revision = ENCODERS[name]["revision"]
    if use_flip_tta:
        tta_text = "flip"
    else:
        tta_text = "notta"
    return f"{name}@{revision}|{tta_text}"


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Make every row length 1, so the dot product of two rows = their cosine similarity.
    The tiny 1e-12 avoids dividing by zero for an all-zero row."""
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / (lengths + 1e-12)).astype(np.float32)


def mirror_all(images: list) -> list:
    """The mirror image (left <-> right) of every picture."""
    mirrored = []
    for image in images:
        mirrored.append(ImageOps.mirror(image))
    return mirrored


class Encoder:
    def __init__(self, name: str, use_flip_tta: bool = True, batch_size: int = 8):
        spec = ENCODERS[name]
        self.name = name
        self.kind = spec["kind"]
        self.hf_id = spec["hf_id"]
        self.revision = spec["revision"]
        self.use_flip_tta = use_flip_tta
        self.batch_size = batch_size
        self.model = None           # filled in by load_encoder() for the real models
        self.processor = None       # turns pictures into the numbers the model expects

    @property
    def stamp(self) -> str:
        # "@property" lets you write encoder.stamp (no brackets), like reading a normal attribute.
        return stamp_of(self.name, self.use_flip_tta)

    def embed(self, images: list) -> np.ndarray:
        """Vectors for a list of images. With flip TTA ("test-time augmentation") we also embed the mirror
        image and add the two: a product photographed from the left then matches one shot from the right."""
        vectors = self.embed_without_tta(images)
        if self.use_flip_tta:
            vectors = vectors + self.embed_without_tta(mirror_all(images))
        return normalise(vectors)

    def embed_without_tta(self, images: list) -> np.ndarray:
        if self.kind == "hist":
            rows = []
            for image in images:
                rows.append(colour_edge_shape_features(image))
            return np.stack(rows)

        # Real models: small batches keep the memory use low.
        batches = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start:start + self.batch_size]
            batches.append(self.embed_batch(batch))
        return np.concatenate(batches)

    def embed_batch(self, batch: list) -> np.ndarray:
        import torch

        inputs = self.processor(images=batch, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():                      # no training: saves memory and time
            if self.kind == "siglip":
                features = self.model.get_image_features(**inputs)
                # transformers 5.x returns an object; older versions returned the numbers directly.
                if not torch.is_tensor(features):
                    features = features.pooler_output
            else:                                         # dino: the CLS token = one summary of the image
                features = self.model(**inputs).pooler_output
        return features.float().cpu().numpy()


def colour_edge_shape_features(image: Image.Image) -> np.ndarray:
    """The offline 'model'. Three parts, joined into one vector:
      1. a colour histogram of the FOREGROUND (how much of each colour the product has)
      2. an edge-direction histogram (are the edges mostly horizontal, vertical, diagonal?)
      3. a coarse silhouette (the product's outline, shrunk to 12x12)
    Foreground = pixels whose colour differs from the photo's border: the border is usually wall or background,
    so this is a crude stand-in for cropping."""
    small = image.resize((96, 96))
    pixels = np.asarray(small, dtype=np.int32)                 # 96 x 96 x 3 numbers (red, green, blue)

    # Background colour = the middle value of all the border pixels.
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    background = np.median(border, axis=0)
    difference = np.abs(pixels - background).sum(axis=2)
    foreground = difference > 60                               # True where the pixel isn't the wall
    if foreground.sum() < 50:                                  # nothing stood out: use the whole picture
        foreground[:] = True

    # 1. Colour histogram: each channel 0..255 -> 8 levels, so 8 x 8 x 8 = 512 colour boxes.
    levels = pixels // 32
    colour_codes = levels[:, :, 0] * 64 + levels[:, :, 1] * 8 + levels[:, :, 2]
    colour = np.bincount(colour_codes[foreground], minlength=512).astype(np.float32)

    # 2. Edge directions: 16 direction boxes, each weighted by how strong the edge is.
    grey = np.asarray(small.convert("L").filter(ImageFilter.SMOOTH), dtype=np.float32)
    change_y, change_x = np.gradient(grey)
    angle_box = ((np.arctan2(change_y, change_x) + np.pi) / (2 * np.pi) * 16).astype(int) % 16
    strength = np.hypot(change_x, change_y)
    edges = np.bincount(angle_box.ravel(), weights=strength.ravel(), minlength=16).astype(np.float32)

    # 3. Silhouette: cut the foreground to its own bounding box, shrink to 12x12, so the SAME shape gives the
    #    same pattern wherever it sits in the photo and however big it is.
    rows_with_product, columns_with_product = np.nonzero(foreground)
    box = foreground[rows_with_product.min():rows_with_product.max() + 1,
                     columns_with_product.min():columns_with_product.max() + 1]
    box_image = Image.fromarray(box.astype(np.uint8) * 255).resize((12, 12))
    shape = np.asarray(box_image, dtype=np.float32).ravel()

    # Each part made length 1, then joined. The 0.3 gives the edges less say than colour and shape.
    colour = colour / (np.linalg.norm(colour) + 1e-9)
    edges = 0.3 * edges / (np.linalg.norm(edges) + 1e-9)
    shape = shape / (np.linalg.norm(shape) + 1e-9)
    return np.concatenate([colour, edges, shape])


def load_encoder(name: str, use_flip_tta: bool = True, batch_size: int = 8) -> Encoder:
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; choose one of {sorted(ENCODERS)}")
    encoder = Encoder(name, use_flip_tta, batch_size)
    if encoder.kind == "hist":
        return encoder                     # the offline model needs nothing loaded

    # Heavy imports only here, so the offline path never needs torch/transformers installed.
    import torch
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16              # half precision: half the memory, only worth it on a GPU
    else:
        device = "cpu"
        dtype = torch.float32

    model = AutoModel.from_pretrained(encoder.hf_id, revision=encoder.revision, dtype=dtype)
    encoder.model = model.to(device).eval()            # eval() = switch off training-only behaviour
    if encoder.kind == "siglip":
        encoder.processor = AutoProcessor.from_pretrained(encoder.hf_id, revision=encoder.revision)
    else:
        encoder.processor = AutoImageProcessor.from_pretrained(encoder.hf_id, revision=encoder.revision)
    return encoder
