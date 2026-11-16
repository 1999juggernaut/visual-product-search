"""Optional cropping: find the product in a busy photo and cut it out before embedding.

WHAT: ``Cropper("owlv2").crop(img)`` -> (cropped_image, "owlv2") when a product was found,
      or (original_image, "crop_fallback") when nothing scored above ``min_score``.
      ``Cropper("none")`` returns the photo unchanged with "none".
WHY OFF BY DEFAULT on the laptop: OWLv2 ("Open-World Localisation", google/owlv2-base-patch16-ensemble,
      ~600 MB) is slow on a CPU. Catalogue photos are already clean; customer photos are not. The Colab
      bake-off measures "crop vs no crop" so we switch it on only if it pays (one knob per arm).
WHY a named fallback: when the cropper finds nothing we still answer, and the trail says crop_fallback, so a
      run of bad answers can be traced to the cropper (named fallback events).
"""

from PIL import Image

OWLV2_ID = "google/owlv2-base-patch16-ensemble"


class Cropper:
    def __init__(self, mode: str = "none", prompts=None, min_score: float = 0.15, pad: float = 0.08):
        self.mode = mode
        if prompts:
            self.prompts = prompts
        else:
            self.prompts = ["a product"]
        self.min_score = min_score         # boxes less confident than this are ignored
        self.pad = pad                     # extra margin around the box, as a share of its size (8%)
        self.model = None
        self.processor = None
        self.device = "cpu"

        if mode == "owlv2":
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            if torch.cuda.is_available():
                self.device = "cuda"
            self.processor = Owlv2Processor.from_pretrained(OWLV2_ID)
            self.model = Owlv2ForObjectDetection.from_pretrained(OWLV2_ID).to(self.device).eval()

    def crop(self, image: Image.Image) -> tuple[Image.Image, str]:
        if self.mode == "none":
            return image, "none"

        import torch
        inputs = self.processor(text=[self.prompts], images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        # OWLv2 pads photos to a square, so its boxes are measured against the LONGER side.
        side = max(image.size)
        found = self.processor.post_process_object_detection(
            outputs, threshold=self.min_score, target_sizes=torch.tensor([[side, side]]))[0]
        if len(found["scores"]) == 0:
            return image, "crop_fallback"

        best = int(found["scores"].argmax())                   # the most confident box
        left, top, right, bottom = found["boxes"][best].tolist()
        margin_x = (right - left) * self.pad
        margin_y = (bottom - top) * self.pad
        left = max(0, left - margin_x)
        top = max(0, top - margin_y)
        right = min(image.width, right + margin_x)
        bottom = min(image.height, bottom + margin_y)

        if right - left < 32 or bottom - top < 32:             # a sliver is not a product
            return image, "crop_fallback"
        return image.crop((left, top, right, bottom)), "owlv2"
