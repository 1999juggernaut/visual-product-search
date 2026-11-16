"""The re-ranker: a vision-language model looks at the photo and the top candidates and picks one.

WHAT: ``Reranker.choose(photo, query_sha, candidates, model)`` -> RerankOutcome
      The model sees the customer photo and N numbered catalogue images and answers in JSON:
          {"choice": 0..N, "confidence": 0..1, "reason": "..."}     (0 = "none of these")
WHY AN INDEX AND NOT A SKU (the model outputs an index, never an id): a model can invent a SKU
      that looks real. A number 1..N can only point at a candidate WE supplied, and we copy every field of the
      answer from our own catalogue row. ``parse_choice`` VALIDATES the number before we trust it.
WHY A CACHE: the same photo + same candidates + same model + same prompt version = the same question. We
      answer it once and keep the answer in data_dir/rerank_cache/ (DEMO_MODE, one bill per
      unique request). The key is a sha256 of those parts, so a prompt change makes new keys automatically.
WHERE IT RUNS (the start of the model name, same as the Colab notebook):
      openrouter:<id>   OpenRouter API (needs VPI_OPENROUTER_API_KEY; the photo LEAVES this machine)
      local:<hf id>     a Hugging Face vision-language model on this machine's GPU (Colab)
"""

import base64
import json
import re
import time

from pydantic import BaseModel

from shared.events import sha256_text
from shared.images import load_path, to_jpeg

PROMPT_VERSION = "v1"
PROMPT = """You are matching a customer's photo of a product to a catalogue.
The FIRST image is the customer's photo. The next {n} images are catalogue products numbered 1 to {n}.
Pick the catalogue product that is the SAME product as in the photo (same shape and same finish).
If none of them is the same product, answer choice 0.
Answer with JSON only: {{"choice": <0-{n}>, "confidence": <0.0-1.0>, "reason": "<one short sentence>"}}"""

# JSON schema for models that support "structured outputs". additionalProperties False = no extra keys.
RERANK_SCHEMA = {
    "name": "rerank_choice",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["choice", "confidence", "reason"],
        "properties": {
            "choice": {"type": "integer"},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
    },
}
# Which JSON mode each free model supports (checked on OpenRouter 2026-10-16). Unknown = prompt only.
JSON_MODES = {
    "nvidia/nemotron-3.5-super-vl:free": "json_schema",
    "google/gemma-4-31b-it:free": "json_object",
    "google/gemma-4-26b-a4b-it:free": "json_object",
}
PHOTO_SIDE = 512            # pixels: smaller images = fewer tokens = cheaper and faster
CANDIDATE_SIDE = 256


class RerankError(Exception):
    """The re-ranker failed (network, bad JSON, invalid choice). identify() catches it and falls back."""


class RerankOutcome(BaseModel):
    choice: int | None          # 0-based position in the candidate list, or None = "none of them match"
    confidence: float
    reason: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cached: bool = False


def parse_choice(text: str, n: int) -> tuple:
    """The validator. Takes the model's raw text; returns (0-based position or None, confidence, reason).
    Some models "think out loud" inside <think> ... </think> first; we drop that and read the LAST {...} block."""
    if text is None:
        text = ""
    # re = "regular expressions", a mini-language for finding patterns in text.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)       # remove the thinking part
    json_blocks = re.findall(r"\{[^{}]*\}", text)                     # every {...} block
    if len(json_blocks) == 0:
        raise RerankError("no JSON object in the answer")

    try:
        data = json.loads(json_blocks[-1])                            # [-1] = the last one
        choice = int(data["choice"])
        confidence = float(data.get("confidence", 0.0))
    except (ValueError, KeyError, TypeError) as error:
        raise RerankError(f"answer is not the agreed JSON: {error}") from None

    if choice < 0 or choice > n:
        raise RerankError(f"choice {choice} is outside 0..{n}")       # an invented candidate: refuse it

    # Keep confidence between 0 and 1, and the reason short.
    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0
    reason = str(data.get("reason", ""))[:200]

    if choice == 0:
        return None, confidence, reason
    return choice - 1, confidence, reason          # the model counts from 1; Python lists count from 0


def as_data_uri(jpeg_bytes: bytes) -> str:
    """Pack JPEG bytes as text ("data URI"), the way the OpenAI-style API accepts images."""
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


class Reranker:
    def __init__(self, settings):
        self.settings = settings
        self.cache_dir = settings.path("rerank_cache")
        self.client = None              # the OpenRouter client, made on first use
        self.local_models = {}          # local models, loaded on first use: {hf id: (processor, model)}

    def cache_key(self, model: str, query_sha: str, candidate_ids: list) -> str:
        parts = [model, PROMPT_VERSION, query_sha]
        for candidate_id in candidate_ids:
            parts.append(candidate_id)
        return sha256_text("|".join(parts))

    def choose(self, photo, query_sha: str, candidates: list, model: str) -> RerankOutcome:
        candidate_ids = []
        for candidate in candidates:
            candidate_ids.append(candidate["image_id"])
        cache_file = self.cache_dir / (self.cache_key(model, query_sha, candidate_ids) + ".json")

        # 1. Asked before? Answer from the cache, for free.
        if self.settings.rerank_cache and cache_file.exists():
            outcome = RerankOutcome.model_validate_json(cache_file.read_text(encoding="utf-8"))
            outcome.cached = True
            return outcome

        # 2. Ask the model.
        prompt = PROMPT.format(n=len(candidates))
        if model.startswith("openrouter:"):
            candidate_jpegs = []
            for candidate in candidates:
                candidate_jpegs.append(to_jpeg(load_path(candidate["image_path"]), CANDIDATE_SIDE))
            photo_jpeg = to_jpeg(photo, PHOTO_SIDE)
            model_id = model.removeprefix("openrouter:")
            text, tokens_in, tokens_out = self.ask_openrouter(model_id, prompt, photo_jpeg, candidate_jpegs)
        elif model.startswith("local:"):
            model_id = model.removeprefix("local:")
            text, tokens_in, tokens_out = self.ask_local(model_id, prompt, photo, candidates)
        else:
            raise RerankError(f"unknown re-ranker {model!r}: use none, openrouter:<id> or local:<id>")

        # 3. Check the answer, then remember it.
        choice, confidence, reason = parse_choice(text, len(candidates))
        outcome = RerankOutcome(choice=choice, confidence=confidence, reason=reason, model=model,
                                tokens_in=tokens_in, tokens_out=tokens_out)
        if self.settings.rerank_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(outcome.model_dump_json(), encoding="utf-8")
        return outcome

    # ----- OpenRouter (an OpenAI-compatible API) -----
    def ask_openrouter(self, model_id, prompt, photo_jpeg, candidate_jpegs):
        if not self.settings.openrouter_api_key:
            raise RerankError("no OpenRouter key: set VPI_OPENROUTER_API_KEY in .env")
        import openai

        if self.client is None:
            self.client = openai.OpenAI(base_url="https://openrouter.ai/api/v1",
                                        api_key=self.settings.openrouter_api_key, timeout=60)

        # The message: the prompt, the photo, then "Candidate 1:" + image, "Candidate 2:" + image, ...
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": as_data_uri(photo_jpeg)}}]
        number = 1
        for jpeg in candidate_jpegs:
            content.append({"type": "text", "text": f"Candidate {number}:"})
            content.append({"type": "image_url", "image_url": {"url": as_data_uri(jpeg)}})
            number += 1

        extra_options = {}
        json_mode = JSON_MODES.get(model_id)
        if json_mode == "json_schema":
            extra_options["response_format"] = {"type": "json_schema", "json_schema": RERANK_SCHEMA}
        elif json_mode == "json_object":
            extra_options["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(3):                      # retry only errors that can succeed on a retry
            try:
                response = self.client.chat.completions.create(
                    model=model_id, messages=[{"role": "user", "content": content}],
                    temperature=0, max_tokens=300, **extra_options)
                text = response.choices[0].message.content or ""
                usage = response.usage
                tokens_in = getattr(usage, "prompt_tokens", 0)      # 0 if the API didn't say
                tokens_out = getattr(usage, "completion_tokens", 0)
                return text, tokens_in, tokens_out
            except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as error:
                last_error = error
                time.sleep(2 ** attempt)               # wait 1 s, then 2 s, then 4 s
            except openai.APIError as error:           # other refusals: retrying won't help
                raise RerankError(f"OpenRouter refused: {error}") from None
        raise RerankError(f"OpenRouter failed 3 times: {last_error}")

    # ----- a local vision-language model (Colab GPU) -----
    def ask_local(self, hf_id, prompt, photo, candidates):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if hf_id not in self.local_models:
            processor = AutoProcessor.from_pretrained(hf_id)
            model = AutoModelForImageTextToText.from_pretrained(hf_id, dtype="auto", device_map="auto").eval()
            self.local_models[hf_id] = (processor, model)
        processor, model = self.local_models[hf_id]

        images = [photo]
        content = [{"type": "text", "text": prompt}, {"type": "image"}]
        number = 1
        for candidate in candidates:
            images.append(load_path(candidate["image_path"]))
            content.append({"type": "text", "text": f"Candidate {number}:"})
            content.append({"type": "image"})
            number += 1

        chat_text = processor.apply_chat_template([{"role": "user", "content": content}],
                                                  add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=[chat_text], images=images, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_length:]                  # only the answer, not the prompt
        answer = processor.decode(new_tokens, skip_special_tokens=True)
        return answer, int(prompt_length), int(len(new_tokens))
