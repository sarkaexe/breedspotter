"""Zero‑shot (CLIP) or fine‑tuned dog‑breed classifier."""
from __future__ import annotations

import pathlib
from typing import List, Tuple, Dict

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

_DEFAULT_MODEL = "openai/clip-vit-base-patch32"

class ZeroShotDogClassifier:
    """Uses CLIP logits to pick the best breed."""

    def __init__(self, breeds: List[str], model_name: str = _DEFAULT_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.breeds = breeds

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Tuple[str, float, List[Tuple[str, float]]]:
        prompts = [f"a photo of a {b}" for b in self.breeds]
        inputs = self.processor(text=prompts, images=image, return_tensors="pt", padding=True).to(self.device)
        logits = self.model(**inputs).logits_per_image  # shape [1, N]
        probs = logits.softmax(dim=1).squeeze(0)
        top_idx = int(probs.argmax())
        ranked = list(zip(self.breeds, probs.tolist()))
        return self.breeds[top_idx], float(probs[top_idx]), ranked


class FineTunedDogClassifier:
    """Loads a custom head trained via `train.py`.  Uses the same CLIP backbone."""

    def __init__(self, weight_path: str, breeds: List[str]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.breeds = breeds
        # Re‑create model + head
        self.model = CLIPModel.from_pretrained(_DEFAULT_MODEL).to(self.device)
        self.head = torch.nn.Linear(self.model.config.projection_dim, len(breeds)).to(self.device)
        self.head.load_state_dict(checkpoint["head"])
        self.model.eval(), self.head.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image):
        image_inputs = self.model.get_image_features(**self.model.get_image_features.tokenizer(image)).to(self.device)
        feats = self.model.get_image_features(pixel_values=image_inputs).float()
        logits = self.head(feats)
        probs = logits.softmax(dim=-1)
        top_idx = int(probs.argmax())
        ranked = list(zip(self.breeds, probs.squeeze(0).tolist()))
        return self.breeds[top_idx], float(probs[0, top_idx]), ranked


def load_classifier(breeds: List[str]) -> ZeroShotDogClassifier | FineTunedDogClassifier:
    """Factory that prefers fine‑tuned weights if present."""
    ckpt = pathlib.Path("weights/dog_classifier.pt")
    if ckpt.exists():
        return FineTunedDogClassifier(str(ckpt), breeds)
    return ZeroShotDogClassifier(breeds)