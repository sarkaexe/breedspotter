"""Minimal fine‑tuning script – optional."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from breedspotter.data import load_metadata

class DogsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: CLIPProcessor, breed2idx: dict[str, int]):
        self.df = df
        self.processor = processor
        self.breed2idx = breed2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # [3, 224, 224]
        label = torch.tensor(self.breed2idx[row["breed"]], dtype=torch.long)
        return pixel_values, label


def fine_tune(epochs: int, batch_size: int, save_path: Path):
    df, breeds = load_metadata()
    breed2idx = {b: i for i, b in enumerate(breeds)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    head = torch.nn.Linear(model.config.projection_dim, len(breeds)).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-5)
    ds = DogsDataset(df, processor, breed2idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    ce = torch.nn.CrossEntropyLoss()
    model.train(), head.train()

    for epoch in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            feats = model.get_image_features(pixel_values=x)
            logits = head(feats)
            loss = ce(logits, y)
            loss.backward()
            opt.step(); opt.zero_grad()
            pbar.set_postfix(loss=loss.item())

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict()}, save_path)
    print(f"✅ Saved fine‑tuned weights to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fine‑tune CLIP on Stanford Dogs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=Path, default=Path("weights/dog_classifier.pt"))
    args = parser.parse_args()
    fine_tune(args.epochs, args.batch_size, args.save_path)