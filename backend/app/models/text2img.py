import cv2
import torch

from transformers import CLIPModel, CLIPProcessor
from app.config import settings
from app.services.executor import run_blocking


class Text2ImgService:
    def __init__(self):
        self.model = None
        self.processor = None

    async def initialize(self):
        self.model = (await run_blocking(CLIPModel.from_pretrained, "openai/clip-vit-base-patch32", cache_dir="pretrained_models")).to(settings.device)
        self.processor = await run_blocking(CLIPProcessor.from_pretrained, "openai/clip-vit-base-patch32", cache_dir="pretrained_models")

    async def encode_image(self, path: str):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot read image {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        inputs = (await run_blocking(self.processor, images=img, return_tensors="pt")).to(settings.device)
        with torch.no_grad():
            emb = (await run_blocking(self.model.get_image_features, **inputs)).detach()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.detach().cpu().numpy()[0]

    async def encode_text(self, text: str):
        inputs = (await run_blocking(self.processor, text=[text], return_tensors="pt")).to(settings.device)
        with torch.no_grad():
            emb = (await run_blocking(self.model.get_text_features, **inputs)).detach()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

text2img = Text2ImgService()