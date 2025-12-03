import logging

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, \
    GPT2Tokenizer, GPT2LMHeadModel
import cv2
import gdown
import os
import zipfile

from app.config import settings
from app.services.executor import run_blocking

logger = logging.getLogger(__name__)

class Caption2JokeService:
    def __init__(self):
        self.tokenizer = None
        self.model = None
    async def initialize(self):
        DEST_DIR = "pretrained_models"
        ZIP_PATH = "pretrained_models/fine_tuned_gpt2_small.zip"
        EXTRACT_DIR = "pretrained_models/fine_tuned_gpt2_small"
        os.makedirs(DEST_DIR, exist_ok=True)
        if not os.path.exists(ZIP_PATH):
            url = f"https://drive.google.com/file/d/1rcNYOL3274bdlW_Q2Jkk5OwSpzVsI_EQ/view?usp=drive_link"
            logger.info("fine_tuned_gpt2_small start downloading")
            gdown.download(url, ZIP_PATH, quiet=False, fuzzy=True)
            logger.info("fine_tuned_gpt2_small downloaded")
        else:
            logger.info("fine_tuned_gpt2_small already downloaded")

        if not os.path.exists(EXTRACT_DIR):
            os.makedirs(EXTRACT_DIR, exist_ok=True)
            logger.info("Unpacking fine_tuned_gpt2_small.zip")
            with zipfile.ZipFile(ZIP_PATH, "r") as z:
                z.extractall(EXTRACT_DIR)
            logger.info("Unpacked fine_tuned_gpt2_small.zip")
        else:
            logger.info("fine_tuned_gpt2_small.zip already unpacked")

        model_path = "pretrained_models/fine_tuned_gpt2_small"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

        # Wrap with DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(settings.device)
        self.model.eval()

    async def process(self, caption: str, max_length=100, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
        input_ids = (await run_blocking(self.tokenizer.encode, caption, return_tensors="pt")).to(settings.device)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = await run_blocking(self.model.generate,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        joke = await run_blocking(self.tokenizer.decode, outputs[0], skip_special_tokens=True)
        return joke



caption2joke = Caption2JokeService()