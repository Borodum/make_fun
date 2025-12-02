import logging

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, \
    GPT2Tokenizer, GPT2LMHeadModel
import cv2
import os

from app.config import settings
from app.services.executor import run_blocking

logger = logging.getLogger(__name__)

class Caption2JokeService:
    def __init__(self):
        self.tokenizer = None
        self.model = None
    async def initialize(self):
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