import base64
import io
import json
import logging
import math
import string
from collections import defaultdict

import nltk
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import numpy as np
import requests
from PIL import Image
from app.config import settings
from app.services.executor import run_blocking
from nltk import word_tokenize
from transformers import BlipProcessor, BlipForConditionalGeneration, ViTModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import gdown
import zipfile

logger = logging.getLogger(__name__)


def clean_tokenize(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    return tokens

def merge_vocabs(vocab1, vocab2):
    # Стартовые специальные токены
    merged_vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
    idx = 4

    # Собираем все уникальные токены, кроме спецтокенов
    special_tokens = {'<pad>', '<start>', '<end>', '<unk>'}
    all_tokens = set(vocab1.keys()).union(set(vocab2.keys())) - special_tokens

    # Добавляем токены в итоговый словарь
    for token in sorted(all_tokens):
        merged_vocab[token] = idx
        idx += 1

    return merged_vocab


def create_vocab(annotations_path, min_freq=5):
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    image_to_captions = defaultdict(list)
    for ann in annotations['annotations']:
        image_to_captions[ann['image_id']].append(ann['caption'])

    word_freq = {}
    for captions in image_to_captions.values():
        for caption in captions:
            tokens = clean_tokenize(caption.lower())
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1

    vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
    idx = 4

    for token, freq in word_freq.items():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1

    return vocab

class CrossAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        assert emb_size % num_heads == 0, "Embedding size must be divisible by number of heads."

        self.W_q = nn.Linear(emb_size, emb_size, bias = False)
        self.W_k = nn.Linear(emb_size, emb_size, bias = False)
        self.W_v = nn.Linear(emb_size, emb_size, bias = False)

        self.out = nn.Linear(emb_size, emb_size)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, y, mask = None):
        batch_size, num_tokens_q, _ = x.shape
        _, num_tokens_k, _ = y.shape

        Q = self.W_q(x)  # (B, num_tokens_q, emb_size)
        K = self.W_k(y) # (B, num_tokens_k, emb_size)
        V = self.W_v(y) # (B, num_tokens_k, emb_size)

        Q = Q.view(batch_size, num_tokens_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, num_tokens_q, head_dim)
        K = K.view(batch_size, num_tokens_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, num_tokens_k, head_dim)
        V = V.view(batch_size, num_tokens_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, num_tokens_k, head_dim)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim) #(B, num_heads, num_tokens_q, num_tokens_k)

        if mask is not None:
            mask = mask.unsqueeze(-1).expand(-1, -1, num_tokens_k) #(B, num_tokens_q, num_tokens_k)
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1) #(B, num_heads, num_tokens_q, num_tokens_k)
            scores = scores.masked_fill(mask == False, float(-1e9))

        attn_weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        attended = attn_weights @ V  # (B, num_heads, num_tokens_q, head_dim)

        concat = attended.transpose(1, 2).contiguous().view(batch_size, num_tokens_q, self.emb_size)  # (B, num_tokens_q, emb_size)
        return self.proj_dropout(self.out(concat))

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads

        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias = False)
        self.out = nn.Linear(dim, dim, bias = False)

        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self,  x, mask = None,  return_attn=False):
        B, num_patches, embed_dim = x.shape

        qkv = self.qkv(x) # (B, num_patches, 3*embed_dim)
        qkv = qkv.reshape(B, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) #(3, B, num_heads, num_patches, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, num_patches, head_dim)

                                        #How important it is for token i to pay attention to token j.
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale #[B, num_heads, N, N]

        if mask is not None:
            # mask: (B, N, N)
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1) #[B, num_heads, N, N]
            attn_scores = attn_scores.masked_fill(mask == False, float(-1e9))

        attn_probs = attn_scores.softmax(dim=-1) #[B, num_heads, N, N]
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = attn_probs @ v  # (B, num_heads, num_patches, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, num_patches, embed_dim)

        if return_attn:
          return self.out(attn_output), attn_probs
        else:
          return self.out(attn_output) #(B, num_patches, embed_dim)

class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout):
        super().__init__()

        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, y, self_mask=None, cross_mask=None):
        x = x + self.attn(self.norm1(x), mask = self_mask)
        x = x + self.cross_attn(self.norm2(x), y, mask = cross_mask)
        x = x + self.mlp(self.norm3(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, mlp_dim, num_layers, dropout=0.1, max_len=512):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = PositionalEncoding(embed_dim, max_len)

        self.layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for i in range(num_layers)
        ])

        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # x — индексы токенов (B, T)
        x = self.embed_tokens(x)  # (B, T, D)
        x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask=self_mask, cross_mask=cross_mask)

        logits = self.output_proj(x)  # (B, T, vocab_size)
        return logits


class ImageCaptioningTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 num_heads,
                 mlp_dim,
                 num_layers,
                 dropout):
        super().__init__()
        # Энкодер изображений (ViT)
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224", add_pooling_layer=False, cache_dir="pretrained_models")

        # Декодер текста
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_len=512
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

    def _create_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
        return mask.bool().unsqueeze(0)  # [seq_len, seq_len]

    def forward(self, images, captions, self_mask=None, cross_mask=None):
        encoder_output = self.encoder(images).last_hidden_state  # (B, 1 + N, D)
        encoder_output_tokens = encoder_output[:, 1:, :]  # (B, N, D)
        logits = self.decoder(captions, encoder_output_tokens, self_mask=self_mask, cross_mask=cross_mask)
        return logits

    def _beam_search(self, image, vocab, beam_width=3, max_length=20):
        self.eval()

        start_token = vocab['<start>']
        end_token = vocab['<end>']
        device = image.device
        batch_size = 1

        beams = [{'tokens': [start_token], 'prob': 0.0}]
        with torch.no_grad():
            encoder_output = self.encoder(image).last_hidden_state  # (B, 1 + N, D)
            encoder_output_tokens = encoder_output[:, 1:, :]  # (B, N, D)
            for i in range(max_length):
                all_cand = []
                for beam in beams:
                    self_mask = self._create_causal_mask(i + 1, device)
                    tokens_tensor = torch.tensor(beam['tokens'], device=device).unsqueeze(0)
                    logits = self.decoder(tokens_tensor, encoder_output_tokens, self_mask=self_mask)
                    logit = logits[:, -1, :]  # последний токен
                    log_probs = F.log_softmax(logit, dim=-1)
                    top_log_probs, top_indices = log_probs.topk(beam_width)
                    for prob, inx in zip(top_log_probs[0], top_indices[0]):
                        all_cand.append({"tokens": beam['tokens'] + [inx.item()], 'prob': beam['prob'] + prob.item()})
                all_cand = sorted(all_cand, key=lambda x: x['prob'], reverse=True)
                beams = all_cand[:beam_width]
                if all([beam['tokens'][-1] == end_token for beam in beams]):
                    break

        best_beam = beams[0]
        output_tokens = best_beam['tokens']
        caption = []
        for token in output_tokens[1:]:
            if token == end_token:
                break
            caption.append(token)

        return torch.tensor(caption).unsqueeze(0)

    def _greedy(self, images, vocab, max_length=20):
        self.eval()

        start_token = vocab['<start>']
        end_token = vocab['<end>']
        device = images.device
        batch_size = images.shape[0]

        encoder_output = self.encoder(images).last_hidden_state  # (B, 1 + N, D)
        encoder_output_tokens = encoder_output[:, 1:, :]  # (B, N, D)
        tokens = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        for i in range(max_length):
            self_mask = self._create_causal_mask(i + 1, device)
            logits = self.decoder(tokens, encoder_output_tokens, self_mask=self_mask)
            logit = logits[:, -1, :]
            pred = logit.argmax(dim=-1)
            tokens = torch.cat([tokens, pred.unsqueeze(-1)], dim=-1)

            if (pred == end_token).all():
                break

        return tokens[:, 1:]

    def sample(self, images, vocab, strategy='greedy', beam_width=3, max_length=20):
        if strategy == "beam":
            if images.shape[0] == 1:
                output = self._beam_search(images, vocab, beam_width, max_length)
            else:
                raise ValueError("Beam search supports only batch size 1. Got batch size {}".format(images.shape[0]))

        elif strategy == "greedy":
            output = self._greedy(images, vocab, max_length)

        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Please choose either 'beam' or 'greedy'.")

        return output


class Img2CaptionService:
    def __init__(self):
        self.model = None
        # self.processor = None
        self.transform = None
        self.big_vocab = None
        self.inv_vocab = None
        self.EMBED_DIM = 768
        self.NUM_HEADS = 12
        self.DEPTH = 6
        self.MLP_DIM = 3072
        self.DROP_RATE = 0.1

    async def initialize(self):
        await run_blocking(nltk.download, 'punkt')
        await run_blocking(nltk.download, 'punkt_tab')

        DEST_DIR = "pretrained_models"
        ZIP_PATH = "pretrained_models/img_2_text.zip"
        EXTRACT_DIR = "pretrained_models/img_2_text"
        os.makedirs(DEST_DIR, exist_ok=True)
        if not os.path.exists(ZIP_PATH):
            url = f"https://drive.google.com/file/d/1f-lJginrQ1FDN_eCW5IwBc9WHa5iaY6N/view?usp=sharing"
            logger.info("img_2_text start downloading")
            gdown.download(url, ZIP_PATH, quiet=False, fuzzy=True)
            logger.info("img_2_text downloaded")
        else:
            logger.info("img_2_text already downloaded")

        if not os.path.exists(EXTRACT_DIR):
            os.makedirs(EXTRACT_DIR, exist_ok=True)
            logger.info("Unpacking img_2_text.zip")
            with zipfile.ZipFile(ZIP_PATH, "r") as z:
                z.extractall(EXTRACT_DIR)
            logger.info("Unpacked img_2_text.zip")
        else:
            logger.info("img_2_text.zip already unpacked")

        vocab_2014 = create_vocab("pretrained_models/img_2_text/captions_train2014.json")
        vocab_2017 = create_vocab("pretrained_models/img_2_text/captions_train2017.json")
        self.big_vocab = merge_vocabs(vocab_2014, vocab_2017)
        self.inv_vocab = {v: k for k, v in self.big_vocab.items()}

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        self.model = ImageCaptioningTransformer(len(self.big_vocab), self.EMBED_DIM, self.NUM_HEADS, self.MLP_DIM, self.DEPTH, self.DROP_RATE)
        checkpoint = torch.load("pretrained_models/img_2_text/vit_epoch_5.pt", map_location=settings.device, weights_only=False)
        state = checkpoint["model_state_dict"]
        state = {k: v for k, v in state.items() if "pooler" not in k}
        self.model.load_state_dict(state)
        self.model.to(settings.device)
        self.model.eval()

    async def process(self, url):
        header, encoded = url.split(",", 1)

        image_bytes = base64.b64decode(encoded)

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        img_tensor = self.transform(pil_image).unsqueeze(0).to(settings.device)

        tokens = await run_blocking(self.model.sample, img_tensor, self.big_vocab, strategy="greedy", max_length=20)
        caption = await run_blocking(self.tokens_to_text, tokens[0].tolist())

        return caption

    def load_image_from_url(self, url):
        img = Image.open(io.BytesIO(requests.get(url).content)).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(settings.device)
        return img_tensor

    def tokens_to_text(self, tokens):
        words = []
        for t in tokens:
            w = self.inv_vocab.get(int(t), "")
            if w in ["<start>", "<end>", "<pad>"]:
                continue
            words.append(w)
        return " ".join(words)

    def get_caption(self, url):
        try:
            image = self.load_image_from_url(url)
        except Exception as e:
            return "url unaccessable"
        tokens = self.model.sample(image, self.big_vocab, strategy="greedy", max_length=20)
        caption = self.tokens_to_text(tokens[0].tolist())
        return caption

img2caption = Img2CaptionService()