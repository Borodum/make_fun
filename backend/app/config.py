import torch
from pydantic.v1 import BaseSettings


class Settings(BaseSettings):
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()