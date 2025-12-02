from fastapi import APIRouter, UploadFile, File

from models.joke_generator import get_joke
from app.schemas.image import ImageUrl
from app.models.img2caption import img2caption
from app.models.caption2joke import caption2joke

router = APIRouter(
    prefix="/images",
    tags=["images"]
)

@router.post("/upload/")
async def upload_image(data: ImageUrl):
    times = 3
    caption = await img2caption.process(data.url)
    jokes = [await caption2joke.process(caption) for _ in range(times)]
    return {"jokes": jokes}


