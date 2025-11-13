from fastapi import APIRouter, UploadFile, File

from models.joke_generator import get_joke
from app.schemas.image import ImageUrl

router = APIRouter(
    prefix="/images",
    tags=["images"]
)

@router.post("/upload/")
async def upload_image(data: ImageUrl):
    times = 3
    jokes = [await get_joke(data.url) for _ in range(times)]
    # jokes = [
    #     "Когда фотошоп решил, что у тебя недостаточно красивый день, он добавил немного фильтров... и твоего бывшего на задний план!",
    #     "Это фото настолько эпичное, что даже камера попросила автограф!",
    #     "Если бы эта картинка могла говорить, она бы сказала: 'Я слишком крута для этого мира!",
    # ]
    return {"jokes": jokes}