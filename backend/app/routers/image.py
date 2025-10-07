from fastapi import APIRouter, UploadFile, File

router = APIRouter(
    prefix="/images",
    tags=["images"]
)

@router.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    jokes = [
        "Когда фотошоп решил, что у тебя недостаточно красивый день, он добавил немного фильтров... и твоего бывшего на задний план!",
        "Это фото настолько эпичное, что даже камера попросила автограф!",
        "Если бы эта картинка могла говорить, она бы сказала: 'Я слишком крута для этого мира!",
    ]
    return {"jokes": jokes}