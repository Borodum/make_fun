from fastapi import APIRouter

from app.schemas.text2img import SearchRequest
from app.services.qdrant import qdrant

router = APIRouter(
    prefix="/text2img",
    tags=["text2img"]
)

@router.post("/upload/")
async def upload_image(data: SearchRequest):
    paths = await qdrant.search_by_text(data.query, data.top_k)
    return {"paths": paths}


