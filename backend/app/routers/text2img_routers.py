import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas.text2img import SearchRequest
from app.services.qdrant import qdrant

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/text2img",
    tags=["text2img"]
)


@router.post("/upload/")
async def upload_image(data: SearchRequest):
    try:
        paths = await qdrant.search_by_text(data.query, data.top_k)
        # Ensure we always return a list of paths (may be empty)
        return {"paths": paths or []}
    except Exception as e:
        logger.exception("Error in /text2img/upload/")
        # Don't propagate server errors to the client as 500; return an empty result
        return {"paths": []}


