import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.text2img import text2img
from app.routers import image
from app.routers import text2img_routers
from app.services.qdrant import qdrant
from app.utils import index_images, download_images

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    qdrant.initialize(host="localhost", port=6333)
    await text2img.initialize()
    await download_images()
    await index_images()
    yield


app = FastAPI(
    docs_url="/api/swagger",
    redoc_url="/api/redocly",
    openapi_url="/api/schema",
    lifespan=lifespan,
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image.router)
app.include_router(text2img_routers.router)