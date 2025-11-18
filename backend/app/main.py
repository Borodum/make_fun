import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
    try:
        qdrant.initialize(host="localhost", port=6333)
    except Exception as e:
        # In development, allow app to start even if Qdrant isn't available.
        logging.warning(f"Qdrant init failed (dev fallback): {e}")
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
# Configure CORS early so it applies to all routes (including mounted static)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve dataset images and other static assets under /static
app.mount("/static", StaticFiles(directory="datasets"), name="static")


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(image.router)
app.include_router(text2img_routers.router)