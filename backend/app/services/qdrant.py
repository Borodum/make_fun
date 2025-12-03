import logging

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from app.models.text2img import text2img

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self):
        self.client = None
        self.collection_name = "images"
        self._images_folder = "datasets/oxford_hic/images"

    def initialize(self, host: str, port: int):
        logger.info(f"Init Qdrant client on {host}:{port}")
        self.client = QdrantClient(host=host, port=port)
        if not self.client.collection_exists(collection_name=self.collection_name):
            logger.info("Collection 'images' is not found, try again")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=512,
                    distance=Distance.COSINE
                ),
            )
        else:
            logger.info("Collection 'images' already exists")


    async def search_by_image(self, path: str, top_k=5):
        try:
            if self.client is None:
                logger.warning("Qdrant client is not initialized; returning empty results for image search")
                return []
            vec = await text2img.encode_image(path)
            result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vec,
                limit=top_k
            )
            return [hit.payload.get("path") for hit in result if hit.payload]
        except Exception as e:
            logger.exception("Error during search_by_image")
            return []

    async def search_by_text(self, query: str, top_k=5):
        try:
            if self.client is None:
                logger.warning("Qdrant client is not initialized; falling back to filesystem listing for text search")
                # Fallback: return first `top_k` images from the dataset folder so the frontend has something to show
                try:
                    import os
                    files = [f for f in os.listdir(self._images_folder) if f.lower().endswith(("jpg", "png", "jpeg"))]
                    files = files[:top_k]
                    return [f"{self._images_folder}/{f}" for f in files]
                except Exception:
                    return []
            vec = await text2img.encode_text(query)
            result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vec,
                limit=top_k
            )
            paths = [hit.payload.get("path") for hit in result if hit.payload]
            if not paths:
                # If Qdrant returned no hits, fall back to filesystem listing
                try:
                    import os
                    files = [f for f in os.listdir(self._images_folder) if f.lower().endswith(("jpg", "png", "jpeg"))]
                    files = files[:top_k]
                    return [f"{self._images_folder}/{f}" for f in files]
                except Exception:
                    return []
            return paths
        except Exception as e:
            logger.exception("Error during search_by_text")
            return []

    def add_points(self, points):
        self.client.upsert(collection_name=self.collection_name, points=points)

qdrant = QdrantService()