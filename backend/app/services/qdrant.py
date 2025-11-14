import logging

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from app.models.text2img import text2img

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self):
        self.client = None
        self.collection_name = "images"

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
        vec = await text2img.encode_image(path)
        result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vec,
            limit=top_k
        )
        return [hit.payload["path"] for hit in result]

    async def search_by_text(self, query: str, top_k=5):
        vec = await text2img.encode_text(query)
        result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vec,
            limit=top_k
        )
        return [hit.payload["path"] for hit in result]

    def add_points(self, points):
        self.client.upsert(self.collection_name, points)

qdrant = QdrantService()