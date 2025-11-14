import csv
import os
import logging

import requests
from tqdm import tqdm

from app.models.text2img import text2img
from app.services.qdrant import qdrant

logger = logging.getLogger(__name__)

CSV_PATH = "datasets/oxford_hic/oxford_hic_image_info.csv"
SAVE_FOLDER = "datasets/oxford_hic/images"

async def download_images():
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    idx = 0
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # for row in tqdm(reader, desc="Downloading images"):
        for row in reader:
            if idx >= 1000:
                break
            image_id = row["image_id"]
            url = row["image_url"]

            ext = url.split(".")[-1]
            if len(ext) > 4:
                ext = "jpg"

            filename = os.path.join(SAVE_FOLDER, f"{image_id}.{ext}")
            idx += 1
            if os.path.exists(filename):
                continue

            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()

                with open(filename, "wb") as img_file:
                    img_file.write(r.content)

            except Exception as e:
                logger.info(f"Failed to download {url}: {e}")
            if idx % 100 == 0:
                logger.info(url + " idx: " + str(idx))

    logger.info("All images downloaded!")

async def index_images():
    folder = "datasets/oxford_hic/images"
    os.makedirs(folder, exist_ok=True)
    points = []
    idx = 0

    for filename in os.listdir(folder):
        if not filename.lower().endswith(("jpg", "png", "jpeg")):
            continue

        path = os.path.join(folder, filename)
        try:
            vector = await text2img.encode_image(path)
        except Exception as e:
            logger.info(f"Failed to process {path}: {e}")
            continue

        points.append({
            "id": idx,
            "vector": vector,
            "payload": {"path": path}
        })
        idx += 1
        if idx % 100 == 0:
            logger.info("index: " + filename + " idx: " + str(idx))
            qdrant.add_points(points)
            points = []

    qdrant.add_points(points)
    logger.info(f"Indexed {idx} images into Qdrant.")