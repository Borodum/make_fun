from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import image

app = FastAPI(
    docs_url="/api/swagger",
    redoc_url="/api/redocly",
    openapi_url="/api/schema",
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image.router)