from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import query, ingestion

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(query.router)
app.include_router(ingestion.router)