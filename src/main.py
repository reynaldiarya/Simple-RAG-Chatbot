from contextlib import asynccontextmanager
from src import logger, settings, router
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Simple RAG Chatbot API...")
    from src.api.routes import get_rag_service
    get_rag_service()
    yield
    logger.info("Stopping Simple RAG Chatbot API...")

app = FastAPI(
    title="Simple RAG Chatbot API",
    lifespan=lifespan
)

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host=settings.host, port=settings.port, reload=True)
