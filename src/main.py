import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from src import logger, settings, router
from src.services.rag import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes services at startup and stores them in app.state."""
    logger.info("Starting Simple RAG Chatbot API...")

    # Initialize RAG service
    app.state.rag_service = RAGService()

    yield

    logger.info("Stopping Simple RAG Chatbot API...")


# Disable API documentation in production
_is_production = os.getenv("ENVIRONMENT", settings.environment).lower() == "production"

app = FastAPI(
    title="Simple RAG Chatbot API",
    lifespan=lifespan,
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
    openapi_url=None if _is_production else "/openapi.json",
)

app.include_router(router, prefix="/api")


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Applies security headers to HTTP responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host=settings.host, port=settings.port, reload=True)
