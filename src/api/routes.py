import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Body, Request
from src.models.schemas import ChatRequest, ChatResponse
from src.services.rag import RAGService

router = APIRouter()

MAX_UPLOAD_SIZE_MB = 10
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = (".txt", ".md", ".pdf", ".png", ".jpg", ".jpeg")

# Magic bytes map for file type verification
MAGIC_BYTES: dict = {
    b"%PDF": {"pdf"},
    b"\x89PNG\r\n\x1a\n": {"png"},
    b"\xff\xd8\xff": {"jpg", "jpeg"},
}
TEXT_EXTENSIONS = {"txt", "md"}


def _verify_magic_bytes(content: bytes, ext: str) -> bool:
    """Verifies file binary signature against its declared extension."""
    if ext in TEXT_EXTENSIONS:
        try:
            content[:2048].decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    for magic, allowed_exts in MAGIC_BYTES.items():
        if content.startswith(magic):
            return ext in allowed_exts

    return False


def get_rag_service(request: Request) -> RAGService:
    """Retrieves the RAGService instance from app state."""
    return request.app.state.rag_service


def get_doc_service(request: Request):
    """Retrieves the DocumentService instance."""
    return request.app.state.rag_service.doc_service


@router.get("/health")
async def health_check(rag_service: RAGService = Depends(get_rag_service)):
    """Provides system health status."""
    checks = {"api": "healthy", "vector_db": "unknown"}
    try:
        count = rag_service.vectorstore._collection.count()
        checks["vector_db"] = f"healthy (indexed chunks: {count})"
    except Exception as e:
        checks["vector_db"] = f"unhealthy ({type(e).__name__})"

    overall = (
        "healthy" if all("unhealthy" not in v for v in checks.values()) else "degraded"
    )
    return {"status": overall, "components": checks}


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, rag_service: RAGService = Depends(get_rag_service)
):
    """Processes a chat query through the RAG pipeline."""
    response = await rag_service.chat(request.query, request.history)
    return response


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...), doc_service=Depends(get_doc_service)
):
    """Uploads and stores a new document."""
    # 1. Validate extension
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if f".{ext}" not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '.{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # 2. Read content and enforce size limit
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) // 1024} KB). Maximum allowed: {MAX_UPLOAD_SIZE_MB} MB.",
        )

    # 3. Verify magic bytes match the declared extension
    if not _verify_magic_bytes(content, ext):
        raise HTTPException(
            status_code=400,
            detail="File content does not match its declared extension. Upload rejected.",
        )

    # Reset file position so save_upload_file can read from the beginning
    await file.seek(0)

    await doc_service.save_upload_file(file)
    return {"message": f"File '{file.filename}' uploaded successfully."}


@router.post("/reindex")
async def reindex_documents(rag_service: RAGService = Depends(get_rag_service)):
    """Rebuilds the vector index from all uploaded documents."""
    num_chunks = await asyncio.to_thread(rag_service.reindex_all)
    return {"message": "Reindexing complete", "chunks_indexed": num_chunks}


@router.post("/reset")
async def reset_all_data(
    rag_service: RAGService = Depends(get_rag_service),
    confirmation: str = Body(..., embed=True),
):
    """Clears all documents and resets the vector database."""
    if confirmation != "CONFIRM_RESET":
        raise HTTPException(
            status_code=400,
            detail='Provide {"confirmation": "CONFIRM_RESET"} in the request body to proceed.',
        )

    result = await asyncio.to_thread(rag_service.reset_all_data)
    return {"message": "All data has been reset", "details": result}
