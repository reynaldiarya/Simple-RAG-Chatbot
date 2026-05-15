from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
import asyncio
from src.models.schemas import ChatRequest, ChatResponse
from src.services.rag import RAGService
from src.services.document import DocumentService

router = APIRouter()

_rag_service: RAGService = None
_doc_service: DocumentService = None

def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

def get_doc_service() -> DocumentService:
    global _doc_service
    if _doc_service is None:
        _doc_service = DocumentService()
    return _doc_service

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Simple RAG Chatbot"}

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    response = await rag_service.chat(request.query, request.history)
    return response

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_service: DocumentService = Depends(get_doc_service)
):
    if not file.filename.endswith(('.txt', '.md', '.pdf')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only txt, md, pdf allowed.")
    
    file_path = await doc_service.save_upload_file(file)
    return {"message": f"File {file.filename} uploaded successfully", "path": file_path}

@router.post("/reindex")
async def reindex_documents(rag_service: RAGService = Depends(get_rag_service)):
    # Run synchronous reindexing in a thread pool to avoid blocking the event loop
    num_chunks = await asyncio.to_thread(rag_service.reindex_all)
    return {"message": "Reindexing complete", "chunks_indexed": num_chunks}

@router.post("/reset")
async def reset_all_data(rag_service: RAGService = Depends(get_rag_service)):
    """Clear all documents and the vector database."""
    result = await asyncio.to_thread(rag_service.reset_all_data)
    return {"message": "All data has been reset", "details": result}
