import os
import shutil
from pathlib import Path
from typing import List
from fastapi import UploadFile, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import fitz
from backend.core import settings, logger

class DocumentService:
    def __init__(self):
        self.docs_dir = Path(settings.documents_dir).resolve()
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        # Setup text splitter with overlap to maintain sentence context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len
        )

    async def save_upload_file(self, upload_file: UploadFile) -> str:
        """Saves the uploaded file to local disk with path validation."""
        # Sanitize filename: extract only the base name without directory components
        safe_filename = Path(upload_file.filename).name
        file_path = self.docs_dir / safe_filename
        file_path_resolved = file_path.resolve()
        
        # Prevent path traversal: ensure the file remains within the documents directory
        if not file_path_resolved.is_relative_to(self.docs_dir):
            raise HTTPException(status_code=400, detail="Invalid filename: path traversal detected")
        
        with open(file_path_resolved, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return str(file_path_resolved)

    def process_file(self, file_path: str) -> List[Document]:
        """Reads a file and converts it into Langchain Document objects."""
        ext = file_path.split('.')[-1].lower()
        content = ""
        
        try:
            if ext == "pdf":
                doc = fitz.open(file_path)
                for page in doc:
                    content += page.get_text() + "\n"
                doc.close()
            elif ext in ["txt", "md"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                logger.warning(f"Unsupported file format: {ext}")
                return []
                
            filename = os.path.basename(file_path)
            # Split large documents into smaller chunks
            chunks = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[{"source": filename}]
            )
            logger.info(f"Processed {filename}: extracted {len(content)} characters, created {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return []

    def load_all_documents(self) -> List[Document]:
        all_docs = []
        for filename in os.listdir(self.docs_dir):
            file_path = self.docs_dir / filename
            if file_path.is_file():
                docs = self.process_file(str(file_path))
                all_docs.extend(docs)
        return all_docs

    def clear_all_documents(self) -> int:
        """Removes all uploaded document files."""
        count = 0
        for filename in os.listdir(self.docs_dir):
            file_path = self.docs_dir / filename
            if file_path.is_file():
                file_path.unlink()
                count += 1
        logger.info(f"Cleared {count} documents from {self.docs_dir}")
        return count
