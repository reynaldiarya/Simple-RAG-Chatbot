from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = []

class SourceNode(BaseModel):
    filename: str
    content_snippet: str
    similarity_score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceNode]
    status: str
