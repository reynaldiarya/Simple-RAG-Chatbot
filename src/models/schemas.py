from pydantic import BaseModel, Field
from typing import List, Literal


class HistoryMessage(BaseModel):
    """Represents a single message in the conversation history."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    history: List[HistoryMessage] = Field(default=[], max_length=10)


class SourceNode(BaseModel):
    filename: str
    content_snippet: str
    similarity_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceNode]
    status: str
