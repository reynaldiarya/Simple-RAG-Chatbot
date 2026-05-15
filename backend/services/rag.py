import asyncio
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.core import settings, logger
from backend.services.document import DocumentService
from backend.models import SourceNode, ChatResponse
from backend.services.ollama import OllamaCloudChat

class RAGService:
    def __init__(self):
        self.vector_db_dir = Path(settings.vector_db_dir).resolve()
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize local embedding model (CPU)
        logger.info(f"Loading Embedding Model: {settings.embedding_model} ...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        logger.info("Embedding Model loaded successfully.")
        
        # Initialize Chroma vectorstore with cosine distance
        self.vectorstore = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embedding_model,
            persist_directory=str(self.vector_db_dir),
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        self.doc_service = DocumentService()
        
        # Initialize Ollama LLM with LangChain wrapper (supports authentication headers)
        self.llm = OllamaCloudChat(
            model_name=settings.ollama_model,
            base_url=settings.ollama_api_base,
            temperature=settings.ollama_temperature,
            top_p=settings.ollama_top_p
        )
        
        # Prompt template with guardrails
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an internal RAG assistant. 
Your job is to answer the user's question based on the context provided below.

RULES:
1. If the context does not contain relevant information to answer the question, reply with: "I don't have enough information in the RAG system to answer that."
2. Keep the answer concise, accurate, and professional.
3. Only use the provided context. Do not use external knowledge.
4. If conversation history is provided, use it to maintain context but never violate rule 3.

CONTEXT:
{context}

CONVERSATION HISTORY (if any):
{history}"""),
            ("human", "{query}")
        ])
        
        # Chain for RAG
        self.chain = self.prompt | self.llm | StrOutputParser()

    def reindex_all(self) -> int:
        """Deletes the old index and recreates it from existing files."""
        try:
            # Delete the old collection
            self.vectorstore.delete_collection()
        except Exception:
            pass  # Collection does not exist yet
        
        # Recreate the vectorstore
        self.vectorstore = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embedding_model,
            persist_directory=str(self.vector_db_dir),
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        docs = self.doc_service.load_all_documents()
        if not docs:
            return 0
        
        # Add documents to the vectorstore (automatically embedded)
        self.vectorstore.add_documents(docs)
        logger.info(f"Successfully indexed {len(docs)} chunks into collection 'rag_collection'.")
        return len(docs)

    def reset_all_data(self) -> dict:
        """Removes all data: vector database and documents."""
        result = {"documents_deleted": 0, "vector_db_cleared": False}
        
        # 1. Clear vector database
        try:
            # Delete collection from memory/API
            self.vectorstore.delete_collection()
            self.vectorstore = None
            
            # Physically remove the vector database directory
            import shutil
            if self.vector_db_dir.exists():
                shutil.rmtree(self.vector_db_dir)
                self.vector_db_dir.mkdir(parents=True, exist_ok=True)
            
            # Reinitialize empty vectorstore
            self.vectorstore = Chroma(
                collection_name="rag_collection",
                embedding_function=self.embedding_model,
                persist_directory=str(self.vector_db_dir),
                collection_metadata={"hnsw:space": "cosine"}
            )
            result["vector_db_cleared"] = True
            logger.info("Vector database cleared successfully.")
        except Exception as e:
            logger.error(f"Error clearing vector database: {str(e)}")
        
        # 2. Clear all documents
        result["documents_deleted"] = self.doc_service.clear_all_documents()
        
        return result

    async def chat(self, query: str, history: list = None) -> ChatResponse:
        """Main logic for RAG Guardrail and Chat."""
        # 1. Retrieval using similarity search (cosine distance)
        # Run synchronous search in a thread pool to avoid blocking the event loop
        results = await asyncio.to_thread(
            self.vectorstore.similarity_search_with_score,
            query=query,
            k=settings.max_retrieved_docs
        )
        
        sources = []
        context_text = ""
        valid_results_found = False
        
        logger.info(f"Query: '{query}'")
        
        for doc, score in results:
            # Cosine distance: 0 = identical, 2 = opposite, threshold 0.3 = cosine_sim >= 0.7
            if score <= settings.similarity_threshold:
                valid_results_found = True
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content
                context_text += f"\n---\nSource: {source}\n{content}\n"
                sources.append(SourceNode(
                    filename=source,
                    content_snippet=content[:150] + "...",
                    similarity_score=round(1.0 - score, 4)  # Convert distance to cosine similarity
                ))
        
        if not valid_results_found:
            return ChatResponse(
                answer="I don't have enough information in the RAG system to answer that.",
                sources=[],
                status="rejected_by_threshold_guardrail"
            )
        
        # Format conversation history
        history_text = ""
        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
        
        # 2. Generate Response via LLM (async)
        try:
            llm_response = await self.chain.ainvoke({
                "context": context_text,
                "history": history_text,
                "query": query
            })
            return ChatResponse(
                answer=llm_response,
                sources=sources,
                status="success"
            )
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return ChatResponse(answer="Error communicating with LLM.", sources=[], status="error")
