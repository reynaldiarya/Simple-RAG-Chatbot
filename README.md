# Simple RAG Chatbot

An enterprise-ready Retrieval-Augmented Generation system for secure, private document intelligence using FastAPI, Streamlit, and Ollama.

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" />
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB.svg" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688.svg" />
  <img src="https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg" />
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-yellow.svg" target="_blank" />
  </a>
  <a href="https://codecov.io/gh/reynaldiarya/Simple-RAG-Chatbot">
    <img src="https://codecov.io/gh/reynaldiarya/Simple-RAG-Chatbot/branch/main/graph/badge.svg" />
  </a>
</p>

## Description

Simple RAG Chatbot provides a high-performance framework for building private RAG systems. It addresses the challenge of interacting with sensitive internal documents by leveraging local embeddings and containerized LLMs through Ollama. The system utilizes a modular service architecture and vector storage to ensure rapid retrieval and accurate responses, making it ideal for organizations that require strict data sovereignty without sacrificing the power of modern AI.

## Features

- **Private Document Retrieval** - Query uploaded documents (PDF, TXT, MD) using semantic search powered by ChromaDB.
- **Modular Service Architecture** - Decoupled backend services for document processing, vector management, and LLM orchestration.
- **Real-time Observability** - Detailed logging system for monitoring embedding model initialization and processing stages.
- **Intelligent Guardrails** - Built-in similarity thresholding to prevent hallucinations when no relevant context is found.
- **Index Management** - Full control over the vector database with capabilities to reindex documents or perform a clean reset.
- **Modern User Interface** - Professional Streamlit dashboard for seamless document management and interactive chat.
- **Singleton Lifecycle Management** - Efficient resource utilization through FastAPI lifespan events and singleton service patterns.

## Tech Stack

- **Backend Framework**: FastAPI (Python)
- **Frontend Framework**: Streamlit
- **LLM Orchestration**: LangChain & LangChain-Ollama
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace (multilingual-e5-large)
- **Document Parsing**: PyMuPDF (Fitz)
- **Validation**: Pydantic v2
- **Environment Management**: Pydantic Settings

## Installation Guide

### Prerequisites

- Python 3.12 or higher
- Ollama installed and running
- Model pulled in Ollama (e.g., `ministral-3:14b-cloud`)

### Local Installation

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/reynaldiarya/Simple-RAG-Chatbot.git
cd Simple-RAG-Chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the environment variables:
```bash
cp .env.example .env
```

5. Start the backend server:
```bash
python -m backend.main
```

6. In a new terminal, start the frontend application:
```bash
streamlit run frontend/app.py
```

## Configuration

The application is configured via environment variables in the `.env` file.

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_API_BASE` | Base URL for the Ollama service | `https://ollama.com` |
| `OLLAMA_MODEL` | The specific LLM model to use | `ministral-3:14b-cloud` |
| `EMBEDDING_MODEL` | The HuggingFace embedding model name | `intfloat/multilingual-e5-large` |
| `CHUNK_SIZE` | Character limit for document splitting | `1000` |
| `CHUNK_OVERLAP` | Overlap between document chunks | `200` |
| `SIMILARITY_THRESHOLD` | Minimum score for context retrieval (0.0 - 1.0) | `0.4` |
| `DOCUMENTS_DIR` | Directory path for uploaded documents | `backend/data/documents` |
| `VECTOR_DB_DIR` | Directory path for ChromaDB persistence | `backend/data/vector_db` |

## Usage

### Document Management

1. **Upload**: Use the sidebar to upload PDF, TXT, or MD files.
2. **Indexing**: Click "Rebuild Vector Index" after uploading to process and embed the new documents.
3. **Reset**: Use "Clear All Data" to wipe the RAG system and physical files completely.

### API Endpoints

#### 1. Chat
Queries the RAG system with optional conversation history.
- **Endpoint**: `POST /api/chat`
- **Body**:
```json
{
  "query": "What is the company policy on remote work?",
  "history": []
}
```

#### 2. Reindex
Manually triggers the document processing and embedding pipeline.
- **Endpoint**: `POST /api/reindex`

## Project Structure

```text
/
├── backend/
│   ├── api/            # API routes and dependency injection
│   ├── core/           # Configuration and logging system
│   ├── models/         # Pydantic schemas and data structures
│   ├── services/       # RAG, Document, and Ollama service logic
│   └── main.py         # Entry point and lifespan management
├── frontend/
│   └── app.py          # Streamlit dashboard and UI logic
├── .env.example        # Environment template
├── requirements.txt    # Project dependencies
└── LICENSE             # MIT License
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full license text.

## Author

Reynaldi Arya