# Simple RAG Chatbot

An enterprise-ready Retrieval-Augmented Generation (RAG) system designed for secure, private document intelligence. Built with FastAPI, Streamlit, and Ollama.

<p align="center">
  <img src="https://img.shields.io/badge/version-1.1.0-blue.svg" />
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB.svg" />
  <img src="https://img.shields.io/badge/FastAPI-0.136+-009688.svg" />
  <img src="https://img.shields.io/badge/Streamlit-1.57+-FF4B4B.svg" />
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-yellow.svg" target="_blank" />
  </a>
</p>

## Description

Simple RAG Chatbot provides a high-performance, secure framework for building private RAG systems. It addresses the challenge of interacting with sensitive internal documents by leveraging local embeddings and LLMs through Ollama (supporting both local and cloud models). The system utilizes a modular service architecture, vector storage, and built-in Optical Character Recognition (OCR) to ensure rapid retrieval and accurate responses across various document types, including images.

Designed with a strong emphasis on security and reliability, this project incorporates prompt injection hardening, deterministic chunk deduplication, strict input validation, and context window limits to ensure enterprise-grade stability.

## Key Features

- **Multi-format Document Retrieval** - Query uploaded documents including text (`.txt`, `.md`), PDFs (`.pdf`), and images (`.png`, `.jpg`, `.jpeg`) via built-in EasyOCR.
- **Robust Security Posture** - Hardened against prompt injections using XML boundary tags, magic-byte file validation, and strict Pydantic input constraints.
- **Context & History Management** - Configurable hard caps on retrieved context size and conversation history length to prevent token exhaustion.
- **Citation Toggles** - Built-in support for source citations (e.g., `[Source: filename.pdf]`) in LLM responses, controllable via environment variables.
- **Modular Architecture** - Decoupled backend services for document processing, vector management (ChromaDB), and LLM orchestration.
- **Index Management** - Full control over the vector database with capabilities to reindex documents or perform a clean, authenticated reset.
- **Modern User Interface** - Professional Streamlit dashboard for seamless document management and interactive chat.

## Tech Stack

- **Backend Framework**: FastAPI (Python)
- **Frontend UI**: Streamlit
- **LLM Orchestration**: Custom LangChain wrapper for Ollama
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace (`intfloat/multilingual-e5-large`)
- **Document Parsing**: PyMuPDF (Fitz) for PDFs, EasyOCR for Images
- **Validation**: Pydantic v2
- **Environment Management**: Pydantic Settings

## Installation Guide

### Prerequisites

- Python 3.12 or higher
- Ollama installed and running (or access to an Ollama Cloud API)
- Desired LLM model pulled in Ollama (e.g., `ministral-3:14b-cloud`)

### Setup Instructions

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/reynaldiarya/Simple-RAG-Chatbot.git
cd Simple-RAG-Chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the environment variables:
```bash
cp .env.example .env
```
*(Edit `.env` to match your specific model and path configurations)*

5. Start the backend server:
```bash
python -m src.main
```

6. In a new terminal (with the virtual environment activated), start the frontend application:
```bash
streamlit run app.py
```

## Configuration

The application is configured via environment variables in the `.env` file. 

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Server environment (`development` or `production`). Production disables Swagger UI. | `development` |
| `OLLAMA_API_BASE` | Base URL for the Ollama service | `https://ollama.com` |
| `OLLAMA_API_KEY` | Optional API key for authenticated Ollama instances | *(empty)* |
| `OLLAMA_MODEL` | The specific LLM model to use | `ministral-3:14b-cloud` |
| `OLLAMA_TEMPERATURE` | Temperature for LLM generation (range 0.0 - 0.5) | `0.1` |
| `EMBEDDING_MODEL` | The HuggingFace embedding model name | `intfloat/multilingual-e5-large` |
| `CHUNK_SIZE` | Character limit for document splitting | `1000` |
| `CHUNK_OVERLAP` | Overlap between document chunks | `200` |
| `SIMILARITY_THRESHOLD` | Minimum score for context retrieval (cosine distance) | `0.3` |
| `MAX_CONTEXT_CHARS` | Hard cap on total characters injected into the LLM prompt | `4000` |
| `MAX_HISTORY_MESSAGES` | Max conversation turns passed to the LLM for continuity | `6` |
| `CITATION_ENABLED` | Toggle `true`/`false` to enable `[Source: filename]` citations | `true` |

## Usage

### Document Management (Via Streamlit UI)

1. **Upload**: Use the sidebar to upload PDF, TXT, MD, or Image files.
2. **Indexing**: Click "Rebuild Vector Index" after uploading to process, run OCR (if needed), and embed the new documents.
3. **Reset**: Use "Clear All Data" to wipe the RAG system completely (requires a two-step confirmation).

### API Endpoints

#### 1. Chat
Queries the RAG system with optional conversation history.
- **Endpoint**: `POST /api/chat`
- **Body**:
```json
{
  "query": "What is the company policy on remote work?",
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help you?"}
  ]
}
```

#### 2. Upload
Uploads a document with strict size and magic-byte validation.
- **Endpoint**: `POST /api/upload`
- **Form Data**: `file` (multipart/form-data)

#### 3. Reset
Destructive action to clear vector data and files.
- **Endpoint**: `POST /api/reset`
- **Body**:
```json
{
  "confirmation": "CONFIRM_RESET"
}
```

## Project Structure

```text
/
├── src/
│   ├── api/            # FastAPI routes and dependency injection
│   ├── core/           # Pydantic configuration and centralized logging
│   ├── models/         # Data schemas and Pydantic validation models
│   ├── services/       # Core business logic (RAG, Document/OCR, Ollama LLM)
│   └── main.py         # Application entry point and lifespan management
├── app.py              # Streamlit frontend dashboard
├── .env.example        # Environment configuration template
├── requirements.txt    # Python dependencies
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