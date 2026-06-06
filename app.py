import streamlit as st
import requests
import os

# Backend API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/api")

# Page configuration
st.set_page_config(
    page_title="Simple RAG Chatbot", layout="wide", initial_sidebar_state="expanded"
)

# Main header
st.markdown('<h1 class="main-header">Simple RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown("##### Query your documents securely using Retrieval-Augmented Generation.")

# Sidebar management
with st.sidebar:
    st.title("Management")
    st.subheader("Document Upload")

    uploaded_file = st.file_uploader(
        "Select a file (PDF, TXT, MD, PNG, JPG, JPEG)",
        type=["txt", "md", "pdf", "png", "jpg", "jpeg"],
    )
    if st.button("Process & Save Document", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Uploading document..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }
                try:
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 200:
                        st.success("Document uploaded successfully.")
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

    st.divider()

    st.subheader("Index Control")
    if st.button("Rebuild Vector Index", use_container_width=True):
        with st.spinner("Processing documents and updating vectors..."):
            try:
                res = requests.post(f"{API_URL}/reindex")
                if res.status_code == 200:
                    chunks = res.json().get("chunks_indexed", 0)
                    st.success(f"Index rebuilt with {chunks} segments.")
                else:
                    st.error("Reindexing failed.")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

    st.divider()

    if st.button("Clear All Data", use_container_width=True, type="secondary"):
        st.session_state["confirm_reset"] = True

    if st.session_state.get("confirm_reset"):
        st.warning(
            "This will permanently delete ALL documents and the vector database."
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm", use_container_width=True, type="primary"):
                with st.spinner("Wiping RAG system..."):
                    try:
                        res = requests.post(
                            f"{API_URL}/reset", json={"confirmation": "CONFIRM_RESET"}
                        )
                        if res.status_code == 200:
                            data = res.json()
                            st.success(
                                f"RAG system cleared. Documents removed: {data['details']['documents_deleted']}"
                            )
                            st.session_state.messages = []
                            st.session_state["confirm_reset"] = False
                            st.rerun()
                        else:
                            st.error("Reset operation failed.")
                            st.session_state["confirm_reset"] = False
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
                        st.session_state["confirm_reset"] = False
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state["confirm_reset"] = False
                st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Retrieved Context"):
                for src in message["sources"]:
                    st.markdown(
                        f"**Source:** {src['filename']} | **Relevance:** {src['similarity_score']}"
                    )
                    st.caption(src["content_snippet"])

# Chat interface
if prompt := st.chat_input("Enter your question..."):
    # User message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant response
    with st.chat_message("assistant"):
        response_container = st.empty()
        with st.spinner("Analyzing RAG system..."):
            try:
                response = requests.post(f"{API_URL}/chat", json={"query": prompt})
                response.raise_for_status()
                data = response.json()

                answer = data["answer"]
                sources = data.get("sources", [])

                # Display final answer
                response_container.markdown(answer)

                # Display sources
                if sources:
                    with st.expander("Retrieved Context"):
                        for src in sources:
                            st.markdown(
                                f"**Source:** {src['filename']} | **Relevance:** {src['similarity_score']}"
                            )
                            st.caption(src["content_snippet"])

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except Exception as e:
                response_container.error(f"System Error: {str(e)}")
