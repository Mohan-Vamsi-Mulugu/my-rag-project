
import os
import tempfile
from pathlib import Path

import streamlit as st

from rag_pipeline import (
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
    AdvancedRAGPipeline,
    load_pdfs,
    split_documents,
    initialize_llm
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="My RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Assistant")

st.write(
    "Upload PDF documents and ask questions "
    "about their content using AI."
)


# ============================================================
# 2. INITIALIZE RAG COMPONENTS
# ============================================================

@st.cache_resource
def initialize_rag():

    embedding_manager = EmbeddingManager()

    vectorstore = VectorStore(
        collection_name="pdf_documents",
        embedding_dim=embedding_manager.dimension
    )

    retriever = RAGRetriever(
        vectorstore,
        embedding_manager
    )

    llm = initialize_llm()

    pipeline = AdvancedRAGPipeline(
        retriever,
        llm
    )

    return pipeline, vectorstore, embedding_manager


try:

    with st.spinner("Initializing RAG system..."):

        pipeline, vectorstore, embedding_manager = (
            initialize_rag()
        )

    st.success("RAG system connected successfully.")

except Exception as e:

    st.error(f"RAG initialization failed: {e}")

    st.stop()


# ============================================================
# 3. VECTOR DATABASE STATUS
# ============================================================

with st.sidebar:

    st.header("Database Status")

    try:

        total_chunks = vectorstore.count()

        st.metric(
            "Indexed document chunks",
            total_chunks
        )

        st.success("Qdrant Cloud connected.")

    except Exception as e:

        st.error(f"Database error: {e}")


# ============================================================
# 4. PDF UPLOAD
# ============================================================

st.subheader("Upload your PDF documents")

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


if st.button(
    "Process PDFs",
    type="primary",
    disabled=not uploaded_files
):

    with st.spinner(
        "Reading PDFs, splitting text and generating embeddings..."
    ):

        try:

            with tempfile.TemporaryDirectory() as temp_dir:

                temp_path = Path(temp_dir)

                # Save uploaded PDFs temporarily
                for uploaded_file in uploaded_files:

                    safe_name = Path(
                        uploaded_file.name
                    ).name

                    file_path = temp_path / safe_name

                    file_path.write_bytes(
                        uploaded_file.getvalue()
                    )

                # Load PDFs
                documents = load_pdfs(temp_dir)

                if not documents:

                    st.warning(
                        "No readable PDF pages found."
                    )

                else:

                    # Split into chunks
                    chunks = split_documents(documents)

                    if not chunks:

                        st.warning(
                            "No document chunks generated."
                        )

                    else:

                        # Generate embeddings
                        texts = [
                            doc.page_content
                            for doc in chunks
                        ]

                        embeddings = (
                            embedding_manager
                            .generate_embeddings(texts)
                        )

                        # Store in Qdrant
                        vectorstore.add_documents(
                            chunks,
                            embeddings
                        )

                        st.success(
                            f"Successfully indexed "
                            f"{len(chunks)} chunks "
                            f"from {len(uploaded_files)} PDFs!"
                        )

                        st.metric(
                            "Total indexed chunks",
                            vectorstore.count()
                        )

        except Exception as e:

            st.error(
                f"PDF processing failed: {e}"
            )


# ============================================================
# 5. CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# 6. QUESTION ANSWERING
# ============================================================

question = st.chat_input(
    "Ask a question about your uploaded PDFs..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching Qdrant and generating your answer..."
        ):

            try:

                result = pipeline.query(
                    question=question,
                    top_k=5,
                    min_score=0.20,
                    summarize=False
                )

                answer = result["answer"]

                st.markdown(answer)

                if result["sources"]:

                    with st.expander(
                        "📄 View document sources"
                    ):

                        for source in result["sources"]:

                            st.markdown(
                                f"**[{source['citation_id']}] "
                                f"{source['source']}**"
                            )

                            st.write(
                                f"Page: {source['page']}"
                            )

                            st.write(
                                source["preview"]
                            )

                            st.caption(
                                "Similarity: "
                                f"{source['similarity_score']}"
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error(
                    f"Question answering failed: {e}"
                )


# ============================================================
# 7. CLEAR CHAT
# ============================================================

if st.sidebar.button("Clear Chat"):

    st.session_state.messages = []

    st.rerun()
