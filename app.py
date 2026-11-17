
import os
import streamlit as st

from pathlib import Path
from rag_pipeline import (
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
    AdvancedRAGPipeline,
    load_pdfs,
    split_documents,
    llm
)

st.set_page_config(
    page_title="My RAG Assistant",
    page_icon="📚"
)

st.title("📚 PDF RAG Assistant")
st.write("Upload PDFs and ask questions about their content.")

# Initialize components once per Streamlit session
@st.cache_resource
def initialize_rag():

    embedding_manager = EmbeddingManager()
    
    vectorstore = VectorStore(
        collection_name="pdf_documents",
        embedding_dim=embedding_manager.model.get_embedding_dimension()
    )

    retriever = RAGRetriever(
        vectorstore,
        embedding_manager
    )

    pipeline = AdvancedRAGPipeline(
        retriever,
        llm
    )

    return pipeline, vectorstore, embedding_manager


pipeline, vectorstore, embedding_manager = initialize_rag()

# PDF upload
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process PDFs"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    else:
        os.makedirs("data", exist_ok=True)

        with st.spinner("Processing documents..."):

            all_documents = []

            for uploaded_file in uploaded_files:

                safe_name = Path(uploaded_file.name).name
                file_path = Path("data") / safe_name

                file_path.write_bytes(
                    uploaded_file.getvalue()
                )

            all_documents = load_pdfs("data/")

            chunks = split_documents(all_documents)

            if chunks:

                texts = [
                    doc.page_content for doc in chunks
                ]

                embeddings = (
                    embedding_manager.generate_embeddings(texts)
                )

                vectorstore.add_documents(
                    chunks,
                    embeddings
                )

                st.success(
                    f"Successfully indexed {len(chunks)} chunks!"
                )
            else:
                st.warning("No document chunks were generated.")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something about your PDFs...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            result = pipeline.query(
                question=question,
                top_k=5,
                min_score=0.20
            )

            answer = result["answer"]

            st.markdown(answer)

            if result["sources"]:

                with st.expander("View sources"):

                    for source in result["sources"]:

                        st.write(
                            f"**{source['source']}** "
                            f"| Page: {source['page']}"
                        )

                        st.write(source["preview"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
