
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import os
import uuid

from qdrant_client import QdrantClient, models
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_cohere import ChatCohere


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

DATA_DIR = "data/"
VECTOR_DB_DIR = "vector_store"
COLLECTION_NAME = "pdf_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "command-a-03-2025"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in .env file")


# ============================================================
# 2. PDF DATA INGESTION
# ============================================================

def load_pdfs(directory: str) -> List[Document]:

    documents = []
    pdf_files = list(Path(directory).rglob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:

        try:
            loader = PyMuPDFLoader(str(pdf_file))
            pages = loader.load()

            for page in pages:
                page.metadata["source_file"] = str(
                    pdf_file.relative_to(Path(directory))
                )
                page.metadata["file_type"] = "pdf"

            documents.extend(pages)

            print(f"Loaded: {pdf_file.name} ({len(pages)} pages)")

        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")

    print(f"Total pages loaded: {len(documents)}")

    return documents


# ============================================================
# 3. DOCUMENT CHUNKING
# ============================================================

def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"Generated {len(chunks)} chunks.")

    return chunks


# ============================================================
# 4. EMBEDDING MANAGER
# ============================================================

class EmbeddingManager:

    def __init__(self, model_name=EMBEDDING_MODEL):

        self.model = SentenceTransformer(model_name)

        print(
            "Embedding model loaded:",
            model_name,
            self.model.get_embedding_dimension()
        )

    def generate_embeddings(
        self,
        texts: List[str]
    ) -> np.ndarray:

        if not texts:
            return np.empty(
                (0, self.model.get_embedding_dimension()),
                dtype=np.float32
            )

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings.astype(np.float32)


# ============================================================
# 5. CHROMADB VECTOR STORE
# ============================================================


class VectorStore:
    """Manages document embeddings in Qdrant Cloud."""

    def __init__(
        self,
        collection_name="pdf_documents",
        embedding_dim=384
    ):

        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                "QDRANT_URL or QDRANT_API_KEY is missing."
            )

        # Connect to Qdrant Cloud
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )

        # Create collection if it does not exist
        if not self.client.collection_exists(
            self.collection_name
        ):

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim,
                    distance=models.Distance.COSINE
                )
            )

            print("Qdrant collection created.")

        else:
            print("Existing Qdrant collection connected.")

        print(
            "Connected to Qdrant Cloud:",
            self.collection_name
        )

    @staticmethod
    def generate_id(doc):

        unique_content = (
            doc.metadata.get("source_file", "")
            + str(doc.metadata.get("page", ""))
            + str(doc.metadata.get("start_index", ""))
            + doc.page_content
        )

        return str(
            uuid.uuid5(uuid.NAMESPACE_URL, unique_content)
        )

    def add_documents(
        self,
        documents,
        embeddings,
        batch_size=100
    ):

        if len(documents) != len(embeddings):
            raise ValueError(
                "Document and embedding counts differ."
            )

        # Remove previous chunks for the files being re-indexed
        source_files = {
            doc.metadata.get("source_file")
            for doc in documents
            if doc.metadata.get("source_file")
        }

        for source in source_files:

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_file",
                                match=models.MatchValue(
                                    value=source
                                )
                            )
                        ]
                    )
                ),
                wait=True
            )

        # Upload in batches
        for start in range(0, len(documents), batch_size):

            batch_docs = documents[
                start:start + batch_size
            ]

            batch_embeddings = embeddings[
                start:start + batch_size
            ]

            points = []

            for doc, embedding in zip(
                batch_docs,
                batch_embeddings
            ):

                metadata = {
                    key: value
                    for key, value in doc.metadata.items()
                    if isinstance(
                        value,
                        (str, int, float, bool)
                    )
                }

                metadata["content_length"] = len(
                    doc.page_content
                )

                payload = {
                    "content": doc.page_content,
                    "metadata": metadata,
                    **metadata
                }

                points.append(
                    models.PointStruct(
                        id=self.generate_id(doc),
                        vector=embedding.tolist(),
                        payload=payload
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

        print(
            "Successfully indexed:",
            len(documents),
            "chunks"
        )

    def count(self):

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True
        )

        return result.count



# ============================================================
# 6. RAG RETRIEVER
# ============================================================

c
class RAGRetriever:
    """Retrieves relevant chunks from Qdrant Cloud."""

    def __init__(
        self,
        vector_store,
        embedding_manager
    ):

        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query,
        top_k=5,
        score_threshold=0.20
    ):

        if self.vector_store.count() == 0:
            return []

        # Generate query embedding
        query_embedding = (
            self.embedding_manager
            .generate_embeddings([query])[0]
        )

        # Search Qdrant
        results = self.vector_store.client.query_points(
            collection_name=self.vector_store.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )

        retrieved_docs = []

        for rank, point in enumerate(
            results.points,
            start=1
        ):

            similarity = point.score

            if similarity >= score_threshold:

                payload = point.payload or {}

                retrieved_docs.append({
                    "id": str(point.id),
                    "content": payload.get("content", ""),
                    "metadata": payload.get("metadata", {}),
                    "similarity_score": similarity,
                    "rank": rank
                })

        return retrieved_docs



# ============================================================
# 7. LLM INITIALIZATION
# ============================================================

llm = ChatCohere(
    model=LLM_MODEL,
    temperature=0,
    cohere_api_key=COHERE_API_KEY
)


# ============================================================
# 8. ADVANCED RAG PIPELINE
# ============================================================

class AdvancedRAGPipeline:

    def __init__(self, retriever, llm):

        self.retriever = retriever
        self.llm = llm
        self.history = []

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.20,
        stream: bool = False,
        summarize: bool = False
    ) -> Dict[str, Any]:

        # RETRIEVAL
        results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            score_threshold=min_score
        )

        if not results:

            answer = "No relevant context found in the documents."

            return {
                "question": question,
                "answer": answer,
                "sources": [],
                "retrieval_score": 0.0,
                "context": "",
                "summary": None
            }

        # CONTEXT PREPARATION
        context_parts = []
        sources = []

        for i, doc in enumerate(results):

            source_id = f"S{i + 1}"

            context_parts.append(
                f"[{source_id}]\n{doc['content']}"
            )

            sources.append({
                "citation_id": source_id,
                "source": doc["metadata"].get(
                    "source_file", "unknown"
                ),
                "page": doc["metadata"].get(
                    "page", "unknown"
                ),
                "similarity_score": round(
                    doc["similarity_score"], 4
                ),
                "preview": doc["content"][:200]
            })

        context = "\n\n".join(context_parts)

        # PROMPT
        prompt = f"""
You are a helpful question-answering assistant.

Answer the question using ONLY the provided context.

Instructions:
1. Do not invent information.
2. If the answer is not present in the context, say so.
3. Provide a clear and concise answer.
4. Cite supporting sources using [S1], [S2], etc.
5. Do not cite sources that do not support your answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        # GENERATION
        if stream:

            answer_parts = []

            for chunk in self.llm.stream(prompt):

                content = chunk.content

                if isinstance(content, str):
                    print(content, end="", flush=True)
                    answer_parts.append(content)

            print()
            answer = "".join(answer_parts)

        else:

            response = self.llm.invoke(prompt)
            answer = response.content

        # RETRIEVAL SCORE (NOT A CALIBRATED CONFIDENCE)
        retrieval_score = max(
            doc["similarity_score"] for doc in results
        )

        # OPTIONAL SUMMARIZATION
        summary = None

        if summarize:

            summary_prompt = f"""
Summarize the following answer in two sentences.
Preserve its key information.

Answer:
{answer}
"""

            summary_response = self.llm.invoke(summary_prompt)
            summary = summary_response.content

        # HISTORY
        history_entry = {
            "question": question,
            "answer": answer,
            "sources": sources
        }

        self.history.append(history_entry)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieval_score": round(retrieval_score, 4),
            "context": context,
            "summary": summary,
            "history": list(self.history)
        }


# ============================================================
# 9. PIPELINE INITIALIZATION
# ============================================================

embedding_manager = EmbeddingManager()

vectorstore = VectorStore()

# INGESTION
documents = load_pdfs(DATA_DIR)

chunks = split_documents(documents)

# EMBEDDING
texts = [doc.page_content for doc in chunks]

embeddings = embedding_manager.generate_embeddings(texts)

# VECTOR DATABASE INDEXING
if chunks:
    vectorstore.add_documents(chunks, embeddings)

# RETRIEVER
rag_retriever = RAGRetriever(
    vectorstore,
    embedding_manager
)

# RAG PIPELINE
rag_pipeline = AdvancedRAGPipeline(
    rag_retriever,
    llm
)


# ============================================================
# 10. QUERY EXECUTION
# ============================================================

result = rag_pipeline.query(
    question="What is feature engineering?",
    top_k=5,
    min_score=0.20,
    stream=False,
    summarize=True
)

print("\nANSWER:\n", result["answer"])

print("\nSOURCES:")
for source in result["sources"]:
    print(source)

print("\nRETRIEVAL SCORE:", result["retrieval_score"])

print("\nSUMMARY:", result["summary"])
