import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import uuid


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Mini RAG",
    page_icon="📚"
)

st.title("📚 Mini RAG: Document Store + Retrieval")
st.caption(
    "PDF → Chunks → Embeddings → ChromaDB → Retrieval → Ollama"
)


# ---------------- EMBEDDING MODEL ----------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# ---------------- CHROMADB ----------------

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(
        path="./chroma_db"
    )


client = get_chroma_client()

collection = client.get_or_create_collection(
    name="documents"
)


# ---------------- SIDEBAR SETTINGS ----------------

st.sidebar.header("Settings")

ollama_model = st.sidebar.text_input(
    "Ollama model",
    value="llama3.2"
)

chunk_size = st.sidebar.slider(
    "Chunk size",
    min_value=200,
    max_value=1500,
    value=500,
    step=100
)

top_k = st.sidebar.slider(
    "Chunks to retrieve",
    min_value=1,
    max_value=5,
    value=3
)


# ---------------- BUILD DOCUMENT STORE ----------------

st.header("1️⃣ Build Document Store")

uploaded_file = st.file_uploader(
    "Upload a text-based PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.button("📥 Process & Store PDF"):

        reader = PdfReader(uploaded_file)

        text = ""

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"


        if not text.strip():
            st.error(
                "No readable text found. "
                "Please upload a text-based PDF."
            )
            st.stop()


        # -------- CHUNKING --------

        chunks = []

        for i in range(
            0,
            len(text),
            chunk_size
        ):
            chunk = text[i:i + chunk_size]

            if chunk.strip():
                chunks.append(chunk)


        # -------- CREATE EMBEDDINGS --------

        with st.spinner(
            "Creating embeddings..."
        ):

            embeddings = embedding_model.encode(
                chunks
            )


        # -------- UNIQUE IDS --------

        ids = [
            f"chunk_{uuid.uuid4()}"
            for _ in chunks
        ]


        # -------- STORE IN CHROMADB --------

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist()
        )


        st.success(
            f"PDF stored successfully. "
            f"{len(chunks)} chunks added to ChromaDB."
        )


# ---------------- QUESTION ANSWERING ----------------

st.header("2️⃣ Ask Questions")

question = st.text_input(
    "Ask a question about your uploaded document",
    placeholder="Example: What is the main topic of the document?"
)


if st.button("🤖 Ask AI"):

    if not question.strip():
        st.warning(
            "Please enter a question."
        )

    elif collection.count() == 0:
        st.warning(
            "Please upload and store a PDF first."
        )

    else:

        # -------- QUESTION EMBEDDING --------

        question_embedding = embedding_model.encode(
            [question]
        )[0]


        # -------- RETRIEVE RELEVANT CHUNKS --------

        results = collection.query(
            query_embeddings=[
                question_embedding.tolist()
            ],
            n_results=min(
                top_k,
                collection.count()
            )
        )


        retrieved_chunks = results[
            "documents"
        ][0]


        context = "\n\n".join(
            retrieved_chunks
        )


        # -------- PROMPT FOR OLLAMA --------

        prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say:

"I don't know based on the provided document."
"""


        # -------- GENERATE ANSWER --------

        try:

            with st.spinner(
                "Generating answer..."
            ):

                response = ollama.chat(
                    model=ollama_model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )


            st.subheader("Answer")

            st.write(
                response["message"]["content"]
            )


            # -------- SHOW RETRIEVED CONTEXT --------

            with st.expander(
                "View Retrieved Context"
            ):

                for i, chunk in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    st.markdown(
                        f"### Chunk {i}"
                    )

                    st.write(chunk)


        except Exception as e:

            st.error(
                "Could not connect to Ollama. "
                "Make sure Ollama is running "
                "and the selected model is installed."
            )

            st.code(str(e))


# ---------------- DATABASE INFO ----------------

st.divider()

st.caption(
    f"Stored chunks in ChromaDB: "
    f"{collection.count()}"
)