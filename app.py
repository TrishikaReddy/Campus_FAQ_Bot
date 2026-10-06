import os
import hashlib
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from dotenv import load_dotenv


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not GEMINI_API_KEY:
    st.error(
        "Gemini API key not found. Please add GEMINI_API_KEY to your .env file."
    )
    st.stop()

# New Gemini SDK
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

st.set_page_config(
    page_title="Campus FAQ Bot",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Load the embedding model only when required
def get_embedding_model():
    return load_embedding_model()


# --------------------------------------------------
# CHROMADB
# --------------------------------------------------

@st.cache_resource
def get_chroma_collection():

    client = chromadb.PersistentClient(
        path="chroma_db"
    )

    collection = client.get_or_create_collection(
        name="campus_faq"
    )

    return collection


collection = get_chroma_collection()


# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_pdf_text(pdf_file):
    """
    Extract text from every page of a PDF.

    Returns:
        list of dictionaries containing page number and text.
    """

    reader = PdfReader(
        BytesIO(pdf_file.getvalue())
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            pages.append(
                {
                    "page": page_number,
                    "text": text.strip()
                }
            )

    return pages


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def create_chunks(
    text,
    chunk_size=1000,
    overlap=150
):
    """
    Split text into overlapping chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(
                chunk.strip()
            )

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# PROCESS DOCUMENTS
# --------------------------------------------------

def process_documents(uploaded_files):
    """
    Extract PDF text, create chunks,
    generate embeddings and store them
    in ChromaDB.
    """

    documents = []
    metadatas = []
    ids = []

    for pdf_file in uploaded_files:

        pages = extract_pdf_text(
            pdf_file
        )

        for page_data in pages:

            page_number = page_data["page"]
            page_text = page_data["text"]

            chunks = create_chunks(
                page_text
            )

            for chunk_number, chunk in enumerate(
                chunks
            ):

                unique_text = (
                    pdf_file.name
                    + str(page_number)
                    + str(chunk_number)
                    + chunk
                )

                chunk_id = hashlib.md5(
                    unique_text.encode("utf-8")
                ).hexdigest()

                documents.append(chunk)

                metadatas.append(
                    {
                        "source": pdf_file.name,
                        "page": page_number,
                        "chunk": chunk_number + 1
                    }
                )

                ids.append(chunk_id)

    if not documents:
        return 0

    # Generate embeddings
    embedding_model = get_embedding_model()

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=False
    ).tolist()

    # Store in ChromaDB
    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    return len(documents)


# --------------------------------------------------
# SEARCH DOCUMENTS
# --------------------------------------------------

def search_documents(
    question,
    number_of_results=5
):
    """
    Convert the question into an embedding
    and search ChromaDB for relevant chunks.
    """

    embedding_model = get_embedding_model()

    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    # Avoid requesting more results than
    # the collection actually contains
    total_documents = collection.count()

    if total_documents == 0:
        return {
            "documents": [],
            "metadatas": [],
            "distances": []
        }

    number_of_results = min(
        number_of_results,
        total_documents
    )

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=number_of_results,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results


# --------------------------------------------------
# GENERATE ANSWER USING GEMINI
# --------------------------------------------------

def generate_answer(
    question,
    search_results
):
    """
    Generate a grounded answer using
    Gemini Interactions API.

    The model is instructed to answer
    only from retrieved college documents.
    """

    if not search_results.get("documents"):

        return (
            "I couldn't find relevant information "
            "in the uploaded college documents."
        )

    if not search_results["documents"][0]:

        return (
            "I couldn't find relevant information "
            "in the uploaded college documents."
        )

    documents = search_results["documents"][0]

    metadatas = search_results["metadatas"][0]

    context_parts = []

    for i, document in enumerate(documents):

        source = metadatas[i]["source"]

        page = metadatas[i]["page"]

        context_parts.append(
            f"Source: {source}, Page: {page}\n"
            f"{document}"
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    prompt = f"""
You are a helpful Campus FAQ Bot.

Your job is to answer the student's question
using ONLY the information provided in the
college document context below.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent college policies.
3. Do not invent attendance requirements.
4. Do not invent dates.
5. Do not invent fees.
6. Do not invent rules.
7. Do not guess when the information is missing.
8. If the answer is not present in the provided
   context, say:

"I couldn't find this information in the
uploaded college documents."

9. Keep the answer clear, concise and
   student-friendly.

Student question:
{question}

College document context:
{context}
"""

    try:

        interaction = gemini_client.interactions.create(
            model=GEMINI_MODEL,
            input=prompt
        )

        answer = interaction.output_text

        if not answer:
            return (
                "I couldn't generate an answer "
                "from the uploaded college documents."
            )

        return answer

    except Exception as e:

        return (
            "⚠️ Gemini could not generate the answer.\n\n"
            f"Error: {str(e)}"
        )


# --------------------------------------------------
# STREAMLIT USER INTERFACE
# --------------------------------------------------

st.title("🎓 Campus FAQ Bot")

st.write(
    "Ask questions about your college handbook, "
    "timetable, rules and other uploaded documents."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📚 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload college documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "🔄 Process Documents",
        use_container_width=True
    )

    if process_button:

        if not uploaded_files:

            st.warning(
                "Please upload at least one PDF."
            )

        else:

            with st.spinner(
                "Reading and processing documents..."
            ):

                try:

                    number_of_chunks = process_documents(
                        uploaded_files
                    )

                    if number_of_chunks == 0:

                        st.warning(
                            "No readable text was found "
                            "in the uploaded PDF."
                        )

                    else:

                        st.success(
                            f"Processed {number_of_chunks} "
                            "document chunks!"
                        )

                except Exception as e:

                    st.error(
                        f"Error processing documents: {str(e)}"
                    )

    st.divider()

    st.subheader("💡 Try asking")

    st.write(
        "• What is the attendance policy?"
    )

    st.write(
        "• What are the college rules?"
    )

    st.write(
        "• What subjects are in the timetable?"
    )

    st.write(
        "• What happens if attendance is below "
        "the requirement?"
    )


# --------------------------------------------------
# KNOWLEDGE BASE STATUS
# --------------------------------------------------

st.subheader("📊 Knowledge Base")

document_count = collection.count()

if document_count == 0:

    st.info(
        "No documents have been processed yet. "
        "Upload a PDF from the sidebar to get started."
    )

else:

    st.success(
        f"Knowledge base contains {document_count} "
        "document chunks."
    )


st.divider()


# --------------------------------------------------
# CHAT AREA
# --------------------------------------------------

st.subheader("💬 Ask your question")

question = st.chat_input(
    "Ask something about your college..."
)


# --------------------------------------------------
# HANDLE QUESTION
# --------------------------------------------------

if question:

    if collection.count() == 0:

        st.warning(
            "Please upload and process a college PDF first."
        )

    else:

        # Display user's question
        with st.chat_message("user"):

            st.write(question)

        # Generate assistant response
        with st.chat_message("assistant"):

            with st.spinner(
                "Searching college documents..."
            ):

                try:

                    search_results = search_documents(
                        question,
                        number_of_results=5
                    )

                    answer = generate_answer(
                        question,
                        search_results
                    )

                except Exception as e:

                    search_results = {
                        "documents": [],
                        "metadatas": [],
                        "distances": []
                    }

                    answer = (
                        "⚠️ Something went wrong while "
                        "searching the knowledge base.\n\n"
                        f"Error: {str(e)}"
                    )

            st.write(answer)

            # ------------------------------------------
            # SOURCES
            # ------------------------------------------

            if search_results.get("metadatas"):

                if search_results["metadatas"][0]:

                    st.divider()

                    st.markdown(
                        "### 📚 Sources"
                    )

                    shown_sources = set()

                    for metadata in (
                        search_results["metadatas"][0]
                    ):

                        source = metadata["source"]

                        page = metadata["page"]

                        source_key = (
                            source,
                            page
                        )

                        if source_key not in shown_sources:

                            st.write(
                                f"📄 **{source}** — "
                                f"Page {page}"
                            )

                            shown_sources.add(
                                source_key
                            )