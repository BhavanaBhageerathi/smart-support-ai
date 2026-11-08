from pathlib import Path
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Get the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Get the knowledge_base folder
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"


def load_documents():

    documents = []

    # Get all text files from knowledge_base
    files = list(KNOWLEDGE_BASE_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError("No text files found in knowledge_base folder")

    for file_path in files:

        # Read the text file
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        # Extract category from document content, if available
        match = re.search(r"Category\s*:\s*(.+)", text, re.IGNORECASE)

        category = match.group(1).strip() if match else "Unknown"

        # Convert the text into a LangChain Document
        document = Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "category": category
            }
        )

        documents.append(document)

    return documents

def split_documents():

    # Load all documents
    documents = load_documents()

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    # Split documents into smaller chunks
    chunks = text_splitter.split_documents(documents)

    return chunks

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Store document chunks in ChromaDB
def create_vector_store():

    # Generate document chunks
    chunks = split_documents()

    # Create persistent ChromaDB storage
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="it_support_knowledge",
        persist_directory=str(BASE_DIR / "chroma_db")
    )

    return vector_store





# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Store document chunks in ChromaDB
def create_vector_store():

    # Initialize persistent ChromaDB
    vector_store = Chroma(
        collection_name="it_support_knowledge",
        embedding_function=embeddings,
        persist_directory=str(BASE_DIR / "chroma_db")
    )

    # Check whether documents are already indexed
    if vector_store._collection.count() == 0:

        print("Creating knowledge base index...")

        chunks = split_documents()

        vector_store.add_documents(chunks)

        print("Documents successfully indexed!")

    else:

        print("Knowledge base already indexed.")
        print("Existing chunks:", vector_store._collection.count())

    return vector_store


def search_knowledge_base(query):  #semantic retrieval

    # Load the existing ChromaDB collection
    vector_store = Chroma(
        collection_name="it_support_knowledge",
        embedding_function=embeddings,
        persist_directory=str(BASE_DIR / "chroma_db")
    )

    # Search for the 3 most relevant chunks
    results = vector_store.similarity_search(
        query,
        k=3
    )

    return results






def generate_recommendation(ticket):

    # Combine ticket details into a search query
    query = f"""
    Title: {ticket['title']}
    Description: {ticket['description']}
    Department: {ticket['requester_department']}
    """

    # Retrieve relevant knowledge
    results = search_knowledge_base(query)

    # Combine retrieved chunks into context
    context = "\n\n".join(
        [
            f"Source: {doc.metadata['source']}\n"
            f"Category: {doc.metadata['category']}\n"
            f"Content: {doc.page_content}"
            for doc in results
        ]
    )

    # Create prompt
    prompt = f"""
    You are an experienced IT support assistant.

    Analyze the following IT support ticket.

    TICKET DETAILS:
    {query}

    RETRIEVED KNOWLEDGE:
    {context}

    Based on the ticket and retrieved knowledge:

    1. Explain the likely issue.
    2. Recommend troubleshooting steps.
    3. Mention whether human intervention is required.

    Use the retrieved knowledge as your primary reference.
    Do not invent company-specific procedures.
    If the retrieved information is insufficient, clearly mention it.
    """

    # Generate response
    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":

    ticket = {
        "title": "Unable to connect to VPN",
        "description": "VPN shows authentication failed even though my password is correct. I have an urgent client meeting.",
        "requester_department": "Sales"
    }

    recommendation = generate_recommendation(ticket)

    print("\nRecommended Resolution:\n")
    print(recommendation)