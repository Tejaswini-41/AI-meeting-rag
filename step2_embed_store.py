import json
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings

INPUT_FILE = "./cleaned_chunks.json"
VECTOR_DB_DIR = "./vector_db"

def load_chunks(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_vector_store(chunks):
    print("🔢 Loading embedding model (first time may download ~90MB)...")
    
    # Free, local embedding model — no API key needed
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    print("📦 Preparing documents...")
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
        )
        documents.append(doc)
    
    print(f"🚀 Embedding {len(documents)} chunks and storing in ChromaDB...")
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    
    print(f"✅ Vector DB saved to: {VECTOR_DB_DIR}")
    return vectorstore

if __name__ == "__main__":
    chunks = load_chunks(INPUT_FILE)
    build_vector_store(chunks)
    print("\n🎉 Done! You can now run step3_query.py")