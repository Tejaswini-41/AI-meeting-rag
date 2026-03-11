from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
# ---- OR if using OpenAI ----
# from langchain_openai import ChatOpenAI

VECTOR_DB_DIR = "./vector_db"

def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})  # top 3 relevant chunks

def ask_question(question, retriever):
    # Retrieve relevant chunks
    relevant_docs = retriever.invoke(question)
    
    # Build context
    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    sources = list(set([doc.metadata["topic"] for doc in relevant_docs]))
    
    # Build prompt
    prompt = f"""You are a helpful assistant answering questions based on KT (Knowledge Transfer) session recordings.

Use ONLY the context below to answer. If the answer is not in the context, say "I don't have information about this in the KT sessions."

Context:
{context}

Question: {question}

Answer:"""
    
    # ---- Option A: Using Ollama (free, local) ----
    # Make sure Ollama is running: ollama run llama3
    llm = Ollama(model="llama3")
    answer = llm.invoke(prompt)
    
    # ---- Option B: Using OpenAI (needs API key) ----
    # llm = ChatOpenAI(model="gpt-4", api_key="your-key")
    # answer = llm.invoke(prompt).content
    
    return answer, sources

def main():
    print("🤖 KT Q&A System Ready!")
    print("📚 Topics available: JAVA, AI, MUT")
    print("Type 'exit' to quit\n")
    
    retriever = load_retriever()
    
    while True:
        question = input("❓ Your Question: ").strip()
        
        if question.lower() == 'exit':
            break
        if not question:
            continue
        
        print("\n🔍 Searching KT sessions...")
        answer, sources = ask_question(question, retriever)
        
        print(f"\n💡 Answer:\n{answer}")
        print(f"\n📂 Source Sessions: {', '.join(sources)}")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()