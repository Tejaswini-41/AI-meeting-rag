import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

VECTOR_DB_DIR = "./vector_db"

st.set_page_config(page_title="KT Q&A Assistant", page_icon="🤖")
st.title("🤖 KT Session Q&A")
st.caption("Ask anything from JAVA, AI, and MUT KT sessions")

@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if question := st.chat_input("Ask from KT sessions..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching KT sessions..."):
            retriever = load_retriever()
            docs = retriever.invoke(question)
            context = "\n\n".join([d.page_content for d in docs])
            sources = list(set([d.metadata["topic"] for d in docs]))
            
            prompt = f"""Answer based on KT sessions only. If not found, say so.

Context: {context}

Question: {question}
Answer:"""
            
            llm = Ollama(model="llama3")
            answer = llm.invoke(prompt)
            
            st.markdown(answer)
            st.caption(f"📂 Sources: {', '.join(sources)}")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer
            })