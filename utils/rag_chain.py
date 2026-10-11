import os
from dotenv import load_dotenv

load_dotenv()

# Choose the model provider in your .env file:
#   LLM_PROVIDER=ollama   (local, nothing leaves your PC)  <- default
#   LLM_PROVIDER=groq | openai | gemini
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()


def get_llm():
    """Return a chat model for the selected provider.
    Imports are inside each branch so you only need the package you use."""

    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )

    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        )

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )

    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=0,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use ollama, groq, openai or gemini."
    )


def create_rag_chain(vectorstore):
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return llm, retriever


def ask_question(llm, retriever, query):
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the context below. "
        "If the answer is not in the context, say \"I couldn't find that in the documents.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    response = llm.invoke(prompt)
    answer = response.content

    # Add source file + page numbers (PyPDFLoader stores them in metadata)
    sources = sorted({
        f"{os.path.basename(d.metadata.get('source', 'unknown'))} (page {d.metadata.get('page', 0) + 1})"
        for d in docs
    })
    if sources:
        answer += "\n\n**Sources:** " + ", ".join(sources)

    return answer


if __name__ == "__main__":
    from vectorstore import load_vectorstore
    db = load_vectorstore()
    llm, retriever = create_rag_chain(db)
    print(f"✅ Answer: {ask_question(llm, retriever, 'What is calisthenics?')}")
