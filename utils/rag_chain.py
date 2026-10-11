import os
import re
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------
# Pick provider + model in your .env (or Streamlit Secrets):
#   LLM_PROVIDER = ollama | groq | openai | gemini | anthropic
#   LLM_MODEL    = any model id for that provider (optional)
# If LLM_MODEL is empty, the default below is used.
# ---------------------------------------------------------------
DEFAULT_MODELS = {
    "ollama": "llama3.2",
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-5-mini",
    "gemini": "gemini-3.5-flash",
    "anthropic": "claude-haiku-5-5",
}

# Handy shortlist (use any of these as LLM_MODEL)
MODEL_IDEAS = {
    "ollama": ["llama3.2", "qwen3:4b", "qwen3:8b", "gemma4", "qwen3.6:27b"],
    "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
    "openai": ["gpt-5-nano", "gpt-5-mini"],
    "gemini": ["gemini-3.5-flash", "gemini-3.1-flash-lite"],
    "anthropic": ["claude-haiku-5-5", "claude-sonnet-5-5"],
}


def get_llm(provider=None, model=None):
    """Return a chat model. Imports sit inside each branch, so you only
    need to install the package for the provider you actually use."""
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),  # default 4096 can truncate context
            temperature=0,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name=model)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        # No temperature: GPT-5 models only accept the default value
        return ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=model)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # No temperature: Gemini 3 models work best at the default value
        return ChatGoogleGenerativeAI(google_api_key=os.getenv("GOOGLE_API_KEY"), model=model)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), model=model)

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Use ollama, groq, openai, gemini or anthropic."
    )


def _to_text(content):
    """Some providers return a list of content blocks; some models add <think> tags."""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def create_rag_chain(vectorstore, provider=None, model=None):
    llm = get_llm(provider, model)
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

    answer = _to_text(llm.invoke(prompt).content)

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
