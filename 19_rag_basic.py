import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from operator import itemgetter


def build_vectorstore(persist_dir: str) -> Chroma:
    pdf_path = Path(__file__).parent / "14_openresume-resume.pdf"
    if not pdf_path.exists():
        # Fallback to README if PDF not found
        corpus_path = Path(__file__).parent / "README.md"
        text = corpus_path.read_text(encoding="utf-8") if corpus_path.exists() else ""
        if not text:
            text = "Resume PDF not found and README missing; using placeholder text."
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        docs = splitter.create_documents(
            [text], metadatas=[{"source": str(corpus_path)}]
        )
    else:
        # Load PDF pages, then split into chunks
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = splitter.split_documents(pages)

    print(f"Loaded {len(docs)} chunks for indexing")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]
    )
    vs = Chroma.from_documents(
        docs, embedding=embeddings, persist_directory=persist_dir
    )
    vs.persist()
    return vs


def get_or_create_vectorstore(base_persist_dir: str) -> Chroma:
    deployment = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]
    persist_dir = f"{base_persist_dir}_{deployment}"
    embeddings = AzureOpenAIEmbeddings(azure_deployment=deployment)
    if Path(persist_dir).exists():
        try:
            return Chroma(embedding_function=embeddings, persist_directory=persist_dir)
        except Exception:
            pass
    return build_vectorstore(persist_dir)


def make_chain(vs: Chroma):
    retriever = vs.as_retriever(search_kwargs={"k": 6})

    system_template = (
        "You are a concise assistant. Use the provided context to answer the user's question. "
        "If the answer is not in the context, say you don't know. Keep answers under 8 sentences.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    prompt = PromptTemplate.from_template(system_template)

    # Use AzureChatOpenAI per user's preference
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        temperature=0,
    )
    parser = StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

    # LCEL chain: take question -> retrieve -> format -> prompt -> llm -> string
    return (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | parser
    )


def main():
    # Accept either OpenAI or Azure OpenAI env configuration
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_azure = bool(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
    if not (has_openai or has_azure):
        raise RuntimeError(
            "Set OPENAI_API_KEY or configure Azure with AZURE_OPENAI_DEPLOYMENT_NAME (and related Azure env vars)."
        )

    persist_dir = str(Path(__file__).parent / ".chroma_rag_resume")
    vs = get_or_create_vectorstore(persist_dir)
    chain = make_chain(vs)

    questions = [
        "Summarize the candidate's experience.",
        "What are the key skills listed?",
        "Give 2 interview questions tailored to this resume.",
    ]

    for q in questions:
        print("\n=== Question ===\n", q)
        # For quick visibility, show top retrieved docs
        top_docs = vs.similarity_search(q, k=3)
        print(f"Retrieved {len(top_docs)} docs. First snippet: ", (top_docs[0].page_content[:180] + "...") if top_docs else "<none>")
        answer = chain.invoke({"question": q})
        print("\n--- Answer ---\n", answer)


if __name__ == "__main__":
    main()
