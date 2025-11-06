"""
INTERVIEW STYLE Q&A:

Q: What is RAG (Retrieval-Augmented Generation) and why is it useful?
A: RAG combines information retrieval with LLM generation. Instead of relying solely
   on the model's training data, RAG retrieves relevant documents, adds them as context,
   and generates answers based on that context. This enables answering questions about
   specific documents the model wasn't trained on.

Q: How does RAG work in practice?
A: (1) Load and split documents into chunks, (2) Create embeddings and store in a
   vector database, (3) When asked a question, retrieve similar chunks, (4) Include
   retrieved chunks as context in the LLM prompt, (5) Generate answer based on context.

Q: What is a vector store and why use it?
A: A vector store (like Chroma) stores document embeddings and enables similarity
   search. When you query, it finds the most semantically similar document chunks
   to your question, even if they don't contain exact keyword matches.

Q: What are the key components of a RAG system?
A: (1) Document loader (loads PDFs, text files, etc.), (2) Text splitter (chunks
   documents), (3) Embeddings model (converts text to vectors), (4) Vector store
   (stores and searches embeddings), (5) Retriever (finds relevant chunks), (6) LLM
   (generates answers from context).

SAMPLE CODE:
"""

import os
from operator import itemgetter
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Q: How do you build a vector store from documents?
# A: Load documents, split into chunks, create embeddings, and store in vector database
def build_vectorstore(persist_dir: str) -> Chroma:
    # Q: How do you handle missing documents?
    # A: Check if file exists, provide fallback (README) or placeholder text
    pdf_path = Path(__file__).parent / "14_openresume-resume.pdf"
    if not pdf_path.exists():
        # Fallback to README if PDF not found
        corpus_path = Path(__file__).parent / "README.md"
        text = corpus_path.read_text(encoding="utf-8") if corpus_path.exists() else ""
        if not text:
            text = "Resume PDF not found and README missing; using placeholder text."
        # Q: How do you split text into chunks?
        # A: Use RecursiveCharacterTextSplitter with chunk_size and chunk_overlap
        #    Overlap ensures context isn't lost at chunk boundaries
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        docs = splitter.create_documents(
            [text], metadatas=[{"source": str(corpus_path)}]
        )
    else:
        # Q: How do you load PDF documents?
        # A: Use PyPDFLoader to load PDF pages, then split into smaller chunks
        # Load PDF pages, then split into chunks
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = splitter.split_documents(pages)

    print(f"Loaded {len(docs)} chunks for indexing")

    # Q: How do you create embeddings?
    # A: Use an embeddings model (AzureOpenAIEmbeddings) to convert text to vectors
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]
    )
    # Q: How do you store documents in a vector database?
    # A: Use Chroma.from_documents() with documents, embeddings, and persist directory
    #    This creates embeddings and stores them for later retrieval
    vs = Chroma.from_documents(
        docs, embedding=embeddings, persist_directory=persist_dir
    )
    vs.persist()  # Save to disk for reuse
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


# Q: How do you create a RAG chain?
# A: Combine retriever (finds relevant docs) with LLM (generates answer from context)
def make_chain(vs: Chroma):
    # Q: How do you create a retriever?
    # A: Convert vector store to retriever with search_kwargs (like k=6 for top 6 results)
    retriever = vs.as_retriever(search_kwargs={"k": 6})

    # Q: How do you design a RAG prompt?
    # A: Include placeholders for context and question - instruct model to use context
    #    and say "don't know" if answer isn't in context
    system_template = (
        "You are a concise assistant. Use the provided context to answer the user's question. "
        "If the answer is not in the context, say you don't know. Keep answers under 8 sentences.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    prompt = PromptTemplate.from_template(system_template)

    # Q: How do you set up the LLM for RAG?
    # A: Use your chat model with low temperature for consistent, factual responses
    # Use AzureChatOpenAI per user's preference
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        temperature=0,
    )
    parser = StrOutputParser()

    # Q: How do you format retrieved documents?
    # A: Create a function that formats document chunks for inclusion in the prompt
    def format_docs(docs):
        return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

    # Q: How do you build the complete RAG chain?
    # A: Use LCEL to chain: question → retrieve → format → prompt → llm → parse
    #    LCEL chain: take question -> retrieve -> format -> prompt -> llm -> string
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
        print(
            f"Retrieved {len(top_docs)} docs. First snippet: ",
            (top_docs[0].page_content[:180] + "...") if top_docs else "<none>",
        )
        answer = chain.invoke({"question": q})
        print("\n--- Answer ---\n", answer)


if __name__ == "__main__":
    main()
