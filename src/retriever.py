from core.config import settings
import time
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf_docs() -> list[Document]:
    docs: list[Document] = []
    start_page = settings.PDF_START_PAGE
    for pdf_path in Path(settings.DATA_DIR).glob("*.pdf"):
        loader = PdfReader(pdf_path)
        for i, page in enumerate(loader.pages):
            # pypdf indexes pages starting from 0, so add 1
            page_number = i + 1

            if page_number < start_page:
                continue

            # Extract the raw text from the current page
            text = (page.extract_text() or "").strip()

            if not text:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_path),
                        "page": i,
                        "source_file": pdf_path.name,
                    },
                )
            )
    return docs

# 2. BUILD : chunk, embed once, and keep it on disk so we don't re-embed [delete db file if changes are made in setting]
def load_store() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL_NAME
    )
    database_path = Path(settings.DATABASE_DIR)
    if database_path.exists():
        return Chroma(
            persist_directory=str(database_path),
            embedding_function=embeddings,
        )
    docs = load_pdf_docs() # agar db file present nahi hai toh pdf load and further chunk karao
    if not docs:
        raise RuntimeError("No PDF documents were loaded.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise RuntimeError("PDFs loaded, but no chunks were created.")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(database_path),
    )
    print(f"Total chunks to embed: {len(chunks)}. Starting batched upload...")
    batch_size = settings.OPENAI_EMBEDDING_BATCH_SIZE
    sleep_delay = settings.OPENAI_EMBEDDING_SLEEP_DELAY

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        try:
            vector_store.add_documents(documents=batch)
        except Exception as exc:
            print(f"\n❌ Embedding error: {type(exc).__name__}: {exc}\n")
            raise

        processed = min(i + batch_size, len(chunks))

        print(
            f"Successfully embedded {processed} "
            f"of {len(chunks)} chunks."
        )

        # Keep this configurable because OpenAI also has rate limits.
        if processed < len(chunks) and sleep_delay > 0:
            time.sleep(sleep_delay)

    return vector_store

def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": settings.TOP_K_CONSTANT})
if __name__ == "__main__":
    retriever = build_retriever()
    result = retriever.invoke("How can I stop reacting to everything so quickly? What does the book suggest?") # demo question
    
    for rs in result :
        page_number = rs.metadata.get("page")
        print(
                f"[{rs.metadata.get('source_file', 'Unknown')}] "
                f"[Page {(page_number + 1) if page_number is not None else 'N/A'}] "
                f"{rs.page_content}...\n"
            )
        
   
