from core.config import settings
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf_docs() -> list[Document]:
    docs: list[Document] = []
    start_page = settings.PDF_START_PAGE
    for pdf_path in Path(settings.DATA_DIR).glob("*.pdf"):
        loader = PyPDFLoader(pdf_path)
        for page in loader.lazy_load():
            page_number = page.metadata.get("page", 0) + 1

        if page_number < start_page:
            continue

        text = page.page_content.strip()

        if not text:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    **page.metadata,
                    "source_file": pdf_path.name,
                },
            )
        )
    return docs

# 2. BUILD : chunk, embed once, and keep it on disk so we don't re-embed [delete db file if changes are made in setting]
def load_store() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL_NAME
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
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(database_path),
    )

def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": settings.TOP_K_CONSTANT})
if __name__ == "__main__":
    retriever = build_retriever()
    result = retriever.invoke("One day we will realize that happiness is not what?") # demo question
    
    for rs in result :
        page_number = rs.metadata.get("page")
        print(
                f"[{rs.metadata.get('source_file', 'Unknown')}] "
                f"[Page {(page_number + 1) if page_number is not None else 'N/A'}] "
                f"{rs.page_content[:150]}...\n"
            )
        
    