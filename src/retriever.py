from core.config import settings
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
    vector_store = Chroma.from_documents(
        embedding=embeddings,
        persist_directory=str(database_path),
    )
    print(f"Total chunks to embed: {len(chunks)}. Starting batched upload...")
    batch_size = settings.EMBEDDING_BATCH_SIZE
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Add current batch to the vector database
        vector_store.add_documents(documents=batch)
        print(f"Successfully embedded chunk {i + 1} of {len(chunks)}.")
        
        # Prevent hitting the Gemini RPM limit on the next loop
        if i + batch_size < len(chunks):
            time.sleep(settings.EMBEDDING_SLEEP_DELAY)
            
    return vector_store

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
        
    