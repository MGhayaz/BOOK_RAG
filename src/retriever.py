from core.config import settings
import os
import glob
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_transcripts():
    docs = []
    for path in glob.glob(f"{settings.DATA_DIR}/*.pdf"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line == "?" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs
# 2. BUILD : chunk, embed once, and keep it on disk so we don't re-embed [delete db file if changes are made in setting]
def load_store():
    embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
    if os.path.exists(settings.DATABASE_DIR):
        return Chroma(persist_directory=settings.DATABASE_DIR, embedding_function=embeddings)
    docs = load_transcripts()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    ).split_documents(docs)

    return Chroma.from_documents(chunks, embeddings, persist_directory=settings.DATABASE_DIR)
def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": settings.TOP_K_CONSTANT})
if __name__ == "__main__":
    retriever = build_retriever()
    result = retriever.invoke("One day we will realize that happiness is not what?")
    for rs in result :
         print(f"[Session {rs.metadata['session']}] {rs.page_content[:150]}...\n")
    