from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()
class Settings(BaseSettings):
    GOOGLE_API_KEY: str = Field(...,min_length=1,)
# yahan bi env ke values dale kyuki agar .env se feilds missing hue toh yahan as default diye jaate [priority env then config]
    MODEL_NAME: str = "gemini-3.5-flash"
    EMBEDDING_MODEL_NAME : str = "gemini-embedding-2-preview"
    DATABASE_DIR : str = "chorma_book_store"
    DATA_DIR : str = "Books"
    CHUNK_SIZE : int = 1000
    CHUNK_OVERLAP : int = 150
    PDF_START_PAGE :int = 15 # PDF page 15 onwards - special case for "thepivotyear" book
    TOP_K_CONSTANT : int = 5
    
    EMBEDDING_BATCH_SIZE = 1 
    EMBEDDING_SLEEP_DELAY = 4.0
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()