from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()
class Settings(BaseSettings):
    GOOGLE_API_KEY: str = Field(...,min_length=1,)
# yahan bi env ke values dale kyuki agar .env se feilds missing hue toh yahan as default diye jaate [priority env then config]
    LLM_MODEL_NAME: str = "gemini-3.5-flash"
    EMBEDDING_MODEL_NAME : str = "gemini-embedding-2-preview"
    JUDGEMENT_MODEL_NAME : str = "gemini-3.1-pro"
    EVAL_THRESHOLD : float = 0.7
    DATABASE_DIR : str = "chorma_book_store"
    DATA_DIR : str = "Books"
    CHUNK_SIZE : int = 900
    CHUNK_OVERLAP : int = 150
    PDF_START_PAGE :int = 15 # PDF page 15 onwards - special case for "thepivotyear" book
    TOP_K_CONSTANT : int = 5
    DATASET_PATH :str = "goldens/dataset.json"
    
    # EMBEDDING_BATCH_SIZE : int = 1 
    # EMBEDDING_SLEEP_DELAY : float = 3.7
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()