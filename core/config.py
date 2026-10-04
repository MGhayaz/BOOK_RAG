
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()
class Settings(BaseSettings):
    GOOGLE_API_KEY: str = Field(...,min_length=1,)
    OPENAI_API_KEY: str = Field(...,min_length=1,)
# yahan bi env ke values dale kyuki agar .env se feilds missing hue toh yahan as default diye jaate [priority env then config]
    LLM_MODEL_NAME: str = "gemini-3.5-flash-lite"
    OPENAI_EMBEDDING_MODEL_NAME : str = "text-embedding-3-large"
    OPENAI_JUDGEMENT_MODEL_NAME : str = "gpt-4o-mini"
    EVAL_THRESHOLD : float = 0.7
    DATABASE_DIR : str = "chorma_book_store"
    DATA_DIR : str = "Books"
    CHUNK_SIZE : int = 450
    CHUNK_OVERLAP : int = 70
    PDF_START_PAGE :int = 15 # PDF page 15 onwards - special case for "thepivotyear" book
    TOP_K_CONSTANT : int = 5
    DATASET_PATH : str = ("goldens/dataset.json")
    
    EMBEDDING_BATCH_SIZE : int = 1 
    EMBEDDING_SLEEP_DELAY : float = 3.7
    
    OPENAI_EMBEDDING_BATCH_SIZE : int = 100 
    OPENAI_EMBEDDING_SLEEP_DELAY : float = 0.3    
    
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()