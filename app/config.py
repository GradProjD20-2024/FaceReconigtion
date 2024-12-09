from pydantic_settings import BaseSettings
from shared.shared import singleton

@singleton
class Settings(BaseSettings):    
    MILVUS_HOST: str
    MILVUS_PORT: str
    
    # Model settings
    MODEL_PATH: str
    SIMILARITY_THRESHOLD: float

    class Config:
        env_file = ".env"
        extra = "allow"