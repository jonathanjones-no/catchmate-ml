from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_service_key: str = ""
    database_url: str = ""
    model_dir: str = "./checkpoints"
    embedding_dim: int = 32
    batch_size: int = 256
    learning_rate: float = 0.001

    class Config:
        env_file = ".env"


settings = Settings()
