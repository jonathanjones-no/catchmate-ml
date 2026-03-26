from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_service_key: SecretStr = SecretStr("")
    database_url: SecretStr = SecretStr("")
    google_bigquery_credentials_b64: SecretStr = SecretStr("")
    bigquery_project: str = "catchmate-production"
    bigquery_dataset: str = "catchmate_analytics"
    model_dir: str = "./checkpoints"
    embedding_dim: int = 32
    batch_size: int = 256
    learning_rate: float = 0.001

    class Config:
        env_file = ".env"


settings = Settings()
