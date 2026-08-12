from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application configuration, loaded from environment variables (.env).

    Every module that needs config (DB connection, Ollama host, etc.) imports
    `settings` from here, instead of calling os.getenv() scattered across
    the codebase. Field names are matched case-insensitively against .env
    keys by pydantic-settings (e.g. postgres_host <-> POSTGRES_HOST).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "crypto_market_db"
    postgres_user: str = "rag_user"
    postgres_password: str = ""

    ollama_host: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_chat_model: str = "llama3.1:8b"

    log_level: str = "INFO"
    environment: str = "development"

    @property
    def postgres_dsn(self) -> str:
        """Connection string used by psycopg to connect to Postgres."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
