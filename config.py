from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, BaseModel


class OpenAISettings(BaseModel):
    API_KEY: SecretStr
    DEFAULT_MODEL: str = "gpt-4o-mini-2024-07-18"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"


class PineconeSettings(BaseModel):
    API_KEY: SecretStr


class Settings(BaseSettings):
    OPENAI: OpenAISettings
    PINECONE: PineconeSettings

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
    )


settings = Settings()
