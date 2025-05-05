from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, BaseModel

class OpenAISettings(BaseModel):
    API_KEY: SecretStr
    MODEL: str = "gpt-4o-mini-2024-07-18"

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
