# app/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    WEBHOOK_SECRET: str
    GEMINI_API_KEY: str
    PINECONE_API_KEY: str
    GITHUB_TOKEN: str

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()