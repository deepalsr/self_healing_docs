from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    WEBHOOK_SECRET: str = ""
    GEMINI_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    GITHUB_TOKEN: str = ""
    GENERATION_MODEL: str = "gemini-3.5-flash"
    ALLOWED_REPOS: str = ""  # comma-separated "owner/repo", empty = allow all (dev default)

    model_config = SettingsConfigDict(env_file=".env")

    def allowed_repos_set(self) -> set[str]:
        return {r.strip() for r in self.ALLOWED_REPOS.split(",") if r.strip()}


settings = Settings()