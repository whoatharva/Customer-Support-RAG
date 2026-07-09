from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Azure OpenAI
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_deployment_name: str
    azure_openai_embedding_deployment: str

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str = "knowledge_base"

    # Supabase
    supabase_url: str
    supabase_key: str

    # Auth
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 30
    admin_username: str
    admin_password: str

    # Langfuse
    langfuse_secret_key: str
    langfuse_public_key: str
    langfuse_base_url: str = "https://us.cloud.langfuse.com"

    # Lightweight LLM (HyDE + relevancy checks) — Gemini primary, Groq fallback
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
