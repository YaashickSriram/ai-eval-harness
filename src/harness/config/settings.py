"""
Settings — type-safe loading of environment variables.

WHY use Pydantic Settings:
  - Validates types at load time (catches "GROQ_API_KEY is missing" early)
  - SecretStr prevents accidental logging of keys
  - Single source of truth : every module imports `settings`, no os.getenv scattered around
  - Industry standard in 2026 : FastAPI, LangChain, most modern Python projects use this pattern
"""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # WHY: extra env vars shouldn't crash us
    )

    # ----- Provider credentials -----
    # WHY SecretStr: prevents accidental exposure in logs/repr.
    # Calling str(settings.groq_api_key) prints "**********", not the key.
    # You must call .get_secret_value() to get the real string : explicit, auditable.
    groq_api_key: SecretStr = Field(..., description="Groq API key")
    gemini_api_key: SecretStr = Field(..., description="Google AI Studio API key")

    # ----- Default models -----
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    gemini_model: str = Field(default="gemini-2.5-flash")

    # ----- Harness behavior -----
    # ----- Harness behavior -----
    harness_log_level: str = Field(default="INFO")
    harness_trace_dir: Path = Field(default=Path("traces"))

    # ----- Network workaround (corporate proxy with SSL inspection) -----
    # WHY: when True, LLM HTTP clients skip TLS verification.
    # Set via HARNESS_INSECURE_SKIP_TLS_VERIFY env var.
    # Should be False in production. See .env.example for context.
    harness_insecure_skip_tls_verify: bool = Field(
        default=False,
        description="Skip TLS cert verification : corporate-proxy workaround only.",
    )


# Singleton instance.
# WHY: import this once, use everywhere. Same pattern as your IMS ConfigLoader.
settings = Settings()  # type: ignore[call-arg]
