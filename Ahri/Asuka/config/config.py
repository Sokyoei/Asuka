"""
config form envioronment variables and .env file
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from Ahri.Asuka import ASUKA_ROOT


class Settings(BaseSettings):
    DEBUG: bool = False

    # dir
    LOG_DIR: Path = ASUKA_ROOT / "logs"
    DOWNLOAD_DIR: Path = ASUKA_ROOT / "downloads"

    model_config = SettingsConfigDict(
        env_file=[ASUKA_ROOT / ".env", ASUKA_ROOT / ".env.dev", ASUKA_ROOT / ".env.prod"],
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
