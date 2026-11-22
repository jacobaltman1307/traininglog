"""Configuration for the TrainingLog server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ServerConfig:
    """Server configuration loaded from environment variables and CLI flags."""

    data_dir: Path = Path("./data")
    host: str = "0.0.0.0"
    port: int = 8765
    api_token: str | None = None
    stall_timeout_sec: int = 300
    stall_check_interval_sec: int = 15

    @property
    def db_path(self) -> Path:
        return self.data_dir / "traininglog.db"

    @property
    def series_dir(self) -> Path:
        return self.data_dir / "series"

    @property
    def blobs_dir(self) -> Path:
        return self.data_dir / "blobs"

    @classmethod
    def from_env(
        cls,
        data_dir: str | Path | None = None,
        host: str | None = None,
        port: int | None = None,
        api_token: str | None = None,
        stall_timeout_sec: int | None = None,
        stall_check_interval_sec: int | None = None,
    ) -> ServerConfig:
        env_data_dir = os.environ.get("TRAININGLOG_DATA_DIR", "./data")
        env_host = os.environ.get("TRAININGLOG_HOST", "0.0.0.0")
        env_port = int(os.environ.get("TRAININGLOG_PORT", "8765"))
        env_token = os.environ.get("TRAININGLOG_API_TOKEN") or None
        env_stall = int(os.environ.get("TRAININGLOG_STALL_TIMEOUT_SEC", "300"))
        env_stall_int = int(os.environ.get("TRAININGLOG_STALL_CHECK_INTERVAL_SEC", "15"))

        resolved_token = api_token if api_token is not None else env_token
        if resolved_token == "":
            resolved_token = None

        return cls(
            data_dir=Path(data_dir if data_dir is not None else env_data_dir),
            host=host if host is not None else env_host,
            port=port if port is not None else env_port,
            api_token=resolved_token,
            stall_timeout_sec=(
                stall_timeout_sec if stall_timeout_sec is not None else env_stall
            ),
            stall_check_interval_sec=(
                stall_check_interval_sec
                if stall_check_interval_sec is not None
                else env_stall_int
            ),
        )
