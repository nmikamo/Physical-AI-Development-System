"""Immutable application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_KEYWORDS: tuple[str, ...] = (
    "humanoid robot",
    "physical AI",
    "bipedal locomotion",
    "whole-body control",
    "imitation learning humanoid",
)


@dataclass(frozen=True)
class AppConfig:
    """Immutable runtime configuration loaded from environment variables."""

    aws_region: str
    s3_bucket: str
    bedrock_model_id: str
    bedrock_summary_model_id: str
    local_data_dir: Path
    scrape_interval_seconds: float
    default_keywords: tuple[str, ...]
    alphaxiv_base_url: str


def load_config() -> AppConfig:
    """Pure-style factory reading environment variables into an immutable AppConfig."""
    raw_keywords = os.getenv("ALPHAXIV_KEYWORDS", "")
    keywords = (
        tuple(k.strip() for k in raw_keywords.split(",") if k.strip())
        if raw_keywords.strip()
        else DEFAULT_KEYWORDS
    )
    interval = max(1.5, float(os.getenv("SCRAPE_INTERVAL_SECONDS", "2.0")))
    return AppConfig(
        aws_region=os.getenv("AWS_REGION", "ap-northeast-3"),
        s3_bucket=os.getenv("S3_BUCKET", "physical-ai-development-system-bucket"),
        bedrock_model_id=os.getenv(
            "BEDROCK_MODEL_ID", "jp.anthropic.claude-sonnet-4-6"
        ),
        bedrock_summary_model_id=os.getenv(
            "BEDROCK_SUMMARY_MODEL_ID", "jp.anthropic.claude-haiku-4-5-20251001-v1:0"
        ),
        local_data_dir=Path(os.getenv("LOCAL_DATA_DIR", "./local_data")),
        scrape_interval_seconds=interval,
        default_keywords=keywords,
        alphaxiv_base_url=os.getenv("ALPHAXIV_BASE_URL", "https://www.alphaxiv.org"),
    )
