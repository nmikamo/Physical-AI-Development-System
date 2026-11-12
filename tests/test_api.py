"""FastAPI endpoint tests."""

from pathlib import Path

from fastapi.testclient import TestClient

from physical_ai_system.config import AppConfig
from physical_ai_system.web.app import create_app


def test_health_and_dashboard_init(tmp_path: Path) -> None:
    cfg = AppConfig(
        aws_region="ap-northeast-3",
        s3_bucket="test-bucket-nonexistent",
        bedrock_model_id="jp.anthropic.claude-sonnet-4-6",
        bedrock_summary_model_id="jp.anthropic.claude-haiku-4-5-20251001-v1:0",
        local_data_dir=tmp_path,
        scrape_interval_seconds=2.0,
        default_keywords=("humanoid robot",),
        alphaxiv_base_url="https://www.alphaxiv.org",
    )
    app = create_app(cfg)
    client = TestClient(app)

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_home = client.get("/")
    assert r_home.status_code == 200
    assert "Physical AI Development System" in r_home.text

    r_init = client.get("/api/dashboard-init")
    assert r_init.status_code == 200
    assert len(r_init.json()["mocaps"]) >= 1
