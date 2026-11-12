"""CLI entrypoint invoked by Linux cron once a month for alphaXiv paper scraping & mining."""

from __future__ import annotations

import json
import logging

from dotenv import load_dotenv

from physical_ai_system.adapters.bedrock_client import BedrockAdvisorClient
from physical_ai_system.adapters.s3_store import S3Store
from physical_ai_system.config import load_config
from physical_ai_system.workflows.paper_pipeline import run_monthly_paper_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()
    cfg = load_config()
    s3 = S3Store(cfg.s3_bucket, cfg.aws_region, cfg.local_data_dir)
    bedrock = BedrockAdvisorClient(
        cfg.aws_region,
        cfg.bedrock_model_id,
        cfg.bedrock_summary_model_id,
    )
    logger.info("Starting monthly alphaXiv scrape with keywords=%s", cfg.default_keywords)
    snapshot = run_monthly_paper_pipeline(cfg, s3, bedrock)
    summary = {
        "generated_at": snapshot.generated_at,
        "period_label": snapshot.period_label,
        "total_papers": snapshot.total_papers,
        "sim_to_real_count": snapshot.sim_to_real_count,
    }
    logger.info("Monthly paper pipeline completed: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
