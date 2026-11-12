"""Monthly paper scraping, text mining, Bedrock summarization, and S3 persistence workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from physical_ai_system.adapters.alphaxiv_scraper import scrape_alphaxiv_papers
from physical_ai_system.adapters.bedrock_client import BedrockAdvisorClient
from physical_ai_system.adapters.s3_store import S3Store
from physical_ai_system.config import AppConfig
from physical_ai_system.domain.models import MonthlyTextMiningSnapshot, PaperRecord
from physical_ai_system.domain.text_mining import (
    build_text_mining_snapshot,
    is_humanoid_or_physical_ai_paper,
)


def merge_papers_pure(
    existing: Sequence[PaperRecord],
    incoming: Sequence[PaperRecord],
) -> tuple[PaperRecord, ...]:
    """Pure deduplication preserving valid robotics papers and updating metadata."""
    merged: dict[str, PaperRecord] = {
        p.paper_id: p
        for p in existing
        if is_humanoid_or_physical_ai_paper(p.title, p.abstract)
    }
    for p in incoming:
        if not is_humanoid_or_physical_ai_paper(p.title, p.abstract):
            continue
        prev = merged.get(p.paper_id)
        if prev and prev.summary_ja and not p.summary_ja:
            from dataclasses import replace

            merged[p.paper_id] = replace(p, summary_ja=prev.summary_ja)
        else:
            merged[p.paper_id] = p
    ordered = sorted(
        merged.values(),
        key=lambda item: (item.published_at, item.paper_id),
        reverse=True,
    )
    return tuple(ordered)


def run_monthly_paper_pipeline(
    config: AppConfig,
    s3: S3Store,
    bedrock: BedrockAdvisorClient,
    keywords: Sequence[str] | None = None,
    max_per_keyword: int = 8,
) -> MonthlyTextMiningSnapshot:
    """Orchestrate rate-limited alphaXiv scrape -> summarize -> text mine -> save to S3."""
    active_keywords = tuple(keywords) if keywords else config.default_keywords

    existing_raw = s3.get_json("papers/catalog.json") or []
    existing_papers = tuple(PaperRecord.from_dict(item) for item in existing_raw)

    scraped = scrape_alphaxiv_papers(
        keywords=active_keywords,
        base_url=config.alphaxiv_base_url,
        interval_seconds=config.scrape_interval_seconds,
        max_per_keyword=max_per_keyword,
    )

    merged = merge_papers_pure(existing_papers, scraped)
    summarized = bedrock.summarize_batch(merged)
    snapshot = build_text_mining_snapshot(summarized, active_keywords)

    month_tag = datetime.now(timezone.utc).strftime("%Y-%m")
    papers_payload = [p.to_dict() for p in summarized]
    s3.put_json("papers/catalog.json", papers_payload)
    s3.put_json(f"papers/monthly_{month_tag}.json", papers_payload)
    s3.put_json("text_mining/latest.json", snapshot.to_dict())
    s3.put_json(f"text_mining/monthly_{month_tag}.json", snapshot.to_dict())
    return snapshot
