"""Amazon Bedrock Converse API adapter with deterministic functional fallback."""

from __future__ import annotations

import logging
from typing import Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from physical_ai_system.domain.models import PaperRecord

logger = logging.getLogger(__name__)


def summarize_paper_extractive_fallback(paper: PaperRecord) -> str:
    """Pure deterministic summary when Bedrock is unreachable or throttled."""
    abstract = paper.abstract.strip()
    if not abstract:
        return f"タイトル: {paper.title}\nアブストラクト情報が取得できなかったため、URLリンク先から原文をご確認ください。"
    sentences = [s.strip() for s in abstract.replace("\n", " ").split(". ") if s.strip()]
    lead = ". ".join(sentences[:2])
    sim_tag = "Sim-to-Real言及あり" if paper.sim_to_real else "一般ヒューマノイド・制御研究"
    kw = ", ".join(paper.keywords_matched) if paper.keywords_matched else "-"
    return (
        f"【概要（自動抽出）】{lead}.\n"
        f"・分類: {sim_tag}\n"
        f"・一致キーワード: {kw}"
    )


class BedrockAdvisorClient:
    """Thin I/O boundary wrapping Amazon Bedrock Converse API."""

    def __init__(
        self,
        region: str,
        advisor_model_id: str,
        summary_model_id: str,
    ) -> None:
        self._region = region
        self._advisor_model_id = advisor_model_id
        self._summary_model_id = summary_model_id
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def _converse(self, model_id: str, prompt: str, max_tokens: int = 800) -> str:
        resp = self._client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.3},
        )
        blocks = resp.get("output", {}).get("message", {}).get("content", [])
        texts = [b["text"] for b in blocks if "text" in b]
        return "\n".join(texts).strip()

    def summarize_paper(self, paper: PaperRecord) -> str:
        """Summarize an academic paper in Japanese via Bedrock (with safe fallback)."""
        if not paper.abstract.strip():
            return summarize_paper_extractive_fallback(paper)
        prompt = (
            "以下の人型ロボット・Physical AIに関する英語論文のタイトルとアブストラクトを、"
            "技術担当者および経営層（部長・役員）向けに日本語で3〜4行で簡潔に要約してください。\n"
            "「目的」「手法」「主要な成果・Sim-to-Realへの示唆」を含めてください。\n\n"
            f"タイトル: {paper.title}\n"
            f"アブストラクト: {paper.abstract}"
        )
        try:
            text = self._converse(self._summary_model_id, prompt, max_tokens=450)
            return text or summarize_paper_extractive_fallback(paper)
        except (BotoCoreError, ClientError, Exception) as exc:
            logger.warning("Bedrock summarize fallback for %s: %s", paper.paper_id, exc)
            return summarize_paper_extractive_fallback(paper)

    def summarize_batch(self, papers: Sequence[PaperRecord]) -> tuple[PaperRecord, ...]:
        """Return a new tuple of PaperRecord instances with Japanese summaries populated."""
        from dataclasses import replace

        results: list[PaperRecord] = []
        for p in papers:
            if p.summary_ja.strip():
                results.append(p)
            else:
                summary = self.summarize_paper(p)
                results.append(replace(p, summary_ja=summary))
        return tuple(results)

    def ask_advisor(self, prompt: str, fallback_text: str) -> tuple[str, str]:
        """Query Bedrock for Antigravity-style advising, returning (reply_text, engine_used)."""
        try:
            text = self._converse(self._advisor_model_id, prompt, max_tokens=900)
            if text:
                return text, f"Amazon Bedrock ({self._advisor_model_id})"
        except (BotoCoreError, ClientError, Exception) as exc:
            logger.warning("Bedrock advisor fallback: %s", exc)
        return fallback_text, "Local Deterministic Advisor (Bedrock Fallback)"
