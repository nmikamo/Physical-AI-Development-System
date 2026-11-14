"""Amazon Bedrock Converse API adapter with rich Japanese technical summary fallback."""

from __future__ import annotations

import logging
import re
from typing import Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from physical_ai_system.domain.models import PaperRecord

logger = logging.getLogger(__name__)

_TECH_RULES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (
        ("tracking", "mimic", "imitation", "mocap", "motion", "retargeting"),
        "人間の全身モーションデータ（MoCap）を規範軌道として人型ロボットへ転移する全身モーショントラッキング・模倣学習手法を提案。",
        "関節軌道追随精度と動的バランスの両立により、複雑な全身動作の再現性を向上。",
    ),
    (
        ("diffusion", "generative", "flow", "trajectory"),
        "拡散モデル（Diffusion Policy）や生成モデルを用いて、多様な全身軌道・行動分布を学習するフレームワークを構築。",
        "マルチモーダルな行動生成と滑らかな軌道補間を実現。",
    ),
    (
        ("rlpd", "awac", "off-policy", "sample", "prior", "actor-critic"),
        "事前デモンストレーションデータとオンライン強化学習を統合し、サンプル効率と学習安定性を高めるオフポリシー強化学習手法を提案。",
        "少ない試行回数で高いタスク成功率を達成し、過学習や方策崩壊を抑制。",
    ),
    (
        ("mpc", "safety", "constraint", "optimal", "qp"),
        "モデル予測制御（MPC）または安全制約と学習ベース方策を融合したハイブリッド全身制御アーキテクチャを設計。",
        "トルク・関節限界を厳守しつつ外乱に対するロバスト性を強化。",
    ),
    (
        ("parkour", "terrain", "visual", "elevation", "stairs", "perceptive"),
        "オンボード視覚・深度知覚と歩容制御を統合し、段差や不整地を踏破する知覚駆動型2足ロコモーションを実現。",
        "未知の不整地環境における適応的フットステップ計画と高い生存率を実証。",
    ),
    (
        ("recovery", "fall", "get-up", "push", "balance", "standing"),
        "強い外乱や転倒状態からの自律立ち上がり・動的バランス復帰を実現する全身強化学習コントローラを構築。",
        "多様な初期姿勢からの復帰成功率と実機耐衝撃性を大幅に改善。",
    ),
    (
        ("manipulation", "loco-manipulation", "contact", "force", "grasp", "bimanual"),
        "2足歩行による移動と両腕による物体操作を協調させる全身ロコマニピュレーション・接触力適応制御を提案。",
        "負荷変動下での安定した歩行維持と精密な双腕作業を両立。",
    ),
    (
        ("teleoperation", "teleop", "vr", "exoskeleton", "demonstration"),
        "低遅延な全身テレオペレーションと実機デモンストレーション収集システムを構築し、模倣学習データパイプラインを確立。",
        "直感的な遠隔操作と高品質な全身モーションデータ獲得を実現。",
    ),
    (
        ("llm", "vlm", "language", "reward", "vision-language"),
        "大規模言語モデル（LLM）/ 視覚言語モデル（VLM）を活用し、高次タスク計画や強化学習の報酬関数自動設計を実現。",
        "人手による報酬調整コストを削減し、自然言語指示からの行動生成を可能化。",
    ),
)


def summarize_paper_extractive_fallback(paper: PaperRecord) -> str:
    """Pure deterministic extractor when Bedrock summary is not generated.

    Explicitly indicates that AI summary is pending and extracts original abstract
    sentences without hallucinating or reusing identical boilerplate templates.
    """
    abstract_clean = " ".join(paper.abstract.strip().split())
    sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", abstract_clean) if s.strip()]

    # Extract first sentence as primary objective/scope
    first_sentence = sentences[0] if sentences else paper.title
    # Extract concluding sentence if available
    conclusion = sentences[-1] if len(sentences) > 1 else ""

    sim_note = (
        "実機転移（Sim-to-Real）関連の記述が原文アブストラクトに含まれています。"
        if paper.sim_to_real
        else "シミュレーション環境での検証を主体とする研究です。"
    )

    kw_str = ", ".join(paper.keywords_matched) if paper.keywords_matched else "humanoid, physical AI"

    lines = [
        "【要約ステータス】AI要約未生成（原文抄録を表示中）",
        f"【研究の主眼】{first_sentence}",
    ]
    if conclusion and conclusion != first_sentence:
        lines.append(f"【結言・成果】{conclusion}")
    lines.extend([
        f"【Sim-to-Real分類】{sim_note}",
        f"【該当キーワード】{kw_str}",
        f"【原文抄録】{abstract_clean}",
    ])
    return "\n".join(lines)


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
        self._bedrock_available: bool | None = None

    def _converse(self, model_id: str, prompt: str, max_tokens: int = 800) -> str:
        if self._bedrock_available is False:
            raise RuntimeError("Bedrock circuit breaker open (use case form pending)")
        try:
            resp = self._client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.3},
            )
            self._bedrock_available = True
            blocks = resp.get("output", {}).get("message", {}).get("content", [])
            texts = [b["text"] for b in blocks if "text" in b]
            return "\n".join(texts).strip()
        except (BotoCoreError, ClientError, Exception) as exc:
            msg = str(exc)
            if "use case details" in msg or "AccessDenied" in msg or "ResourceNotFound" in msg:
                self._bedrock_available = False
            raise

    def summarize_paper(self, paper: PaperRecord) -> str:
        """Summarize an academic paper in Japanese via Bedrock (with structured Japanese fallback)."""
        if not paper.abstract.strip():
            return summarize_paper_extractive_fallback(paper)
        prompt = (
            "以下の人型ロボット・Physical AIに関する英語論文のタイトルとアブストラクトを、"
            "技術担当者および経営層（部長・役員）向けに日本語で4行程度で構造化して要約してください。\n"
            "【研究目的・手法】【技術的成果】【Sim-to-Real・応用示唆】の見出しを含めてください。\n\n"
            f"タイトル: {paper.title}\n"
            f"アブストラクト: {paper.abstract}"
        )
        try:
            text = self._converse(self._summary_model_id, prompt, max_tokens=450)
            return text or summarize_paper_extractive_fallback(paper)
        except Exception as exc:
            logger.debug("Bedrock summarize fallback for %s: %s", paper.paper_id, exc)
            return summarize_paper_extractive_fallback(paper)

    def summarize_batch(self, papers: Sequence[PaperRecord]) -> tuple[PaperRecord, ...]:
        """Return a new tuple of PaperRecord instances with structured Japanese summaries populated."""
        from dataclasses import replace

        results: list[PaperRecord] = []
        for p in papers:
            # Retain if already populated with structured or extracted summary
            if p.summary_ja.strip() and (
                "【研究目的・手法】" in p.summary_ja
                or "【要約ステータス】" in p.summary_ja
                or "【研究の主眼】" in p.summary_ja
            ):
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
        except Exception as exc:
            logger.debug("Bedrock advisor fallback: %s", exc)
        return fallback_text, "Antigravity AI Engine (Rule + Metric Diagnostics)"
