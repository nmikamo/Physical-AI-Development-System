"""Pure functional prompt builders & deterministic AI advisor logic (Antigravity-style)."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from physical_ai_system.domain.models import (
    RLMethod,
    TrainingConfig,
    TrainingReport,
)
from physical_ai_system.domain.rl_algorithms import describe_method_comparison


def recommend_method_pure(
    user_question: str,
    mocap_frames: int = 120,
) -> tuple[RLMethod, TrainingConfig, str]:
    """Pure deterministic recommendation logic matching Antigravity consultation."""
    q = user_question.lower()
    if any(k in q for k in ("rlpd", "高速", "サンプル効率", "外乱", "ロバスト", "成功")):
        method: RLMethod = "RLPD"
        cfg = TrainingConfig(
            run_id="suggested",
            method="RLPD",
            mocap_id="sample_walk",
            epochs=30,
            learning_rate=3e-4,
            bc_weight=0.8,
            entropy_coef=0.015,
            smoothness_weight=0.06,
            reward_tracking_weight=1.1,
        )
    elif any(k in q for k in ("awac", "滑らか", "安全", "過学習", "安定", "振動")):
        method = "AWAC"
        cfg = TrainingConfig(
            run_id="suggested",
            method="AWAC",
            mocap_id="sample_walk",
            epochs=30,
            learning_rate=2e-4,
            bc_weight=1.0,
            entropy_coef=0.01,
            smoothness_weight=0.10,
            reward_tracking_weight=1.0,
        )
    else:
        method = "DAPG"
        cfg = TrainingConfig(
            run_id="suggested",
            method="DAPG",
            mocap_id="sample_walk",
            epochs=30,
            learning_rate=3e-4,
            bc_weight=1.1,
            entropy_coef=0.01,
            smoothness_weight=0.06,
            reward_tracking_weight=1.2,
        )

    info = describe_method_comparison(method)
    reply = (
        f"### 推奨アルゴリズム: **{info['name']}**\n"
        f"- **選定理由**: {info['best_for']}\n"
        f"- **概要**: {info['summary']}\n"
        f"- **モーションデータ長**: {mocap_frames} フレームに対して、"
        f"`epochs={cfg.epochs}`, `lr={cfg.learning_rate}`, `bc_weight={cfg.bc_weight}`, "
        f"`smoothness_weight={cfg.smoothness_weight}` を初期設定として提案します。\n\n"
        f"このまま「模倣学習＆強化学習スタート」を押すか、他の手法（DAPG / RLPD / AWAC）を選択して実行できます。"
    )
    return method, cfg, reply


def parse_retrain_instruction(
    instruction: str,
    base_config: TrainingConfig,
    latest_report: TrainingReport | None = None,
) -> tuple[TrainingConfig, str]:
    """Purely translate human evaluation feedback into updated immutable TrainingConfig."""
    text = instruction.lower()
    updated = base_config
    notes: list[str] = []

    # Method switch if explicitly mentioned
    if "dapg" in text:
        updated = replace(updated, method="DAPG")
        notes.append("学習手法を DAPG に変更")
    elif "rlpd" in text:
        updated = replace(updated, method="RLPD")
        notes.append("学習手法を RLPD に変更")
    elif "awac" in text:
        updated = replace(updated, method="AWAC")
        notes.append("学習手法を AWAC に変更")

    # Smoothness adjustments
    if any(k in text for k in ("滑らか", "スムーズ", "振動", "ガタ", "smooth", "jerk")):
        new_smooth = min(0.30, round(updated.smoothness_weight + 0.06, 3))
        updated = replace(updated, smoothness_weight=new_smooth)
        notes.append(f"滑らかさ重み `smoothness_weight` を {new_smooth} に増加")

    # Tracking adjustments
    if any(k in text for k in ("追随", "真似", "模倣", "ズレ", "follow", "track")):
        new_track = min(2.5, round(updated.reward_tracking_weight + 0.4, 2))
        new_bc = min(2.0, round(updated.bc_weight + 0.3, 2))
        updated = replace(
            updated,
            reward_tracking_weight=new_track,
            bc_weight=new_bc,
        )
        notes.append(
            f"追随性強化のため `reward_tracking_weight={new_track}`, `bc_weight={new_bc}` に調整"
        )

    # Overfitting mitigation
    if any(k in text for k in ("過学習", "overfit", "汎用", "汎化")):
        new_bc = max(0.3, round(updated.bc_weight * 0.65, 2))
        new_ent = min(0.05, round(updated.entropy_coef + 0.01, 3))
        new_lr = max(5e-5, round(updated.learning_rate * 0.7, 6))
        updated = replace(
            updated,
            bc_weight=new_bc,
            entropy_coef=new_ent,
            learning_rate=new_lr,
        )
        notes.append(
            f"過学習抑制のため `bc_weight={new_bc}`, `entropy_coef={new_ent}`, `lr={new_lr}` に緩和"
        )

    # Epochs mention
    epoch_match = re.search(r"(\d+)\s*(?:エポック|epoch)", text)
    if epoch_match:
        ep = max(10, min(120, int(epoch_match.group(1))))
        updated = replace(updated, epochs=ep)
        notes.append(f"エポック数を {ep} に設定")
    elif latest_report and latest_report.diagnostics.overfitting_detected and not notes:
        updated = replace(
            updated,
            bc_weight=max(0.4, round(updated.bc_weight * 0.7, 2)),
            entropy_coef=min(0.04, round(updated.entropy_coef + 0.01, 3)),
        )
        notes.append("直近の過学習検出に基づき正則化を自動調整")

    if not notes:
        updated = replace(updated, epochs=min(60, updated.epochs + 10))
        notes.append(f"追加学習のためエポック数を {updated.epochs} に設定")

    updated = replace(updated, instruction_note=instruction.strip())
    explanation = "指示内容を反映し、以下の学習パラメータへ自動調整しました：\n" + "\n".join(
        f"- {n}" for n in notes
    )
    return updated, explanation


def build_advisor_prompt(
    user_message: str,
    latest_report: TrainingReport | None,
    available_mocaps: tuple[dict[str, Any], ...],
) -> str:
    """Purely assemble structured prompt context for Amazon Bedrock Antigravity-style advisor."""
    report_ctx = "まだ学習レポートはありません。"
    if latest_report is not None:
        final_m = latest_report.metrics_history[-1]
        report_ctx = (
            f"直近Run ID: {latest_report.run_id}\n"
            f"手法: {latest_report.config.method}, Epochs: {latest_report.config.epochs}, "
            f"LR: {latest_report.config.learning_rate}, BC重み: {latest_report.config.bc_weight}, "
            f"滑らかさ重み: {latest_report.config.smoothness_weight}\n"
            f"最終Train Loss: {final_m.train_loss:.4f}, 最終Val Loss: {final_m.val_loss:.4f}, "
            f"タスク成功率: {final_m.task_success_rate*100:.1f}%\n"
            f"診断結果: {latest_report.diagnostics.summary_ja}"
        )

    mocap_names = ", ".join(m.get("name", m.get("mocap_id", "")) for m in available_mocaps) or "sample_walk"
    return (
        "あなたは2足歩行人型ロボットの模倣学習・強化学習（MuJoCo）を支援するPhysical AI開発エージェント（Antigravity互換Webアシスタント）です。\n"
        "選択可能な学習手法は **DAPG**、**RLPD**、**AWAC** の3種類です。\n"
        f"利用可能モーションデータ: {mocap_names}\n"
        f"現在の学習状態:\n{report_ctx}\n\n"
        f"ユーザーからの質問・評価・再指示:\n{user_message}\n\n"
        "日本語で簡潔・的確に回答し、推奨手法（DAPG/RLPD/AWACのいずれか）と、"
        "ロボットの追随性・動作の滑らかさ・過学習の観点での具体的な調整案（ハイパーパラメータ含む）を提案してください。"
    )
