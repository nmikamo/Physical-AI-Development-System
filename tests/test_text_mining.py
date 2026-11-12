"""Unit tests for pure functional text mining and monthly aggregation."""

from physical_ai_system.domain.models import PaperRecord
from physical_ai_system.domain.text_mining import (
    aggregate_monthly_counts,
    build_text_mining_snapshot,
    detect_sim_to_real,
    is_humanoid_or_physical_ai_paper,
)


def _sample_papers() -> tuple[PaperRecord, ...]:
    return (
        PaperRecord(
            paper_id="2601.0001",
            title="Whole-Body Motion Tracking for Bipedal Humanoid Robots",
            abstract="We propose a diffusion policy for humanoid locomotion and sim-to-real transfer.",
            authors=("Alice",),
            published_at="2026-01-10",
            month_key="2026-01",
            url="https://www.alphaxiv.org/abs/2601.0001",
            keywords_matched=("humanoid robot",),
            sim_to_real=True,
        ),
        PaperRecord(
            paper_id="2601.0002",
            title="MPC and Safety Constraints for Bipedal Walking",
            abstract="Model predictive control stabilizes dynamic balance on uneven terrain.",
            authors=("Bob",),
            published_at="2026-01-22",
            month_key="2026-01",
            url="https://www.alphaxiv.org/abs/2601.0002",
            keywords_matched=("bipedal locomotion",),
            sim_to_real=False,
        ),
        PaperRecord(
            paper_id="2602.0003",
            title="RLPD with Prior Motion Capture Data on Humanoid",
            abstract="Off-policy reinforcement learning with domain randomization improves sim-to-real.",
            authors=("Carol",),
            published_at="2026-02-05",
            month_key="2026-02",
            url="https://www.alphaxiv.org/abs/2602.0003",
            keywords_matched=("physical AI",),
            sim_to_real=True,
        ),
        PaperRecord(
            paper_id="2602.0004",
            title="LLM Reward Design for Teleoperation Loco-Manipulation",
            abstract="Vision language models guide whole-body manipulation and fall recovery.",
            authors=("Dave",),
            published_at="2026-02-14",
            month_key="2026-02",
            url="https://www.alphaxiv.org/abs/2602.0004",
            keywords_matched=("whole-body control",),
            sim_to_real=False,
        ),
    )


def test_detect_sim_to_real() -> None:
    assert detect_sim_to_real("Zero-shot Sim-to-Real bipedal control")
    assert detect_sim_to_real("Trained with domain randomization")
    assert not detect_sim_to_real("Pure analytical kinematics")


def test_aggregate_monthly_counts() -> None:
    counts = aggregate_monthly_counts(_sample_papers())
    assert counts == (("2026-01", 2), ("2026-02", 2))


def test_build_text_mining_snapshot() -> None:
    snap = build_text_mining_snapshot(_sample_papers(), ("humanoid robot", "physical AI"))
    assert snap.total_papers == 4
    assert snap.sim_to_real_count == 2
    assert len(snap.points) == 4
    assert len(snap.density_grid.x_grid) == 56
    assert len(snap.density_grid.z_matrix) == 56
    assert all(pt.url.startswith("https://www.alphaxiv.org/abs/") for pt in snap.points)
    assert is_humanoid_or_physical_ai_paper("Humanoid Locomotion", "Bipedal robot in MuJoCo")
    assert not is_humanoid_or_physical_ai_paper("Self-Evolving Autonomous GIS", "Geographic systems")
