"""Integration tests for MuJoCo humanoid simulation & mocap parsing."""

from pathlib import Path

from physical_ai_system.adapters.mujoco_engine import (
    generate_sample_biped_walk_mocap,
    parse_mocap_bytes,
    run_imitation_and_rl_training,
)
from physical_ai_system.domain.models import TrainingConfig


def test_parse_csv_and_bvh_mocap() -> None:
    csv_bytes = b"j1,j2,j3\n0.1,0.2,0.3\n0.2,0.3,0.4\n"
    traj = parse_mocap_bytes("walk.csv", csv_bytes, mocap_id="m_csv")
    assert traj.num_frames == 2
    assert traj.num_joints == 11

    bvh_bytes = b"HIERARCHY\nMOTION\nFrames: 2\nFrame Time: 0.0333\n0 0 0 1 2 3 4 5 6 7 8 9 10 11\n0 0 0 2 3 4 5 6 7 8 9 10 11 12\n"
    traj_bvh = parse_mocap_bytes("walk.bvh", bvh_bytes, mocap_id="m_bvh")
    assert traj_bvh.num_frames == 2
    assert traj_bvh.num_joints == 11


def test_mujoco_training_end_to_end(tmp_path: Path) -> None:
    mocap = generate_sample_biped_walk_mocap(num_frames=30)
    cfg = TrainingConfig(
        run_id="test_run",
        method="RLPD",
        mocap_id=mocap.mocap_id,
        epochs=5,
    )
    report, video_bytes, frames_payload = run_imitation_and_rl_training(
        config=cfg,
        mocap=mocap,
        output_dir=tmp_path,
    )
    assert len(report.metrics_history) == 5
    assert len(frames_payload["frames"]) == 30
    assert report.diagnostics.tracking_score > 0
    assert isinstance(video_bytes, bytes)
