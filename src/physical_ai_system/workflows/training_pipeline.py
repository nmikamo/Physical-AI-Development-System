"""Imitation + Reinforcement Learning & MuJoCo evaluation workflow persisting to S3 report/."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from physical_ai_system.adapters.mujoco_engine import (
    DEFAULT_HUMANOID_XML,
    generate_sample_biped_walk_mocap,
    parse_mocap_bytes,
    run_imitation_and_rl_training,
)
from physical_ai_system.adapters.s3_store import S3Store
from physical_ai_system.domain.models import (
    EpochMetrics,
    MocapTrajectory,
    TrainingConfig,
    TrainingReport,
)


def load_or_create_mocap(s3: S3Store, mocap_id: str) -> MocapTrajectory:
    """Load uploaded mocap trajectory from S3 mocap/ or synthesize default sample."""
    meta = s3.get_json(f"mocap/{mocap_id}/meta.json")
    if meta and "filename" in meta:
        raw = s3.get_bytes(f"mocap/{mocap_id}/{meta['filename']}")
        if raw:
            return parse_mocap_bytes(meta["filename"], raw, mocap_id=mocap_id)

    sample = generate_sample_biped_walk_mocap()
    if mocap_id == sample.mocap_id:
        # Persist sample metadata once so it shows up cleanly in S3 mocap/
        s3.put_json(
            f"mocap/{sample.mocap_id}/meta.json",
            {
                "mocap_id": sample.mocap_id,
                "name": sample.name,
                "filename": "sample_biped_walk.csv",
                "fps": sample.fps,
                "frames": sample.num_frames,
                "joints": sample.num_joints,
            },
        )
    return sample


def save_uploaded_mocap(
    s3: S3Store,
    mocap_id: str,
    filename: str,
    raw_bytes: bytes,
) -> MocapTrajectory:
    """Persist uploaded mocap data to S3 mocap/ and return parsed MocapTrajectory."""
    traj = parse_mocap_bytes(filename, raw_bytes, mocap_id=mocap_id)
    s3.put_bytes(f"mocap/{mocap_id}/{filename}", raw_bytes)
    s3.put_json(
        f"mocap/{mocap_id}/meta.json",
        {
            "mocap_id": traj.mocap_id,
            "name": traj.name,
            "filename": filename,
            "source_format": traj.source_format,
            "fps": traj.fps,
            "frames": traj.num_frames,
            "joints": traj.num_joints,
        },
    )
    return traj


def execute_training_workflow(
    s3: S3Store,
    config: TrainingConfig,
    local_runs_dir: Path,
    xml_text: str | None = None,
    progress_cb: Callable[[int, int, EpochMetrics], None] | None = None,
) -> TrainingReport:
    """Run imitation + RL training in MuJoCo and persist video/3D frames/report to S3 report/."""
    mocap = load_or_create_mocap(s3, config.mocap_id)
    report, video_bytes, frames_payload = run_imitation_and_rl_training(
        config=config,
        mocap=mocap,
        xml_text=xml_text or DEFAULT_HUMANOID_XML,
        output_dir=local_runs_dir,
        progress_cb=progress_cb,
    )
    if video_bytes:
        s3.put_bytes(report.video_s3_key, video_bytes, content_type="video/mp4")
    s3.put_json(report.frames_s3_key, frames_payload)
    s3.put_json(report.report_s3_key, report.to_dict())
    s3.put_json("report/latest.json", report.to_dict())
    return report
