"""Immutable domain data models (Functional Programming core)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

RLMethod = Literal["DAPG", "RLPD", "AWAC"]


@dataclass(frozen=True)
class PaperRecord:
    """Single scraped academic paper with alphaxiv/arXiv metadata and summary."""

    paper_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    published_at: str
    month_key: str
    url: str
    keywords_matched: tuple[str, ...]
    sim_to_real: bool
    summary_ja: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["authors"] = list(self.authors)
        data["keywords_matched"] = list(self.keywords_matched)
        return data

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PaperRecord":
        return PaperRecord(
            paper_id=str(data["paper_id"]),
            title=str(data["title"]),
            abstract=str(data.get("abstract", "")),
            authors=tuple(data.get("authors", ())),
            published_at=str(data.get("published_at", "")),
            month_key=str(data.get("month_key", "")),
            url=str(data["url"]),
            keywords_matched=tuple(data.get("keywords_matched", ())),
            sim_to_real=bool(data.get("sim_to_real", False)),
            summary_ja=str(data.get("summary_ja", "")),
        )


@dataclass(frozen=True)
class TextMiningPoint:
    """2D projected paper point on the topic density map."""

    paper_id: str
    title: str
    url: str
    month_key: str
    x: float
    y: float
    cluster_id: int
    cluster_label: str
    sim_to_real: bool
    summary_ja: str


@dataclass(frozen=True)
class ClusterSummary:
    """Cluster centroid and representative terms for map annotations."""

    cluster_id: int
    label: str
    center_x: float
    center_y: float
    count: int
    top_terms: tuple[str, ...]


@dataclass(frozen=True)
class DensityGrid:
    """2D KDE density grid for contour heatmap rendering."""

    x_grid: tuple[float, ...]
    y_grid: tuple[float, ...]
    z_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class MonthlyTextMiningSnapshot:
    """Aggregated monthly text mining snapshot saved to S3."""

    generated_at: str
    period_label: str
    total_papers: int
    sim_to_real_count: int
    keywords_used: tuple[str, ...]
    monthly_counts: tuple[tuple[str, int], ...]
    top_terms: tuple[tuple[str, float], ...]
    clusters: tuple[ClusterSummary, ...]
    points: tuple[TextMiningPoint, ...]
    density_grid: DensityGrid

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "period_label": self.period_label,
            "total_papers": self.total_papers,
            "sim_to_real_count": self.sim_to_real_count,
            "keywords_used": list(self.keywords_used),
            "monthly_counts": [
                {"month": m, "count": c} for m, c in self.monthly_counts
            ],
            "top_terms": [{"term": t, "score": s} for t, s in self.top_terms],
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "label": c.label,
                    "center_x": c.center_x,
                    "center_y": c.center_y,
                    "count": c.count,
                    "top_terms": list(c.top_terms),
                }
                for c in self.clusters
            ],
            "points": [asdict(p) for p in self.points],
            "density_grid": {
                "x_grid": list(self.density_grid.x_grid),
                "y_grid": list(self.density_grid.y_grid),
                "z_matrix": [list(row) for row in self.density_grid.z_matrix],
            },
        }


@dataclass(frozen=True)
class MocapTrajectory:
    """Immutable motion-capture reference trajectory (T x D joint targets)."""

    mocap_id: str
    name: str
    source_format: str
    fps: float
    joint_names: tuple[str, ...]
    joint_positions: tuple[tuple[float, ...], ...]  # shape: (T, num_joints)
    root_positions: tuple[tuple[float, float, float], ...]  # shape: (T, 3)

    @property
    def num_frames(self) -> int:
        return len(self.joint_positions)

    @property
    def num_joints(self) -> int:
        return len(self.joint_positions[0]) if self.joint_positions else 0


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable hyperparameters for Imitation + Reinforcement Learning."""

    run_id: str
    method: RLMethod
    mocap_id: str
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 3e-4
    bc_weight: float = 1.0
    entropy_coef: float = 0.01
    smoothness_weight: float = 0.05
    reward_tracking_weight: float = 1.0
    seed: int = 42
    robot_xml_name: str = "humanoid_default.xml"
    instruction_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpochMetrics:
    """Single epoch metrics for Train Loss, Validation Loss, and Task Success Rate."""

    epoch: int
    train_loss: float
    val_loss: float
    task_success_rate: float
    mean_return: float
    tracking_error: float
    jerk_smoothness: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationDiagnostics:
    """Automated evaluation diagnostics answering tracking, smoothness, overfitting."""

    tracking_score: float
    smoothness_score: float
    overfitting_gap: float
    overfitting_detected: bool
    is_smooth: bool
    follows_human_motion: bool
    summary_ja: str
    recommended_adjustments: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["recommended_adjustments"] = list(self.recommended_adjustments)
        return data


@dataclass(frozen=True)
class TrainingReport:
    """Full immutable training & MuJoCo evaluation report saved to S3 report/."""

    run_id: str
    created_at: str
    config: TrainingConfig
    metrics_history: tuple[EpochMetrics, ...]
    diagnostics: EvaluationDiagnostics
    video_s3_key: str
    frames_s3_key: str
    report_s3_key: str
    video_local_url: str
    frames_local_url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "config": self.config.to_dict(),
            "metrics_history": [m.to_dict() for m in self.metrics_history],
            "diagnostics": self.diagnostics.to_dict(),
            "video_s3_key": self.video_s3_key,
            "frames_s3_key": self.frames_s3_key,
            "report_s3_key": self.report_s3_key,
            "video_local_url": self.video_local_url,
            "frames_local_url": self.frames_local_url,
        }
