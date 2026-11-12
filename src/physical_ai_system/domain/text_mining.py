"""Pure functional text mining, topic clustering, 2D KDE density grid, and monthly aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from physical_ai_system.domain.models import (
    ClusterSummary,
    DensityGrid,
    MonthlyTextMiningSnapshot,
    PaperRecord,
    TextMiningPoint,
)

TOPIC_STOPWORDS: frozenset[str] = frozenset(
    {
        "using",
        "based",
        "approach",
        "method",
        "methods",
        "paper",
        "propose",
        "proposed",
        "results",
        "show",
        "demonstrate",
        "robot",
        "robots",
        "learning",
        "humanoid",
        "physical",
        "framework",
        "model",
        "models",
        "task",
        "tasks",
        "data",
        "via",
        "towards",
        "novel",
        "new",
        "real",
        "world",
        "performance",
        "system",
        "systems",
    }
)


def aggregate_monthly_counts(
    papers: Sequence[PaperRecord],
) -> tuple[tuple[str, int], ...]:
    """Purely aggregate paper counts per YYYY-MM month key in chronological order."""
    counts: dict[str, int] = defaultdict(int)
    for p in papers:
        key = p.month_key or "unknown"
        counts[key] += 1
    return tuple(sorted(counts.items(), key=lambda item: item[0]))


def detect_sim_to_real(text: str) -> bool:
    """Pure predicate checking whether a paper discusses Sim-to-Real transfer."""
    lowered = text.lower()
    markers = ("sim-to-real", "sim2real", "simulation-to-real", "domain randomization")
    return any(m in lowered for m in markers)


def _build_corpus(papers: Sequence[PaperRecord]) -> tuple[str, ...]:
    return tuple(f"{p.title}. {p.abstract}" for p in papers)


def _compute_2d_coords(
    tfidf_matrix: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Project TF-IDF vectors to 2D coordinates deterministically."""
    n_samples, n_features = tfidf_matrix.shape
    if n_samples == 0:
        return np.zeros((0, 2), dtype=float)
    if n_samples == 1:
        return np.array([[0.0, 0.0]], dtype=float)
    if n_samples == 2 or n_features < 2:
        angles = np.linspace(0.0, np.pi, n_samples)
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)

    n_components = min(2, n_samples - 1, n_features)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    coords = svd.fit_transform(tfidf_matrix)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((n_samples, 1), dtype=float)])

    # Normalize to [-1, 1] range for stable plotting
    max_abs = np.max(np.abs(coords), axis=0)
    max_abs = np.where(max_abs < 1e-8, 1.0, max_abs)
    return coords / max_abs


def _compute_density_grid(
    coords: np.ndarray,
    grid_size: int = 48,
) -> DensityGrid:
    """Purely compute 2D KDE contour grid over projected coordinates."""
    x_lin = np.linspace(-1.25, 1.25, grid_size)
    y_lin = np.linspace(-1.25, 1.25, grid_size)
    if len(coords) < 3:
        xx, yy = np.meshgrid(x_lin, y_lin)
        z = np.exp(-(xx**2 + yy**2) * 2.0)
        return DensityGrid(
            x_grid=tuple(float(v) for v in x_lin),
            y_grid=tuple(float(v) for v in y_lin),
            z_matrix=tuple(tuple(float(c) for c in row) for row in z),
        )

    jitter = np.random.default_rng(42).normal(0.0, 1e-3, size=coords.shape)
    pts = (coords + jitter).T
    try:
        kde = gaussian_kde(pts, bw_method=0.42)
        xx, yy = np.meshgrid(x_lin, y_lin)
        grid_coords = np.vstack([xx.ravel(), yy.ravel()])
        z = kde(grid_coords).reshape(grid_size, grid_size)
    except Exception:
        xx, yy = np.meshgrid(x_lin, y_lin)
        z = np.exp(-(xx**2 + yy**2) * 1.5)

    z_max = float(np.max(z)) or 1.0
    z_norm = z / z_max
    return DensityGrid(
        x_grid=tuple(float(v) for v in x_lin),
        y_grid=tuple(float(v) for v in y_lin),
        z_matrix=tuple(tuple(float(c) for c in row) for row in z_norm),
    )


def _label_cluster(terms: Sequence[str], cluster_id: int) -> str:
    """Purely derive human-readable cluster label from top TF-IDF terms."""
    joined = " ".join(terms).lower()
    if any(k in joined for k in ("motion", "tracking", "imitation", "teleoperation", "diffusion")):
        return "全身モーショントラッキング・模倣"
    if any(k in joined for k in ("bipedal", "gait", "locomotion", "walking", "parkour")):
        return "2足歩行・不整地歩容制御"
    if any(k in joined for k in ("sim", "domain", "transfer", "policy", "reinforcement")):
        return "強化学習・Sim-to-Real転移"
    if any(k in joined for k in ("vlm", "llm", "language", "vision", "reward")):
        return "LLM/VLM統合・視覚制御"
    if any(k in joined for k in ("manipulation", "hand", "grasp", "contact", "force")):
        return "全身ロコマニピュレーション"
    if any(k in joined for k in ("mpc", "balance", "fall", "recovery", "control")):
        return "動的バランス・転倒復帰制御"
    head = " / ".join(terms[:3]) if terms else f"Cluster {cluster_id + 1}"
    return f"研究群{cluster_id + 1}: {head}"


def build_text_mining_snapshot(
    papers: Sequence[PaperRecord],
    keywords_used: Sequence[str],
    generated_at: str | None = None,
) -> MonthlyTextMiningSnapshot:
    """Pure functional pipeline: PaperRecords -> TF-IDF -> 2D SVD -> KMeans -> KDE -> Snapshot."""
    ts = generated_at or datetime.now(timezone.utc).isoformat()
    monthly_counts = aggregate_monthly_counts(papers)
    period_label = (
        f"{monthly_counts[0][0]} ~ {monthly_counts[-1][0]}"
        if monthly_counts
        else datetime.now(timezone.utc).strftime("%Y-%m")
    )
    sim_count = sum(1 for p in papers if p.sim_to_real)

    if not papers:
        empty_grid = _compute_density_grid(np.zeros((0, 2), dtype=float))
        return MonthlyTextMiningSnapshot(
            generated_at=ts,
            period_label=period_label,
            total_papers=0,
            sim_to_real_count=0,
            keywords_used=tuple(keywords_used),
            monthly_counts=(),
            top_terms=(),
            clusters=(),
            points=(),
            density_grid=empty_grid,
        )

    corpus = _build_corpus(papers)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=300,
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf_sparse = vectorizer.fit_transform(corpus)
    tfidf = tfidf_sparse.toarray()
    feature_names = tuple(str(t) for t in vectorizer.get_feature_names_out())

    # Global top terms (filtering overly generic words)
    mean_scores = np.mean(tfidf, axis=0)
    term_scores = sorted(
        (
            (term, float(score))
            for term, score in zip(feature_names, mean_scores)
            if term.lower() not in TOPIC_STOPWORDS and len(term) > 2
        ),
        key=lambda x: x[1],
        reverse=True,
    )[:25]

    coords = _compute_2d_coords(tfidf)
    density_grid = _compute_density_grid(coords)

    n_clusters = max(1, min(6, len(papers) // 3 or 1))
    if len(papers) >= n_clusters and n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
    else:
        labels = np.zeros(len(papers), dtype=int)

    cluster_indices: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(labels):
        cluster_indices[int(cid)].append(idx)

    clusters_list: list[ClusterSummary] = []
    cluster_label_map: dict[int, str] = {}
    for cid in sorted(cluster_indices.keys()):
        idxs = cluster_indices[cid]
        sub_tfidf = np.mean(tfidf[idxs], axis=0)
        top_idx = np.argsort(sub_tfidf)[::-1]
        c_terms = tuple(
            feature_names[i]
            for i in top_idx
            if feature_names[i].lower() not in TOPIC_STOPWORDS
        )[:5]
        c_label = _label_cluster(c_terms, cid)
        cluster_label_map[cid] = c_label
        center = np.mean(coords[idxs], axis=0)
        clusters_list.append(
            ClusterSummary(
                cluster_id=cid,
                label=c_label,
                center_x=float(center[0]),
                center_y=float(center[1]),
                count=len(idxs),
                top_terms=c_terms,
            )
        )

    points = tuple(
        TextMiningPoint(
            paper_id=paper.paper_id,
            title=paper.title,
            url=paper.url,
            month_key=paper.month_key,
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            cluster_id=int(labels[i]),
            cluster_label=cluster_label_map[int(labels[i])],
            sim_to_real=paper.sim_to_real,
            summary_ja=paper.summary_ja,
        )
        for i, paper in enumerate(papers)
    )

    return MonthlyTextMiningSnapshot(
        generated_at=ts,
        period_label=period_label,
        total_papers=len(papers),
        sim_to_real_count=sim_count,
        keywords_used=tuple(keywords_used),
        monthly_counts=monthly_counts,
        top_terms=tuple(term_scores),
        clusters=tuple(clusters_list),
        points=points,
        density_grid=density_grid,
    )
