"""Pure functional text mining, topic clustering, 2D KDE density grid, and monthly aggregation."""

from __future__ import annotations

from collections import defaultdict
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
        "robotics",
        "learning",
        "humanoid",
        "humanoids",
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
        "enable",
        "enables",
        "general",
        "present",
        "introduce",
        "control",
    }
)

CLUSTER_ARCHETYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "全身モーショントラッキング\n拡散軌道生成・模倣",
        ("tracking", "motion", "imitation", "diffusion", "retargeting", "mocap", "mimic"),
    ),
    (
        "オフポリシーRL・探索\nサンプル効率・潜在表現",
        ("off-policy", "rlpd", "awac", "sample", "latent", "exploration", "prior", "q-learning"),
    ),
    (
        "LLM・VLM統合\n自動報酬設計・行動計画",
        ("llm", "vlm", "vision-language", "language", "reward", "foundation", "semantic"),
    ),
    (
        "MPC融合・安全制約\n並列微分可能シム",
        ("mpc", "predictive", "safety", "constraint", "differentiable", "parallel", "optimal"),
    ),
    (
        "視覚知覚・不整地歩行\nヒューマノイドパルクール",
        ("parkour", "terrain", "visual", "perception", "elevation", "stairs", "agile", "vision"),
    ),
    (
        "2足歩行・歩容制御\n(周期歩容・動的歩行)",
        ("bipedal", "biped", "gait", "walking", "stepping", "locomotion", "periodic", "cassie"),
    ),
    (
        "転倒復帰・立ち上がり\n動的バランス制御",
        ("recovery", "fall", "get-up", "standing", "balance", "push", "disturbance", "stability"),
    ),
    (
        "全身ロコマニピュレーション\n接触・力適応制御",
        ("loco-manipulation", "manipulation", "contact", "force", "grasping", "payload", "compliance"),
    ),
    (
        "遠隔動作模倣\nテレオペレーション",
        ("teleoperation", "teleop", "vr", "exoskeleton", "bimanual", "demonstration", "avatar"),
    ),
    (
        "Sim-to-Real転移\nドメインランダマイゼーション",
        ("sim-to-real", "sim2real", "randomization", "transfer", "deployment", "zero-shot", "hardware"),
    ),
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
    """Pure predicate checking whether a paper discusses Sim-to-Real or hardware deployment."""
    lowered = text.lower()
    markers = (
        "sim-to-real",
        "sim2real",
        "simulation-to-real",
        "domain randomization",
        "real-world deployment",
        "real humanoid",
        "zero-shot transfer",
        "physical robot",
        "hardware experiment",
        "on a real",
    )
    return any(m in lowered for m in markers)


def is_humanoid_or_physical_ai_paper(title: str, abstract: str) -> bool:
    """Pure relevance filter ensuring only robotics / humanoid / locomotion papers pass."""
    text = f"{title} {abstract}".lower()
    must_have_any = (
        "humanoid",
        "biped",
        "bipedal",
        "legged",
        "locomotion",
        "whole-body",
        "loco-manipulation",
        "robot",
        "mujoco",
        "isaac",
        "teleoperation",
        "imitation learning",
        "motion tracking",
    )
    exclude_domains = (
        "autonomous gis",
        "geographic information",
        "dce-ivim",
        "mri",
        "cancer",
        "clinical trial",
        "astronomy",
    )
    if any(ex in text for ex in exclude_domains):
        return False
    return any(k in text for k in must_have_any)


def _build_corpus(papers: Sequence[PaperRecord]) -> tuple[str, ...]:
    return tuple(f"{p.title}. {p.title}. {p.abstract}" for p in papers)


def _compute_2d_coords(
    tfidf_matrix: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Project TF-IDF vectors to zero-centered, well-spread 2D coordinates in [-0.95, 0.95]."""
    n_samples, n_features = tfidf_matrix.shape
    if n_samples == 0:
        return np.zeros((0, 2), dtype=float)
    if n_samples == 1:
        return np.array([[0.0, 0.0]], dtype=float)
    if n_samples == 2 or n_features < 3:
        angles = np.linspace(0.0, np.pi, n_samples)
        return np.stack([np.cos(angles) * 0.6, np.sin(angles) * 0.6], axis=1)

    n_components = min(3, n_samples - 1, n_features)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    raw = svd.fit_transform(tfidf_matrix)

    # Use components (1, 2) if available since component 0 in TF-IDF mainly captures document length,
    # or zero-center components (0, 1)
    if raw.shape[1] >= 3:
        coords = np.column_stack([raw[:, 0] - np.mean(raw[:, 0]), raw[:, 1] - np.mean(raw[:, 1])])
        coords[:, 0] += 0.35 * (raw[:, 2] - np.mean(raw[:, 2]))
    elif raw.shape[1] == 2:
        coords = raw - np.mean(raw, axis=0, keepdims=True)
    else:
        coords = np.hstack([raw - np.mean(raw), np.zeros((n_samples, 1), dtype=float)])

    # Rank-blend with Gaussian spread so points fill the 2D plane organically without overlapping
    rng = np.random.default_rng(seed)
    for dim in range(2):
        col = coords[:, dim]
        std = float(np.std(col))
        if std > 1e-8:
            col = (col - float(np.mean(col))) / std
        ranks = np.argsort(np.argsort(col)).astype(float)
        uniform = (ranks / max(1.0, float(n_samples - 1))) * 1.7 - 0.85
        coords[:, dim] = np.clip(0.55 * np.tanh(col * 0.75) + 0.45 * uniform, -0.98, 0.98)

    coords += rng.normal(0.0, 0.03, size=coords.shape)
    return np.clip(coords, -1.0, 1.0)


def _compute_density_grid(
    coords: np.ndarray,
    grid_size: int = 56,
) -> DensityGrid:
    """Purely compute smooth 2D KDE contour grid over projected coordinates."""
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
        kde = gaussian_kde(pts, bw_method=0.38)
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
        z_matrix=tuple(tuple(float(round(c, 4)) for c in row) for row in z_norm),
    )


def _assign_unique_cluster_labels(
    cluster_terms: dict[int, Sequence[str]],
    cluster_docs: dict[int, str],
) -> dict[int, str]:
    """Purely assign distinct, non-duplicate Japanese robotics labels to every cluster."""
    used: set[str] = set()
    assigned: dict[int, str] = {}

    for cid in sorted(cluster_terms.keys()):
        terms = cluster_terms[cid]
        blob = (cluster_docs.get(cid, "") + " " + " ".join(terms)).lower()
        best_label = ""
        best_score = -1
        for label, markers in CLUSTER_ARCHETYPES:
            if label in used:
                continue
            score = sum(3 if m in " ".join(terms).lower() else (1 if m in blob else 0) for m in markers)
            if score > best_score:
                best_score = score
                best_label = label

        if not best_label or best_label in used:
            for label, _ in CLUSTER_ARCHETYPES:
                if label not in used:
                    best_label = label
                    break
        if not best_label:
            head = "・".join(terms[:2]) if terms else f"テーマ{cid + 1}"
            best_label = f"ヒューマノイド制御 ({head})"

        used.add(best_label)
        assigned[cid] = best_label

    return assigned


def build_text_mining_snapshot(
    papers: Sequence[PaperRecord],
    keywords_used: Sequence[str],
    generated_at: str | None = None,
) -> MonthlyTextMiningSnapshot:
    """Pure functional pipeline: PaperRecords -> TF-IDF -> 2D SVD -> KMeans -> KDE -> Snapshot."""
    ts = generated_at or datetime.now(timezone.utc).isoformat()
    monthly_counts = aggregate_monthly_counts(papers)
    period_label = (
        f"{monthly_counts[0][0].replace('-', '.')} - {monthly_counts[-1][0].replace('-', '.')}"
        if monthly_counts
        else datetime.now(timezone.utc).strftime("%Y.%m")
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
        max_features=400,
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf_sparse = vectorizer.fit_transform(corpus)
    tfidf = tfidf_sparse.toarray()
    feature_names = tuple(str(t) for t in vectorizer.get_feature_names_out())

    mean_scores = np.mean(tfidf, axis=0)
    term_scores = sorted(
        (
            (term, round(float(score), 4))
            for term, score in zip(feature_names, mean_scores)
            if term.lower() not in TOPIC_STOPWORDS and len(term) > 2
        ),
        key=lambda x: x[1],
        reverse=True,
    )[:25]

    coords = _compute_2d_coords(tfidf)
    density_grid = _compute_density_grid(coords, grid_size=56)

    n_clusters = max(1, min(8, len(papers) // 4 or 1))
    if len(papers) >= n_clusters and n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
    else:
        labels = np.zeros(len(papers), dtype=int)

    cluster_indices: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(labels):
        cluster_indices[int(cid)].append(idx)

    cluster_terms_map: dict[int, tuple[str, ...]] = {}
    cluster_docs_map: dict[int, str] = {}
    for cid in sorted(cluster_indices.keys()):
        idxs = cluster_indices[cid]
        sub_tfidf = np.mean(tfidf[idxs], axis=0)
        top_idx = np.argsort(sub_tfidf)[::-1]
        c_terms = tuple(
            feature_names[i]
            for i in top_idx
            if feature_names[i].lower() not in TOPIC_STOPWORDS
        )[:5]
        cluster_terms_map[cid] = c_terms
        cluster_docs_map[cid] = " ".join(corpus[i] for i in idxs)

    cluster_label_map = _assign_unique_cluster_labels(cluster_terms_map, cluster_docs_map)

    clusters_list: list[ClusterSummary] = []
    for cid in sorted(cluster_indices.keys()):
        idxs = cluster_indices[cid]
        center = np.mean(coords[idxs], axis=0)
        clusters_list.append(
            ClusterSummary(
                cluster_id=cid,
                label=cluster_label_map[cid],
                center_x=round(float(center[0]), 4),
                center_y=round(float(center[1]), 4),
                count=len(idxs),
                top_terms=cluster_terms_map[cid],
            )
        )

    points = tuple(
        TextMiningPoint(
            paper_id=paper.paper_id,
            title=paper.title,
            url=paper.url,
            month_key=paper.month_key,
            published_at=paper.published_at,
            authors=paper.authors,
            keywords_matched=paper.keywords_matched,
            x=round(float(coords[i, 0]), 4),
            y=round(float(coords[i, 1]), 4),
            cluster_id=int(labels[i]),
            cluster_label=cluster_label_map[int(labels[i])].replace("\n", "・"),
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
