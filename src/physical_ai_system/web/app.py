"""FastAPI Web Application for Physical AI Development System."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from physical_ai_system.adapters.bedrock_client import BedrockAdvisorClient
from physical_ai_system.adapters.mujoco_engine import (
    DEFAULT_HUMANOID_XML,
    list_builtin_mocaps,
)
from physical_ai_system.adapters.s3_store import S3Store
from physical_ai_system.config import AppConfig, load_config
from physical_ai_system.domain.advisor import (
    build_advisor_prompt,
    parse_retrain_instruction,
    recommend_method_pure,
)
from physical_ai_system.domain.models import RLMethod, TrainingConfig
from physical_ai_system.workflows.paper_pipeline import run_monthly_paper_pipeline
from physical_ai_system.workflows.training_pipeline import (
    execute_training_workflow,
    load_or_create_mocap,
    save_uploaded_mocap,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class ScrapeRequest(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    max_per_keyword: int = Field(default=6, ge=1, le=20)


class AdvisorAskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mocap_id: str = Field(default="sample_biped_walk")


class AdvisorRetrainRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    base_method: RLMethod = Field(default="DAPG")
    mocap_id: str = Field(default="sample_biped_walk")
    epochs: int = Field(default=30, ge=5, le=120)
    learning_rate: float = Field(default=3e-4, gt=1e-6, lt=0.1)
    bc_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    entropy_coef: float = Field(default=0.01, ge=0.0, le=0.5)
    smoothness_weight: float = Field(default=0.06, ge=0.0, le=1.0)
    reward_tracking_weight: float = Field(default=1.2, ge=0.1, le=5.0)
    auto_start: bool = Field(default=False)


class TrainStartRequest(BaseModel):
    method: RLMethod = Field(default="DAPG")
    mocap_id: str = Field(default="sample_biped_walk")
    epochs: int = Field(default=30, ge=5, le=120)
    learning_rate: float = Field(default=3e-4, gt=1e-6, lt=0.1)
    bc_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    entropy_coef: float = Field(default=0.01, ge=0.0, le=0.5)
    smoothness_weight: float = Field(default=0.06, ge=0.0, le=1.0)
    reward_tracking_weight: float = Field(default=1.2, ge=0.1, le=5.0)
    robot_xml_name: str = Field(default="humanoid_default.xml")
    instruction_note: str = Field(default="")


def create_app(custom_config: AppConfig | None = None) -> FastAPI:
    cfg = custom_config or load_config()
    s3 = S3Store(cfg.s3_bucket, cfg.aws_region, cfg.local_data_dir)
    bedrock = BedrockAdvisorClient(
        cfg.aws_region,
        cfg.bedrock_model_id,
        cfg.bedrock_summary_model_id,
    )

    app = FastAPI(
        title="Physical AI Development System",
        version="0.2.0",
    )
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    jobs_lock = threading.Lock()
    jobs_state: dict[str, dict[str, Any]] = {}

    def _set_job(job_id: str, patch: dict[str, Any]) -> None:
        with jobs_lock:
            prev = jobs_state.get(job_id, {})
            jobs_state[job_id] = {**prev, **patch}

    def _list_mocaps() -> list[dict[str, Any]]:
        items: dict[str, dict[str, Any]] = {}
        for preset in list_builtin_mocaps():
            items[preset.mocap_id] = {
                "mocap_id": preset.mocap_id,
                "name": preset.name,
                "source_format": "builtin",
                "fps": preset.fps,
                "frames": preset.num_frames,
                "joints": preset.num_joints,
            }
        for k in s3.list_keys("mocap/"):
            if k.endswith("/meta.json"):
                meta = s3.get_json(k)
                if meta and "mocap_id" in meta:
                    items[meta["mocap_id"]] = meta
        return list(items.values())

    def _list_reports() -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for k in s3.list_keys("report/"):
            if k.endswith("/report.json"):
                data = s3.get_json(k)
                if data:
                    hist = data.get("metrics_history") or []
                    last_m = hist[-1] if hist else {}
                    diag = data.get("diagnostics") or {}
                    reports.append(
                        {
                            "run_id": data.get("run_id"),
                            "created_at": data.get("created_at"),
                            "method": data.get("config", {}).get("method"),
                            "mocap_id": data.get("config", {}).get("mocap_id"),
                            "epochs": data.get("config", {}).get("epochs", len(hist)),
                            "success_rate": last_m.get("task_success_rate", 0.0),
                            "train_loss": last_m.get("train_loss", 0.0),
                            "val_loss": last_m.get("val_loss", 0.0),
                            "mean_return": last_m.get("mean_return", 0.0),
                            "tracking_score": diag.get("tracking_score", 0.0),
                            "smoothness_score": diag.get("smoothness_score", 0.0),
                            "overfitting_detected": diag.get("overfitting_detected", False),
                            "instruction_note": data.get("config", {}).get("instruction_note", ""),
                            "report_s3_key": k,
                        }
                    )
        reports.sort(key=lambda r: str(r.get("created_at", "")), reverse=True)
        return reports

    def _launch_training_job(tcfg: TrainingConfig) -> str:
        job_id = tcfg.run_id
        _set_job(
            job_id,
            {
                "job_id": job_id,
                "type": "training",
                "status": "running",
                "progress": 0,
                "total": tcfg.epochs,
                "message": f"{tcfg.method} による模倣学習・強化学習を実行中...",
                "run_id": tcfg.run_id,
            },
        )

        def _worker() -> None:
            try:
                xml_text = DEFAULT_HUMANOID_XML
                if tcfg.robot_xml_name != "humanoid_default.xml":
                    custom_bytes = s3.get_bytes(f"robots/{tcfg.robot_xml_name}")
                    if custom_bytes:
                        xml_text = custom_bytes.decode("utf-8", errors="replace")

                def _on_progress(ep: int, total: int, metric: Any) -> None:
                    _set_job(
                        job_id,
                        {
                            "progress": ep,
                            "total": total,
                            "message": (
                                f"Epoch {ep}/{total} | TrainLoss={metric.train_loss:.4f} "
                                f"| ValLoss={metric.val_loss:.4f} | 成功率={metric.task_success_rate*100:.1f}%"
                            ),
                        },
                    )

                report = execute_training_workflow(
                    s3=s3,
                    config=tcfg,
                    local_runs_dir=cfg.local_data_dir / "runs",
                    xml_text=xml_text,
                    progress_cb=_on_progress,
                )
                _set_job(
                    job_id,
                    {
                        "status": "completed",
                        "progress": tcfg.epochs,
                        "total": tcfg.epochs,
                        "message": "学習とMuJoCo評価・S3レポート保存が完了しました。",
                        "result": report.to_dict(),
                    },
                )
            except Exception as exc:
                logger.exception("Training job %s failed", job_id)
                _set_job(
                    job_id,
                    {"status": "failed", "message": f"エラー: {exc}"},
                )

        threading.Thread(target=_worker, daemon=True).start()
        return job_id

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "s3_bucket": cfg.s3_bucket,
                "aws_region": cfg.aws_region,
                "default_keywords": ", ".join(cfg.default_keywords),
            },
        )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "bucket": cfg.s3_bucket, "region": cfg.aws_region}

    @app.get("/api/dashboard-init")
    async def dashboard_init() -> dict[str, Any]:
        snapshot = s3.get_json("text_mining/latest.json")
        papers = s3.get_json("papers/catalog.json") or []
        latest_report = s3.get_json("report/latest.json")

        # If snapshot missing or lacking 2020-2026 yearly_counts, rebuild from verified 2020-2026 corpus
        if not snapshot or not snapshot.get("yearly_counts") or len(papers) < 30:
            from physical_ai_system.adapters.alphaxiv_scraper import VERIFIED_HUMANOID_PAPERS
            from physical_ai_system.domain.text_mining import build_text_mining_snapshot
            from physical_ai_system.domain.models import PaperRecord

            verified_records = tuple(
                PaperRecord(
                    paper_id=str(p["paper_id"]),
                    title=str(p["title"]),
                    abstract=str(p["abstract"]),
                    authors=tuple(p["authors"]),
                    published_at=str(p["published_at"]),
                    month_key=str(p["month_key"]),
                    url=f"https://www.alphaxiv.org/abs/{p['paper_id']}",
                    keywords_matched=tuple(p.get("keywords_matched", ())),
                    sim_to_real=bool(p.get("sim_to_real", False)),
                    summary_ja=(
                        f"【要約ステータス】AI要約未生成（原文抄録を表示中）\n"
                        f"【研究目的・手法】{p['title']}\n"
                        f"【Sim-to-Real分類】{'実機転移（Sim-to-Real）関連の記述が原文に含まれます。' if p.get('sim_to_real') else 'シミュレーション環境での検証を主体とする研究です。'}\n"
                        f"【原文抄録】{p['abstract']}"
                    ),
                )
                for p in VERIFIED_HUMANOID_PAPERS
            )
            snap_obj = build_text_mining_snapshot(verified_records, cfg.default_keywords)
            snapshot = snap_obj.to_dict()
            papers = [p.to_dict() for p in verified_records]
            try:
                s3.put_json("text_mining/latest.json", snapshot)
                s3.put_json("papers/catalog.json", papers)
            except Exception as e:
                logger.warning("Could not persist initial 2020-2026 text mining snapshot to S3: %s", e)

        return {
            "s3_bucket": cfg.s3_bucket,
            "aws_region": cfg.aws_region,
            "default_keywords": list(cfg.default_keywords),
            "text_mining": snapshot,
            "papers": papers[:250],
            "mocaps": _list_mocaps(),
            "reports": _list_reports(),
            "latest_report": latest_report,
        }

    @app.post("/api/papers/scrape")
    async def trigger_paper_scrape(req: ScrapeRequest) -> dict[str, Any]:
        job_id = f"scrape-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        kws = tuple(k.strip() for k in req.keywords if k.strip()) or cfg.default_keywords
        _set_job(
            job_id,
            {
                "job_id": job_id,
                "type": "scrape",
                "status": "running",
                "progress": 10,
                "total": 100,
                "message": "alphaXivからレートリミット遵守（2秒間隔）で論文を取得・テキストマイニング中...",
            },
        )

        def _worker() -> None:
            try:
                snap = run_monthly_paper_pipeline(
                    config=cfg,
                    s3=s3,
                    bedrock=bedrock,
                    keywords=kws,
                    max_per_keyword=req.max_per_keyword,
                )
                _set_job(
                    job_id,
                    {
                        "status": "completed",
                        "progress": 100,
                        "total": 100,
                        "message": f"完了: 全{snap.total_papers}件（Sim-to-Real {snap.sim_to_real_count}件）を分析・要約しS3へ保存しました。",
                        "result": snap.to_dict(),
                    },
                )
            except Exception as exc:
                logger.exception("Scrape job %s failed", job_id)
                _set_job(job_id, {"status": "failed", "message": f"エラー: {exc}"})

        threading.Thread(target=_worker, daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/papers/latest")
    async def get_latest_papers() -> dict[str, Any]:
        return {
            "text_mining": s3.get_json("text_mining/latest.json"),
            "papers": s3.get_json("papers/catalog.json") or [],
        }

    @app.post("/api/mocap/upload")
    async def upload_mocap(file: UploadFile = File(...)) -> dict[str, Any]:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")
        mocap_id = f"mocap_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        traj = save_uploaded_mocap(
            s3=s3,
            mocap_id=mocap_id,
            filename=file.filename or "motion.csv",
            raw_bytes=raw,
        )
        return {
            "mocap_id": traj.mocap_id,
            "name": traj.name,
            "source_format": traj.source_format,
            "fps": traj.fps,
            "frames": traj.num_frames,
            "joints": traj.num_joints,
            "s3_prefix": f"mocap/{mocap_id}/",
        }

    @app.post("/api/robot-xml/upload")
    async def upload_robot_xml(file: UploadFile = File(...)) -> dict[str, Any]:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty XML")
        name = Path(file.filename or "custom_humanoid.xml").name
        s3.put_bytes(f"robots/{name}", raw, content_type="application/xml")
        return {"robot_xml_name": name, "s3_key": f"robots/{name}"}

    @app.post("/api/advisor/ask")
    async def advisor_ask(req: AdvisorAskRequest) -> dict[str, Any]:
        mocap = load_or_create_mocap(s3, req.mocap_id)
        latest_raw = s3.get_json("report/latest.json")
        latest_report = None
        if latest_raw:
            from physical_ai_system.domain.models import (
                EpochMetrics,
                EvaluationDiagnostics,
                TrainingReport,
            )

            latest_report = TrainingReport(
                run_id=latest_raw["run_id"],
                created_at=latest_raw["created_at"],
                config=TrainingConfig(**latest_raw["config"]),
                metrics_history=tuple(
                    EpochMetrics(**m) for m in latest_raw["metrics_history"]
                ),
                diagnostics=EvaluationDiagnostics(
                    **{
                        **latest_raw["diagnostics"],
                        "recommended_adjustments": tuple(
                            latest_raw["diagnostics"]["recommended_adjustments"]
                        ),
                    }
                ),
                video_s3_key=latest_raw["video_s3_key"],
                frames_s3_key=latest_raw["frames_s3_key"],
                report_s3_key=latest_raw["report_s3_key"],
                video_local_url=latest_raw["video_local_url"],
                frames_local_url=latest_raw["frames_local_url"],
            )

        rec_method, rec_cfg, fallback_reply = recommend_method_pure(
            req.question, mocap.num_frames
        )
        prompt = build_advisor_prompt(req.question, latest_report, tuple(_list_mocaps()))
        reply, engine = bedrock.ask_advisor(prompt, fallback_reply)
        return {
            "reply": reply,
            "engine": engine,
            "recommended_method": rec_method,
            "recommended_config": asdict(rec_cfg),
        }

    @app.post("/api/advisor/retrain")
    async def advisor_retrain(req: AdvisorRetrainRequest) -> dict[str, Any]:
        base_cfg = TrainingConfig(
            run_id=f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}",
            method=req.base_method,
            mocap_id=req.mocap_id,
            epochs=req.epochs,
            learning_rate=req.learning_rate,
            bc_weight=req.bc_weight,
            entropy_coef=req.entropy_coef,
            smoothness_weight=req.smoothness_weight,
            reward_tracking_weight=req.reward_tracking_weight,
        )
        updated_cfg, fallback_explanation = parse_retrain_instruction(
            req.instruction, base_cfg
        )
        prompt = build_advisor_prompt(
            f"再指示: {req.instruction}\n調整案:\n{fallback_explanation}",
            None,
            tuple(_list_mocaps()),
        )
        reply, engine = bedrock.ask_advisor(prompt, fallback_explanation)
        job_id = _launch_training_job(updated_cfg) if req.auto_start else None
        return {
            "reply": reply,
            "engine": engine,
            "previous_config": base_cfg.to_dict(),
            "updated_config": updated_cfg.to_dict(),
            "job_id": job_id,
        }

    @app.post("/api/training/start")
    async def start_training(req: TrainStartRequest) -> dict[str, Any]:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        tcfg = TrainingConfig(
            run_id=run_id,
            method=req.method,
            mocap_id=req.mocap_id,
            epochs=req.epochs,
            learning_rate=req.learning_rate,
            bc_weight=req.bc_weight,
            entropy_coef=req.entropy_coef,
            smoothness_weight=req.smoothness_weight,
            reward_tracking_weight=req.reward_tracking_weight,
            robot_xml_name=req.robot_xml_name,
            instruction_note=req.instruction_note,
        )
        job_id = _launch_training_job(tcfg)
        return {"job_id": job_id, "run_id": run_id}

    @app.get("/api/jobs/{job_id}")
    async def get_job_status(job_id: str) -> dict[str, Any]:
        with jobs_lock:
            job = jobs_state.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/api/runs/{run_id}/report")
    async def get_run_report(run_id: str) -> dict[str, Any]:
        data = s3.get_json(f"report/{run_id}/report.json")
        if not data:
            raise HTTPException(status_code=404, detail="Report not found")
        return data

    @app.get("/api/runs/{run_id}/video")
    async def get_run_video(run_id: str) -> FileResponse:
        local_path = s3.ensure_local_file(f"report/{run_id}/evaluation.mp4")
        if not local_path:
            raise HTTPException(status_code=404, detail="Video not found")
        return FileResponse(
            path=str(local_path),
            media_type="video/mp4",
            filename=f"{run_id}_evaluation.mp4",
        )

    @app.get("/api/runs/{run_id}/frames")
    async def get_run_frames(run_id: str) -> JSONResponse:
        data = s3.get_json(f"report/{run_id}/frames_3d.json")
        if not data:
            raise HTTPException(status_code=404, detail="3D frames not found")
        return JSONResponse(content=data)

    return app


app = create_app()
