/* Physical AI Development System Frontend (Functional & Declarative UI) */

const state = {
  papers: [],
  snapshot: null,
  suggestedConfig: null,
  currentReport: null,
  frames3D: [],
  frameIndex: 0,
  playing3D: true,
  threeCtx: null,
};

const qs = (sel) => document.querySelector(sel);
const qsa = (sel) => Array.from(document.querySelectorAll(sel));

async function apiJson(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return res.json();
}

function initTabs() {
  qsa(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      qsa(".tab-btn").forEach((b) => b.classList.remove("active"));
      qsa(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const target = qs(`#tab-${btn.dataset.tab}`);
      if (target) target.classList.add("active");
      if (btn.dataset.tab === "eval" && state.threeCtx) {
        resizeThree();
      }
      window.dispatchEvent(new Event("resize"));
    });
  });
}

function pollJob(jobId, bannerEl, onDone) {
  bannerEl.className = "job-banner";
  bannerEl.textContent = "処理を開始しました...";
  const timer = setInterval(async () => {
    try {
      const job = await apiJson(`/api/jobs/${jobId}`);
      bannerEl.textContent = `${job.message} (${job.progress}/${job.total})`;
      if (job.status === "completed") {
        clearInterval(timer);
        bannerEl.className = "job-banner done";
        onDone(job.result);
      } else if (job.status === "failed") {
        clearInterval(timer);
        bannerEl.className = "job-banner fail";
      }
    } catch (err) {
      clearInterval(timer);
      bannerEl.className = "job-banner fail";
      bannerEl.textContent = `ジョブ監視エラー: ${err.message}`;
    }
  }, 1500);
}

/* ==================== TAB 1: alphaXiv & Text Mining ==================== */

function renderDensityMap(snapshot) {
  const el = qs("#densityMapPlot");
  if (!snapshot || !snapshot.points || snapshot.points.length === 0) {
    el.innerHTML = `<div class="paper-detail-empty">まだ論文データがありません。「論文収集・分析・Bedrock要約を実行」を押してください。</div>`;
    return;
  }

  const dg = snapshot.density_grid;
  const contourTrace = {
    type: "contour",
    x: dg.x_grid,
    y: dg.y_grid,
    z: dg.z_matrix,
    colorscale: "Jet",
    showscale: true,
    contours: { coloring: "heatmap", showlines: false },
    opacity: 0.82,
    hoverinfo: "skip",
    colorbar: { title: "密度", tickfont: { color: "#e6edf7" }, titlefont: { color: "#e6edf7" } },
  };

  const simPts = snapshot.points.filter((p) => p.sim_to_real);
  const genPts = snapshot.points.filter((p) => !p.sim_to_real);

  const makeScatter = (pts, name, color) => ({
    type: "scatter",
    mode: "markers",
    name,
    x: pts.map((p) => p.x),
    y: pts.map((p) => p.y),
    customdata: pts,
    text: pts.map((p) => `${p.title}<br>[${p.month_key}] ${p.cluster_label}<br>(クリックで別タブで論文を開く)`),
    hovertemplate: "%{text}<extra></extra>",
    marker: {
      size: 10,
      color,
      line: { width: 1.5, color: "#0b111b" },
    },
  });

  const annotations = (snapshot.clusters || []).map((c) => ({
    x: c.center_x,
    y: c.center_y,
    text: `<b>${c.label}</b> (${c.count}件)`,
    showarrow: false,
    bgcolor: "rgba(255,255,255,0.88)",
    bordercolor: "#1e293b",
    borderwidth: 1,
    borderpad: 4,
    font: { color: "#0f172a", size: 11 },
  }));

  const layout = {
    title: {
      text: `${snapshot.period_label} | 全${snapshot.total_papers}件（うちSim-to-Real ${snapshot.sim_to_real_count}件）`,
      font: { color: "#e6edf7", size: 14 },
    },
    paper_bgcolor: "#0b111b",
    plot_bgcolor: "#0b111b",
    margin: { l: 35, r: 20, t: 44, b: 35 },
    xaxis: { showgrid: false, zeroline: false, showticklabels: false, range: [-1.25, 1.25] },
    yaxis: { showgrid: false, zeroline: false, showticklabels: false, range: [-1.25, 1.25] },
    legend: { orientation: "h", y: -0.05, font: { color: "#e6edf7" } },
    annotations,
  };

  Plotly.newPlot(el, [contourTrace, makeScatter(genPts, "一般ヒューマノイド研究", "#ffffff"), makeScatter(simPts, "Sim-to-Real 研究", "#fde047")], layout, { responsive: true });

  el.on("plotly_click", (evt) => {
    const pt = evt?.points?.[0]?.customdata;
    if (!pt) return;
    showPaperDetail(pt);
    if (pt.url) {
      window.open(pt.url, "_blank", "noopener,noreferrer");
    }
  });
}

function showPaperDetail(paper) {
  const box = qs("#paperDetailBox");
  box.className = "paper-detail";
  box.innerHTML = `
    <h3>${escapeHtml(paper.title)}</h3>
    <div class="meta">
      公開月: ${escapeHtml(paper.month_key || "-")} |
      クラスタ: ${escapeHtml(paper.cluster_label || "-")} |
      Sim-to-Real: ${paper.sim_to_real ? "Yes" : "No"}
    </div>
    <p><a href="${escapeAttr(paper.url)}" target="_blank" rel="noopener noreferrer" class="btn btn-primary btn-sm">alphaXiv で論文原文を開く ↗</a></p>
    <div class="summary-box">${escapeHtml(paper.summary_ja || "要約データなし")}</div>
  `;
}

function renderMonthlyAndTerms(snapshot) {
  const mEl = qs("#monthlyTrendPlot");
  const tEl = qs("#topTermsPlot");
  if (!snapshot) return;

  const months = (snapshot.monthly_counts || []).map((d) => d.month);
  const counts = (snapshot.monthly_counts || []).map((d) => d.count);
  Plotly.newPlot(
    mEl,
    [{ type: "bar", x: months, y: counts, marker: { color: "#3b82f6" } }],
    {
      paper_bgcolor: "#0b111b",
      plot_bgcolor: "#0b111b",
      font: { color: "#e6edf7" },
      margin: { l: 40, r: 16, t: 16, b: 40 },
      xaxis: { title: "月" },
      yaxis: { title: "論文数" },
    },
    { responsive: true }
  );

  const terms = (snapshot.top_terms || []).slice(0, 12).reverse();
  Plotly.newPlot(
    tEl,
    [{ type: "bar", orientation: "h", x: terms.map((t) => t.score), y: terms.map((t) => t.term), marker: { color: "#f97316" } }],
    {
      paper_bgcolor: "#0b111b",
      plot_bgcolor: "#0b111b",
      font: { color: "#e6edf7" },
      margin: { l: 130, r: 16, t: 16, b: 36 },
    },
    { responsive: true }
  );
}

function renderPapersTable(papers, filterText = "") {
  const tbody = qs("#papersTableBody");
  const q = filterText.trim().toLowerCase();
  const filtered = papers.filter((p) => {
    if (!q) return true;
    return `${p.title} ${p.summary_ja} ${p.month_key}`.toLowerCase().includes(q);
  });
  tbody.innerHTML = filtered
    .map(
      (p) => `
      <tr>
        <td>${escapeHtml(p.month_key)}</td>
        <td><a href="${escapeAttr(p.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(p.title)} ↗</a></td>
        <td>${p.sim_to_real ? '<span class="pill pill-ok">Yes</span>' : "No"}</td>
        <td>${escapeHtml(p.summary_ja || "-")}</td>
      </tr>`
    )
    .join("");
}

function applyPaperSnapshot(snapshot, papers) {
  state.snapshot = snapshot;
  state.papers = papers || [];
  if (snapshot) {
    qs("#kpiTotalPapers").textContent = snapshot.total_papers;
    qs("#kpiSim2Real").textContent = snapshot.sim_to_real_count;
    qs("#kpiPeriod").textContent = snapshot.period_label;
  }
  renderDensityMap(snapshot);
  renderMonthlyAndTerms(snapshot);
  renderPapersTable(state.papers, qs("#paperSearchInput").value);
}

/* ==================== TAB 2: AI Advisor & Training ==================== */

function appendChat(role, text) {
  const log = qs("#advisorChatLog");
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function populateMocaps(mocaps) {
  const sel = qs("#mocapSelect");
  sel.innerHTML = (mocaps || [])
    .map((m) => `<option value="${escapeAttr(m.mocap_id)}">${escapeHtml(m.name)} (${m.frames} frames)</option>`)
    .join("");
}

function applyConfigToForm(cfg) {
  if (!cfg) return;
  if (cfg.method) qs("#methodSelect").value = cfg.method;
  if (cfg.epochs) qs("#epochsInput").value = cfg.epochs;
  if (cfg.learning_rate) qs("#lrInput").value = cfg.learning_rate;
  if (cfg.bc_weight !== undefined) qs("#bcWeightInput").value = cfg.bc_weight;
  if (cfg.smoothness_weight !== undefined) qs("#smoothWeightInput").value = cfg.smoothness_weight;
  if (cfg.reward_tracking_weight !== undefined) qs("#trackWeightInput").value = cfg.reward_tracking_weight;
  if (cfg.entropy_coef !== undefined) qs("#entropyInput").value = cfg.entropy_coef;
}

/* ==================== TAB 3: MuJoCo Evaluation & Reports ==================== */

function populateReports(reports) {
  const sel = qs("#reportSelect");
  sel.innerHTML = (reports || [])
    .map(
      (r) =>
        `<option value="${escapeAttr(r.run_id)}">${escapeHtml(r.run_id)} | ${escapeHtml(r.method)} | 成功率 ${(
          (r.success_rate || 0) * 100
        ).toFixed(1)}%</option>`
    )
    .join("");
}

async function renderReport(report) {
  if (!report) return;
  state.currentReport = report;
  qs("#evalRunBadge").textContent = `${report.run_id} (${report.config.method})`;

  // Video
  const vid = qs("#mujocoVideo");
  vid.src = `${report.video_local_url}?t=${Date.now()}`;

  // Loss & Success Rate Charts
  const history = report.metrics_history || [];
  const epochs = history.map((m) => m.epoch);
  Plotly.newPlot(
    qs("#lossPlot"),
    [
      { type: "scatter", mode: "lines+markers", name: "Train Loss", x: epochs, y: history.map((m) => m.train_loss), line: { color: "#3b82f6", width: 2.5 } },
      { type: "scatter", mode: "lines+markers", name: "Validation Loss", x: epochs, y: history.map((m) => m.val_loss), line: { color: "#f97316", width: 2.5, dash: "dot" } },
    ],
    {
      paper_bgcolor: "#0b111b",
      plot_bgcolor: "#0b111b",
      font: { color: "#e6edf7" },
      margin: { l: 48, r: 16, t: 20, b: 40 },
      xaxis: { title: "Epoch" },
      yaxis: { title: "Loss" },
    },
    { responsive: true }
  );

  Plotly.newPlot(
    qs("#successPlot"),
    [
      {
        type: "scatter",
        mode: "lines+markers",
        name: "タスク成功率 (%)",
        x: epochs,
        y: history.map((m) => Number((m.task_success_rate * 100).toFixed(2))),
        line: { color: "#10b981", width: 3 },
        fill: "tozeroy",
        fillcolor: "rgba(16,185,129,0.15)",
      },
    ],
    {
      paper_bgcolor: "#0b111b",
      plot_bgcolor: "#0b111b",
      font: { color: "#e6edf7" },
      margin: { l: 48, r: 16, t: 20, b: 40 },
      xaxis: { title: "Epoch" },
      yaxis: { title: "Success Rate (%)", range: [0, 105] },
    },
    { responsive: true }
  );

  // Diagnostics
  const d = report.diagnostics;
  qs("#diagnosticsBox").innerHTML = `
    <div class="diag-pills">
      <span class="pill ${d.follows_human_motion ? "diag-ok" : "diag-warn"}">追随性: ${d.tracking_score}/100</span>
      <span class="pill ${d.is_smooth ? "diag-ok" : "diag-warn"}">滑らかさ: ${d.smoothness_score}/100</span>
      <span class="pill ${!d.overfitting_detected ? "diag-ok" : "diag-warn"}">過学習: ${d.overfitting_detected ? "検出あり" : "なし"} (差 ${d.overfitting_gap})</span>
    </div>
    <p>${escapeHtml(d.summary_ja)}</p>
    <ul>${(d.recommended_adjustments || []).map((a) => `<li>${escapeHtml(a)}</li>`).join("")}</ul>
  `;

  // 3D Frames
  try {
    const fdata = await apiJson(report.frames_local_url);
    state.frames3D = fdata.frames || [];
    state.frameIndex = 0;
    initThreeIfNeeded();
  } catch (err) {
    console.warn("3D frames load failed:", err);
  }
}

/* ==================== Three.js Humanoid Skeleton Viewer ==================== */

const BONE_LINKS = [
  ["pelvis", "torso"],
  ["torso", "head"],
  ["torso", "right_upper_arm"],
  ["right_upper_arm", "right_forearm"],
  ["torso", "left_upper_arm"],
  ["left_upper_arm", "left_forearm"],
  ["pelvis", "right_thigh"],
  ["right_thigh", "right_shin"],
  ["right_shin", "right_foot"],
  ["pelvis", "left_thigh"],
  ["left_thigh", "left_shin"],
  ["left_shin", "left_foot"],
];

function initThreeIfNeeded() {
  if (state.threeCtx || typeof THREE === "undefined") return;
  const container = qs("#threeContainer");
  const w = container.clientWidth || 500;
  const h = container.clientHeight || 340;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x080d16);

  const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
  camera.position.set(2.4, 1.5, 2.4);
  camera.lookAt(0, 0.85, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(w, h);
  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  const grid = new THREE.GridHelper(12, 24, 0x3b82f6, 0x1e293b);
  scene.add(grid);
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));

  const jointMeshes = {};
  const jointGeo = new THREE.SphereGeometry(0.055, 16, 16);
  const jointMat = new THREE.MeshStandardMaterial({ color: 0xf97316 });
  const headMat = new THREE.MeshStandardMaterial({ color: 0x93c5fd });

  const allNodes = new Set(BONE_LINKS.flat());
  allNodes.forEach((n) => {
    const mesh = new THREE.Mesh(jointGeo, n === "head" ? headMat : jointMat);
    if (n === "head") mesh.scale.set(1.6, 1.6, 1.6);
    scene.add(mesh);
    jointMeshes[n] = mesh;
  });

  const boneLines = BONE_LINKS.map(() => {
    const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    const mat = new THREE.LineBasicMaterial({ color: 0xe2e8f0, linewidth: 2 });
    const line = new THREE.Line(geom, mat);
    scene.add(line);
    return line;
  });

  state.threeCtx = { scene, camera, renderer, jointMeshes, boneLines };

  let lastTick = 0;
  function animate(now) {
    requestAnimationFrame(animate);
    if (state.frames3D.length > 0 && state.playing3D && now - lastTick > 65) {
      lastTick = now;
      state.frameIndex = (state.frameIndex + 1) % state.frames3D.length;
    }
    if (state.frames3D.length > 0) {
      const f = state.frames3D[state.frameIndex];
      qs("#frameCounter").textContent = `Frame ${state.frameIndex + 1}/${state.frames3D.length}`;
      const bodies = f.bodies || {};
      Object.entries(bodies).forEach(([k, pos]) => {
        if (jointMeshes[k]) {
          // MuJoCo is Z-up (x, y, z); Three.js is Y-up (x, z, -y)
          jointMeshes[k].position.set(pos[0], pos[2], -pos[1]);
        }
      });
      BONE_LINKS.forEach(([a, b], idx) => {
        if (bodies[a] && bodies[b]) {
          const va = new THREE.Vector3(bodies[a][0], bodies[a][2], -bodies[a][1]);
          const vb = new THREE.Vector3(bodies[b][0], bodies[b][2], -bodies[b][1]);
          boneLines[idx].geometry.setFromPoints([va, vb]);
        }
      });
      if (bodies.pelvis) {
        camera.lookAt(bodies.pelvis[0], 0.85, -bodies.pelvis[1]);
        camera.position.x = bodies.pelvis[0] + 2.2;
      }
    }
    renderer.render(scene, camera);
  }
  requestAnimationFrame(animate);
}

function resizeThree() {
  if (!state.threeCtx) return;
  const container = qs("#threeContainer");
  const w = container.clientWidth || 500;
  const h = container.clientHeight || 340;
  state.threeCtx.camera.aspect = w / h;
  state.threeCtx.camera.updateProjectionMatrix();
  state.threeCtx.renderer.setSize(w, h);
}

/* ==================== Helpers & Bootstrap ==================== */

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
function escapeAttr(str) {
  return escapeHtml(str).replace(/'/g, "&#39;");
}

async function bootstrap() {
  initTabs();

  try {
    const init = await apiJson("/api/dashboard-init");
    applyPaperSnapshot(init.text_mining, init.papers);
    populateMocaps(init.mocaps);
    populateReports(init.reports);
    if (init.latest_report) {
      await renderReport(init.latest_report);
    }
  } catch (err) {
    qs("#healthStatus").textContent = "Init Error";
    console.error(err);
  }

  qs("#paperSearchInput").addEventListener("input", (e) => {
    renderPapersTable(state.papers, e.target.value);
  });

  qs("#btnScrape").addEventListener("click", async () => {
    const kws = qs("#keywordsInput")
      .value.split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    const maxPer = Number(qs("#maxPerKw").value || 6);
    const banner = qs("#scrapeJobBanner");
    banner.classList.remove("hidden");
    const { job_id } = await apiJson("/api/papers/scrape", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keywords: kws, max_per_keyword: maxPer }),
    });
    pollJob(job_id, banner, async () => {
      const latest = await apiJson("/api/papers/latest");
      applyPaperSnapshot(latest.text_mining, latest.papers);
    });
  });

  qs("#btnAskAdvisor").addEventListener("click", async () => {
    const q = qs("#advisorQuestion").value.trim();
    if (!q) return;
    appendChat("user", q);
    qs("#advisorQuestion").value = "";
    const res = await apiJson("/api/advisor/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, mocap_id: qs("#mocapSelect").value }),
    });
    qs("#advisorEngineBadge").textContent = res.engine;
    appendChat("ai", res.reply);
    state.suggestedConfig = res.recommended_config;
    qs("#applySuggestionWrap").classList.remove("hidden");
  });

  qs("#btnApplySuggested").addEventListener("click", () => {
    applyConfigToForm(state.suggestedConfig);
  });

  qs("#btnUploadMocap").addEventListener("click", async () => {
    const file = qs("#mocapFile").files[0];
    if (!file) return alert("CSV / NPZ / BVH ファイルを選択してください");
    const fd = new FormData();
    fd.append("file", file);
    const res = await apiJson("/api/mocap/upload", { method: "POST", body: fd });
    const init = await apiJson("/api/dashboard-init");
    populateMocaps(init.mocaps);
    qs("#mocapSelect").value = res.mocap_id;
    alert(`S3保存完了: ${res.s3_prefix} (${res.frames} frames)`);
  });

  qs("#btnUploadXml").addEventListener("click", async () => {
    const file = qs("#robotXmlFile").files[0];
    if (!file) return alert("MJCF (.xml) ファイルを選択してください");
    const fd = new FormData();
    fd.append("file", file);
    const res = await apiJson("/api/robot-xml/upload", { method: "POST", body: fd });
    qs("#robotXmlName").value = res.robot_xml_name;
    alert(`ロボットXMLをS3へ保存しました: ${res.s3_key}`);
  });

  qs("#btnStartTrain").addEventListener("click", async () => {
    const banner = qs("#trainJobBanner");
    banner.classList.remove("hidden");
    const payload = {
      method: qs("#methodSelect").value,
      mocap_id: qs("#mocapSelect").value,
      epochs: Number(qs("#epochsInput").value),
      learning_rate: Number(qs("#lrInput").value),
      bc_weight: Number(qs("#bcWeightInput").value),
      smoothness_weight: Number(qs("#smoothWeightInput").value),
      reward_tracking_weight: Number(qs("#trackWeightInput").value),
      entropy_coef: Number(qs("#entropyInput").value),
      robot_xml_name: qs("#robotXmlName").value,
    };
    const { job_id } = await apiJson("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    pollJob(job_id, banner, async (report) => {
      await renderReport(report);
      const init = await apiJson("/api/dashboard-init");
      populateReports(init.reports);
      qs("#reportSelect").value = report.run_id;
      qs('.tab-btn[data-tab="eval"]').click();
    });
  });

  qs("#btnLoadReport").addEventListener("click", async () => {
    const runId = qs("#reportSelect").value;
    if (!runId) return;
    const report = await apiJson(`/api/runs/${runId}/report`);
    await renderReport(report);
  });

  qs("#btnToggle3D").addEventListener("click", () => {
    state.playing3D = !state.playing3D;
  });

  qs("#btnRetrainWithAI").addEventListener("click", async () => {
    const instr = qs("#retrainInstruction").value.trim();
    if (!instr) return alert("評価と再学習の指示を入力してください");
    const curCfg = state.currentReport ? state.currentReport.config : {};
    const res = await apiJson("/api/advisor/retrain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        instruction: instr,
        base_method: curCfg.method || qs("#methodSelect").value,
        mocap_id: curCfg.mocap_id || qs("#mocapSelect").value,
        epochs: curCfg.epochs || Number(qs("#epochsInput").value),
        learning_rate: curCfg.learning_rate || Number(qs("#lrInput").value),
        bc_weight: curCfg.bc_weight ?? Number(qs("#bcWeightInput").value),
        entropy_coef: curCfg.entropy_coef ?? Number(qs("#entropyInput").value),
        smoothness_weight: curCfg.smoothness_weight ?? Number(qs("#smoothWeightInput").value),
        reward_tracking_weight: curCfg.reward_tracking_weight ?? Number(qs("#trackWeightInput").value),
        auto_start: true,
      }),
    });
    const rbox = qs("#retrainResponseBox");
    rbox.classList.remove("hidden");
    rbox.textContent = `[${res.engine}]\n${res.reply}`;
    applyConfigToForm(res.updated_config);
    if (res.job_id) {
      const banner = qs("#trainJobBanner");
      banner.classList.remove("hidden");
      pollJob(res.job_id, banner, async (report) => {
        await renderReport(report);
        const init = await apiJson("/api/dashboard-init");
        populateReports(init.reports);
        qs("#reportSelect").value = report.run_id;
      });
    }
  });

  window.addEventListener("resize", resizeThree);
}

document.addEventListener("DOMContentLoaded", bootstrap);
