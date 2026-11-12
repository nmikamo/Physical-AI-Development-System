/* Physical AI Development System Frontend v2 (High-Operability Interactive Studio) */

const state = {
  papers: [],
  snapshot: null,
  activeClusterFilter: null,
  sim2RealOnly: false,
  searchQuery: "",
  selectedPaperId: null,
  suggestedConfig: null,
  currentReport: null,
  reportsList: [],
  frames3D: [],
  jointNames: [],
  frameIndex: 0,
  playing3D: true,
  speed3D: 1.0,
  showRefSkeleton: true,
  threeCtx: null,
};

const qs = (sel) => document.querySelector(sel);
const qsa = (sel) => Array.from(document.querySelectorAll(sel));

function showToast(msg) {
  const t = qs("#toastBox");
  if (!t) return;
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.add("hidden"), 3500);
}

async function apiJson(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return res.json();
}

function switchTab(tabId, scrollSelector = null) {
  qsa(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === tabId));
  qsa(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tabId}`));
  if (tabId === "eval" && state.threeCtx) {
    setTimeout(resizeThree, 60);
  }
  setTimeout(() => window.dispatchEvent(new Event("resize")), 80);
  if (scrollSelector) {
    const target = qs(scrollSelector);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      target.classList.remove("pulse-highlight");
      void target.offsetWidth;
      target.classList.add("pulse-highlight");
    }
  }
}

function initNavigation() {
  qsa(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  qsa(".step-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      switchTab(chip.dataset.gotoTab, chip.dataset.gotoTarget);
    });
  });
}

function pollJob(jobId, bannerEl, textEl, fillEl, onDone) {
  bannerEl.className = "job-banner";
  textEl.textContent = "ジョブを開始しました...";
  if (fillEl) fillEl.style.width = "10%";

  const timer = setInterval(async () => {
    try {
      const job = await apiJson(`/api/jobs/${jobId}`);
      const pct = Math.min(100, Math.max(8, Math.round(((job.progress || 1) / Math.max(1, job.total || 100)) * 100)));
      textEl.textContent = `${job.message} (${job.progress}/${job.total})`;
      if (fillEl) fillEl.style.width = `${pct}%`;

      if (job.status === "completed") {
        clearInterval(timer);
        bannerEl.className = "job-banner done";
        if (fillEl) fillEl.style.width = "100%";
        onDone(job.result);
      } else if (job.status === "failed") {
        clearInterval(timer);
        bannerEl.className = "job-banner fail";
      }
    } catch (err) {
      clearInterval(timer);
      bannerEl.className = "job-banner fail";
      textEl.textContent = `通信エラー: ${err.message}`;
    }
  }, 1200);
}

/* ==================== TAB A: alphaXiv & Text Mining ==================== */

function getFilteredPoints() {
  if (!state.snapshot || !state.snapshot.points) return [];
  return state.snapshot.points.filter((pt) => {
    if (state.activeClusterFilter !== null && pt.cluster_id !== state.activeClusterFilter) return false;
    if (state.sim2RealOnly && !pt.sim_to_real) return false;
    if (state.searchQuery) {
      const blob = `${pt.title} ${pt.summary_ja} ${pt.month_key} ${(pt.keywords_matched || []).join(" ")}`.toLowerCase();
      if (!blob.includes(state.searchQuery.toLowerCase())) return false;
    }
    return true;
  });
}

function renderClusterFilterPills(snapshot) {
  const wrap = qs("#clusterFilterPills");
  if (!snapshot || !snapshot.clusters) {
    wrap.innerHTML = "";
    return;
  }
  const allBtn = `<button class="cluster-pill ${state.activeClusterFilter === null ? "active" : ""}" data-cid="all">すべて (${snapshot.total_papers})</button>`;
  const cBtns = snapshot.clusters
    .map((c) => {
      const shortLabel = c.label.replace(/\n/g, "・");
      const active = state.activeClusterFilter === c.cluster_id ? "active" : "";
      return `<button class="cluster-pill ${active}" data-cid="${c.cluster_id}">${escapeHtml(shortLabel)} (${c.count})</button>`;
    })
    .join("");
  wrap.innerHTML = allBtn + cBtns;

  wrap.querySelectorAll(".cluster-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      const cid = btn.dataset.cid;
      state.activeClusterFilter = cid === "all" ? null : Number(cid);
      renderClusterFilterPills(state.snapshot);
      renderDensityMap(state.snapshot);
      renderPapersTable();
    });
  });
}

function renderDensityMap(snapshot) {
  const el = qs("#densityMapPlot");
  if (!snapshot || !snapshot.points || snapshot.points.length === 0) {
    el.innerHTML = `<div class="paper-detail-empty">論文データがありません。「論文収集・テキストマイニング・要約更新を実行」を押してください。</div>`;
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
    contours: { coloring: "heatmap", showlines: true, labelfont: { size: 9, color: "rgba(255,255,255,0.4)" } },
    line: { width: 0.5, color: "rgba(255,255,255,0.22)" },
    opacity: 0.85,
    hoverinfo: "skip",
    colorbar: { title: "研究密度", tickfont: { color: "#ebf1fa" }, titlefont: { color: "#ebf1fa" }, thickness: 14 },
  };

  const pts = getFilteredPoints();
  const simPts = pts.filter((p) => p.sim_to_real);
  const genPts = pts.filter((p) => !p.sim_to_real);

  const makeScatter = (list, name, color, symbol) => ({
    type: "scatter",
    mode: "markers",
    name,
    x: list.map((p) => p.x),
    y: list.map((p) => p.y),
    customdata: list,
    text: list.map(
      (p) =>
        `<b>${escapeHtml(p.title)}</b><br>公開月: ${p.month_key} | ${escapeHtml(p.cluster_label)}<br>👉 クリックで要約表示＆alphaXiv論文を開く`
    ),
    hovertemplate: "%{text}<extra></extra>",
    marker: {
      size: 11,
      symbol,
      color,
      line: { width: 1.8, color: "#080e18" },
    },
  });

  const annotations = (snapshot.clusters || [])
    .filter((c) => state.activeClusterFilter === null || c.cluster_id === state.activeClusterFilter)
    .map((c) => ({
      x: c.center_x,
      y: c.center_y,
      text: `<b>${c.label.replace(/\n/g, "<br>")}</b>`,
      showarrow: false,
      bgcolor: "rgba(255, 255, 255, 0.90)",
      bordercolor: "#1e293b",
      borderwidth: 1.2,
      borderpad: 5,
      font: { color: "#0f172a", size: 10.5 },
    }));

  const layout = {
    title: {
      text: `${snapshot.period_label} | 全${snapshot.total_papers}件（うちSim-to-Real ${snapshot.sim_to_real_count}件）`,
      font: { color: "#ebf1fa", size: 14 },
    },
    paper_bgcolor: "#080e18",
    plot_bgcolor: "#080e18",
    margin: { l: 25, r: 20, t: 42, b: 35 },
    xaxis: { showgrid: false, zeroline: false, showticklabels: false, range: [-1.18, 1.18] },
    yaxis: { showgrid: false, zeroline: false, showticklabels: false, range: [-1.18, 1.18] },
    legend: { orientation: "h", y: -0.03, font: { color: "#ebf1fa" } },
    annotations,
  };

  Plotly.react(
    el,
    [
      contourTrace,
      makeScatter(genPts, "シミュレーション・基礎制御研究", "#ffffff", "circle"),
      makeScatter(simPts, "Sim-to-Real 実機転移研究", "#fde047", "diamond"),
    ],
    layout,
    { responsive: true, displayModeBar: false }
  );

  if (!el._clickBound) {
    el._clickBound = true;
    el.on("plotly_click", (evt) => {
      const pt = evt?.points?.[0]?.customdata;
      if (!pt) return;
      selectPaper(pt, qs("#chkOpenTabOnClick").checked);
    });
  }
}

function formatStructuredSummaryHtml(summaryText) {
  const lines = String(summaryText || "要約データなし").split("\n").filter(Boolean);
  return lines
    .map((line) => {
      const m = line.match(/^【([^】]+)】(.*)$/);
      if (m) {
        return `<div class="summary-line"><span class="summary-tag">${escapeHtml(m[1])}</span>${escapeHtml(m[2])}</div>`;
      }
      return `<div class="summary-line">${escapeHtml(line)}</div>`;
    })
    .join("");
}

function selectPaper(paper, openNewTab = false) {
  state.selectedPaperId = paper.paper_id;
  qs("#selectedPaperIdBadge").textContent = paper.paper_id;
  const box = qs("#paperDetailBox");
  box.className = "paper-detail";
  const authors = (paper.authors || []).join(", ") || "Authors on alphaXiv";
  box.innerHTML = `
    <h3>${escapeHtml(paper.title)}</h3>
    <div class="meta-grid">
      <span class="pill">公開: ${escapeHtml(paper.published_at || paper.month_key)}</span>
      <span class="pill pill-s3">${escapeHtml(paper.cluster_label || "Physical AI")}</span>
      <span class="pill ${paper.sim_to_real ? "pill-ok" : ""}">${paper.sim_to_real ? "Sim-to-Real 実機研究" : "Simulation 研究"}</span>
    </div>
    <div class="muted" style="margin-bottom:10px;">著者: ${escapeHtml(authors)}</div>
    <a href="${escapeAttr(paper.url)}" target="_blank" rel="noopener noreferrer" class="btn btn-primary" style="width:100%; margin-bottom:10px;">
      🔗 alphaXiv で論文原文を別画面で開く (${escapeHtml(paper.paper_id)}) ↗
    </a>
    <div class="summary-section">
      ${formatStructuredSummaryHtml(paper.summary_ja)}
    </div>
  `;
  renderPapersTable();
  if (openNewTab && paper.url) {
    window.open(paper.url, "_blank", "noopener,noreferrer");
  }
}

function renderMonthlyAndTerms(snapshot) {
  const mEl = qs("#monthlyTrendPlot");
  const tEl = qs("#topTermsPlot");
  if (!snapshot) return;

  const months = (snapshot.monthly_counts || []).map((d) => d.month);
  const counts = (snapshot.monthly_counts || []).map((d) => d.count);
  Plotly.react(
    mEl,
    [
      {
        type: "bar",
        x: months,
        y: counts,
        marker: { color: "#3b82f6", line: { color: "#60a5fa", width: 1 } },
        hovertemplate: "%{x}: %{y}件 (クリックで絞り込み)<extra></extra>",
      },
    ],
    {
      paper_bgcolor: "#080e18",
      plot_bgcolor: "#080e18",
      font: { color: "#ebf1fa", size: 11 },
      margin: { l: 40, r: 14, t: 12, b: 48 },
      xaxis: { title: "公開月 (YYYY-MM)", tickangle: -35 },
      yaxis: { title: "論文数", gridcolor: "#1e293b" },
    },
    { responsive: true, displayModeBar: false }
  );

  if (!mEl._clickBound) {
    mEl._clickBound = true;
    mEl.on("plotly_click", (evt) => {
      const month = evt?.points?.[0]?.x;
      if (!month) return;
      qs("#paperSearchInput").value = month;
      state.searchQuery = month;
      renderDensityMap(state.snapshot);
      renderPapersTable();
      showToast(`公開月「${month}」で論文を絞り込みました`);
    });
  }

  const terms = (snapshot.top_terms || []).slice(0, 12).reverse();
  Plotly.react(
    tEl,
    [
      {
        type: "bar",
        orientation: "h",
        x: terms.map((t) => t.score),
        y: terms.map((t) => t.term),
        marker: { color: "#f97316" },
        hovertemplate: "%{y}: %{x:.3f} (クリックで絞り込み)<extra></extra>",
      },
    ],
    {
      paper_bgcolor: "#080e18",
      plot_bgcolor: "#080e18",
      font: { color: "#ebf1fa", size: 11 },
      margin: { l: 135, r: 16, t: 12, b: 36 },
      xaxis: { title: "TF-IDF 重要度", gridcolor: "#1e293b" },
    },
    { responsive: true, displayModeBar: false }
  );

  if (!tEl._clickBound) {
    tEl._clickBound = true;
    tEl.on("plotly_click", (evt) => {
      const term = evt?.points?.[0]?.y;
      if (!term) return;
      qs("#paperSearchInput").value = term;
      state.searchQuery = term;
      renderDensityMap(state.snapshot);
      renderPapersTable();
      showToast(`キーワード「${term}」で論文を絞り込みました`);
    });
  }
}

function renderPapersTable() {
  const tbody = qs("#papersTableBody");
  const pts = getFilteredPoints();
  qs("#tableCountHint").textContent = `全 ${pts.length} 件表示中（行クリックで上部に詳細表示、タイトルクリックで別タブ遷移）`;

  tbody.innerHTML = pts
    .map((p) => {
      const isSel = p.paper_id === state.selectedPaperId ? "selected-row" : "";
      const shortSum = String(p.summary_ja || "")
        .split("\n")
        .slice(0, 2)
        .join(" / ");
      return `
      <tr class="${isSel}" data-pid="${escapeAttr(p.paper_id)}">
        <td><span class="pill">${escapeHtml(p.month_key)}</span></td>
        <td>
          <a href="${escapeAttr(p.url)}" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation();">
            ${escapeHtml(p.title)} ↗
          </a>
        </td>
        <td><small>${escapeHtml(p.cluster_label || "-")}</small></td>
        <td>${p.sim_to_real ? '<span class="pill pill-ok">Sim-to-Real</span>' : '<span class="pill">Sim</span>'}</td>
        <td><small>${escapeHtml(shortSum)}</small></td>
      </tr>`;
    })
    .join("");

  tbody.querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => {
      const pid = tr.dataset.pid;
      const found = (state.snapshot?.points || []).find((p) => p.paper_id === pid);
      if (found) selectPaper(found, false);
    });
  });
}

function applyPaperSnapshot(snapshot, papers) {
  state.snapshot = snapshot;
  state.papers = papers || [];
  if (snapshot) {
    qs("#kpiTotalPapers").textContent = `${snapshot.total_papers} 件`;
    qs("#kpiSim2Real").textContent = `${snapshot.sim_to_real_count} 件`;
    qs("#kpiPeriod").textContent = snapshot.period_label;
  }
  renderClusterFilterPills(snapshot);
  renderDensityMap(snapshot);
  renderMonthlyAndTerms(snapshot);
  renderPapersTable();
  if (snapshot?.points?.length > 0 && !state.selectedPaperId) {
    selectPaper(snapshot.points[0], false);
  }
}

/* ==================== TAB B: AI Advisor & Training Studio ==================== */

function renderMarkdownLite(text) {
  return escapeHtml(text)
    .replace(/^### (.*)$/gm, "<strong>$1</strong>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br/>");
}

function appendChat(role, text) {
  const log = qs("#advisorChatLog");
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  div.innerHTML = role === "ai" ? renderMarkdownLite(text) : escapeHtml(text);
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function selectAlgoCard(method, markRecommended = false) {
  qs("#methodSelect").value = method;
  qsa(".algo-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.method === method);
  });
  if (markRecommended) {
    ["DAPG", "RLPD", "AWAC"].forEach((m) => {
      const badge = qs(`#recBadge-${m}`);
      if (badge) badge.classList.toggle("hidden", m !== method);
    });
  }
}

function populateMocaps(mocaps) {
  const sel = qs("#mocapSelect");
  const prev = sel.value;
  sel.innerHTML = (mocaps || [])
    .map(
      (m) =>
        `<option value="${escapeAttr(m.mocap_id)}">${escapeHtml(m.name)} (${m.frames} frames / ${m.fps}Hz)</option>`
    )
    .join("");
  if (prev && Array.from(sel.options).some((o) => o.value === prev)) {
    sel.value = prev;
  }
}

function applyConfigToForm(cfg, prevCfg = null) {
  if (!cfg) return;
  if (cfg.method) selectAlgoCard(cfg.method, true);

  const map = [
    ["epochs", "#epochsInput"],
    ["learning_rate", "#lrInput"],
    ["bc_weight", "#bcWeightInput"],
    ["smoothness_weight", "#smoothWeightInput"],
    ["reward_tracking_weight", "#trackWeightInput"],
    ["entropy_coef", "#entropyInput"],
  ];

  map.forEach(([key, inputSel]) => {
    const input = qs(inputSel);
    const diffEl = qs(`#diff-${key}`);
    if (cfg[key] !== undefined && input) {
      const oldVal = prevCfg && prevCfg[key] !== undefined ? prevCfg[key] : Number(input.value);
      const newVal = cfg[key];
      input.value = newVal;
      if (diffEl) {
        diffEl.textContent = oldVal !== newVal ? `(${oldVal} → ${newVal})` : "";
      }
    }
  });
  qs("#paramDiffHint").classList.remove("hidden");
  showToast(`AI推奨設定（${cfg.method}）をフォームに反映しました`);
}

/* ==================== TAB C: MuJoCo Verification, Graphs & Retrain ==================== */

function populateReports(reports) {
  state.reportsList = reports || [];
  const sel = qs("#reportSelect");
  sel.innerHTML = state.reportsList
    .map(
      (r) =>
        `<option value="${escapeAttr(r.run_id)}">${escapeHtml(r.run_id)} | ${escapeHtml(r.method)} (${escapeHtml(
          r.mocap_id
        )}) | 成功率 ${((r.success_rate || 0) * 100).toFixed(1)}%</option>`
    )
    .join("");

  const tbody = qs("#reportsComparisonBody");
  tbody.innerHTML = state.reportsList
    .map((r) => {
      const isCur = state.currentReport && state.currentReport.run_id === r.run_id ? "selected-row" : "";
      const sr = ((r.success_rate || 0) * 100).toFixed(1);
      return `
      <tr class="${isCur}" data-runid="${escapeAttr(r.run_id)}">
        <td><code>${escapeHtml(r.run_id)}</code></td>
        <td><strong>${escapeHtml(r.method)}</strong></td>
        <td>${escapeHtml(r.mocap_id)}</td>
        <td>${r.epochs || "-"}</td>
        <td><span class="pill pill-ok">${sr}%</span></td>
        <td>${r.tracking_score ?? "-"}/100</td>
        <td>${r.smoothness_score ?? "-"}/100</td>
        <td>${Number(r.train_loss || 0).toFixed(4)} / ${Number(r.val_loss || 0).toFixed(4)}</td>
        <td>${r.overfitting_detected ? '<span class="pill pill-warn">要注意</span>' : '<span class="pill pill-ok">正常</span>'}</td>
        <td><button class="btn btn-secondary btn-xs">表示</button></td>
      </tr>`;
    })
    .join("");

  tbody.querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", async () => {
      const runId = tr.dataset.runid;
      if (!runId) return;
      qs("#reportSelect").value = runId;
      const rep = await apiJson(`/api/runs/${runId}/report`);
      await renderReport(rep);
      showToast(`レポート ${runId} を読み込みました`);
    });
  });
}

async function renderReport(report) {
  if (!report) return;
  state.currentReport = report;
  const history = report.metrics_history || [];
  const lastM = history[history.length - 1] || {};
  const d = report.diagnostics || {};

  // Update KPI strip
  qs("#evalKpiMethod").textContent = `${report.config.method} / ${report.config.mocap_id}`;
  qs("#evalKpiSuccess").textContent = `${((lastM.task_success_rate || 0) * 100).toFixed(1)}%`;
  qs("#evalKpiTracking").textContent = `${d.tracking_score ?? "-"} / 100`;
  qs("#evalKpiSmooth").textContent = `${d.smoothness_score ?? "-"} / 100`;
  qs("#evalKpiOverfit").textContent = d.overfitting_detected
    ? `検出 (Gap ${d.overfitting_gap})`
    : `問題なし (Gap ${d.overfitting_gap})`;

  // Video
  const vid = qs("#mujocoVideo");
  const vidUrl = `${report.video_local_url}?t=${Date.now()}`;
  vid.src = vidUrl;
  qs("#btnDownloadVideo").href = report.video_local_url;

  const epochs = history.map((m) => m.epoch);
  const darkLayout = (xTitle, yTitle, extra = {}) => ({
    paper_bgcolor: "#080e18",
    plot_bgcolor: "#080e18",
    font: { color: "#ebf1fa", size: 11 },
    margin: { l: 48, r: 20, t: 18, b: 42 },
    xaxis: { title: xTitle, gridcolor: "#1e293b" },
    yaxis: { title: yTitle, gridcolor: "#1e293b", ...extra },
    legend: { orientation: "h", y: 1.12 },
  });

  // 1. Loss Plot
  Plotly.react(
    qs("#lossPlot"),
    [
      {
        type: "scatter",
        mode: "lines+markers",
        name: "Train Loss",
        x: epochs,
        y: history.map((m) => m.train_loss),
        line: { color: "#3b82f6", width: 2.5 },
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: "Validation Loss",
        x: epochs,
        y: history.map((m) => m.val_loss),
        line: { color: "#f97316", width: 2.5, dash: "dot" },
      },
    ],
    darkLayout("Epoch", "Loss"),
    { responsive: true, displayModeBar: false }
  );

  // 2. Success Rate & Survival Steps Plot
  Plotly.react(
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
        fillcolor: "rgba(16,185,129,0.14)",
      },
      {
        type: "bar",
        name: "生存ステップ比 (%)",
        x: epochs,
        y: history.map((m) => Math.min(100, Math.round(((m.survival_steps || 90) / 90) * 100))),
        marker: { color: "rgba(59,130,246,0.22)" },
      },
    ],
    darkLayout("Epoch", "Success Rate (%)", { range: [0, 105] }),
    { responsive: true, displayModeBar: false }
  );

  // 3. Episode Return Curve (Raw, Moving Avg, Peak)
  const returns = history.map((m) => m.mean_return);
  const movAvg = returns.map((_, i) => {
    const slice = returns.slice(Math.max(0, i - 3), i + 1);
    return Number((slice.reduce((a, b) => a + b, 0) / slice.length).toFixed(3));
  });
  const peak = Math.max(...returns, 0);
  Plotly.react(
    qs("#returnPlot"),
    [
      {
        type: "scatter",
        mode: "lines",
        name: "Raw Episode Return",
        x: epochs,
        y: returns,
        line: { color: "rgba(96,165,250,0.55)", width: 1.5 },
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: "Moving Average Return",
        x: epochs,
        y: movAvg,
        line: { color: "#ef4444", width: 3 },
      },
      {
        type: "scatter",
        mode: "lines",
        name: `Peak Return (${peak.toFixed(1)})`,
        x: epochs,
        y: epochs.map(() => peak),
        line: { color: "#fde047", width: 1.5, dash: "dash" },
      },
    ],
    darkLayout("Epoch", "Cumulative Return"),
    { responsive: true, displayModeBar: false }
  );

  // Diagnostics Box
  qs("#diagnosticsBox").innerHTML = `
    <div class="diag-pills">
      <span class="pill ${d.follows_human_motion ? "diag-ok" : "diag-warn"}">
        追随性: ${d.tracking_score}/100 (${d.follows_human_motion ? "良好" : "要改善"})
      </span>
      <span class="pill ${d.is_smooth ? "diag-ok" : "diag-warn"}">
        滑らかさ: ${d.smoothness_score}/100 (${d.is_smooth ? "滑らか" : "振動あり"})
      </span>
      <span class="pill ${!d.overfitting_detected ? "diag-ok" : "diag-warn"}">
        過学習: ${d.overfitting_detected ? "検出あり" : "なし"} (Val-Train差 ${d.overfitting_gap})
      </span>
    </div>
    <p><strong>総合判定:</strong> ${escapeHtml(d.summary_ja)}</p>
    <div class="muted" style="margin-bottom:4px;">AI推奨アクション:</div>
    <ul style="margin:0; padding-left:18px;">
      ${(d.recommended_adjustments || []).map((a) => `<li>${escapeHtml(a)}</li>`).join("")}
    </ul>
  `;

  // Highlight active row in comparison table
  populateReports(state.reportsList);

  // Load 3D Frames & Joint Tracking Plot
  try {
    const fdata = await apiJson(report.frames_local_url);
    state.frames3D = fdata.frames || [];
    state.jointNames = fdata.joint_names || [];
    state.frameIndex = 0;
    const slider = qs("#frameSlider");
    if (slider) {
      slider.max = Math.max(0, state.frames3D.length - 1);
      slider.value = 0;
    }
    populateJointSelector();
    renderJointTrackingPlot();
    initThreeIfNeeded();
  } catch (err) {
    console.warn("3D frames load failed:", err);
  }
}

function populateJointSelector() {
  const sel = qs("#jointPlotSelect");
  if (!sel || !state.jointNames.length) return;
  const labelsJa = {
    waist_pitch: "腰ピッチ (waist_pitch)",
    right_shoulder_pitch: "右肩ピッチ (right_shoulder)",
    right_elbow: "右肘 (right_elbow)",
    left_shoulder_pitch: "左肩ピッチ (left_shoulder)",
    left_elbow: "左肘 (left_elbow)",
    right_hip_pitch: "右股関節 (right_hip_pitch)",
    right_knee: "右膝 (right_knee)",
    right_ankle: "右足首 (right_ankle)",
    left_hip_pitch: "左股関節 (left_hip_pitch)",
    left_knee: "左膝 (left_knee)",
    left_ankle: "左足首 (left_ankle)",
  };
  sel.innerHTML = state.jointNames
    .map((jn, idx) => `<option value="${idx}" ${jn === "right_hip_pitch" ? "selected" : ""}>${labelsJa[jn] || jn}</option>`)
    .join("");
}

function renderJointTrackingPlot() {
  const el = qs("#jointTrackingPlot");
  if (!el || !state.frames3D.length) return;
  const jIdx = Number(qs("#jointPlotSelect")?.value || 5);
  const steps = state.frames3D.map((f) => f.step);
  const refVals = state.frames3D.map((f) => (f.qpos_ref ? f.qpos_ref[jIdx] : 0));
  const polVals = state.frames3D.map((f) => (f.qpos_policy ? f.qpos_policy[jIdx] : 0));

  Plotly.react(
    el,
    [
      {
        type: "scatter",
        mode: "lines",
        name: "人間お手本 MoCap (Reference)",
        x: steps,
        y: refVals,
        line: { color: "#10b981", width: 2.5, dash: "dash" },
      },
      {
        type: "scatter",
        mode: "lines",
        name: "ロボット実軌道 (Policy)",
        x: steps,
        y: polVals,
        line: { color: "#f97316", width: 2.5 },
      },
    ],
    {
      paper_bgcolor: "#080e18",
      plot_bgcolor: "#080e18",
      font: { color: "#ebf1fa", size: 11 },
      margin: { l: 48, r: 16, t: 18, b: 42 },
      xaxis: { title: "Simulation Step", gridcolor: "#1e293b" },
      yaxis: { title: "Joint Angle (rad)", gridcolor: "#1e293b" },
      legend: { orientation: "h", y: 1.12 },
    },
    { responsive: true, displayModeBar: false }
  );
}

/* ==================== Three.js 3D Viewer with Orbit & Dual Skeleton ==================== */

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
  const w = container.clientWidth || 540;
  const h = container.clientHeight || 320;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x070b12);

  const camera = new THREE.PerspectiveCamera(44, w / h, 0.1, 100);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(w, h);
  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  const grid = new THREE.GridHelper(16, 32, 0x3b82f6, 0x1e293b);
  scene.add(grid);
  scene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.7);
  dirLight.position.set(3, 6, 4);
  scene.add(dirLight);

  const allNodes = Array.from(new Set(BONE_LINKS.flat()));

  // Policy Skeleton (Orange/Blue solid)
  const policyJoints = {};
  const jointGeo = new THREE.SphereGeometry(0.052, 16, 16);
  const jointMat = new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.3 });
  const headMat = new THREE.MeshStandardMaterial({ color: 0x60a5fa, roughness: 0.3 });

  allNodes.forEach((n) => {
    const m = new THREE.Mesh(jointGeo, n === "head" ? headMat : jointMat);
    if (n === "head") m.scale.set(1.65, 1.65, 1.65);
    scene.add(m);
    policyJoints[n] = m;
  });

  const policyBones = BONE_LINKS.map(() => {
    const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    const mat = new THREE.LineBasicMaterial({ color: 0xf8fafc, linewidth: 2 });
    const line = new THREE.Line(geom, mat);
    scene.add(line);
    return line;
  });

  // Reference Human Skeleton (Emerald green ghost)
  const refGroup = new THREE.Group();
  scene.add(refGroup);
  const refJoints = {};
  const refGeo = new THREE.SphereGeometry(0.04, 12, 12);
  const refMat = new THREE.MeshBasicMaterial({ color: 0x10b981, wireframe: true });

  allNodes.forEach((n) => {
    const m = new THREE.Mesh(refGeo, refMat);
    if (n === "head") m.scale.set(1.5, 1.5, 1.5);
    refGroup.add(m);
    refJoints[n] = m;
  });

  const refBones = BONE_LINKS.map(() => {
    const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    const mat = new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.75 });
    const line = new THREE.Line(geom, mat);
    refGroup.add(line);
    return line;
  });

  // Built-in Mouse Orbit & Wheel Zoom Controller
  const orbit = { theta: 0.75, phi: 0.32, radius: 2.65, dragging: false, lastX: 0, lastY: 0 };
  container.addEventListener("mousedown", (e) => {
    orbit.dragging = true;
    orbit.lastX = e.clientX;
    orbit.lastY = e.clientY;
  });
  window.addEventListener("mouseup", () => {
    orbit.dragging = false;
  });
  window.addEventListener("mousemove", (e) => {
    if (!orbit.dragging) return;
    const dx = e.clientX - orbit.lastX;
    const dy = e.clientY - orbit.lastY;
    orbit.lastX = e.clientX;
    orbit.lastY = e.clientY;
    orbit.theta -= dx * 0.008;
    orbit.phi = Math.max(0.05, Math.min(1.35, orbit.phi + dy * 0.008));
  });
  container.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      orbit.radius = Math.max(1.2, Math.min(6.0, orbit.radius + e.deltaY * 0.002));
    },
    { passive: false }
  );

  state.threeCtx = { scene, camera, renderer, policyJoints, policyBones, refGroup, refJoints, refBones, orbit };

  let lastTick = 0;
  function animate(now) {
    requestAnimationFrame(animate);
    const interval = 55 / Math.max(0.2, state.speed3D);
    if (state.frames3D.length > 0 && state.playing3D && now - lastTick > interval) {
      lastTick = now;
      state.frameIndex = (state.frameIndex + 1) % state.frames3D.length;
      const slider = qs("#frameSlider");
      if (slider) slider.value = state.frameIndex;
    }

    refGroup.visible = state.showRefSkeleton;

    let targetX = 0;
    let targetZ = 0;
    if (state.frames3D.length > 0) {
      const f = state.frames3D[state.frameIndex] || state.frames3D[0];
      qs("#frameCounter").textContent = `Frame ${state.frameIndex + 1}/${state.frames3D.length}`;
      const bodies = f.bodies || {};
      const bodiesRef = f.bodies_ref || bodies;

      Object.entries(bodies).forEach(([k, pos]) => {
        if (policyJoints[k]) policyJoints[k].position.set(pos[0], pos[2], -pos[1]);
      });
      BONE_LINKS.forEach(([a, b], idx) => {
        if (bodies[a] && bodies[b]) {
          policyBones[idx].geometry.setFromPoints([
            new THREE.Vector3(bodies[a][0], bodies[a][2], -bodies[a][1]),
            new THREE.Vector3(bodies[b][0], bodies[b][2], -bodies[b][1]),
          ]);
        }
      });

      // Offset reference skeleton slightly in lateral Z (0.35m) so both are clearly visible side-by-side & overlaid
      const zOffset = 0.32;
      Object.entries(bodiesRef).forEach(([k, pos]) => {
        if (refJoints[k]) refJoints[k].position.set(pos[0], pos[2], -pos[1] - zOffset);
      });
      BONE_LINKS.forEach(([a, b], idx) => {
        if (bodiesRef[a] && bodiesRef[b]) {
          refBones[idx].geometry.setFromPoints([
            new THREE.Vector3(bodiesRef[a][0], bodiesRef[a][2], -bodiesRef[a][1] - zOffset),
            new THREE.Vector3(bodiesRef[b][0], bodiesRef[b][2], -bodiesRef[b][1] - zOffset),
          ]);
        }
      });

      if (bodies.pelvis) {
        targetX = bodies.pelvis[0];
        targetZ = -bodies.pelvis[1];
      }
    }

    const camX = targetX + orbit.radius * Math.cos(orbit.phi) * Math.cos(orbit.theta);
    const camY = 0.85 + orbit.radius * Math.sin(orbit.phi);
    const camZ = targetZ + orbit.radius * Math.cos(orbit.phi) * Math.sin(orbit.theta);
    camera.position.set(camX, camY, camZ);
    camera.lookAt(targetX, 0.85, targetZ - (state.showRefSkeleton ? 0.15 : 0));

    renderer.render(scene, camera);
  }
  requestAnimationFrame(animate);
}

function resizeThree() {
  if (!state.threeCtx) return;
  const container = qs("#threeContainer");
  const w = container.clientWidth || 540;
  const h = container.clientHeight || 320;
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

async function askAdvisorQuestion(q) {
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
  qs("#suggestedMethodText").textContent = res.recommended_method;
  qs("#applySuggestionWrap").classList.remove("hidden");
  selectAlgoCard(res.recommended_method, true);
}

async function bootstrap() {
  initNavigation();

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

  // Tab A events
  qs("#chkSim2RealOnly").addEventListener("change", (e) => {
    state.sim2RealOnly = e.target.checked;
    renderDensityMap(state.snapshot);
    renderPapersTable();
  });

  qs("#btnResetMapFilter").addEventListener("click", () => {
    state.activeClusterFilter = null;
    state.sim2RealOnly = false;
    state.searchQuery = "";
    qs("#chkSim2RealOnly").checked = false;
    qs("#paperSearchInput").value = "";
    renderClusterFilterPills(state.snapshot);
    renderDensityMap(state.snapshot);
    renderPapersTable();
  });

  qs("#paperSearchInput").addEventListener("input", (e) => {
    state.searchQuery = e.target.value.trim();
    renderDensityMap(state.snapshot);
    renderPapersTable();
  });

  qs("#btnClearPaperSearch").addEventListener("click", () => {
    qs("#paperSearchInput").value = "";
    state.searchQuery = "";
    renderDensityMap(state.snapshot);
    renderPapersTable();
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
    pollJob(job_id, banner, qs("#scrapeJobText"), qs("#scrapeProgressFill"), async () => {
      const latest = await apiJson("/api/papers/latest");
      applyPaperSnapshot(latest.text_mining, latest.papers);
      showToast("論文データの収集・テキストマイニングが完了しました");
    });
  });

  // Tab B events: One-click question chips & Algo cards
  qsa(".chip-btn").forEach((btn) => {
    btn.addEventListener("click", () => askAdvisorQuestion(btn.dataset.ask));
  });

  qs("#btnAskAdvisor").addEventListener("click", () => {
    askAdvisorQuestion(qs("#advisorQuestion").value.trim());
  });
  qs("#advisorQuestion").addEventListener("keydown", (e) => {
    if (e.key === "Enter") askAdvisorQuestion(qs("#advisorQuestion").value.trim());
  });

  qs("#btnApplySuggested").addEventListener("click", () => {
    applyConfigToForm(state.suggestedConfig);
  });

  qsa(".algo-card").forEach((card) => {
    card.addEventListener("click", () => selectAlgoCard(card.dataset.method, false));
  });

  qs("#btnUploadMocap").addEventListener("click", async () => {
    const file = qs("#mocapFile").files[0];
    if (!file) return showToast("CSV / NPZ / BVH ファイルを選択してください");
    const fd = new FormData();
    fd.append("file", file);
    const res = await apiJson("/api/mocap/upload", { method: "POST", body: fd });
    const init = await apiJson("/api/dashboard-init");
    populateMocaps(init.mocaps);
    qs("#mocapSelect").value = res.mocap_id;
    showToast(`モーションデータをS3へ保存しました: ${res.name} (${res.frames} frames)`);
  });

  qs("#btnUploadXml").addEventListener("click", async () => {
    const file = qs("#robotXmlFile").files[0];
    if (!file) return showToast("MJCF (.xml) ファイルを選択してください");
    const fd = new FormData();
    fd.append("file", file);
    const res = await apiJson("/api/robot-xml/upload", { method: "POST", body: fd });
    qs("#robotXmlName").value = res.robot_xml_name;
    qs("#currentXmlLabel").textContent = `現在: ${res.robot_xml_name} (S3保存済み)`;
    showToast(`カスタムロボットXMLをS3へ保存しました: ${res.s3_key}`);
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
    pollJob(job_id, banner, qs("#trainJobText"), qs("#trainProgressFill"), async (report) => {
      const init = await apiJson("/api/dashboard-init");
      populateReports(init.reports);
      qs("#reportSelect").value = report.run_id;
      await renderReport(report);
      switchTab("eval", "#sectionStep3");
      showToast(`学習が完了しました (${report.run_id})！MuJoCo動画とグラフを表示します`);
    });
  });

  // Tab C events
  qs("#btnLoadReport").addEventListener("click", async () => {
    const runId = qs("#reportSelect").value;
    if (!runId) return;
    const report = await apiJson(`/api/runs/${runId}/report`);
    await renderReport(report);
    showToast(`レポート ${runId} を表示しました`);
  });

  qsa(".speed-vid-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      qsa(".speed-vid-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const vid = qs("#mujocoVideo");
      if (vid) vid.playbackRate = Number(btn.dataset.vrate || 1.0);
    });
  });

  qs("#chkShowRefSkeleton").addEventListener("change", (e) => {
    state.showRefSkeleton = e.target.checked;
  });

  qs("#btnToggle3D").addEventListener("click", () => {
    state.playing3D = !state.playing3D;
    qs("#btnToggle3D").textContent = state.playing3D ? "一時停止" : "再生";
  });

  qs("#btnStepPrev").addEventListener("click", () => {
    if (!state.frames3D.length) return;
    state.playing3D = false;
    qs("#btnToggle3D").textContent = "再生";
    state.frameIndex = (state.frameIndex - 1 + state.frames3D.length) % state.frames3D.length;
    qs("#frameSlider").value = state.frameIndex;
  });

  qs("#btnStepNext").addEventListener("click", () => {
    if (!state.frames3D.length) return;
    state.playing3D = false;
    qs("#btnToggle3D").textContent = "再生";
    state.frameIndex = (state.frameIndex + 1) % state.frames3D.length;
    qs("#frameSlider").value = state.frameIndex;
  });

  qs("#frameSlider").addEventListener("input", (e) => {
    state.playing3D = false;
    qs("#btnToggle3D").textContent = "再生";
    state.frameIndex = Number(e.target.value);
  });

  qs("#speed3DSelect").addEventListener("change", (e) => {
    state.speed3D = Number(e.target.value || 1.0);
  });

  qsa(".cam-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!state.threeCtx) return;
      const preset = btn.dataset.cam;
      if (preset === "side") {
        state.threeCtx.orbit.theta = Math.PI / 2;
        state.threeCtx.orbit.phi = 0.15;
      } else if (preset === "front") {
        state.threeCtx.orbit.theta = 0.0;
        state.threeCtx.orbit.phi = 0.18;
      } else {
        state.threeCtx.orbit.theta = 0.75;
        state.threeCtx.orbit.phi = 0.32;
      }
    });
  });

  qs("#jointPlotSelect").addEventListener("change", () => {
    renderJointTrackingPlot();
  });

  qsa(".eval-chip-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      qs("#retrainInstruction").value = btn.dataset.instr;
      qs("#retrainInstruction").focus();
    });
  });

  qs("#btnRetrainWithAI").addEventListener("click", async () => {
    const instr = qs("#retrainInstruction").value.trim();
    if (!instr) return showToast("評価コメントまたは上の評価チップを選択してください");
    const curCfg = state.currentReport ? state.currentReport.config : {};
    const banner = qs("#evalTrainJobBanner");
    banner.classList.remove("hidden");

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
    rbox.innerHTML = `<strong>[${escapeHtml(res.engine)}]</strong><br/>${renderMarkdownLite(res.reply)}`;
    applyConfigToForm(res.updated_config, res.previous_config);

    if (res.job_id) {
      pollJob(res.job_id, banner, qs("#evalTrainJobText"), qs("#evalTrainProgressFill"), async (report) => {
        const init = await apiJson("/api/dashboard-init");
        populateReports(init.reports);
        qs("#reportSelect").value = report.run_id;
        await renderReport(report);
        showToast(`AI再指示による再学習が完了しました (${report.run_id})！`);
      });
    }
  });

  window.addEventListener("resize", resizeThree);
}

document.addEventListener("DOMContentLoaded", bootstrap);
