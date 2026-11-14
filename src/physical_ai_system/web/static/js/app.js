/* Physical AI Development System Frontend v3 (Enhanced Operability & Accessibility) */

const state = {
  papers: [],
  snapshot: null,
  activeClusterFilter: null,
  sim2RealOnly: false,
  selectedYear: "all",
  trendViewMode: "yearly",
  searchQuery: "",
  selectedPaperId: null,
  suggestedConfig: null,
  pendingRetrainConfig: null,
  currentReport: null,
  reportsList: [],
  mocapsList: [],
  frames3D: [],
  jointNames: [],
  frameIndex: 0,
  playing3D: true,
  speed3D: 1.0,
  showRefSkeleton: true,
  threeCtx: null,
  stepStates: {
    step1: "未相談",
    step2: "待機中",
    step3: "待機中",
    step4: "未評価",
  },
};

const qs = (sel) => document.querySelector(sel);
const qsa = (sel) => Array.from(document.querySelectorAll(sel));

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

function showToast(msg) {
  const t = qs("#toastBox");
  if (!t) return;
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.add("hidden"), 3500);
}

function showInlineError(elSel, msg) {
  const el = qs(elSel);
  if (!el) return;
  if (!msg) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.textContent = `⚠ ${msg}`;
  el.classList.remove("hidden");
}

function setBtnLoading(btnSel, isLoading, loadingText = "処理中...") {
  const btn = qs(btnSel);
  if (!btn) return;
  if (isLoading) {
    btn.dataset.origText = btn.dataset.origText || btn.querySelector(".btn-text")?.textContent || btn.textContent;
    btn.disabled = true;
    btn.classList.add("btn-loading");
    const tSpan = btn.querySelector(".btn-text");
    if (tSpan) tSpan.textContent = loadingText;
  } else {
    btn.disabled = false;
    btn.classList.remove("btn-loading");
    const tSpan = btn.querySelector(".btn-text");
    if (tSpan && btn.dataset.origText) tSpan.textContent = btn.dataset.origText;
  }
}

async function apiJson(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const errBody = await res.json();
      detail = errBody.detail || errBody.message || JSON.stringify(errBody);
    } catch {
      const text = await res.text();
      if (text) detail = text;
    }
    throw new Error(detail);
  }
  return res.json();
}

/* ==================== Navigation & Guided Stepper ==================== */

function updateStepStatus(stepNum, statusText, stateClass = "") {
  state.stepStates[`step${stepNum}`] = statusText;
  const tag = qs(`#stepState${stepNum}`);
  if (tag) {
    tag.textContent = statusText;
    tag.className = "step-state-tag";
    if (stateClass) tag.classList.add(stateClass);
  }
}

function switchTab(tabId, scrollSelector = null) {
  qsa(".tab-btn").forEach((b) => {
    const isActive = b.dataset.tab === tabId;
    b.classList.toggle("active", isActive);
    b.setAttribute("aria-selected", isActive ? "true" : "false");
  });
  qsa(".tab-panel").forEach((p) => {
    const isActive = p.id === `tab-${tabId}`;
    p.classList.toggle("active", isActive);
  });

  const vTitle = qs("#viewTitle");
  if (vTitle) {
    const titles = {
      papers: "論文を探す",
      train: "モデル学習",
      eval: "結果を評価",
    };
    vTitle.textContent = titles[tabId] || "ダッシュボード";
  }

  const stepper = qs("#workflowStepper");
  if (stepper) {
    // Show stepper only in train and eval tabs
    stepper.style.display = tabId === "papers" ? "none" : "flex";
  }

  if (tabId === "eval" && state.threeCtx) {
    setTimeout(resizeThree, 60);
  }
  setTimeout(() => window.dispatchEvent(new Event("resize")), 80);

  if (scrollSelector) {
    const target = qs(scrollSelector);
    if (target) {
      setTimeout(() => {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 50);
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

  // System info drawer toggle
  const infoBtn = qs("#btnSystemInfoToggle");
  const infoDrawer = qs("#systemInfoDrawer");
  const closeInfoBtn = qs("#btnCloseSystemInfo");
  if (infoBtn && infoDrawer) {
    infoBtn.addEventListener("click", () => {
      const isExpanded = infoBtn.getAttribute("aria-expanded") === "true";
      infoBtn.setAttribute("aria-expanded", isExpanded ? "false" : "true");
      infoDrawer.classList.toggle("hidden", isExpanded);
    });
    if (closeInfoBtn) {
      closeInfoBtn.addEventListener("click", () => {
        infoBtn.setAttribute("aria-expanded", "false");
        infoDrawer.classList.add("hidden");
      });
    }
  }

  // Score Help toggle
  const scoreHelpBtn = qs("#btnToggleScoreHelp");
  const scoreHelpBox = qs("#scoreHelpBox");
  if (scoreHelpBtn && scoreHelpBox) {
    scoreHelpBtn.addEventListener("click", () => {
      const isExpanded = scoreHelpBtn.getAttribute("aria-expanded") === "true";
      scoreHelpBtn.setAttribute("aria-expanded", isExpanded ? "false" : "true");
      scoreHelpBox.classList.toggle("hidden", isExpanded);
    });
  }
}

function pollJob(jobId, bannerEl, textEl, fillEl, onDone, onFail) {
  bannerEl.className = "job-banner";
  textEl.textContent = "処理を開始しました...";
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
        if (onDone) onDone(job.result);
      } else if (job.status === "failed") {
        clearInterval(timer);
        bannerEl.className = "job-banner fail";
        textEl.textContent = `失敗: ${job.message}`;
        if (onFail) onFail(job.message);
      }
    } catch (err) {
      clearInterval(timer);
      bannerEl.className = "job-banner fail";
      textEl.textContent = `通信エラー: ${err.message}`;
      if (onFail) onFail(err.message);
    }
  }, 1200);
}

/* ==================== TAB 1: 論文を探す (Paper Search & Text Mining) ==================== */

function getFilteredPoints() {
  if (!state.snapshot || !state.snapshot.points) return [];
  return state.snapshot.points.filter((pt) => {
    if (state.activeClusterFilter !== null && pt.cluster_id !== state.activeClusterFilter) return false;
    if (state.sim2RealOnly && !pt.sim_to_real) return false;
    if (state.selectedYear && state.selectedYear !== "all") {
      const yr = (pt.month_key || "").slice(0, 4);
      if (yr !== state.selectedYear) return false;
    }
    if (state.searchQuery) {
      const blob = `${pt.title} ${pt.summary_ja} ${pt.month_key} ${(pt.keywords_matched || []).join(" ")}`.toLowerCase();
      if (!blob.includes(state.searchQuery.toLowerCase())) return false;
    }
    return true;
  });
}

function setYearFilter(year) {
  state.selectedYear = year;
  qsa(".year-pill").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.year === year);
  });
  renderDensityMap(state.snapshot);
  renderPapersTable();
  updateActiveFiltersBar();
  const label = year === "all" ? "全期間 (2020〜)" : `${year}年度`;
  showToast(`発行年度「${label}」で論文を絞り込みました`);
}

function initYearFilterPills() {
  qsa(".year-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      setYearFilter(btn.dataset.year || "all");
    });
  });
}

function updateActiveFiltersBar() {
  const bar = qs("#activeFiltersBar");
  const tagsContainer = qs("#activeFilterTags");
  if (!bar || !tagsContainer) return;

  const tags = [];
  if (state.selectedYear && state.selectedYear !== "all") {
    tags.push(`年度: ${state.selectedYear}年`);
  }
  if (state.activeClusterFilter !== null) {
    const c = (state.snapshot?.clusters || []).find((x) => x.cluster_id === state.activeClusterFilter);
    tags.push(`クラスタ: ${c ? c.label.replace(/\n/g, "・") : state.activeClusterFilter}`);
  }
  if (state.sim2RealOnly) {
    tags.push("Sim-to-Real のみ");
  }
  if (state.searchQuery) {
    tags.push(`検索: "${state.searchQuery}"`);
  }

  if (tags.length === 0) {
    bar.classList.add("hidden");
    tagsContainer.innerHTML = "";
  } else {
    bar.classList.remove("hidden");
    tagsContainer.innerHTML = tags.map((t) => `<span class="active-tag">${escapeHtml(t)}</span>`).join("");
  }
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
      updateActiveFiltersBar();
    });
  });
}

function renderDensityMap(snapshot) {
  const el = qs("#densityMapPlot");
  if (!snapshot || !snapshot.points || snapshot.points.length === 0) {
    el.innerHTML = `<div class="paper-detail-empty">論文データがありません。「論文を更新」ボタンを押してください。</div>`;
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
    contours: { coloring: "heatmap", showlines: true, labelfont: { size: 10, color: "rgba(15,23,42,0.6)" } },
    line: { width: 0.6, color: "rgba(15,23,42,0.2)" },
    opacity: 0.85,
    hoverinfo: "skip",
    colorbar: {
      title: "研究密度",
      tickfont: { color: "#475569", size: 12 },
      titlefont: { color: "#0f172a", size: 13 },
      thickness: 16,
    },
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
        `<b>${escapeHtml(p.title)}</b><br>公開: ${p.month_key} | ${escapeHtml(p.cluster_label)}<br>👉 クリックで詳細表示`
    ),
    hovertemplate: "%{text}<extra></extra>",
    marker: {
      size: 12,
      symbol,
      color,
      line: { width: 1.5, color: "#ffffff" },
    },
  });

  const annotations = (snapshot.clusters || [])
    .filter((c) => state.activeClusterFilter === null || c.cluster_id === state.activeClusterFilter)
    .map((c) => ({
      x: c.center_x,
      y: c.center_y,
      text: `<b>${c.label.replace(/\n/g, "<br>")}</b>`,
      showarrow: false,
      bgcolor: "rgba(255, 255, 255, 0.95)",
      bordercolor: "#cbd5e1",
      borderwidth: 1.2,
      borderpad: 5,
      font: { color: "#0f172a", size: 11 },
    }));

  const layout = {
    title: {
      text: `${snapshot.period_label} | 全${snapshot.total_papers}件 (Sim-to-Real: ${snapshot.sim_to_real_count}件)`,
      font: { color: "#0f172a", size: 14 },
    },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#f8fafc",
    font: { color: "#334155", size: 13 },
    margin: { l: 25, r: 20, t: 45, b: 35 },
    xaxis: { showgrid: false, zeroline: false, showticklabels: false, range: [-1.18, 1.18] },
    yaxis: { showgrid: false, zeroline: false, showticklabels: false, range: [-1.18, 1.18] },
    legend: { orientation: "h", y: -0.04, font: { color: "#475569", size: 12 } },
    annotations,
  };

  Plotly.react(
    el,
    [
      contourTrace,
      makeScatter(genPts, "シミュレーション研究", "#3b82f6", "circle"),
      makeScatter(simPts, "Sim-to-Real 実機研究", "#ea580c", "diamond"),
    ],
    layout,
    { responsive: true, displayModeBar: false }
  );

  if (!el._clickBound) {
    el._clickBound = true;
    el.on("plotly_click", (evt) => {
      const pt = evt?.points?.[0]?.customdata;
      if (!pt) return;
      selectPaper(pt, false);
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

function selectPaper(paper, scrollToDetail = false) {
  state.selectedPaperId = paper.paper_id;
  qs("#selectedPaperIdBadge").textContent = paper.paper_id;
  const box = qs("#paperDetailBox");
  box.className = "paper-detail";
  const authors = (paper.authors || []).join(", ") || "著者情報（alphaXiv掲載）";

  box.innerHTML = `
    <h3>${escapeHtml(paper.title)}</h3>
    <div class="meta-grid">
      <span class="pill">公開: ${escapeHtml(paper.published_at || paper.month_key)}</span>
      <span class="pill pill-s3">${escapeHtml(paper.cluster_label || "Physical AI")}</span>
      <span class="pill ${paper.sim_to_real ? "pill-ok" : ""}">${paper.sim_to_real ? "Sim-to-Real 実機研究" : "Simulation 研究"}</span>
    </div>
    <div class="paper-authors">著者: ${escapeHtml(authors)}</div>
    <a href="${escapeAttr(paper.url)}" target="_blank" rel="noopener noreferrer" class="btn btn-primary" style="width:100%; margin-bottom:14px;">
      🔗 alphaXiv で論文原文を開く (${escapeHtml(paper.paper_id)}) ↗
    </a>
    <div class="summary-section">
      ${formatStructuredSummaryHtml(paper.summary_ja || paper.abstract)}
    </div>
  `;

  renderPapersTable();

  // On smaller screens, scroll user smoothly to the detail panel
  if (scrollToDetail || window.innerWidth <= 1024) {
    const detailContainer = qs("#paperDetailContainer");
    if (detailContainer) {
      detailContainer.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }
}

function renderYearlyTrendPlot(snapshot) {
  const yEl = qs("#yearlyTrendPlot");
  if (!yEl || !snapshot) return;

  const yearly = snapshot.yearly_counts || [];
  if (yearly.length === 0) {
    yEl.innerHTML = `<div class="table-empty">年度別データがありません</div>`;
    return;
  }

  const years = yearly.map((d) => `${d.year}年`);
  const simCounts = yearly.map((d) => d.sim_count);
  const pureSimCounts = yearly.map((d) => Math.max(0, d.count - d.sim_count));

  const traces = [
    {
      type: "bar",
      name: "Sim-to-Real 実機研究",
      x: years,
      y: simCounts,
      marker: { color: "#2563eb", line: { color: "#1d4ed8", width: 1 } },
      customdata: yearly.map((d) => d.year),
      hovertemplate: "<b>%{x}</b><br>Sim-to-Real: %{y}件<br>👉 クリックで該当年度に絞り込み<extra></extra>",
    },
    {
      type: "bar",
      name: "Simulation 研究",
      x: years,
      y: pureSimCounts,
      marker: { color: "#94a3b8", line: { color: "#64748b", width: 1 } },
      customdata: yearly.map((d) => d.year),
      hovertemplate: "<b>%{x}</b><br>Simulation: %{y}件<br>👉 クリックで該当年度に絞り込み<extra></extra>",
    },
  ];

  const layout = {
    barmode: "stack",
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#f8fafc",
    font: { color: "#334155", size: 13 },
    margin: { l: 45, r: 16, t: 20, b: 45 },
    xaxis: { title: "発行年度 (2020年〜)", tickfont: { size: 12, color: "#475569" }, titlefont: { size: 13, color: "#1e293b" } },
    yaxis: { title: "論文数 (件)", gridcolor: "#e2e8f0", tickfont: { size: 12, color: "#475569" }, titlefont: { size: 13, color: "#1e293b" } },
    legend: { orientation: "h", y: 1.15, font: { size: 12, color: "#475569" } },
  };

  Plotly.react(yEl, traces, layout, { responsive: true, displayModeBar: false });

  if (!yEl._clickBound) {
    yEl._clickBound = true;
    yEl.on("plotly_click", (evt) => {
      const yr = evt?.points?.[0]?.customdata;
      if (!yr) return;
      setYearFilter(String(yr));
    });
  }
}

function initTrendPlotToggle() {
  const btnYearly = qs("#btnTrendYearly");
  const btnMonthly = qs("#btnTrendMonthly");
  const yearlyPlot = qs("#yearlyTrendPlot");
  const monthlyPlot = qs("#monthlyTrendPlot");
  const title = qs("#trendPlotTitle");

  if (!btnYearly || !btnMonthly) return;

  btnYearly.addEventListener("click", () => {
    btnYearly.classList.add("active");
    btnMonthly.classList.remove("active");
    yearlyPlot.classList.remove("hidden");
    monthlyPlot.classList.add("hidden");
    state.trendViewMode = "yearly";
    if (title) title.textContent = "年度別 論文推移 (2020年〜)";
    Plotly.Plots.resize(yearlyPlot);
  });

  btnMonthly.addEventListener("click", () => {
    btnMonthly.classList.add("active");
    btnYearly.classList.remove("active");
    monthlyPlot.classList.remove("hidden");
    yearlyPlot.classList.add("hidden");
    state.trendViewMode = "monthly";
    if (title) title.textContent = "月別 論文推移";
    Plotly.Plots.resize(monthlyPlot);
  });
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
        marker: { color: "#2563eb", line: { color: "#1d4ed8", width: 1.2 } },
        hovertemplate: "%{x}: %{y}件 (クリックで絞り込み)<extra></extra>",
      },
    ],
    {
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#f8fafc",
      font: { color: "#334155", size: 13 },
      margin: { l: 45, r: 16, t: 16, b: 50 },
      xaxis: { title: "公開月", tickangle: -30, tickfont: { size: 12, color: "#475569" }, titlefont: { size: 13, color: "#1e293b" } },
      yaxis: { title: "論文数", gridcolor: "#e2e8f0", tickfont: { size: 12, color: "#475569" }, titlefont: { size: 13, color: "#1e293b" } },
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
      updateActiveFiltersBar();
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
        marker: { color: "#ea580c" },
        hovertemplate: "%{y}: %{x:.3f} (クリックで絞り込み)<extra></extra>",
      },
    ],
    {
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#f8fafc",
      font: { color: "#334155", size: 13 },
      margin: { l: 140, r: 16, t: 16, b: 40 },
      xaxis: { title: "TF-IDF 重要度", gridcolor: "#e2e8f0", tickfont: { size: 12, color: "#475569" }, titlefont: { size: 13, color: "#1e293b" } },
      yaxis: { tickfont: { size: 12, color: "#475569" } },
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
      updateActiveFiltersBar();
      showToast(`キーワード「${term}」で論文を絞り込みました`);
    });
  }
}

function renderPapersTable() {
  const tbody = qs("#papersTableBody");
  const emptyState = qs("#tableEmptyState");
  const pts = getFilteredPoints();

  const totalCount = state.snapshot?.total_papers || 0;
  qs("#tableCountHint").textContent = `全 ${totalCount} 件中 ${pts.length} 件を表示中（行クリックで詳細表示）`;

  if (pts.length === 0) {
    tbody.innerHTML = "";
    emptyState.classList.remove("hidden");
    const reasons = [];
    if (state.searchQuery) reasons.push(`検索語「${state.searchQuery}」`);
    if (state.sim2RealOnly) reasons.push("Sim-to-Real のみ");
    if (state.activeClusterFilter !== null) reasons.push("特定クラスタ");
    qs("#tableEmptyMessage").textContent = `条件（${reasons.join(" ＋ ")}）に一致する論文が見つかりませんでした。`;
    return;
  }

  emptyState.classList.add("hidden");
  tbody.innerHTML = pts
    .map((p) => {
      const isSel = p.paper_id === state.selectedPaperId ? "selected-row" : "";
      // Keep summary short in table (1-2 sentences)
      const rawText = String(p.summary_ja || p.abstract || "").replace(/【[^】]+】/g, "");
      const shortSum = rawText
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean)
        .slice(0, 1)
        .join(" ")
        .slice(0, 80);

      return `
      <tr class="${isSel}" data-pid="${escapeAttr(p.paper_id)}" tabindex="0" role="button" aria-label="${escapeAttr(p.title)}">
        <td><span class="pill">${escapeHtml(p.month_key)}</span></td>
        <td>
          <strong>${escapeHtml(p.title)}</strong>
        </td>
        <td><small>${escapeHtml(p.cluster_label || "-")}</small></td>
        <td>${p.sim_to_real ? '<span class="pill pill-ok">Sim-to-Real</span>' : '<span class="pill">Simulation</span>'}</td>
        <td><span style="color:var(--text-muted); font-size:0.875rem;">${escapeHtml(shortSum)}...</span></td>
      </tr>`;
    })
    .join("");

  tbody.querySelectorAll("tr").forEach((tr) => {
    const handleSelect = () => {
      const pid = tr.dataset.pid;
      const found = (state.snapshot?.points || []).find((p) => p.paper_id === pid);
      if (found) selectPaper(found, false);
    };
    tr.addEventListener("click", handleSelect);
    tr.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        handleSelect();
      }
    });
  });
}

function applyPaperSnapshot(snapshot, papers) {
  state.snapshot = snapshot;
  state.papers = papers || [];
  if (snapshot) {
    qs("#kpiTotalPapers").textContent = `${snapshot.total_papers} 件`;
    qs("#kpiSim2Real").textContent = `${snapshot.sim_to_real_count} 件`;
    qs("#kpiPeriod").textContent = snapshot.period_label || "-";
    qs("#kpiLastUpdated").textContent = new Date().toLocaleDateString("ja-JP");
  }
  renderClusterFilterPills(snapshot);
  renderYearlyTrendPlot(snapshot);
  renderMonthlyAndTerms(snapshot);
  renderDensityMap(snapshot);
  renderPapersTable();
  updateActiveFiltersBar();
  if (snapshot?.points?.length > 0 && !state.selectedPaperId) {
    selectPaper(snapshot.points[0], false);
  }
}

/* ==================== TAB 2: 学習する (AI Advisor & Training Setup) ==================== */

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

function selectAlgoMethod(method, markRecommended = false) {
  qs("#methodSelect").value = method;
  const radio = qs(`input[name="rlMethodRadio"][value="${method}"]`);
  if (radio) radio.checked = true;

  if (markRecommended) {
    ["DAPG", "RLPD", "AWAC"].forEach((m) => {
      const badge = qs(`#recBadge-${m}`);
      if (badge) badge.classList.toggle("hidden", m !== method);
    });
  }
  updatePreflightPreview();
}

function populateMocaps(mocaps) {
  state.mocapsList = mocaps || [];
  const sel = qs("#mocapSelect");
  const prev = sel.value;
  sel.innerHTML = state.mocapsList
    .map(
      (m) =>
        `<option value="${escapeAttr(m.mocap_id)}">${escapeHtml(m.name)} (${m.frames}フレーム / ${m.fps}Hz)</option>`
    )
    .join("");
  if (prev && Array.from(sel.options).some((o) => o.value === prev)) {
    sel.value = prev;
  }
  updatePreflightPreview();
}

function updatePreflightPreview() {
  const mVal = qs("#methodSelect")?.value || "DAPG";
  const selMocapOpt = qs("#mocapSelect")?.selectedOptions?.[0];
  const mocapName = selMocapOpt ? selMocapOpt.text.split(" (")[0] : "-";
  const epochs = qs("#epochsInput")?.value || "25";
  const smooth = qs("#smoothWeightInput")?.value || "0.06";

  if (qs("#previewMethod")) qs("#previewMethod").textContent = mVal;
  if (qs("#previewMocap")) qs("#previewMocap").textContent = mocapName;
  if (qs("#previewEpochs")) qs("#previewEpochs").textContent = `${epochs} 回`;
  if (qs("#previewSmooth")) qs("#previewSmooth").textContent = smooth;
}

function applyConfigToForm(cfg, prevCfg = null) {
  if (!cfg) return;
  if (cfg.method) selectAlgoMethod(cfg.method, true);

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
  updatePreflightPreview();
  showToast(`AI推奨設定 (${cfg.method}) をフォームへ反映しました`);
}

async function askAdvisorQuestion(q) {
  if (!q) return;
  showInlineError("#advisorErrorBanner", "");
  appendChat("user", q);
  qs("#advisorQuestion").value = "";
  setBtnLoading("#btnAskAdvisor", true, "相談中...");

  try {
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
    selectAlgoMethod(res.recommended_method, true);
    updateStepStatus(1, "相談済", "state-done");
  } catch (err) {
    showInlineError("#advisorErrorBanner", `AI相談の送信に失敗しました: ${err.message}。再度お試しください。`);
  } finally {
    setBtnLoading("#btnAskAdvisor", false);
  }
}

/* ==================== TAB 3: 結果を評価 (MuJoCo Evaluation & Retrain) ==================== */

function populateReports(reports) {
  state.reportsList = reports || [];
  const sel = qs("#reportSelect");
  const prevVal = sel.value;
  sel.innerHTML = state.reportsList
    .map(
      (r) =>
        `<option value="${escapeAttr(r.run_id)}">${escapeHtml(r.run_id)} | ${escapeHtml(r.method)} | 成功率 ${((r.success_rate || 0) * 100).toFixed(1)}%</option>`
    )
    .join("");

  if (prevVal && Array.from(sel.options).some((o) => o.value === prevVal)) {
    sel.value = prevVal;
  }

  const tbody = qs("#reportsComparisonBody");
  tbody.innerHTML = state.reportsList
    .map((r) => {
      const isCur = state.currentReport && state.currentReport.run_id === r.run_id ? "selected-row" : "";
      const sr = ((r.success_rate || 0) * 100).toFixed(1);
      return `
      <tr class="${isCur}" data-runid="${escapeAttr(r.run_id)}" tabindex="0" role="button" aria-label="Run ${escapeAttr(r.run_id)}">
        <td><code>${escapeHtml(r.run_id)}</code></td>
        <td><strong>${escapeHtml(r.method)}</strong></td>
        <td>${escapeHtml(r.mocap_id)}</td>
        <td>${r.epochs || "-"}</td>
        <td><span class="pill pill-ok">${sr}%</span></td>
        <td>${r.tracking_score ?? "-"}/100</td>
        <td>${r.smoothness_score ?? "-"}/100</td>
        <td>${Number(r.train_loss || 0).toFixed(4)} / ${Number(r.val_loss || 0).toFixed(4)}</td>
        <td>${r.overfitting_detected ? '<span class="pill pill-warn">⚠ 要注意</span>' : '<span class="pill pill-ok">✔ 正常</span>'}</td>
        <td><button class="btn btn-secondary btn-xs">表示</button></td>
      </tr>`;
    })
    .join("");

  tbody.querySelectorAll("tr").forEach((tr) => {
    const handleLoad = async () => {
      const runId = tr.dataset.runid;
      if (!runId) return;
      qs("#reportSelect").value = runId;
      try {
        const rep = await apiJson(`/api/runs/${runId}/report`);
        await renderReport(rep);
        showToast(`レポート ${runId} を表示しました`);
      } catch (err) {
        showToast(`読込エラー: ${err.message}`);
      }
    };
    tr.addEventListener("click", handleLoad);
    tr.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        handleLoad();
      }
    });
  });
}

async function renderReport(report) {
  if (!report) return;
  state.currentReport = report;
  const history = report.metrics_history || [];
  const lastM = history[history.length - 1] || {};
  const d = report.diagnostics || {};

  // Update Run Meta Context Header
  qs("#evalRunId").textContent = report.run_id;
  qs("#evalMethod").textContent = report.config.method;
  qs("#evalMocap").textContent = report.config.mocap_id;
  qs("#evalCreatedAt").textContent = report.created_at ? new Date(report.created_at).toLocaleString("ja-JP") : "-";

  // Update KPI Score Grid
  qs("#evalKpiSuccess").textContent = `${((lastM.task_success_rate || 0) * 100).toFixed(1)}%`;
  qs("#evalKpiTracking").textContent = `${d.tracking_score ?? "-"} 点`;
  qs("#evalKpiSmooth").textContent = `${d.smoothness_score ?? "-"} 点`;
  qs("#evalKpiOverfit").textContent = d.overfitting_detected ? "⚠ 検出あり" : "✔ 正常";
  qs("#evalKpiOverfitGap").textContent = `乖離幅: ${d.overfitting_gap || "0.0000"}`;

  // Update Diagnostics Details
  qs("#diagnosticsDetails").innerHTML = `
    <p style="margin: 0 0 8px; font-weight: 600; font-size: 1rem; color: var(--text-main);">
      総合判定: ${escapeHtml(d.summary_ja || "正常に完了しました。")}
    </p>
    <div style="margin-top: 10px;">
      <span style="font-weight: 600; font-size: 0.875rem; color: var(--text-muted);">AIからの推奨改善アクション:</span>
      <ul style="margin: 6px 0 0; padding-left: 20px;">
        ${(d.recommended_adjustments || []).map((a) => `<li style="margin-bottom: 4px;">${escapeHtml(a)}</li>`).join("")}
      </ul>
    </div>
  `;

  // Video
  const vid = qs("#mujocoVideo");
  const vidUrl = `${report.video_local_url}?t=${Date.now()}`;
  vid.src = vidUrl;
  qs("#btnDownloadVideo").href = report.video_local_url;

  // Light Plotly layout with 13-14px font
  const epochs = history.map((m) => m.epoch);
  const lightLayout = (xTitle, yTitle, extra = {}) => ({
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#f8fafc",
    font: { color: "#334155", size: 13 },
    margin: { l: 52, r: 20, t: 20, b: 46 },
    xaxis: { title: xTitle, gridcolor: "#e2e8f0", tickfont: { size: 12, color: "#64748b" }, titlefont: { size: 13, color: "#1e293b" } },
    yaxis: { title: yTitle, gridcolor: "#e2e8f0", tickfont: { size: 12, color: "#64748b" }, titlefont: { size: 13, color: "#1e293b" }, ...extra },
    legend: { orientation: "h", y: 1.14, font: { size: 12, color: "#475569" } },
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
    lightLayout("Epoch (エポック)", "Loss (損失)"),
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
    lightLayout("Epoch (エポック)", "成功率 (%)", { range: [0, 105] }),
    { responsive: true, displayModeBar: false }
  );

  // 3. Episode Return Curve
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
        name: "エピソード累積報酬",
        x: epochs,
        y: returns,
        line: { color: "rgba(96,165,250,0.55)", width: 1.5 },
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: "移動平均リターン",
        x: epochs,
        y: movAvg,
        line: { color: "#ef4444", width: 3 },
      },
      {
        type: "scatter",
        mode: "lines",
        name: `最高リターン (${peak.toFixed(1)})`,
        x: epochs,
        y: epochs.map(() => peak),
        line: { color: "#f59e0b", width: 1.5, dash: "dash" },
      },
    ],
    lightLayout("Epoch (エポック)", "累積報酬"),
    { responsive: true, displayModeBar: false }
  );

  // Highlight active row in comparison table
  populateReports(state.reportsList);

  // Update Stepper state
  updateStepStatus(3, "確認中", "state-current");

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
        name: "人間のお手本 MoCap",
        x: steps,
        y: refVals,
        line: { color: "#10b981", width: 2.5, dash: "dash" },
      },
      {
        type: "scatter",
        mode: "lines",
        name: "ロボット実測軌道",
        x: steps,
        y: polVals,
        line: { color: "#f97316", width: 2.5 },
      },
    ],
    {
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#f8fafc",
      font: { color: "#334155", size: 13 },
      margin: { l: 52, r: 16, t: 20, b: 46 },
      xaxis: { title: "シミュレーションステップ", gridcolor: "#e2e8f0", tickfont: { size: 12, color: "#64748b" }, titlefont: { size: 13, color: "#1e293b" } },
      yaxis: { title: "関節角度 (rad)", gridcolor: "#e2e8f0", tickfont: { size: 12, color: "#64748b" }, titlefont: { size: 13, color: "#1e293b" } },
      legend: { orientation: "h", y: 1.14, font: { size: 12, color: "#475569" } },
    },
    { responsive: true, displayModeBar: false }
  );
}

/* ==================== Three.js 3D Viewer ==================== */

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
  const h = container.clientHeight || 340;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x060a12);

  const camera = new THREE.PerspectiveCamera(44, w / h, 0.1, 100);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(w, h);
  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  const grid = new THREE.GridHelper(16, 32, 0x2563eb, 0x1e293b);
  scene.add(grid);
  scene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.7);
  dirLight.position.set(3, 6, 4);
  scene.add(dirLight);

  const allNodes = Array.from(new Set(BONE_LINKS.flat()));

  // Policy Skeleton (Orange)
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

  // Reference Human Skeleton (Green Ghost)
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
  const h = container.clientHeight || 340;
  state.threeCtx.camera.aspect = w / h;
  state.threeCtx.camera.updateProjectionMatrix();
  state.threeCtx.renderer.setSize(w, h);
}

/* ==================== Initialization & Event Binding ==================== */

async function bootstrap() {
  initNavigation();

  // Support ?tab=papers | train | eval from URL query
  const urlParams = new URLSearchParams(window.location.search);
  const initialTab = urlParams.get("tab") || "papers";
  switchTab(initialTab);

  // Check backend health
  try {
    const health = await apiJson("/api/health");
    if (health.status === "ok") {
      qs("#healthDot").className = "status-dot status-ok";
      qs("#healthStatus").textContent = "正常稼働中 (HTTP 200)";
    }
  } catch (err) {
    qs("#healthDot").className = "status-dot status-warn";
    qs("#healthStatus").textContent = "オフライン";
  }

  // Load Dashboard Data
  try {
    const init = await apiJson("/api/dashboard-init");
    applyPaperSnapshot(init.text_mining, init.papers);
    populateMocaps(init.mocaps);
    populateReports(init.reports);
    if (init.latest_report) {
      await renderReport(init.latest_report);
    }
  } catch (err) {
    console.error("Dashboard init error:", err);
    showToast("データの読み込みに失敗しました。再読み込みしてください。");
  }

  // Initialize Year Filter Pills & Trend Toggle
  initYearFilterPills();
  initTrendPlotToggle();

  // --- Tab 1 Events ---
  qs("#chkSim2RealOnly").addEventListener("change", (e) => {
    state.sim2RealOnly = e.target.checked;
    renderDensityMap(state.snapshot);
    renderPapersTable();
    updateActiveFiltersBar();
  });

  qs("#paperSearchInput").addEventListener("input", (e) => {
    state.searchQuery = e.target.value.trim();
    renderDensityMap(state.snapshot);
    renderPapersTable();
    updateActiveFiltersBar();
  });

  const clearSearchHandler = () => {
    qs("#paperSearchInput").value = "";
    state.searchQuery = "";
    renderDensityMap(state.snapshot);
    renderPapersTable();
    updateActiveFiltersBar();
  };
  qs("#btnClearPaperSearch").addEventListener("click", clearSearchHandler);

  const resetAllFiltersHandler = () => {
    state.selectedYear = "all";
    state.activeClusterFilter = null;
    state.sim2RealOnly = false;
    state.searchQuery = "";
    if (qs("#chkSim2RealOnly")) qs("#chkSim2RealOnly").checked = false;
    if (qs("#paperSearchInput")) qs("#paperSearchInput").value = "";
    qsa(".year-pill").forEach((btn) => btn.classList.toggle("active", btn.dataset.year === "all"));
    renderClusterFilterPills(state.snapshot);
    renderDensityMap(state.snapshot);
    renderPapersTable();
    updateActiveFiltersBar();
    showToast("絞り込み条件をすべて解除しました");
  };
  qs("#btnResetMapFilter").addEventListener("click", resetAllFiltersHandler);
  qs("#btnResetAllFilters").addEventListener("click", resetAllFiltersHandler);
  qs("#btnEmptyReset").addEventListener("click", resetAllFiltersHandler);

  qs("#btnScrape").addEventListener("click", async () => {
    showInlineError("#scrapeErrorBanner", "");
    const kws = qs("#keywordsInput")
      .value.split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    const maxPer = Number(qs("#maxPerKw").value || 6);

    if (kws.length === 0) {
      showInlineError("#scrapeErrorBanner", "検索キーワードを1つ以上入力してください。");
      return;
    }

    setBtnLoading("#btnScrape", true, "更新中...");
    const banner = qs("#scrapeJobBanner");
    banner.classList.remove("hidden");

    try {
      const { job_id } = await apiJson("/api/papers/scrape", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keywords: kws, max_per_keyword: maxPer }),
      });
      pollJob(
        job_id,
        banner,
        qs("#scrapeJobText"),
        qs("#scrapeProgressFill"),
        async () => {
          setBtnLoading("#btnScrape", false);
          const latest = await apiJson("/api/papers/latest");
          applyPaperSnapshot(latest.text_mining, latest.papers);
          showToast("論文データの収集・テキストマイニングが完了しました");
        },
        (errMsg) => {
          setBtnLoading("#btnScrape", false);
          showInlineError("#scrapeErrorBanner", `論文更新エラー: ${errMsg}`);
        }
      );
    } catch (err) {
      setBtnLoading("#btnScrape", false);
      banner.classList.add("hidden");
      showInlineError("#scrapeErrorBanner", `リクエスト送信エラー: ${err.message}`);
    }
  });

  // --- Tab 2 Events ---
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

  // Algorithm Radios
  qsa('input[name="rlMethodRadio"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      selectAlgoMethod(e.target.value, false);
    });
  });

  qs("#mocapSelect").addEventListener("change", updatePreflightPreview);
  qs("#epochsInput").addEventListener("input", updatePreflightPreview);
  qs("#smoothWeightInput").addEventListener("input", updatePreflightPreview);

  qs("#btnUploadMocap").addEventListener("click", async () => {
    const file = qs("#mocapFile").files[0];
    if (!file) return showToast("CSV / NPZ / BVH ファイルを選択してください");
    setBtnLoading("#btnUploadMocap", true, "送信中...");
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await apiJson("/api/mocap/upload", { method: "POST", body: fd });
      const init = await apiJson("/api/dashboard-init");
      populateMocaps(init.mocaps);
      qs("#mocapSelect").value = res.mocap_id;
      updatePreflightPreview();
      showToast(`モーションデータをS3へ保存しました: ${res.name}`);
    } catch (err) {
      showToast(`アップロードエラー: ${err.message}`);
    } finally {
      setBtnLoading("#btnUploadMocap", false);
    }
  });

  qs("#btnUploadXml").addEventListener("click", async () => {
    const file = qs("#robotXmlFile").files[0];
    if (!file) return showToast("MJCF (.xml) ファイルを選択してください");
    setBtnLoading("#btnUploadXml", true, "登録中...");
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await apiJson("/api/robot-xml/upload", { method: "POST", body: fd });
      qs("#robotXmlName").value = res.robot_xml_name;
      qs("#currentXmlLabel").textContent = `現在: ${res.robot_xml_name} (S3登録済)`;
      showToast(`カスタムロボットXMLをS3へ保存しました: ${res.s3_key}`);
    } catch (err) {
      showToast(`XML登録エラー: ${err.message}`);
    } finally {
      setBtnLoading("#btnUploadXml", false);
    }
  });

  qs("#btnStartTrain").addEventListener("click", async () => {
    showInlineError("#trainErrorBanner", "");
    const epochs = Number(qs("#epochsInput").value);
    if (isNaN(epochs) || epochs < 5 || epochs > 120) {
      showInlineError("#trainErrorBanner", "エポック数は 5 から 120 の範囲で指定してください。");
      qs("#epochsInput").focus();
      return;
    }

    setBtnLoading("#btnStartTrain", true, "学習実行中...");
    const banner = qs("#trainJobBanner");
    banner.classList.remove("hidden");
    updateStepStatus(2, "実行中", "state-current");

    const payload = {
      method: qs("#methodSelect").value,
      mocap_id: qs("#mocapSelect").value,
      epochs,
      learning_rate: Number(qs("#lrInput").value),
      bc_weight: Number(qs("#bcWeightInput").value),
      smoothness_weight: Number(qs("#smoothWeightInput").value),
      reward_tracking_weight: Number(qs("#trackWeightInput").value),
      entropy_coef: Number(qs("#entropyInput").value),
      robot_xml_name: qs("#robotXmlName").value,
    };

    try {
      const { job_id } = await apiJson("/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      pollJob(
        job_id,
        banner,
        qs("#trainJobText"),
        qs("#trainProgressFill"),
        async (report) => {
          setBtnLoading("#btnStartTrain", false);
          updateStepStatus(2, "完了", "state-done");
          const init = await apiJson("/api/dashboard-init");
          populateReports(init.reports);
          qs("#reportSelect").value = report.run_id;
          await renderReport(report);
          switchTab("eval", "#sectionStep3");
          showToast(`学習が完了しました (${report.run_id})！結果を表示します`);
        },
        (errMsg) => {
          setBtnLoading("#btnStartTrain", false);
          updateStepStatus(2, "エラー", "");
          showInlineError("#trainErrorBanner", `学習エラー: ${errMsg}`);
        }
      );
    } catch (err) {
      setBtnLoading("#btnStartTrain", false);
      banner.classList.add("hidden");
      updateStepStatus(2, "エラー", "");
      showInlineError("#trainErrorBanner", `開始リクエスト失敗: ${err.message}`);
    }
  });

  // --- Tab 3 Events ---
  qs("#btnLoadReport").addEventListener("click", async () => {
    const runId = qs("#reportSelect").value;
    if (!runId) return;
    setBtnLoading("#btnLoadReport", true, "読込中...");
    try {
      const report = await apiJson(`/api/runs/${runId}/report`);
      await renderReport(report);
      showToast(`レポート ${runId} を表示しました`);
    } catch (err) {
      showToast(`読込エラー: ${err.message}`);
    } finally {
      setBtnLoading("#btnLoadReport", false);
    }
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

  // Step 4: AI Retrain Plan Creation (Step 1 of 2: Does NOT train yet!)
  qs("#btnCreateRetrainPlan").addEventListener("click", async () => {
    showInlineError("#retrainErrorBanner", "");
    const instr = qs("#retrainInstruction").value.trim();
    if (!instr) {
      showInlineError("#retrainErrorBanner", "評価コメントまたは指示テンプレートを選択してください。");
      return;
    }

    setBtnLoading("#btnCreateRetrainPlan", true, "提案作成中...");
    const curCfg = state.currentReport ? state.currentReport.config : {};

    try {
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
          auto_start: false, // Pure proposal generation!
        }),
      });

      state.pendingRetrainConfig = res.updated_config;
      qs("#retrainEngineBadge").textContent = res.engine;
      qs("#retrainExplanation").innerHTML = renderMarkdownLite(res.reply);

      // Render Diff Table
      const paramLabels = {
        method: "学習アルゴリズム",
        epochs: "エポック数",
        learning_rate: "学習率",
        bc_weight: "模倣正則化重み",
        smoothness_weight: "滑らかさ重み",
        reward_tracking_weight: "追随報酬重み",
        entropy_coef: "探索エントロピー",
      };

      const diffRows = [];
      const prev = res.previous_config || {};
      const next = res.updated_config || {};
      Object.keys(next).forEach((k) => {
        if (paramLabels[k] && prev[k] !== undefined) {
          const isChanged = prev[k] !== next[k];
          diffRows.push(`
            <tr style="${isChanged ? 'background: rgba(37,99,235,0.15);' : ''}">
              <td><strong>${paramLabels[k]}</strong></td>
              <td>${prev[k]}</td>
              <td><strong style="color: ${isChanged ? '#34d399' : 'inherit'};">${next[k]}</strong></td>
              <td>${isChanged ? '変更あり' : '維持'}</td>
            </tr>
          `);
        }
      });
      qs("#retrainDiffTableBody").innerHTML = diffRows.join("");

      qs("#retrainConfirmBox").classList.remove("hidden");
      qs("#retrainConfirmBox").scrollIntoView({ behavior: "smooth", block: "start" });
      updateStepStatus(4, "変更案作成済", "state-current");
      showToast("AIによるパラメータ変更案を作成しました。確認して実行してください。");
    } catch (err) {
      showInlineError("#retrainErrorBanner", `提案作成エラー: ${err.message}`);
    } finally {
      setBtnLoading("#btnCreateRetrainPlan", false);
    }
  });

  qs("#btnCancelRetrain").addEventListener("click", () => {
    qs("#retrainConfirmBox").classList.add("hidden");
    state.pendingRetrainConfig = null;
    showToast("再学習をキャンセルしました");
  });

  // Step 4: Execute Confirmed Retrain (Step 2 of 2: Actually runs training!)
  qs("#btnExecuteRetrainConfirmed").addEventListener("click", async () => {
    if (!state.pendingRetrainConfig) return;
    const banner = qs("#evalTrainJobBanner");
    banner.classList.remove("hidden");
    setBtnLoading("#btnExecuteRetrainConfirmed", true, "再学習中...");
    updateStepStatus(4, "再学習中", "state-current");

    try {
      const { job_id } = await apiJson("/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.pendingRetrainConfig),
      });

      pollJob(
        job_id,
        banner,
        qs("#evalTrainJobText"),
        qs("#evalTrainProgressFill"),
        async (report) => {
          setBtnLoading("#btnExecuteRetrainConfirmed", false);
          qs("#retrainConfirmBox").classList.add("hidden");
          state.pendingRetrainConfig = null;
          updateStepStatus(4, "再学習完了", "state-done");
          const init = await apiJson("/api/dashboard-init");
          populateReports(init.reports);
          qs("#reportSelect").value = report.run_id;
          await renderReport(report);
          showToast(`再学習が完了しました (${report.run_id})！新結果を表示します`);
        },
        (errMsg) => {
          setBtnLoading("#btnExecuteRetrainConfirmed", false);
          updateStepStatus(4, "再学習エラー", "");
          showInlineError("#retrainErrorBanner", `再学習エラー: ${errMsg}`);
        }
      );
    } catch (err) {
      setBtnLoading("#btnExecuteRetrainConfirmed", false);
      banner.classList.add("hidden");
      showInlineError("#retrainErrorBanner", `再学習開始失敗: ${err.message}`);
    }
  });

  window.addEventListener("resize", resizeThree);
}

document.addEventListener("DOMContentLoaded", bootstrap);
