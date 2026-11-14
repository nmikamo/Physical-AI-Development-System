(() => {
  "use strict";

  /* =========================================================
   * 1. Immutable Constants & Pure Helpers
   * ========================================================= */
  const PAGE_META = Object.freeze({
    dashboard: { title: "Dashboard", breadcrumb: "Dashboard" },
    papers: { title: "Papers & Text Mining", breadcrumb: "Papers & Mining" },
    train: { title: "AI Advisor & Training", breadcrumb: "Training" },
    eval: { title: "Evaluation & 3D", breadcrumb: "Evaluation & 3D" },
    system: { title: "System & AWS", breadcrumb: "System & AWS" },
  });

  const COMMAND_ITEMS = Object.freeze([
    { id: "nav-dashboard", label: "Dashboard (Overview) へ移動", group: "Navigation", action: { type: "nav", page: "dashboard" } },
    { id: "nav-papers", label: "Papers & Text Mining へ移動", group: "Navigation", action: { type: "nav", page: "papers" } },
    { id: "nav-train", label: "AI Advisor & Training へ移動", group: "Navigation", action: { type: "nav", page: "train" } },
    { id: "nav-eval", label: "Evaluation & 3D へ移動", group: "Navigation", action: { type: "nav", page: "eval" } },
    { id: "nav-system", label: "System & AWS へ移動", group: "Navigation", action: { type: "nav", page: "system" } },
    { id: "act-refresh", label: "バックエンドデータを再読み込み", group: "Actions", action: { type: "refresh" } },
    { id: "act-theme", label: "Light / Dark テーマを切り替え", group: "Actions", action: { type: "theme" } },
    { id: "act-rail", label: "サイドバーのコンパクト表示を切り替え", group: "Actions", action: { type: "rail" } },
  ]);

  const SKELETON_EDGES = Object.freeze([
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
  ]);

  const readThemeTokens = () => {
    const styles = getComputedStyle(document.documentElement);
    const get = (name, fallback) => styles.getPropertyValue(name).trim() || fallback;
    return Object.freeze({
      primary: get("--primary", "#1abb9c"),
      blue: get("--blue", "#066fd1"),
      azure: get("--azure", "#4299e1"),
      green: get("--green", "#2fb344"),
      yellow: get("--yellow", "#f59f00"),
      purple: get("--purple", "#ae3ec9"),
      red: get("--red", "#d63939"),
      text: get("--text", "#1e2633"),
      textMuted: get("--text-muted", "#7e8896"),
      border: get("--border-color-light", "#eff0f3"),
      surface: get("--bg-surface", "#ffffff"),
    });
  };

  const escapeHtml = (str) =>
    String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");

  const filterPoints = (points, { simOnly, cluster, year, search }) => {
    const q = (search || "").trim().toLowerCase();
    return (points || []).filter((p) => {
      if (simOnly && !p.sim_to_real) return false;
      if (cluster !== "all" && p.cluster_label !== cluster) return false;
      const pYear = String(p.published_at || p.month_key || "").slice(0, 4);
      if (year !== "all" && pYear !== year) return false;
      if (q) {
        const hay = `${p.paper_id} ${p.title} ${p.cluster_label} ${(p.keywords_matched || []).join(" ")}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  };

  /* =========================================================
   * 2. Pure ECharts 6 Option Builders
   * ========================================================= */
  const buildTrendOption = (snapshot, mode, t) => {
    const yearly = snapshot?.yearly_counts || [];
    const monthly = snapshot?.monthly_counts || [];
    const isYearly = mode === "yearly" && yearly.length > 0;
    const labels = isYearly ? yearly.map((y) => y.year) : monthly.map((m) => m.month);
    const counts = isYearly ? yearly.map((y) => y.count) : monthly.map((m) => m.count);
    const simCounts = isYearly ? yearly.map((y) => y.sim_count) : null;

    const series = [
      {
        name: "Total Papers",
        type: "bar",
        barMaxWidth: 26,
        data: counts,
        itemStyle: { color: t.azure, borderRadius: [4, 4, 0, 0] },
      },
    ];
    if (simCounts) {
      series.push({
        name: "Sim-to-Real",
        type: "line",
        smooth: true,
        data: simCounts,
        itemStyle: { color: t.primary },
        lineStyle: { width: 2.5 },
        areaStyle: { color: t.primary, opacity: 0.1 },
      });
    }

    return {
      tooltip: { trigger: "axis" },
      legend: { bottom: 0, textStyle: { color: t.textMuted, fontSize: 11 } },
      grid: { top: 20, left: 42, right: 20, bottom: 36 },
      xAxis: { type: "category", data: labels, axisLabel: { color: t.textMuted, fontSize: 11 } },
      yAxis: {
        type: "value",
        splitLine: { lineStyle: { color: t.border, type: "dashed" } },
        axisLabel: { color: t.textMuted, fontSize: 11 },
      },
      series,
    };
  };

  const buildAlgoPieOption = (reports, t) => {
    const counts = { DAPG: 0, RLPD: 0, AWAC: 0 };
    (reports || []).forEach((r) => {
      if (counts[r.method] !== undefined) counts[r.method] += 1;
    });
    const total = counts.DAPG + counts.RLPD + counts.AWAC;
    const data =
      total > 0
        ? [
            { value: counts.DAPG, name: "DAPG", itemStyle: { color: t.primary } },
            { value: counts.RLPD, name: "RLPD", itemStyle: { color: t.blue } },
            { value: counts.AWAC, name: "AWAC", itemStyle: { color: t.purple } },
          ].filter((d) => d.value > 0)
        : [{ value: 1, name: "No Runs Yet", itemStyle: { color: t.border } }];

    return {
      tooltip: { trigger: "item", formatter: "{b}: {c} ({d}%)" },
      legend: { bottom: 0, textStyle: { color: t.textMuted, fontSize: 11 } },
      series: [
        {
          name: "Algorithm",
          type: "pie",
          radius: ["52%", "74%"],
          center: ["50%", "44%"],
          label: { show: false },
          data,
        },
      ],
    };
  };

  const buildMiningMapOption = (points, t) => {
    const palette = [t.primary, t.blue, t.purple, t.yellow, t.green, t.red];
    const groups = new Map();
    points.forEach((p) => {
      const key = p.cluster_label || "Other";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push({
        value: [p.x, p.y],
        paper: p,
        symbolSize: p.sim_to_real ? 15 : 10,
      });
    });

    const series = Array.from(groups.entries()).map(([label, data], idx) => ({
      name: label,
      type: "scatter",
      data,
      itemStyle: {
        color: palette[idx % palette.length],
        borderColor: t.surface,
        borderWidth: 1.5,
      },
    }));

    return {
      tooltip: {
        trigger: "item",
        formatter: (params) => {
          const p = params.data?.paper;
          if (!p) return "";
          return `<strong>${escapeHtml(p.paper_id)}</strong><br/>${escapeHtml(p.title)}<br/><span style="opacity:.8">${escapeHtml(p.cluster_label)} (${escapeHtml(p.month_key)})</span>`;
        },
      },
      legend: { bottom: 0, type: "scroll", textStyle: { color: t.textMuted, fontSize: 11 } },
      grid: { top: 20, left: 42, right: 20, bottom: 42 },
      xAxis: {
        type: "value",
        splitLine: { lineStyle: { color: t.border, type: "dashed" } },
        axisLabel: { color: t.textMuted, fontSize: 11 },
      },
      yAxis: {
        type: "value",
        splitLine: { lineStyle: { color: t.border, type: "dashed" } },
        axisLabel: { color: t.textMuted, fontSize: 11 },
      },
      series,
    };
  };

  const buildTopTermsOption = (snapshot, t) => {
    const terms = (snapshot?.top_terms || []).slice(0, 10).reverse();
    return {
      tooltip: { trigger: "axis" },
      grid: { top: 12, left: 115, right: 24, bottom: 24 },
      xAxis: {
        type: "value",
        splitLine: { lineStyle: { color: t.border, type: "dashed" } },
        axisLabel: { color: t.textMuted, fontSize: 11 },
      },
      yAxis: {
        type: "category",
        data: terms.map((item) => item.term),
        axisLabel: { color: t.textMuted, fontSize: 11 },
      },
      series: [
        {
          type: "bar",
          data: terms.map((item) => Number(item.score.toFixed(3))),
          itemStyle: { color: t.primary, borderRadius: [0, 4, 4, 0] },
        },
      ],
    };
  };

  const buildEvalMetricsOption = (report, t) => {
    const hist = report?.metrics_history || [];
    const epochs = hist.map((m) => `Ep ${m.epoch}`);
    const trainLoss = hist.map((m) => Number(m.train_loss.toFixed(4)));
    const valLoss = hist.map((m) => Number(m.val_loss.toFixed(4)));
    const success = hist.map((m) => Number((m.task_success_rate * 100).toFixed(1)));

    return {
      tooltip: { trigger: "axis" },
      legend: { data: ["Train Loss", "Val Loss", "Task Success (%)"], bottom: 0, textStyle: { color: t.textMuted, fontSize: 11 } },
      grid: { top: 24, left: 48, right: 48, bottom: 38 },
      xAxis: { type: "category", data: epochs, axisLabel: { color: t.textMuted, fontSize: 11 } },
      yAxis: [
        { type: "value", name: "Loss", splitLine: { lineStyle: { color: t.border, type: "dashed" } }, axisLabel: { color: t.textMuted, fontSize: 11 } },
        { type: "value", name: "Success %", min: 0, max: 100, splitLine: { show: false }, axisLabel: { color: t.textMuted, fontSize: 11 } },
      ],
      series: [
        { name: "Train Loss", type: "line", smooth: true, data: trainLoss, itemStyle: { color: t.blue } },
        { name: "Val Loss", type: "line", smooth: true, data: valLoss, itemStyle: { color: t.yellow } },
        { name: "Task Success (%)", type: "line", yAxisIndex: 1, smooth: true, data: success, itemStyle: { color: t.primary }, areaStyle: { color: t.primary, opacity: 0.1 } },
      ],
    };
  };

  /* =========================================================
   * 3. API Client & Polling Helper
   * ========================================================= */
  const fetchJson = async (url, options = {}) => {
    const res = await fetch(url, options);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    return data;
  };

  const pollJob = (jobId, onProgress) =>
    new Promise((resolve, reject) => {
      const tick = async () => {
        try {
          const job = await fetchJson(`/api/jobs/${encodeURIComponent(jobId)}`);
          onProgress(job);
          if (job.status === "completed") {
            resolve(job);
          } else if (job.status === "failed") {
            reject(new Error(job.message || "Job failed"));
          } else {
            setTimeout(tick, 1000);
          }
        } catch (err) {
          reject(err);
        }
      };
      tick();
    });

  /* =========================================================
   * 4. Application State & Controllers
   * ========================================================= */
  const state = {
    page: "dashboard",
    theme: localStorage.getItem("pai_theme") || "light",
    dashTrendMode: "yearly",
    papersTrendMode: "yearly",
    paperFilters: { simOnly: false, cluster: "all", year: "all", search: "" },
    selectedAlgo: "DAPG",
    robotXmlName: "humanoid_default.xml",
    lastSuggestion: null,
    snapshot: null,
    papers: [],
    mocaps: [],
    reports: [],
    currentReport: null,
    currentFrames: [],
    frameIndex: 0,
    threePlaying: true,
    threeCtx: null,
    charts: new Map(),
  };

  const showToast = (message, variant = "success") => {
    const host = document.getElementById("toastHost");
    if (!host) return;
    const el = document.createElement("div");
    el.className = `toast toast-${variant}`;
    el.textContent = message;
    host.appendChild(el);
    requestAnimationFrame(() => el.classList.add("show"));
    setTimeout(() => {
      el.classList.remove("show");
      setTimeout(() => el.remove(), 220);
    }, 3200);
  };

  const ensureChart = (domId, optionBuilder) => {
    if (typeof window.echarts === "undefined") return null;
    const el = document.getElementById(domId);
    if (!el || el.offsetParent === null) return null;
    let instance = state.charts.get(domId);
    if (!instance) {
      instance = window.echarts.init(el);
      state.charts.set(domId, instance);
      if (domId === "chartMiningMap") {
        instance.on("click", (params) => {
          if (params?.data?.paper) renderPaperDetail(params.data.paper);
        });
      }
    }
    instance.setOption(optionBuilder(readThemeTokens()), true);
    instance.resize();
    return instance;
  };

  const renderVisibleCharts = () => {
    const t = readThemeTokens();
    if (state.page === "dashboard") {
      ensureChart("chartDashActivity", () => buildTrendOption(state.snapshot, state.dashTrendMode, t));
      ensureChart("chartDashAlgoPie", () => buildAlgoPieOption(state.reports, t));
    } else if (state.page === "papers") {
      const pts = filterPoints(state.snapshot?.points || [], state.paperFilters);
      ensureChart("chartMiningMap", () => buildMiningMapOption(pts, t));
      ensureChart("chartPapersTrend", () => buildTrendOption(state.snapshot, state.papersTrendMode, t));
      ensureChart("chartTopTerms", () => buildTopTermsOption(state.snapshot, t));
    } else if (state.page === "eval") {
      ensureChart("chartEvalMetrics", () => buildEvalMetricsOption(state.currentReport, t));
    }
  };

  const renderPaperDetail = (paper) => {
    if (!paper) return;
    document.getElementById("detailPaperId").textContent = paper.paper_id;
    document.getElementById("detailPaperTitle").textContent = paper.title;
    document.getElementById("detailPaperCluster").textContent = paper.cluster_label || (paper.sim_to_real ? "Sim2Real" : "Simulation");
    document.getElementById("detailPaperYear").textContent = String(paper.published_at || paper.month_key || "-");
    document.getElementById("detailPaperSummary").textContent = paper.summary_ja || paper.abstract || "";
    const link = document.getElementById("detailPaperLink");
    if (link) {
      if (paper.url) {
        link.href = paper.url;
        link.classList.remove("hidden");
      } else {
        link.classList.add("hidden");
      }
    }
  };

  const renderPapersView = () => {
    const points = state.snapshot?.points || [];
    const filtered = filterPoints(points, state.paperFilters);
    const countHint = document.getElementById("papersCountHint");
    if (countHint) countHint.textContent = `(${filtered.length} / ${points.length})`;

    const tbody = document.getElementById("papersTbody");
    if (tbody) {
      tbody.innerHTML = filtered
        .slice(0, 120)
        .map(
          (p) => `
        <tr data-paper-id="${escapeHtml(p.paper_id)}" style="cursor:pointer">
          <td class="cell-mono">${escapeHtml(p.paper_id)}</td>
          <td class="cell-strong">${escapeHtml(p.title)}</td>
          <td><span class="pill-badge pill-blue">${escapeHtml(p.cluster_label || "-")}</span></td>
          <td class="cell-mono">${escapeHtml(p.published_at || p.month_key)}</td>
          <td>${p.sim_to_real ? '<span class="status status-green">Sim2Real</span>' : '<span class="status status-yellow">Sim</span>'}</td>
        </tr>`
        )
        .join("");

      tbody.querySelectorAll("tr[data-paper-id]").forEach((row) => {
        row.addEventListener("click", () => {
          const found = points.find((pt) => pt.paper_id === row.dataset.paperId);
          if (found) renderPaperDetail(found);
        });
      });
    }

    if (filtered.length > 0 && document.getElementById("detailPaperId").textContent === "-") {
      renderPaperDetail(filtered[0]);
    }
    renderVisibleCharts();
  };

  const renderFilterPills = () => {
    const clusters = state.snapshot?.clusters || [];
    const clusterHost = document.getElementById("clusterFilterPills");
    if (clusterHost) {
      const items = ["all", ...clusters.map((c) => c.label)];
      clusterHost.innerHTML = items
        .map(
          (lbl) =>
            `<button type="button" class="filter-chip ${state.paperFilters.cluster === lbl ? "active" : ""}" data-cluster="${escapeHtml(lbl)}">${lbl === "all" ? "全クラスタ" : escapeHtml(lbl)}</button>`
        )
        .join("");
      clusterHost.querySelectorAll("[data-cluster]").forEach((btn) => {
        btn.addEventListener("click", () => {
          state.paperFilters.cluster = btn.dataset.cluster;
          renderFilterPills();
          renderPapersView();
        });
      });
    }

    const years = (state.snapshot?.yearly_counts || []).map((y) => y.year);
    const yearHost = document.getElementById("yearFilterPills");
    if (yearHost) {
      const items = ["all", ...years];
      yearHost.innerHTML = items
        .map(
          (yr) =>
            `<button type="button" class="filter-chip ${state.paperFilters.year === yr ? "active" : ""}" data-year="${escapeHtml(yr)}">${yr === "all" ? "全年度" : escapeHtml(yr)}</button>`
        )
        .join("");
      yearHost.querySelectorAll("[data-year]").forEach((btn) => {
        btn.addEventListener("click", () => {
          state.paperFilters.year = btn.dataset.year;
          renderFilterPills();
          renderPapersView();
        });
      });
    }
  };

  const renderDashboardView = () => {
    const snap = state.snapshot;
    const reports = state.reports || [];
    const latest = state.currentReport;

    const totalPapers = snap?.total_papers ?? state.papers.length;
    const simCount = snap?.sim_to_real_count ?? 0;
    const simRatio = totalPapers > 0 ? `${((simCount / totalPapers) * 100).toFixed(1)}%` : "0%";

    document.getElementById("dashStatPapers").textContent = String(totalPapers);
    document.getElementById("dashStatPeriod").textContent = snap?.period_label || "2020–2026";
    document.getElementById("dashStatSim2Real").textContent = String(simCount);
    document.getElementById("dashStatSimRatio").textContent = simRatio;
    document.getElementById("dashStatRuns").textContent = String(reports.length);
    document.getElementById("dashStatMocaps").textContent = String(state.mocaps.length);
    document.getElementById("dashChartStat").textContent = `${totalPapers} Papers`;

    if (latest && latest.metrics_history?.length) {
      const last = latest.metrics_history[latest.metrics_history.length - 1];
      document.getElementById("dashStatSuccess").textContent = `${(last.task_success_rate * 100).toFixed(1)}%`;
      document.getElementById("dashStatSuccessMethod").textContent = latest.config?.method || "-";
      document.getElementById("dashStatSuccessSub").textContent = `${latest.run_id} (${latest.config?.mocap_id})`;
    } else if (reports.length > 0) {
      const r0 = reports[0];
      document.getElementById("dashStatSuccess").textContent = `${(r0.success_rate * 100).toFixed(1)}%`;
      document.getElementById("dashStatSuccessMethod").textContent = r0.method || "-";
      document.getElementById("dashStatSuccessSub").textContent = `${r0.run_id} (${r0.mocap_id})`;
    }

    // Activity feed
    const feed = document.getElementById("dashActivityFeed");
    if (feed) {
      const items = [];
      if (snap) {
        items.push({
          tag: "AX",
          bg: "var(--blue)",
          text: `<strong>Text Mining</strong>: ${snap.total_papers}件 (Sim2Real ${snap.sim_to_real_count}件)`,
          time: (snap.generated_at || "").slice(0, 16).replace("T", " "),
        });
      }
      reports.slice(0, 4).forEach((r) => {
        items.push({
          tag: r.method || "RL",
          bg: "var(--primary)",
          text: `<strong>${escapeHtml(r.run_id)}</strong> (${escapeHtml(r.method)}) 成功率 ${(r.success_rate * 100).toFixed(1)}%`,
          time: String(r.created_at || "").slice(0, 16).replace("T", " "),
        });
      });
      feed.innerHTML = items
        .map(
          (it) => `
        <li class="activity-item">
          <div class="activity-avatar" style="background:${it.bg}">${escapeHtml(it.tag)}</div>
          <div>
            <div class="activity-body">${it.text}</div>
            <div class="activity-time">${escapeHtml(it.time)}</div>
          </div>
        </li>`
        )
        .join("");
    }

    // Runs table
    const tbody = document.getElementById("dashRunsTbody");
    if (tbody) {
      tbody.innerHTML = reports.length
        ? reports
            .slice(0, 6)
            .map(
              (r) => `
          <tr data-run-id="${escapeHtml(r.run_id)}" style="cursor:pointer">
            <td class="cell-mono">${escapeHtml(r.run_id)}</td>
            <td><span class="pill-badge pill-teal">${escapeHtml(r.method)}</span></td>
            <td class="cell-strong">${escapeHtml(r.mocap_id)}</td>
            <td class="cell-mono">${(r.success_rate * 100).toFixed(1)}%</td>
            <td class="cell-mono">${Number(r.tracking_score || 0).toFixed(2)}</td>
            <td>${escapeHtml(String(r.created_at || "").slice(0, 16).replace("T", " "))}</td>
          </tr>`
            )
            .join("")
        : '<tr><td colspan="6" style="text-align:center;color:var(--text-muted);padding:20px">まだ学習レポートがありません。Training画面から学習を実行してください。</td></tr>';

      tbody.querySelectorAll("tr[data-run-id]").forEach((row) => {
        row.addEventListener("click", async () => {
          await loadRunEvaluation(row.dataset.runId);
          navigateTo("eval");
        });
      });
    }

    renderVisibleCharts();
  };

  const renderMocapSelect = () => {
    const sel = document.getElementById("mocapSelect");
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = state.mocaps
      .map(
        (m) =>
          `<option value="${escapeHtml(m.mocap_id)}">${escapeHtml(m.name || m.mocap_id)} (${m.frames} frames, ${m.joints} joints)</option>`
      )
      .join("");
    if (prev && state.mocaps.some((m) => m.mocap_id === prev)) sel.value = prev;
  };

  const renderRunSelect = () => {
    const sel = document.getElementById("evalRunSelect");
    if (!sel) return;
    sel.innerHTML = state.reports.length
      ? state.reports
          .map(
            (r) =>
              `<option value="${escapeHtml(r.run_id)}">${escapeHtml(r.run_id)} [${escapeHtml(r.method)} / ${(r.success_rate * 100).toFixed(1)}%]</option>`
          )
          .join("")
      : '<option value="">No runs available</option>';
    if (state.currentReport?.run_id) sel.value = state.currentReport.run_id;
  };

  const renderEvaluationView = () => {
    const rep = state.currentReport;
    document.getElementById("evalRunMethodBadge").textContent = rep ? `${rep.config?.method} · ${rep.run_id}` : "-";
    document.getElementById("threeRunTag").textContent = rep ? `${rep.config?.method} ${rep.run_id}` : "No Run";

    const diag = rep?.diagnostics;
    document.getElementById("diagTracking").textContent = diag ? Number(diag.tracking_score).toFixed(3) : "-";
    document.getElementById("diagSmoothness").textContent = diag ? Number(diag.smoothness_score).toFixed(3) : "-";
    document.getElementById("diagOverfit").textContent = diag ? (diag.overfitting_detected ? "Detected" : "Healthy") : "-";
    document.getElementById("diagSummaryJa").textContent =
      diag?.summary_ja || "学習レポートを選択または新規学習を実行すると、軌道追従性・滑らかさ・過学習の自動診断結果が表示されます。";

    const video = document.getElementById("evalVideoPlayer");
    const fallback = document.getElementById("evalVideoFallback");
    if (rep?.run_id && video && fallback) {
      video.src = `/api/runs/${encodeURIComponent(rep.run_id)}/video`;
      video.classList.remove("hidden");
      fallback.classList.add("hidden");
    } else if (video && fallback) {
      video.classList.add("hidden");
      fallback.classList.remove("hidden");
    }

    renderVisibleCharts();
  };

  const loadRunEvaluation = async (runId) => {
    if (!runId) return;
    try {
      const [rep, framesRes] = await Promise.all([
        fetchJson(`/api/runs/${encodeURIComponent(runId)}/report`),
        fetchJson(`/api/runs/${encodeURIComponent(runId)}/frames`).catch(() => ({ frames: [] })),
      ]);
      state.currentReport = rep;
      state.currentFrames = framesRes.frames || [];
      state.frameIndex = 0;
      renderRunSelect();
      renderEvaluationView();
    } catch (err) {
      showToast(`レポート読み込み失敗: ${err.message}`, "error");
    }
  };

  const loadDashboardInit = async () => {
    const pill = document.getElementById("healthPill");
    const statusText = document.getElementById("healthStatusText");
    try {
      const data = await fetchJson("/api/dashboard-init");
      state.snapshot = data.text_mining;
      state.papers = data.papers || [];
      state.mocaps = data.mocaps || [];
      state.reports = data.reports || [];
      state.currentReport = data.latest_report || null;

      if (pill && statusText) {
        pill.classList.add("is-ok");
        statusText.textContent = "API Connected";
      }

      renderFilterPills();
      renderMocapSelect();
      renderRunSelect();
      renderDashboardView();
      renderPapersView();
      renderEvaluationView();

      if (state.currentReport?.run_id) {
        const framesRes = await fetchJson(`/api/runs/${encodeURIComponent(state.currentReport.run_id)}/frames`).catch(() => ({ frames: [] }));
        state.currentFrames = framesRes.frames || [];
      }
    } catch (err) {
      if (statusText) statusText.textContent = "Offline";
      showToast(`初期データ取得に失敗しました: ${err.message}`, "error");
    }
  };

  /* =========================================================
   * 5. Three.js 3D Skeleton Viewer (Policy + Reference)
   * ========================================================= */
  const initThreeSkeleton = () => {
    if (typeof window.THREE === "undefined" || state.threeCtx) return;
    const container = document.getElementById("threeStage");
    if (!container) return;

    const width = container.clientWidth || 500;
    const height = container.clientHeight || 300;
    const scene = new window.THREE.Scene();
    const camera = new window.THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(2.2, 1.5, 2.8);
    camera.lookAt(0, 0.85, 0);

    const renderer = new window.THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);
    scene.add(new window.THREE.GridHelper(6, 12, 0x1abb9c, 0x334155));

    const buildSkeletonGroup = (jointColor, lineColor) => {
      const group = new window.THREE.Group();
      const joints = {};
      const jointGeo = new window.THREE.SphereGeometry(0.038, 10, 10);
      const jointMat = new window.THREE.MeshBasicMaterial({ color: jointColor });
      const lineMat = new window.THREE.LineBasicMaterial({ color: lineColor });
      const edges = [];

      SKELETON_EDGES.forEach(([a, b]) => {
        if (!joints[a]) {
          joints[a] = new window.THREE.Mesh(jointGeo, jointMat);
          group.add(joints[a]);
        }
        if (!joints[b]) {
          joints[b] = new window.THREE.Mesh(jointGeo, jointMat);
          group.add(joints[b]);
        }
        const geo = new window.THREE.BufferGeometry().setFromPoints([
          new window.THREE.Vector3(),
          new window.THREE.Vector3(),
        ]);
        const line = new window.THREE.Line(geo, lineMat);
        group.add(line);
        edges.push({ a, b, line });
      });
      scene.add(group);
      return { group, joints, edges };
    };

    const policySkel = buildSkeletonGroup(0x1abb9c, 0x5eead4);
    const refSkel = buildSkeletonGroup(0xf59f00, 0xfcd34d);
    state.threeCtx = { scene, camera, renderer, policySkel, refSkel };

    const updateSkeleton = (skel, bodies) => {
      if (!bodies) return;
      Object.entries(bodies).forEach(([name, pos]) => {
        if (skel.joints[name] && Array.isArray(pos) && pos.length === 3) {
          // MuJoCo uses Z-up; Three.js uses Y-up
          skel.joints[name].position.set(pos[0], pos[2], -pos[1]);
        }
      });
      skel.edges.forEach(({ a, b, line }) => {
        if (skel.joints[a] && skel.joints[b]) {
          line.geometry.setFromPoints([skel.joints[a].position, skel.joints[b].position]);
        }
      });
    };

    let tick = 0;
    const animate = () => {
      requestAnimationFrame(animate);
      if (state.page !== "eval") return;
      const showRef = document.getElementById("chkShowRefSkeleton")?.checked ?? true;
      refSkel.group.visible = showRef && state.currentFrames.length > 0;

      if (state.currentFrames.length > 0) {
        if (state.threePlaying && tick % 2 === 0) {
          state.frameIndex = (state.frameIndex + 1) % state.currentFrames.length;
        }
        const f = state.currentFrames[state.frameIndex];
        updateSkeleton(policySkel, f.bodies);
        updateSkeleton(refSkel, f.bodies_ref);
        document.getElementById("threeFrameTag").textContent = `Frame ${state.frameIndex + 1} / ${state.currentFrames.length}`;
      } else {
        document.getElementById("threeFrameTag").textContent = "No 3D Frames";
      }
      tick += 1;
      policySkel.group.rotation.y += 0.004;
      refSkel.group.rotation.y = policySkel.group.rotation.y;
      renderer.render(scene, camera);
    };
    animate();
  };

  /* =========================================================
   * 6. Navigation, Theme, Command Palette
   * ========================================================= */
  const applyTheme = (theme) => {
    state.theme = theme;
    if (theme === "dark") {
      document.documentElement.setAttribute("data-theme", "dark");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    localStorage.setItem("pai_theme", theme);
    const darkToggle = document.getElementById("settingsDarkToggle");
    if (darkToggle) {
      darkToggle.classList.toggle("on", theme === "dark");
      darkToggle.setAttribute("aria-checked", String(theme === "dark"));
    }
    document.getElementById("themeToggleBtn")?.setAttribute("aria-pressed", String(theme === "dark"));
    renderVisibleCharts();
  };

  const toggleRailMode = () => {
    const isRail = document.body.classList.toggle("sidebar-rail");
    const railToggle = document.getElementById("settingsRailToggle");
    if (railToggle) {
      railToggle.classList.toggle("on", isRail);
      railToggle.setAttribute("aria-checked", String(isRail));
    }
    setTimeout(renderVisibleCharts, 240);
  };

  const navigateTo = (pageId, focusSelector = null) => {
    if (!PAGE_META[pageId]) return;
    state.page = pageId;
    document.body.setAttribute("data-page", pageId);

    document.querySelectorAll(".page-section").forEach((sec) => {
      sec.classList.toggle("active", sec.id === `page-${pageId}`);
    });
    document.querySelectorAll("[data-nav]").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.nav === pageId);
    });
    const bc = document.getElementById("breadcrumbCurrent");
    if (bc) bc.textContent = PAGE_META[pageId].breadcrumb;

    if (window.innerWidth <= 768) {
      document.getElementById("sidebar")?.classList.remove("open");
      document.body.classList.remove("sidebar-open");
      document.getElementById("sidebarBackdrop")?.setAttribute("hidden", "");
    }

    requestAnimationFrame(() => {
      renderVisibleCharts();
      if (pageId === "eval") initThreeSkeleton();
      if (focusSelector) {
        document.querySelector(focusSelector)?.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  };

  const initCommandPalette = () => {
    const backdrop = document.getElementById("cmdkBackdrop");
    const input = document.getElementById("cmdkInput");
    const list = document.getElementById("cmdkList");
    if (!backdrop || !input || !list) return;

    const renderList = (query = "") => {
      const q = query.trim().toLowerCase();
      const matched = COMMAND_ITEMS.filter((c) => c.label.toLowerCase().includes(q) || c.group.toLowerCase().includes(q));
      list.innerHTML = matched.length
        ? matched
            .map(
              (item, idx) => `
          <button type="button" class="cmdk-item ${idx === 0 ? "active" : ""}" data-cmd-id="${item.id}">
            <span class="cmdk-item-label">${escapeHtml(item.label)}</span>
            <span class="cmdk-item-kbd">${escapeHtml(item.group)}</span>
          </button>`
            )
            .join("")
        : '<div class="cmdk-empty">一致するコマンドがありません</div>';

      list.querySelectorAll(".cmdk-item").forEach((btn) => {
        btn.addEventListener("click", () => {
          const cmd = COMMAND_ITEMS.find((c) => c.id === btn.dataset.cmdId);
          backdrop.classList.add("hidden");
          if (!cmd) return;
          if (cmd.action.type === "nav") navigateTo(cmd.action.page);
          if (cmd.action.type === "theme") applyTheme(state.theme === "dark" ? "light" : "dark");
          if (cmd.action.type === "rail") toggleRailMode();
          if (cmd.action.type === "refresh") loadDashboardInit();
        });
      });
    };

    const openCmdk = () => {
      backdrop.classList.remove("hidden");
      input.value = "";
      renderList("");
      setTimeout(() => input.focus(), 20);
    };

    document.getElementById("cmdkTrigger")?.addEventListener("click", openCmdk);
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) backdrop.classList.add("hidden");
    });
    input.addEventListener("input", () => renderList(input.value));
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        backdrop.classList.contains("hidden") ? openCmdk() : backdrop.classList.add("hidden");
      } else if (e.key === "Escape") {
        backdrop.classList.add("hidden");
      }
    });
  };

  /* =========================================================
   * 7. Event Wiring (Full Backend Integration)
   * ========================================================= */
  const collectTrainingPayload = () => ({
    method: state.selectedAlgo,
    mocap_id: document.getElementById("mocapSelect").value || "sample_biped_walk",
    epochs: Number(document.getElementById("epochsInput").value || 30),
    learning_rate: Number(document.getElementById("lrInput").value || 0.0003),
    bc_weight: Number(document.getElementById("bcWeightInput").value || 1.0),
    entropy_coef: Number(document.getElementById("entropyInput").value || 0.01),
    smoothness_weight: Number(document.getElementById("smoothInput").value || 0.06),
    reward_tracking_weight: Number(document.getElementById("rewardTrackInput").value || 1.2),
    robot_xml_name: state.robotXmlName,
    instruction_note: document.getElementById("instructionNoteInput").value || "",
  });

  const startTrainingWithPayload = async (payload) => {
    const banner = document.getElementById("trainJobBanner");
    const txt = document.getElementById("trainJobText");
    const pct = document.getElementById("trainJobPct");
    const bar = document.getElementById("trainProgressBar");
    banner.classList.remove("hidden");
    txt.textContent = "学習ジョブを開始しています...";
    pct.textContent = "0%";
    bar.style.width = "4%";

    try {
      const { job_id, run_id } = await fetchJson("/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      await pollJob(job_id, (job) => {
        const p = job.total > 0 ? Math.round((job.progress / job.total) * 100) : 0;
        txt.textContent = job.message || "学習中...";
        pct.textContent = `${p}%`;
        bar.style.width = `${p}%`;
      });
      showToast(`学習完了 (${run_id})。評価画面を更新しました。`);
      await loadDashboardInit();
      await loadRunEvaluation(run_id);
      navigateTo("eval");
    } catch (err) {
      showToast(`学習エラー: ${err.message}`, "error");
    }
  };

  document.addEventListener("DOMContentLoaded", () => {
    applyTheme(state.theme);
    initCommandPalette();
    loadDashboardInit();

    document.querySelectorAll("[data-nav]").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.preventDefault();
        navigateTo(el.dataset.nav, el.dataset.focus || null);
      });
    });

    document.querySelectorAll(".nav-tree > .nav-toggle").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tree = btn.closest(".nav-tree");
        const isOpen = tree.classList.toggle("open");
        btn.setAttribute("aria-expanded", String(isOpen));
      });
    });

    document.getElementById("sidebarToggle")?.addEventListener("click", () => {
      if (window.innerWidth <= 768) {
        const open = document.getElementById("sidebar")?.classList.toggle("open");
        document.body.classList.toggle("sidebar-open", Boolean(open));
        const bd = document.getElementById("sidebarBackdrop");
        if (bd) open ? bd.removeAttribute("hidden") : bd.setAttribute("hidden", "");
      } else {
        toggleRailMode();
      }
    });

    document.getElementById("sidebarBackdrop")?.addEventListener("click", () => {
      document.getElementById("sidebar")?.classList.remove("open");
      document.body.classList.remove("sidebar-open");
      document.getElementById("sidebarBackdrop")?.setAttribute("hidden", "");
    });

    document.getElementById("themeToggleBtn")?.addEventListener("click", () => applyTheme(state.theme === "dark" ? "light" : "dark"));
    document.getElementById("settingsDarkToggle")?.addEventListener("click", () => applyTheme(state.theme === "dark" ? "light" : "dark"));
    document.getElementById("settingsRailToggle")?.addEventListener("click", toggleRailMode);
    document.getElementById("btnRefreshAll")?.addEventListener("click", () => {
      loadDashboardInit();
      showToast("最新データを再読み込みしました");
    });

    // Dashboard trend tabs
    document.querySelectorAll("#dashActivityTabs .chart-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll("#dashActivityTabs .chart-tab").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        state.dashTrendMode = tab.dataset.range;
        renderVisibleCharts();
      });
    });

    // Papers trend tabs & filters
    document.getElementById("btnTrendYearly")?.addEventListener("click", (e) => {
      state.papersTrendMode = "yearly";
      e.target.classList.add("active");
      document.getElementById("btnTrendMonthly")?.classList.remove("active");
      renderVisibleCharts();
    });
    document.getElementById("btnTrendMonthly")?.addEventListener("click", (e) => {
      state.papersTrendMode = "monthly";
      e.target.classList.add("active");
      document.getElementById("btnTrendYearly")?.classList.remove("active");
      renderVisibleCharts();
    });

    document.querySelectorAll("[data-sim-filter]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-sim-filter]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.paperFilters.simOnly = btn.dataset.simFilter === "sim2real";
        renderPapersView();
      });
    });

    document.getElementById("paperSearchInput")?.addEventListener("input", (e) => {
      state.paperFilters.search = e.target.value;
      renderPapersView();
    });

    document.getElementById("btnResetPaperFilters")?.addEventListener("click", () => {
      state.paperFilters = { simOnly: false, cluster: "all", year: "all", search: "" };
      document.getElementById("paperSearchInput").value = "";
      document.querySelectorAll("[data-sim-filter]").forEach((b) => b.classList.toggle("active", b.dataset.simFilter === "all"));
      renderFilterPills();
      renderPapersView();
      showToast("論文フィルタをリセットしました");
    });

    // Paper scrape job
    document.getElementById("btnScrape")?.addEventListener("click", async () => {
      const kws = (document.getElementById("keywordsInput").value || "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const maxPer = Number(document.getElementById("maxPerKw").value || 6);
      const banner = document.getElementById("scrapeJobBanner");
      const txt = document.getElementById("scrapeJobText");
      const pct = document.getElementById("scrapeJobPct");
      const bar = document.getElementById("scrapeProgressBar");
      banner.classList.remove("hidden");
      try {
        const { job_id } = await fetchJson("/api/papers/scrape", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ keywords: kws, max_per_keyword: maxPer }),
        });
        await pollJob(job_id, (job) => {
          const p = job.total > 0 ? Math.round((job.progress / job.total) * 100) : 0;
          txt.textContent = job.message || "取得中...";
          pct.textContent = `${p}%`;
          bar.style.width = `${p}%`;
        });
        await loadDashboardInit();
        showToast("論文スクレイピングとテキストマイニングが完了しました");
      } catch (err) {
        showToast(`スクレイピング失敗: ${err.message}`, "error");
      }
    });

    // Algorithm selector
    document.querySelectorAll("#algoSelector .algo-card").forEach((card) => {
      card.addEventListener("click", () => {
        document.querySelectorAll("#algoSelector .algo-card").forEach((c) => c.classList.remove("selected"));
        card.classList.add("selected");
        state.selectedAlgo = card.dataset.algo;
      });
    });

    // Advisor Ask
    document.getElementById("btnAskAdvisor")?.addEventListener("click", async () => {
      const input = document.getElementById("advisorQuestionInput");
      const log = document.getElementById("advisorChatLog");
      const q = input.value.trim();
      if (!q) return;

      const uDiv = document.createElement("div");
      uDiv.className = "chat-msg user";
      uDiv.textContent = q;
      log.appendChild(uDiv);
      input.value = "";

      try {
        const res = await fetchJson("/api/advisor/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: q,
            mocap_id: document.getElementById("mocapSelect").value || "sample_biped_walk",
          }),
        });
        document.getElementById("advisorEngineBadge").textContent = res.engine || "Bedrock";
        state.lastSuggestion = res;
        const bDiv = document.createElement("div");
        bDiv.className = "chat-msg assistant";
        bDiv.innerHTML = `<div class="chat-msg-meta">AI Advisor (${escapeHtml(res.engine)})</div><div style="white-space:pre-line">${escapeHtml(res.reply)}</div>`;
        log.appendChild(bDiv);
        log.scrollTop = log.scrollHeight;

        document.getElementById("suggestedMethodText").textContent = res.recommended_method;
        document.getElementById("applySuggestionWrap").classList.remove("hidden");
      } catch (err) {
        showToast(`AI相談エラー: ${err.message}`, "error");
      }
    });

    document.getElementById("btnApplySuggested")?.addEventListener("click", () => {
      const s = state.lastSuggestion;
      if (!s) return;
      state.selectedAlgo = s.recommended_method;
      document.querySelectorAll("#algoSelector .algo-card").forEach((c) => {
        c.classList.toggle("selected", c.dataset.algo === s.recommended_method);
      });
      const cfg = s.recommended_config || {};
      if (cfg.epochs !== undefined) document.getElementById("epochsInput").value = cfg.epochs;
      if (cfg.learning_rate !== undefined) document.getElementById("lrInput").value = cfg.learning_rate;
      if (cfg.bc_weight !== undefined) document.getElementById("bcWeightInput").value = cfg.bc_weight;
      if (cfg.entropy_coef !== undefined) document.getElementById("entropyInput").value = cfg.entropy_coef;
      if (cfg.smoothness_weight !== undefined) document.getElementById("smoothInput").value = cfg.smoothness_weight;
      if (cfg.reward_tracking_weight !== undefined) document.getElementById("rewardTrackInput").value = cfg.reward_tracking_weight;
      showToast(`推奨設定 (${s.recommended_method}) をフォームに適用しました`);
    });

    // Uploads
    document.getElementById("btnUploadMocap")?.addEventListener("click", async () => {
      const file = document.getElementById("mocapUploadInput").files?.[0];
      if (!file) return showToast("Mocapファイルを選択してください", "warning");
      const fd = new FormData();
      fd.append("file", file);
      try {
        const res = await fetchJson("/api/mocap/upload", { method: "POST", body: fd });
        state.mocaps.push(res);
        renderMocapSelect();
        document.getElementById("mocapSelect").value = res.mocap_id;
        showToast(`Mocapをアップロードしました: ${res.name}`);
      } catch (err) {
        showToast(`Mocapアップロード失敗: ${err.message}`, "error");
      }
    });

    document.getElementById("btnUploadRobotXml")?.addEventListener("click", async () => {
      const file = document.getElementById("robotXmlInput").files?.[0];
      if (!file) return showToast("XMLファイルを選択してください", "warning");
      const fd = new FormData();
      fd.append("file", file);
      try {
        const res = await fetchJson("/api/robot-xml/upload", { method: "POST", body: fd });
        state.robotXmlName = res.robot_xml_name;
        document.getElementById("robotXmlNameLabel").textContent = res.robot_xml_name;
        showToast(`Robot XML を設定しました: ${res.robot_xml_name}`);
      } catch (err) {
        showToast(`XMLアップロード失敗: ${err.message}`, "error");
      }
    });

    // Start training
    document.getElementById("btnStartTrain")?.addEventListener("click", () => {
      startTrainingWithPayload(collectTrainingPayload());
    });

    // Evaluation Run Selector & 3D Playback
    document.getElementById("evalRunSelect")?.addEventListener("change", (e) => {
      if (e.target.value) loadRunEvaluation(e.target.value);
    });
    document.getElementById("btnToggle3DPlay")?.addEventListener("click", () => {
      state.threePlaying = !state.threePlaying;
      showToast(state.threePlaying ? "3D 骨格再生を再開しました" : "3D 骨格再生を一時停止しました");
    });

    // Retrain Preview & Auto-Start
    const handleRetrain = async (autoStart) => {
      const inst = document.getElementById("retrainInstructionInput").value.trim();
      if (!inst) return showToast("再学習への指示を入力してください", "warning");
      const cfg = state.currentReport?.config || collectTrainingPayload();
      try {
        const res = await fetchJson("/api/advisor/retrain", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            instruction: inst,
            base_method: cfg.method || "DAPG",
            mocap_id: cfg.mocap_id || "sample_biped_walk",
            epochs: cfg.epochs || 30,
            learning_rate: cfg.learning_rate || 0.0003,
            bc_weight: cfg.bc_weight ?? 1.0,
            entropy_coef: cfg.entropy_coef ?? 0.01,
            smoothness_weight: cfg.smoothness_weight ?? 0.06,
            reward_tracking_weight: cfg.reward_tracking_weight ?? 1.2,
            auto_start: autoStart,
          }),
        });
        const box = document.getElementById("retrainResultBox");
        box.classList.remove("hidden");
        box.textContent = `[Engine: ${res.engine}]\n${res.reply}`;
        if (autoStart && res.job_id) {
          showToast(`再学習ジョブを開始しました (${res.job_id})`);
          await pollJob(res.job_id, () => {});
          await loadDashboardInit();
          await loadRunEvaluation(res.updated_config?.run_id);
        }
      } catch (err) {
        showToast(`再学習エラー: ${err.message}`, "error");
      }
    };

    document.getElementById("btnRetrainPreview")?.addEventListener("click", () => handleRetrain(false));
    document.getElementById("btnRetrainAndStart")?.addEventListener("click", () => handleRetrain(true));

    window.addEventListener("resize", () => {
      state.charts.forEach((chart) => chart.resize());
    });
  });
})();
