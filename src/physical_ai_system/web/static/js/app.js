(() => {
  "use strict";

  /* =========================================================
   * 1. Immutable Configuration & Mock Data
   * ========================================================= */
  const PAGE_META = Object.freeze({
    dashboard: { title: "Dashboard", breadcrumb: "Dashboard" },
    papers: { title: "Papers & Text Mining", breadcrumb: "Papers & Mining" },
    train: { title: "AI Advisor & Training", breadcrumb: "Training" },
    eval: { title: "Evaluation & 3D", breadcrumb: "Evaluation & 3D" },
    system: { title: "System & AWS", breadcrumb: "System & AWS" },
  });

  const MOCK_PAPERS = Object.freeze([
    {
      id: "arXiv:2601.0412",
      title: "Whole-Body Humanoid Locomotion via Diffusion-Guided DAPG and Sim2Real Domain Randomization",
      cluster: "Sim2Real / Locomotion",
      year: 2026,
      sim2real: true,
      x: 2.1,
      y: 3.4,
      summary: "2足歩行ヒューマノイドの頑健な歩行制御に向けて、モーションキャプチャ由来のデモ方策をDAPGで正則化しつつ、MuJoCo上の摩擦・質量ランダム化により実機転移（Sim2Real）成功率を89%以上に向上させた手法を提案。",
    },
    {
      id: "arXiv:2511.1890",
      title: "RLPD-Biped: High-Sample-Efficiency Offline-to-Online Reinforcement Learning for Humanoid Gait",
      cluster: "RL Algorithms",
      year: 2025,
      sim2real: true,
      x: -1.4,
      y: 2.6,
      summary: "オフラインのモーションキャプチャデータとオンライン探索を50:50で混合するRLPDを用い、従来比1/3の環境ステップ数で安定した2足歩行方策を獲得。",
    },
    {
      id: "arXiv:2508.0731",
      title: "Conservative Advantage-Weighted Actor-Critic for Rough Terrain Humanoid Stepping",
      cluster: "RL Algorithms",
      year: 2025,
      sim2real: false,
      x: -2.2,
      y: -0.8,
      summary: "不整地歩行データセットに対するAWACの保守的方策更新を検証し、分布外アクションによる転倒を抑制する正則化係数の設計指針を提示。",
    },
    {
      id: "arXiv:2412.1104",
      title: "Zero-Shot Sim2Real Transfer of Torque-Controlled Bipedal Walking Policies in MuJoCo",
      cluster: "Sim2Real / Locomotion",
      year: 2024,
      sim2real: true,
      x: 1.7,
      y: 1.9,
      summary: "MuJoCoの接触ソルバとアクチュエータ遅延モデルを同定し、追加の実機ファインチューニングなしでゼロショット歩行転移を達成。",
    },
    {
      id: "arXiv:2405.0392",
      title: "Retargeting Human Motion Capture to 21-DoF Humanoid Robots via Kinematic Constraints",
      cluster: "Mocap Retargeting",
      year: 2024,
      sim2real: false,
      x: 0.4,
      y: -2.1,
      summary: "BVH/NPZ形式の人間モーションキャプチャデータを21自由度ヒューマノイドモデルへ運動学的整合性を保ちながら高速リターゲティングするパイプライン。",
    },
    {
      id: "arXiv:2309.1422",
      title: "Adaptive Push Recovery for Bipedal Robots Using Reward Shaping and Curriculum Learning",
      cluster: "Control & Safety",
      year: 2023,
      sim2real: true,
      x: 3.1,
      y: -1.1,
      summary: "外乱インパルスに対するバランス回復をカリキュラム学習で段階的に強化し、横方向プッシュに対する耐性を大幅に改善。",
    },
  ]);

  const ACTIVITY_SERIES = Object.freeze({
    "7d": {
      labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
      success: [76, 79, 81, 84, 86, 88, 89.4],
      papers: [12, 18, 15, 22, 27, 24, 30],
    },
    "30d": {
      labels: ["W1", "W2", "W3", "W4"],
      success: [68, 75, 83, 89.4],
      papers: [32, 41, 38, 37],
    },
    yearly: {
      labels: ["2020", "2021", "2022", "2023", "2024", "2025", "2026"],
      success: [52, 60, 67, 74, 81, 86, 89.4],
      papers: [14, 19, 25, 31, 40, 52, 64],
    },
  });

  const COMMAND_ITEMS = Object.freeze([
    { id: "nav-dashboard", label: "Dashboard (Overview) へ移動", group: "Navigation", action: { type: "nav", page: "dashboard" } },
    { id: "nav-papers", label: "Papers & Text Mining へ移動", group: "Navigation", action: { type: "nav", page: "papers" } },
    { id: "nav-train", label: "AI Advisor & Training へ移動", group: "Navigation", action: { type: "nav", page: "train" } },
    { id: "nav-eval", label: "Evaluation & 3D へ移動", group: "Navigation", action: { type: "nav", page: "eval" } },
    { id: "nav-system", label: "System & AWS へ移動", group: "Navigation", action: { type: "nav", page: "system" } },
    { id: "act-theme", label: "Light / Dark テーマを切り替え", group: "Actions", action: { type: "theme" } },
    { id: "act-rail", label: "サイドバーのコンパクト表示を切り替え", group: "Actions", action: { type: "rail" } },
  ]);

  /* =========================================================
   * 2. Pure Functions (Theme Tokens & ECharts Options)
   * ========================================================= */
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

  const buildActivityChartOption = (range, t) => {
    const data = ACTIVITY_SERIES[range] || ACTIVITY_SERIES["7d"];
    return {
      tooltip: { trigger: "axis" },
      legend: {
        data: ["Policy Success (%)", "Indexed Papers"],
        bottom: 0,
        textStyle: { color: t.textMuted, fontSize: 11 },
      },
      grid: { top: 20, left: 40, right: 40, bottom: 36 },
      xAxis: {
        type: "category",
        data: data.labels,
        axisLine: { lineStyle: { color: t.border } },
        axisLabel: { color: t.textMuted, fontSize: 11 },
      },
      yAxis: [
        {
          type: "value",
          name: "%",
          min: 40,
          max: 100,
          splitLine: { lineStyle: { color: t.border, type: "dashed" } },
          axisLabel: { color: t.textMuted, fontSize: 11 },
        },
        {
          type: "value",
          name: "Papers",
          splitLine: { show: false },
          axisLabel: { color: t.textMuted, fontSize: 11 },
        },
      ],
      series: [
        {
          name: "Policy Success (%)",
          type: "line",
          smooth: true,
          data: data.success,
          itemStyle: { color: t.primary },
          lineStyle: { width: 2.5 },
          areaStyle: { color: t.primary, opacity: 0.12 },
        },
        {
          name: "Indexed Papers",
          type: "bar",
          yAxisIndex: 1,
          barWidth: 16,
          data: data.papers,
          itemStyle: { color: t.azure, borderRadius: [4, 4, 0, 0], opacity: 0.78 },
        },
      ],
    };
  };

  const buildAlgoPieOption = (t) => ({
    tooltip: { trigger: "item", formatter: "{b}: {c} runs ({d}%)" },
    legend: { bottom: 0, textStyle: { color: t.textMuted, fontSize: 11 } },
    series: [
      {
        name: "Algorithm",
        type: "pie",
        radius: ["52%", "74%"],
        center: ["50%", "44%"],
        label: { show: false },
        data: [
          { value: 6, name: "DAPG", itemStyle: { color: t.primary } },
          { value: 4, name: "RLPD", itemStyle: { color: t.blue } },
          { value: 2, name: "AWAC", itemStyle: { color: t.purple } },
        ],
      },
    ],
  });

  const buildMiningMapOption = (papers, t) => ({
    tooltip: {
      trigger: "item",
      formatter: (params) => {
        const p = params.data.paper;
        return `<strong>${p.id}</strong><br/>${p.title}<br/><span style="opacity:.8">${p.cluster} (${p.year})</span>`;
      },
    },
    grid: { top: 24, left: 42, right: 24, bottom: 36 },
    xAxis: {
      name: "Dim 1",
      splitLine: { lineStyle: { color: t.border, type: "dashed" } },
      axisLabel: { color: t.textMuted, fontSize: 11 },
    },
    yAxis: {
      name: "Dim 2",
      splitLine: { lineStyle: { color: t.border, type: "dashed" } },
      axisLabel: { color: t.textMuted, fontSize: 11 },
    },
    series: [
      {
        type: "scatter",
        symbolSize: (val, params) => (params.data.paper.sim2real ? 20 : 14),
        data: papers.map((p) => ({
          value: [p.x, p.y],
          paper: p,
          itemStyle: {
            color: p.sim2real ? t.primary : t.blue,
            borderColor: t.surface,
            borderWidth: 2,
          },
        })),
      },
    ],
  });

  const buildYearlyTrendOption = (t) => ({
    tooltip: { trigger: "axis" },
    legend: { data: ["Total Papers", "Sim2Real"], bottom: 0, textStyle: { color: t.textMuted, fontSize: 11 } },
    grid: { top: 20, left: 40, right: 20, bottom: 36 },
    xAxis: {
      type: "category",
      data: ["2020", "2021", "2022", "2023", "2024", "2025", "2026"],
      axisLabel: { color: t.textMuted, fontSize: 11 },
    },
    yAxis: {
      type: "value",
      splitLine: { lineStyle: { color: t.border, type: "dashed" } },
      axisLabel: { color: t.textMuted, fontSize: 11 },
    },
    series: [
      { name: "Total Papers", type: "bar", data: [14, 19, 25, 31, 40, 52, 64], itemStyle: { color: t.azure, borderRadius: [4, 4, 0, 0] } },
      { name: "Sim2Real", type: "line", smooth: true, data: [3, 5, 8, 12, 18, 25, 34], itemStyle: { color: t.primary }, lineStyle: { width: 2.5 } },
    ],
  });

  const buildTopTermsOption = (t) => ({
    tooltip: { trigger: "axis" },
    grid: { top: 12, left: 110, right: 24, bottom: 24 },
    xAxis: {
      type: "value",
      splitLine: { lineStyle: { color: t.border, type: "dashed" } },
      axisLabel: { color: t.textMuted, fontSize: 11 },
    },
    yAxis: {
      type: "category",
      data: ["whole-body", "mocap", "bipedal", "imitation", "humanoid", "sim2real"],
      axisLabel: { color: t.textMuted, fontSize: 11 },
    },
    series: [
      {
        type: "bar",
        data: [38, 46, 55, 62, 74, 88],
        itemStyle: { color: t.primary, borderRadius: [0, 4, 4, 0] },
      },
    ],
  });

  const buildEvalMetricsOption = (t) => {
    const epochs = Array.from({ length: 15 }, (_, i) => `Ep ${(i + 1) * 2}`);
    const trainLoss = epochs.map((_, i) => Number((0.95 * Math.exp(-i * 0.16) + 0.08).toFixed(3)));
    const valLoss = epochs.map((_, i) => Number((1.02 * Math.exp(-i * 0.14) + 0.11).toFixed(3)));
    const success = epochs.map((_, i) => Number((22 + 68 * (1 - Math.exp(-i * 0.2))).toFixed(1)));
    return {
      tooltip: { trigger: "axis" },
      legend: { data: ["Train Loss", "Val Loss", "Task Success (%)"], bottom: 0, textStyle: { color: t.textMuted, fontSize: 11 } },
      grid: { top: 24, left: 46, right: 46, bottom: 38 },
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

  const filterPapers = (papers, mode) => (mode === "sim2real" ? papers.filter((p) => p.sim2real) : papers);

  /* =========================================================
   * 3. Application State & DOM Controllers
   * ========================================================= */
  const state = {
    page: "dashboard",
    theme: localStorage.getItem("pai_theme") || "light",
    activityRange: "7d",
    paperFilter: "all",
    selectedAlgo: "DAPG",
    charts: new Map(),
    threePlaying: true,
    threeInitialized: false,
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
    }, 3000);
  };

  const renderPaperDetail = (paper) => {
    document.getElementById("detailPaperId").textContent = paper.id;
    document.getElementById("detailPaperTitle").textContent = paper.title;
    document.getElementById("detailPaperCluster").textContent = paper.cluster;
    document.getElementById("detailPaperYear").textContent = String(paper.year);
    document.getElementById("detailPaperSummary").textContent = paper.summary;
  };

  const renderPapersTable = (papers) => {
    const tbody = document.getElementById("mockPapersTbody");
    if (!tbody) return;
    tbody.innerHTML = papers
      .map(
        (p) => `
      <tr data-paper-id="${p.id}" style="cursor:pointer">
        <td class="cell-mono">${p.id}</td>
        <td class="cell-strong">${p.title}</td>
        <td><span class="pill-badge pill-blue">${p.cluster}</span></td>
        <td class="cell-mono">${p.year}</td>
        <td>${p.sim2real ? '<span class="status status-green">Yes</span>' : '<span class="status status-yellow">No</span>'}</td>
      </tr>`
      )
      .join("");

    tbody.querySelectorAll("tr[data-paper-id]").forEach((row) => {
      row.addEventListener("click", () => {
        const found = MOCK_PAPERS.find((p) => p.id === row.dataset.paperId);
        if (found) renderPaperDetail(found);
      });
    });
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
      ensureChart("chartDashActivity", () => buildActivityChartOption(state.activityRange, t));
      ensureChart("chartDashAlgoPie", () => buildAlgoPieOption(t));
    } else if (state.page === "papers") {
      ensureChart("chartMiningMap", () => buildMiningMapOption(filterPapers(MOCK_PAPERS, state.paperFilter), t));
      ensureChart("chartYearlyTrend", () => buildYearlyTrendOption(t));
      ensureChart("chartTopTerms", () => buildTopTermsOption(t));
    } else if (state.page === "eval") {
      ensureChart("chartEvalMetrics", () => buildEvalMetricsOption(t));
    }
  };

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
    const topToggle = document.getElementById("themeToggleBtn");
    if (topToggle) topToggle.setAttribute("aria-pressed", String(theme === "dark"));
    renderVisibleCharts();
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
      if (pageId === "eval" && !state.threeInitialized) initThreeSkeleton();
      if (focusSelector) {
        const target = document.querySelector(focusSelector);
        if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  };

  const initThreeSkeleton = () => {
    if (typeof window.THREE === "undefined") return;
    const container = document.getElementById("threeStage");
    if (!container) return;
    state.threeInitialized = true;

    const width = container.clientWidth || 500;
    const height = container.clientHeight || 300;
    const scene = new window.THREE.Scene();
    const camera = new window.THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(2.4, 1.5, 3.2);
    camera.lookAt(0, 0.8, 0);

    const renderer = new window.THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    scene.add(new window.THREE.GridHelper(6, 12, 0x1abb9c, 0x334155));

    const humanoid = new window.THREE.Group();
    const matJoint = new window.THREE.MeshBasicMaterial({ color: 0x1abb9c });
    const matLimb = new window.THREE.LineBasicMaterial({ color: 0x94a3b8 });

    const makeJoint = (x, y, z) => {
      const m = new window.THREE.Mesh(new window.THREE.SphereGeometry(0.06, 12, 12), matJoint);
      m.position.set(x, y, z);
      humanoid.add(m);
      return m;
    };

    const pelvis = makeJoint(0, 0.9, 0);
    const chest = makeJoint(0, 1.4, 0);
    const head = makeJoint(0, 1.72, 0);
    const lHip = makeJoint(-0.14, 0.88, 0);
    const rHip = makeJoint(0.14, 0.88, 0);
    const lFoot = makeJoint(-0.14, 0.08, 0);
    const rFoot = makeJoint(0.14, 0.08, 0);

    const connect = (a, b) => {
      const geo = new window.THREE.BufferGeometry().setFromPoints([a.position, b.position]);
      const line = new window.THREE.Line(geo, matLimb);
      humanoid.add(line);
      return { line, a, b };
    };

    const links = [
      connect(pelvis, chest),
      connect(chest, head),
      connect(pelvis, lHip),
      connect(pelvis, rHip),
      connect(lHip, lFoot),
      connect(rHip, rFoot),
    ];

    scene.add(humanoid);

    let frame = 0;
    const animate = () => {
      requestAnimationFrame(animate);
      if (state.threePlaying && state.page === "eval") {
        frame = (frame + 1) % 120;
        const phase = (frame / 120) * Math.PI * 2;
        lFoot.position.z = Math.sin(phase) * 0.28;
        lFoot.position.y = 0.08 + Math.max(0, Math.cos(phase)) * 0.12;
        rFoot.position.z = -Math.sin(phase) * 0.28;
        rFoot.position.y = 0.08 + Math.max(0, -Math.cos(phase)) * 0.12;
        humanoid.rotation.y += 0.008;
        links.forEach(({ line, a, b }) => {
          line.geometry.setFromPoints([a.position, b.position]);
        });
        const tag = document.getElementById("threeFrameTag");
        if (tag) tag.textContent = `Frame ${frame} / 120`;
      }
      renderer.render(scene, camera);
    };
    animate();
  };

  const initCommandPalette = () => {
    const backdrop = document.getElementById("cmdkBackdrop");
    const input = document.getElementById("cmdkInput");
    const list = document.getElementById("cmdkList");
    if (!backdrop || !input || !list) return;

    const renderList = (query = "") => {
      const q = query.trim().toLowerCase();
      const matched = COMMAND_ITEMS.filter((c) => c.label.toLowerCase().includes(q) || c.group.toLowerCase().includes(q));
      if (!matched.length) {
        list.innerHTML = '<div class="cmdk-empty">一致するコマンドがありません</div>';
        return;
      }
      list.innerHTML = matched
        .map(
          (item, idx) => `
        <button type="button" class="cmdk-item ${idx === 0 ? "active" : ""}" data-cmd-id="${item.id}">
          <span class="cmdk-item-label">${item.label}</span>
          <span class="cmdk-item-kbd">${item.group}</span>
        </button>`
        )
        .join("");

      list.querySelectorAll(".cmdk-item").forEach((btn) => {
        btn.addEventListener("click", () => {
          const cmd = COMMAND_ITEMS.find((c) => c.id === btn.dataset.cmdId);
          closeCmdk();
          if (!cmd) return;
          if (cmd.action.type === "nav") navigateTo(cmd.action.page);
          if (cmd.action.type === "theme") applyTheme(state.theme === "dark" ? "light" : "dark");
          if (cmd.action.type === "rail") toggleRailMode();
        });
      });
    };

    const openCmdk = () => {
      backdrop.classList.remove("hidden");
      input.value = "";
      renderList("");
      setTimeout(() => input.focus(), 20);
    };

    const closeCmdk = () => backdrop.classList.add("hidden");

    document.getElementById("cmdkTrigger")?.addEventListener("click", openCmdk);
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) closeCmdk();
    });
    input.addEventListener("input", () => renderList(input.value));
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        backdrop.classList.contains("hidden") ? openCmdk() : closeCmdk();
      } else if (e.key === "Escape" && !backdrop.classList.contains("hidden")) {
        closeCmdk();
      }
    });
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

  const checkBackendHealth = async () => {
    const pill = document.getElementById("healthPill");
    const text = document.getElementById("healthStatusText");
    if (!pill || !text) return;
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        pill.classList.remove("is-mock");
        pill.classList.add("is-ok");
        text.textContent = "API Connected (Mock UI)";
      }
    } catch {
      // Static template preview fallback
    }
  };

  /* =========================================================
   * 4. Bootstrap & Event Wiring
   * ========================================================= */
  document.addEventListener("DOMContentLoaded", () => {
    applyTheme(state.theme);
    renderPapersTable(MOCK_PAPERS);
    initCommandPalette();
    checkBackendHealth();

    // Navigation clicks
    document.querySelectorAll("[data-nav]").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.preventDefault();
        navigateTo(el.dataset.nav, el.dataset.focus || null);
      });
    });

    // Sidebar accordion toggles
    document.querySelectorAll(".nav-tree > .nav-toggle").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tree = btn.closest(".nav-tree");
        const isOpen = tree.classList.toggle("open");
        btn.setAttribute("aria-expanded", String(isOpen));
      });
    });

    // Sidebar toggle (desktop rail vs mobile drawer)
    document.getElementById("sidebarToggle")?.addEventListener("click", () => {
      if (window.innerWidth <= 768) {
        const sidebar = document.getElementById("sidebar");
        const backdrop = document.getElementById("sidebarBackdrop");
        const open = sidebar?.classList.toggle("open");
        document.body.classList.toggle("sidebar-open", Boolean(open));
        if (backdrop) open ? backdrop.removeAttribute("hidden") : backdrop.setAttribute("hidden", "");
      } else {
        toggleRailMode();
      }
    });

    document.getElementById("sidebarBackdrop")?.addEventListener("click", () => {
      document.getElementById("sidebar")?.classList.remove("open");
      document.body.classList.remove("sidebar-open");
      document.getElementById("sidebarBackdrop")?.setAttribute("hidden", "");
    });

    // Theme toggles
    document.getElementById("themeToggleBtn")?.addEventListener("click", () => {
      applyTheme(state.theme === "dark" ? "light" : "dark");
    });
    document.getElementById("settingsDarkToggle")?.addEventListener("click", () => {
      applyTheme(state.theme === "dark" ? "light" : "dark");
    });
    document.getElementById("settingsRailToggle")?.addEventListener("click", toggleRailMode);

    // Dashboard activity range tabs
    document.querySelectorAll("#dashActivityTabs .chart-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll("#dashActivityTabs .chart-tab").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        state.activityRange = tab.dataset.range;
        renderVisibleCharts();
      });
    });

    // Papers filter chips
    document.querySelectorAll("[data-paper-filter]").forEach((chip) => {
      chip.addEventListener("click", () => {
        document.querySelectorAll("[data-paper-filter]").forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        state.paperFilter = chip.dataset.paperFilter;
        renderPapersTable(filterPapers(MOCK_PAPERS, state.paperFilter));
        renderVisibleCharts();
      });
    });

    document.getElementById("btnResetPaperFilters")?.addEventListener("click", () => {
      state.paperFilter = "all";
      document.querySelectorAll("[data-paper-filter]").forEach((c) => {
        c.classList.toggle("active", c.dataset.paperFilter === "all");
      });
      renderPapersTable(MOCK_PAPERS);
      renderVisibleCharts();
      showToast("論文フィルタをリセットしました");
    });

    // Mock action buttons
    document.getElementById("btnMockScrape")?.addEventListener("click", () => {
      showToast("alphaXiv スクレイピングをモック実行しました (間隔 ≥ 2.0s)");
    });

    document.getElementById("btnQuickToast")?.addEventListener("click", () => {
      showToast("システムステータス: すべて正常稼働中 (Gentelella v4 UI)");
    });

    // Algorithm selector cards
    document.querySelectorAll("#algoSelector .algo-card").forEach((card) => {
      card.addEventListener("click", () => {
        document.querySelectorAll("#algoSelector .algo-card").forEach((c) => c.classList.remove("selected"));
        card.classList.add("selected");
        state.selectedAlgo = card.dataset.algo;
        showToast(`アルゴリズムを ${state.selectedAlgo} に切り替えました`);
      });
    });

    // Mock AI Advisor chat
    document.getElementById("btnMockAskAdvisor")?.addEventListener("click", () => {
      const input = document.getElementById("advisorQuestionInput");
      const log = document.getElementById("advisorChatLog");
      const q = input?.value.trim();
      if (!q || !log) return;
      const userDiv = document.createElement("div");
      userDiv.className = "chat-msg user";
      userDiv.textContent = q;
      log.appendChild(userDiv);

      const botDiv = document.createElement("div");
      botDiv.className = "chat-msg assistant";
      botDiv.innerHTML = `<div class="chat-msg-meta">AI Advisor (Template Mock)</div>ご質問ありがとうございます。現在の選択手法 <strong>${state.selectedAlgo}</strong> に対して、<code>smoothness_weight=0.08</code> への微調整を推奨します。`;
      log.appendChild(botDiv);
      log.scrollTop = log.scrollHeight;
      input.value = "";
    });

    document.getElementById("btnStartMockTrain")?.addEventListener("click", () => {
      showToast(`${state.selectedAlgo} 学習ジョブをキューに追加しました (Mock)`);
    });

    document.getElementById("btnToggle3DPlay")?.addEventListener("click", () => {
      state.threePlaying = !state.threePlaying;
      showToast(state.threePlaying ? "3D 骨格再生を再開しました" : "3D 骨格再生を一時停止しました");
    });

    window.addEventListener("resize", () => {
      state.charts.forEach((chart) => chart.resize());
    });

    renderVisibleCharts();
  });
})();
