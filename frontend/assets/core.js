/* =========================================================
   ModelForge · 应用外壳（导航 / 路由 / 顶栏 / AI 助手 / 工作台）
   ========================================================= */
(function (global) {
  const MF = global.MF;
  const { esc, fmt, icon } = MF;
  MF.views = MF.views || {};

  /* ---------------- 通用片段 ---------------- */
  const nowStamp = () => {
    const d = new Date(), p = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  };
  MF.nowStamp = nowStamp;

  MF.pageHead = function (title, sub, actions) {
    return `<div class="page-head"><div><h1>${title}</h1>${sub ? `<p>${sub}</p>` : ''}</div>${actions ? `<div class="actions">${actions}</div>` : ''}</div>`;
  };

  MF.stagePill = function (stage) {
    const map = { candidate: ['候选', 'pill-info'], production: ['生产', 'pill-success'], staging: ['预发布', 'pill-warn'], dev: ['开发', 'pill-info'], archived: ['已归档', 'pill-muted'] };
    const m = map[stage] || ['未知', 'pill-muted'];
    return `<span class="pill ${m[1]}"><span class="pdot"></span>${m[0]}</span>`;
  };
  MF.taskPill = (task) => `<span class="pill pill-info">${esc(MF.TASK_LABELS[task] || task || '—')}</span>`;
  MF.expStatus = function (status) {
    const map = { running: ['训练中', 'pill-info'], done: ['已完成', 'pill-success'], failed: ['失败', 'pill-danger'], queued: ['排队中', 'pill-muted'] };
    const m = map[status] || ['未知', 'pill-muted'];
    return `<span class="pill ${m[1]}">${status === 'running' ? '<span class="pdot pulse"></span>' : '<span class="pdot"></span>'}${m[0]}</span>`;
  };

  MF.statCard = function (o) {
    return `<div class="card stat-card">
      <div class="stat-top"><div class="stat-icon ${o.tone || ''}">${icon(o.icon)}</div><div class="stat-label">${o.label}</div></div>
      <div class="stat-value">${o.value}${o.suffix ? `<small>${o.suffix}</small>` : ''}</div>
      <div class="stat-foot"><div>${o.delta ? `<span class="stat-delta ${o.down ? 'down' : ''}">${icon('trend', 'icon-sm')}${o.delta}<span class="muted">${o.note || ''}</span></span>` : `<span class="small muted">${o.note || ''}</span>`}</div>
        ${o.spark ? MF.charts.spark(o.spark, o.color || '#2563eb') : ''}</div>
    </div>`;
  };

  MF.sectionCard = function (title, sub, actions, body, extraClass) {
    return `<div class="card ${extraClass || ''}">
      <div class="card-head"><div><div class="section-title">${title}</div>${sub ? `<div class="section-sub">${sub}</div>` : ''}</div>${actions ? `<div class="actions">${actions}</div>` : ''}</div>
      <div class="card-body">${body}</div></div>`;
  };

  /* ---------------- 菜单 ---------------- */
  function openMenu(anchor, html) {
    const old = MF.$('.menu-float'); if (old) old.remove();
    const menu = document.createElement('div');
    menu.className = 'menu menu-float';
    menu.innerHTML = html;
    document.body.appendChild(menu);
    const r = anchor.getBoundingClientRect();
    menu.style.top = `${r.bottom + 8}px`;
    menu.style.right = `${Math.max(12, window.innerWidth - r.right)}px`;
    menu.style.position = 'fixed';
    setTimeout(() => {
      const away = (e) => { if (!menu.contains(e.target) && !anchor.contains(e.target)) { menu.remove(); document.removeEventListener('click', away); } };
      document.addEventListener('click', away);
    }, 0);
    return menu;
  }

  function openNotifications(anchor) {
    openMenu(anchor, `<div style="padding:10px 10px 6px;font-size:12px;font-weight:700;color:#0f172a">通知</div>
      ${[['模型训练完成', '客户流失预测 v2.2 已完成，AUC 0.941', '3 分钟前'], ['服务健康提醒', 'house-price-api 延迟升高至 74ms', '26 分钟前'], ['数据质量报告', '客户流失预测集质量分提升至 91', '昨天 14:32']]
        .map(([t, d, time]) => `<div class="menu-item" style="align-items:flex-start"><div style="width:26px;height:26px;border-radius:7px;background:var(--primary-50);color:var(--primary-600);display:grid;place-items:center">${icon('bell', 'icon-sm')}</div>
          <div style="flex:1"><div style="font-weight:600;color:#0f172a">${t}</div><div class="tiny muted" style="margin-top:2px">${d}</div><div class="tiny" style="color:#94a3b8;margin-top:2px">${time}</div></div></div>`).join('')}
      <div class="menu-sep"></div><div class="menu-item" style="justify-content:center;color:var(--primary-600)">全部标记为已读</div>`);
  }

  function openUserMenu(anchor) {
    const user = MF.state.user;
    const menu = openMenu(anchor, `<div style="padding:9px 10px;border-bottom:1px solid var(--line);margin-bottom:4px"><div style="font-size:12px;font-weight:700">${esc(user.name)}</div><div class="tiny muted">${esc(user.account)} · ${esc(user.role)}</div></div>
      <div class="menu-item" data-user-profile>${icon('user', 'icon-sm')}账号信息</div>
      <div class="menu-item" data-user-settings>${icon('settings', 'icon-sm')}大模型配置</div>
      <div class="menu-sep"></div>
      <div class="menu-item danger" data-user-logout>${icon('logout', 'icon-sm')}退出登录</div>`);
    MF.$('[data-user-profile]', menu).onclick = () => { menu.remove(); MF.toast('账号信息', `${user.name} · ${user.role} · 该账号由平台管理员维护。`, 'info'); };
    MF.$('[data-user-settings]', menu).onclick = () => { menu.remove(); location.hash = '#/settings'; };
    MF.$('[data-user-logout]', menu).onclick = async () => {
      menu.remove();
      if (await MF.confirm({ title: '退出登录', message: '退出后需要重新登录才能继续使用平台。', okText: '退出登录', danger: true })) {
        MF.session.clear(); location.replace('login.html');
      }
    };
  }

  /* ---------------- 全局搜索 ---------------- */
  function openSearch() {
    const pages = [
      { type: '页面', name: '工作台', meta: '总览与近期任务', href: '#/overview' },
      { type: '页面', name: '我的数据', meta: '上传、查看与清洗', href: '#/datasets' },
      { type: '页面', name: '新建模型', meta: '选数据 → 自动训练', href: '#/build' },
      { type: '页面', name: '我的模型', meta: '训练结果与效果验证', href: '#/models' },
      { type: '页面', name: '试一下', meta: '用真实数据测试模型', href: '#/inference' },
      { type: '页面', name: '服务管理', meta: '对外提供接口', href: '#/services' },
      { type: '页面', name: '大模型配置', meta: 'AI 服务设置', href: '#/settings' }
    ];
    MF.modal({
      title: '搜索', desc: '快速跳转到页面、数据集或模型', slim: true, foot: null,
      body: `<div class="search-box" style="width:100%;background:#fff;height:40px">${icon('search', 'icon-sm')}<input id="gs-input" placeholder="输入名称，例如：流失、模型、服务" autofocus></div>
        <div id="gs-list" style="margin-top:10px;max-height:320px;overflow-y:auto"></div>`,
      onMount(el, close) {
        const input = MF.$('#gs-input', el), list = MF.$('#gs-list', el);
        let items = pages.slice();
        const draw = () => {
          const q = input.value.trim().toLowerCase();
          const res = items.filter((i) => !q || (i.name + i.meta + i.type).toLowerCase().indexOf(q) >= 0).slice(0, 10);
          list.innerHTML = res.length ? res.map((i, idx) => `<a class="menu-item" href="${i.href}" data-gs="${idx}" style="color:var(--text)"><span class="tag">${i.type}</span><div style="flex:1"><div style="font-weight:600;font-size:12px">${esc(i.name)}</div><div class="tiny muted">${esc(i.meta)}</div></div>${icon('chevron-right', 'icon-sm')}</a>`).join('')
            : `<div class="empty" style="padding:26px"><p>没有匹配结果</p></div>`;
          MF.$$('a[data-gs]', list).forEach((a) => (a.onclick = close));
        };
        input.oninput = draw;
        input.onkeydown = (e) => { if (e.key === 'Enter') { const first = MF.$('a[data-gs]', list); if (first) first.click(); } };
        draw();
        // 异步补充真实数据集与模型
        Promise.all([MF.api.listDatasets().catch(() => []), MF.api.listModels().catch(() => [])]).then(([ds, ms]) => {
          items = pages.concat(
            ds.map((d) => ({ type: '数据集', name: d.name, meta: `${MF.TASK_LABELS[d.task_hint] || d.task_hint} · ${fmt(d.rows)} 行`, href: `#/datasets/${d.id}` })),
            ms.map((m) => ({ type: '模型', name: m.name, meta: `v${m.version} · ${m.architecture}`, href: `#/models/${m.id}` }))
          );
          draw();
        });
        setTimeout(() => input.focus(), 50);
      }
    });
  }

  /* ---------------- AI 助手（接入默认大模型） ---------------- */
  const ASSISTANT_SYSTEM = '你是 ModelForge 神经网络模型生产平台的智能助手。用简洁专业的中文回答用户关于数据上传、清洗、EDA、网络结构、训练超参、评估指标、模型登记与服务发布的问题。若用户提到具体数据集/模型，建议其到对应详情页查看；不要编造平台不存在的功能。';

  function openAssistant() {
    if (MF.$('.drawer')) return;
    const mask = document.createElement('div'); mask.className = 'drawer-mask';
    const drawer = document.createElement('div'); drawer.className = 'drawer';
    drawer.innerHTML = `
      <div class="drawer-head"><div class="ds-ico" style="width:30px;height:30px;background:var(--primary-50)">${icon('sparkles', 'icon-sm')}</div>
        <div><div class="t">平台 AI 助手</div><div class="tiny muted" data-assistant-provider>内置平台向导</div></div>
        <button class="icon-btn ml-auto" data-close-drawer>${icon('x')}</button></div>
      <div class="drawer-body" data-chat></div>
      <div class="drawer-foot"><input class="input" placeholder="询问数据、建模、训练或发布问题…" data-chat-input>
        <button class="send" data-chat-send>${icon('arrow-right')}</button></div>`;
    document.body.appendChild(mask); document.body.appendChild(drawer);
    MF.api.listProviders().then((list) => {
      const def = (list || []).find((p) => p.is_default);
      const el = MF.$('[data-assistant-provider]', drawer);
      if (el && def) el.textContent = `${def.name} · ${def.model || ''}`;
    }).catch(() => {});
    const chat = MF.$('[data-chat]', drawer);
    chat.innerHTML = `<div class="chat-msg ai"><div class="who">AI</div><div class="chat-bubble">我是平台助手，可以基于当前数据资产、模型版本与训练结果为你提供建议。试试下面的问题：
      <div class="chat-suggest"><button>这份数据需要怎样清洗？</button><button>推荐什么网络结构？</button><button>如何提升流失类召回率？</button></div></div></div>`;
    MF.$('[data-close-drawer]', drawer).onclick = () => { drawer.remove(); mask.remove(); };
    mask.onclick = () => { drawer.remove(); mask.remove(); };
    const send = async (text) => {
      const q = (text || MF.$('[data-chat-input]', drawer).value || '').trim();
      if (!q) return;
      MF.$('[data-chat-input]', drawer).value = '';
      chat.insertAdjacentHTML('beforeend', `<div class="chat-msg me"><div class="who">我</div><div class="chat-bubble">${esc(q)}</div></div>
        <div class="chat-msg ai" data-thinking><div class="who">AI</div><div class="chat-bubble"><span class="pulse">正在请求大模型…</span></div></div>`);
      chat.scrollTop = chat.scrollHeight;
      const thinking = MF.$('[data-thinking]', chat);
      try {
        const res = await MF.api.llmChat([{ role: 'user', content: q }], ASSISTANT_SYSTEM);
        const reply = (res && res.reply) || '（未收到回复）';
        if (thinking) thinking.querySelector('.chat-bubble').innerHTML = esc(reply).replace(/\n/g, '<br>');
      } catch (e) {
        if (thinking) thinking.querySelector('.chat-bubble').innerHTML =
          `<span style="color:#dc2626">调用失败：</span>${esc(e.message || e)}<br><span class="tiny muted">请在「大模型配置」中确认默认服务已配置且测试连接通过。</span>`;
      }
      chat.scrollTop = chat.scrollHeight;
    };
    MF.$('[data-chat-send]', drawer).onclick = () => send();
    MF.$('[data-chat-input]', drawer).onkeydown = (e) => { if (e.key === 'Enter') send(); };
    MF.$$('.chat-suggest button', drawer).forEach((b) => (b.onclick = () => send(b.textContent)));
    setTimeout(() => MF.$('[data-chat-input]', drawer).focus(), 80);
  }
  MF.openAssistant = openAssistant;

  /* ---------------- 路由 ---------------- */
  function renderNav(active) {
    const groups = [
      { label: '开始', items: [['overview', 'home', '工作台']] },
      { label: '做模型', items: [['datasets', 'database', '我的数据'], ['build', 'sparkles', '新建模型'], ['models', 'box', '我的模型'], ['inference', 'cpu', '试一下']] },
      { label: '上线', items: [['services', 'rocket', '服务管理']] },
      { label: '设置', items: [['settings', 'settings', 'AI 服务配置']] }
    ];
    MF.$('#side-nav').innerHTML = groups.map((g) => `<div class="nav-group"><div class="nav-group-label">${g.label}</div>
      ${g.items.map(([key, ico, label]) => `<a class="nav-item ${active === key ? 'active' : ''}" href="#/${key}">${icon(ico)}<span>${label}</span>${key === 'build' ? '<span class="nav-badge">AI</span>' : ''}</a>`).join('')}</div>`).join('');
    // 侧栏大模型状态：真实读取默认 Provider
    MF.api.listProviders().then((list) => {
      const box = MF.$('#llm-status');
      if (!box) return;
      const def = (list || []).find((p) => p.is_default);
      box.innerHTML = def
        ? `<span class="dot" style="background:${def.status === 'active' ? '#0f9d76' : '#f59e0b'}"></span><span class="label">大模型服务</span><span class="value">${esc(def.name)} · ${def.status === 'active' ? '在线' : '未验证'}</span>`
        : `<span class="dot" style="background:#94a3b8"></span><span class="label">大模型服务</span><span class="value">未配置</span>`;
    }).catch(() => {});
  }

  function setCrumbs(page, r) {
    const map = { overview: ['工作台'], datasets: ['我的数据'], build: ['新建模型'], train: ['训练记录'], models: ['我的模型'], inference: ['试一下'], services: ['服务管理'], settings: ['AI 服务配置'] };
    let items = (map[page] || ['工作台']).slice();
    if (page === 'dataset') items = ['数据资产', '数据集详情'];
    if (page === 'model') items = ['模型仓库', '模型详情'];
    render(items);
    // 详情页异步补全实体名称
    if (page === 'dataset' && r.id) {
      MF.api.getDataset(r.id).then((d) => render(['数据资产', (d && d.name) || '数据集详情'])).catch(() => {});
    }
    if (page === 'model' && r.id) {
      MF.api.getModel(r.id).then((m) => render(['模型仓库', m && m.name ? `${m.name} v${m.version}` : '模型详情'])).catch(() => {});
    }
    function render(list) {
      MF.$('#crumbs').innerHTML = list.map((t, i) => `${i ? '<span class="sep">/</span>' : ''}<span class="${i === list.length - 1 ? 'cur' : ''}">${esc(t)}</span>`).join('');
    }
  }

  function parseRoute() {
    const raw = location.hash.replace(/^#/, '') || '/overview';
    const [path, qs] = raw.split('?');
    const parts = path.split('/').filter(Boolean);
    return { page: parts[0] || 'overview', id: parts[1] || null, query: new URLSearchParams(qs || ''), hash: raw };
  }
  MF.parseRoute = parseRoute;

  function notFound(r, view) {
    view.innerHTML = `<div class="card card-pad"><div class="empty"><div class="ico">${icon('alert')}</div><h4>页面不存在</h4><p>请从左侧导航进入对应功能。</p><a class="btn btn-primary" href="#/overview">返回工作台</a></div></div>`;
  }

  function route() {
    const r = parseRoute();
    const activeMap = { dataset: 'datasets', model: 'models' };
    renderNav(activeMap[r.page] || r.page);
    setCrumbs(r.page, r);
    const view = MF.$('#view');
    if (!view) return;
    view.className = 'content';
    const fn = MF.views[r.page];
    if (fn) fn(r, view); else notFound(r, view);
    view.classList.add('view');
    window.scrollTo(0, 0);
  }
  MF.route = route;

  /* ---------------- 工作台 ---------------- */
  MF.views.overview = function (r, view) {
    const user = MF.state.user;
    const hour = new Date().getHours();
    const greet = hour < 6 ? '凌晨好' : hour < 11 ? '上午好' : hour < 14 ? '中午好' : hour < 18 ? '下午好' : '晚上好';
    const savedDraft = MF.wizardStore.load();
    const draft = savedDraft && !savedDraft.done ? savedDraft : null;
    const ARCH = { mlp: 'MLP', cnn: 'CNN', emb_mlp: 'EMB_MLP' };

    const head = MF.pageHead(`${greet}，${esc(user.name)}`, '上传你的数据，系统会自动帮你训练出一个能用的模型。',
      `<button class="btn" data-upload>${icon('upload')}上传数据</button><a class="btn btn-primary" href="#/build?new=1">${icon('plus')}新建模型</a>`);
    view.innerHTML = `${head}<div class="card card-pad"><div class="notice">${icon('refresh', 'icon-sm spin')}<div>正在加载工作台数据…</div></div></div>`;
    MF.$('[data-upload]', view).onclick = () => MF.openUploadModal && MF.openUploadModal();

    Promise.all([
      MF.api.listDatasets(),
      MF.api.listModels(),
      MF.api.listServices(),
      MF.api.trainJobs(),
      MF.api.listProviders().catch(() => [])
    ]).then(([datasets, models, services, jobs, providers]) => render(datasets, models, services, jobs, providers))
      .catch((e) => {
        view.innerHTML = `${head}<div class="card card-pad"><div class="notice danger">${icon('alert', 'icon-sm')}<div><b>工作台数据加载失败</b><br>${esc(e.message || e)}</div></div></div>`;
        MF.$('[data-upload]', view).onclick = () => MF.openUploadModal && MF.openUploadModal();
      });

    function render(datasets, models, services, jobs, providers) {
      const running = services.filter((s) => s.status === 'running').length;
      const doneJobs = jobs.filter((j) => j.status === 'done').length;
      const kpis = [
        { label: '训练好的模型', value: models.length, suffix: '个', icon: 'box', note: '可以拿去用' },
        { label: '对外服务中', value: running, suffix: '个', icon: 'rocket', tone: 'green', note: `共 ${services.length} 个服务` },
        { label: '训练次数', value: jobs.length, suffix: '次', icon: 'cpu', tone: 'purple', note: `已完成 ${doneJobs} 次` },
        { label: '我的数据', value: datasets.length, suffix: '份', icon: 'database', tone: 'amber', note: '已上传的数据' }
      ];

      const expRows = jobs.slice(0, 5).map((j) => {
        const pct = Math.min(100, Math.round(Number(j.progress || 0) * 100));
        const arch = (j.metrics || {}).arch || (j.config || {}).arch || '';
        return `<tr>
        <td><div class="cell-main">${esc(j.dataset_name || '训练任务')}</div><div class="cell-sub mono">${esc(j.id)}</div></td>
        <td>${MF.taskPill(j.task_hint)}</td>
        <td class="small muted">${esc((ARCH[arch] || arch || '—').toUpperCase())}</td>
        <td>${MF.expStatus(j.status)}</td>
        <td style="min-width:120px"><div class="flex items-center gap-8"><div class="bar ${j.status === 'failed' ? 'warn' : ''}" style="flex:1"><span style="width:${pct}%"></span></div><span class="tiny muted">${pct}%</span></div></td>
        <td class="small muted nowrap">${esc(j.created_at)}</td>
        <td><a class="btn btn-sm btn-ghost" href="#/train?job=${esc(j.id)}">查看</a></td></tr>`;
      }).join('');
      const expTable = jobs.length
        ? `<div class="table-wrap" style="margin:0 -20px -20px"><table class="tbl"><thead><tr><th>训练任务</th><th>任务类型</th><th>网络结构</th><th>状态</th><th>进度</th><th>开始时间</th><th></th></tr></thead><tbody>${expRows}</tbody></table></div>`
        : `<div class="empty" style="padding:30px"><div class="ico">${icon('cpu')}</div><p>还没有训练任务</p><a class="btn btn-sm btn-primary mt-8" href="#/build?new=1">开始第一个训练</a></div>`;

      const serviceRows = services.slice(0, 3).map((s) => `<div class="flex items-center gap-12" style="padding:10px 0;border-bottom:1px solid #eef2f8">
        <span class="badge-dot ${s.status === 'running' ? '' : 'gray'}"></span>
        <div style="min-width:0;flex:1"><div class="small" style="font-weight:600;color:#1e293b">${esc(s.name)}</div>
          <div class="tiny muted">${esc(s.model_name || '—')}${s.model_version ? ' v' + esc(s.model_version) : ''} · ${s.status === 'running' ? '运行中' : '已停止'}</div></div>
        <div class="text-right"><div class="small" style="font-weight:600">${fmt(s.requests_total || 0)}</div><div class="tiny muted">累计调用</div></div></div>`).join('');
      const serviceBody = services.length
        ? serviceRows
        : `<div class="empty" style="padding:24px"><p>还没有发布服务</p><a class="btn btn-sm btn-ghost mt-8" href="#/services">去发布</a></div>`;

      const defProvider = (providers || []).find((p) => p.is_default);
      const dirty = datasets.filter((d) => { const st = ((d.meta || {}).stats) || {}; return (st.missing_cells || 0) > 0 || (st.duplicate_rows || 0) > 0; });
      const insight = !datasets.length
        ? '还没有数据集。点击「导入数据」上传 CSV 或图像 ZIP，开始第一个建模任务。'
        : dirty.length
          ? `有 ${dirty.length} 个数据集存在缺失值或重复记录（如「${esc(dirty[0].name)}」），建议先在数据资产页执行清洗，再开始训练。`
          : '数据资产状态良好。可以从「新建模型」开始，按引导完成训练、评估与服务发布。';

      const modelRows = models.slice(0, 3).map((m) => `<a class="flex items-center gap-10" href="#/models/${esc(m.id)}" style="padding:9px 0;border-bottom:1px solid #eef2f8;color:inherit">
        <div class="ds-ico">${icon('box')}</div><div style="flex:1;min-width:0"><div class="small" style="font-weight:600">${esc(m.name)}</div><div class="tiny muted">v${esc(m.version)} · ${esc(ARCH[m.architecture] || m.architecture || '—')}</div></div>${MF.stagePill(m.stage)}</a>`).join('');
      const modelBody = models.length
        ? modelRows
        : `<div class="empty" style="padding:24px"><p>模型仓库为空</p><a class="btn btn-sm btn-ghost mt-8" href="#/build?new=1">训练并登记第一个模型</a></div>`;

      view.innerHTML = `
        ${head}
        <div class="grid grid-4">${kpis.map(MF.statCard).join('')}</div>
        <div class="grid grid-2-1 mt-20">
          ${MF.sectionCard('近期训练任务', '训练记录与实时状态', `<a class="btn btn-sm btn-ghost" href="#/train">全部任务 ${icon('arrow-right', 'icon-sm')}</a>`, expTable, 'card-accent')}
          <div style="display:flex;flex-direction:column;gap:18px">
            <div class="card card-pad">
              <div class="section-title">${icon('sparkles', 'icon-sm')} 三步做出一个模型</div>
              <div class="section-sub" style="margin-bottom:14px">不用懂机器学习，跟着走就行</div>
              <div class="kv-list">
                <div class="kv-row"><span class="k">1 · 把数据传上来</span><span class="v">系统自动看懂格式</span></div>
                <div class="kv-row"><span class="k">2 · 点「开始训练」</span><span class="v">自动选模型和参数</span></div>
                <div class="kv-row"><span class="k">3 · 看结果，拿来用</span><span class="v">能测效果也能对外发布</span></div>
              </div>
              <div class="mt-16">
                ${draft ? `<a class="btn btn-primary btn-block" href="#/build?resume=1">${icon('play', 'icon-sm')}继续上次 · 第 ${draft.step || 1} 步</a><button class="btn btn-ghost btn-block mt-8" data-discard-draft>放弃草稿</button>`
                  : `<a class="btn btn-primary btn-block" href="#/build?new=1">${icon('sparkles', 'icon-sm')}开始训练模型</a>`}
              </div>
            </div>
            <div class="card card-pad" style="background:linear-gradient(150deg,#f7faff,#f2f7ff);border-color:var(--primary-100)">
              <div class="flex items-center gap-8"><span class="badge-ai">${icon('sparkles', 'icon-sm')}AI 洞察</span><span class="tiny muted">${esc(defProvider ? defProvider.name : '未配置大模型')}</span></div>
              <p class="small mt-12" style="color:#475569;line-height:1.7">${insight}</p>
              <button class="btn btn-sm btn-soft mt-12" data-open-ai>${icon('sparkles', 'icon-sm')}让助手继续分析</button>
            </div>
          </div>
        </div>
        <div class="grid grid-2-1 mt-20">
          ${MF.sectionCard('服务健康度', '在线服务运行状态', `<a class="btn btn-sm btn-ghost" href="#/services">服务管理 ${icon('arrow-right', 'icon-sm')}</a>`, serviceBody)}
          ${MF.sectionCard('最近模型', '模型仓库中的最新版本', `<a class="btn btn-sm btn-ghost" href="#/models">查看全部</a>`, modelBody)}
        </div>`;

      MF.$('[data-upload]', view).onclick = () => MF.openUploadModal && MF.openUploadModal();
      MF.$('[data-open-ai]', view).onclick = openAssistant;
      const discard = MF.$('[data-discard-draft]', view);
      if (discard) discard.onclick = async () => {
        if (await MF.confirm({ title: '放弃草稿', message: '草稿中的步骤配置将被清除，已保存的数据与模型不受影响。', okText: '放弃草稿', danger: true })) {
          MF.wizardStore.clear(); route(); MF.toast('已清除草稿', '', 'success');
        }
      };
    }
  };

  /* ---------------- 启动 ---------------- */
  MF.boot = async function () {
    if (!MF.session.guard()) return;
    // 向后端校验 token 有效性；401 时 api.js 会自动清理会话并跳回登录页
    try {
      const me = await MF.api.me();
      const s = MF.session.get();
      MF.session.set(Object.assign({}, s, {
        account: me.account, name: me.name, role: me.role, org: me.org || ''
      }));
    } catch (e) {
      if (e && /无法连接后端/.test(e.message || '')) {
        MF.toast('后端服务未启动', e.message, 'error', 6000);
      }
      return;
    }
    const user = MF.state.user;
    MF.$$('[data-user-name]').forEach((el) => (el.textContent = user.name));
    MF.$$('[data-user-role]').forEach((el) => (el.textContent = user.role));
    MF.$$('[data-user-initial]').forEach((el) => (el.textContent = (user.name || '访客').slice(-2)));
    MF.$('#btn-global-search').onclick = openSearch;
    const gsInput = MF.$('#global-search');
    if (gsInput) {
      gsInput.onfocus = () => { gsInput.blur(); openSearch(); };
      gsInput.onkeydown = (e) => { if (e.key === 'Enter') { gsInput.blur(); openSearch(); } };
    }
    MF.$('#btn-ai').onclick = openAssistant;
    MF.$('#btn-notify').onclick = (e) => openNotifications(e.currentTarget);
    MF.$('#top-user').onclick = (e) => openUserMenu(e.currentTarget);
    document.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openSearch(); }
    });
    window.addEventListener('hashchange', route);
    if (!location.hash) location.hash = '#/overview'; else route();
    if (!location.hash) route();
  };
})(window);
