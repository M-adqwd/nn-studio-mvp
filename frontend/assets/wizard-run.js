/* =========================================================
   ModelForge · 训练监控 / 运行页（真实后端 API 版）
   路由：#/train（最近任务列表） · #/train?job=<id>（单任务监控）
   每 1.5s 轮询 MF.api.trainJob(id)，支持刷新恢复。
   ========================================================= */
(function (global) {
  const MF = global.MF;
  const { esc, fmt, icon } = MF;

  let pollTimer = null;
  function stopPoll() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }
  window.addEventListener('hashchange', stopPoll);

  const loadingHTML = (t) => `<div class="card card-pad"><div class="notice">${icon('refresh', 'icon-sm spin')}<div>${esc(t || '正在加载…')}</div></div></div>`;
  const errHTML = (e) => `<div class="card card-pad"><div class="notice danger">${icon('alert', 'icon-sm')}<div><b>请求失败</b><br>${esc(e.message || e)}</div></div>
    <div class="flex gap-10 mt-12"><button class="btn btn-sm" data-retry>${icon('refresh', 'icon-sm')}重试</button><a class="btn btn-sm btn-ghost" href="#/train">任务列表</a></div></div>`;
  function bindRetry(view) {
    const retry = MF.$('[data-retry]', view);
    if (retry) retry.onclick = () => MF.route();
  }

  function sec(title, num, body, sub) {
    return `<div class="card card-pad" style="margin-bottom:14px">
      <div class="wiz-section-title">${num ? `<span class="num">${num}</span>` : ''}${title}</div>
      ${sub ? `<div class="section-sub" style="margin:-4px 0 12px ${num ? '28px' : '0'}">${sub}</div>` : ''}
      ${body}</div>`;
  }
  const round = (v, p) => { const k = Math.pow(10, p || 4); return Math.round(Number(v) * k) / k; };

  function historyOf(job) { return (job.metrics && job.metrics.history) || []; }
  function bestOf(job) { return (job.metrics && job.metrics.best_val) || historyOf(job).slice(-1)[0] || null; }
  function isReg(job) { return ((job.metrics && job.metrics.task) || '') === 'regression' || job.task_hint === 'tabular_regression'; }
  function lastOf(job) { const h = historyOf(job); return h.length ? h[h.length - 1] : null; }
  const pctOf = (job) => Math.min(100, Math.round(Number(job.progress || 0) * 100));

  function logsHTML(job) {
    const logs = (job.logs || []).slice().reverse();
    if (!logs.length) return '<div class="ln"><span class="hi">等待任务日志…</span></div>';
    return logs.map((x) => `<div class="ln">${esc(x)}</div>`).join('');
  }

  function drawChart(job) {
    const el = MF.$('#tr-loss-chart');
    if (!el) return;
    const h = historyOf(job);
    if (!h.length) {
      const stage = (job.metrics && job.metrics.stage) || '';
      el.innerHTML = '<div class="chart-empty">' +
        (stage ? esc(stage) + '…' : '等待第一轮训练数据…') + '</div>';
      return;
    }
    // 多轮迭代时标出「换方案」的分界，让用户知道曲线为什么会有转折
    const markers = [];
    for (let i = 1; i < h.length; i++) {
      const prev = h[i - 1].round_epoch;
      const cur = h[i].round_epoch;
      if (prev !== undefined && cur !== undefined && cur <= prev) {
        markers.push({ index: i, label: '换方案', color: '#f59e0b' });
      }
    }
    // 检测任务没有 loss 曲线，改画 mAP
    const isDet = h[0] && (h[0]['metrics/mAP50(B)'] !== undefined
      || h[0].mAP50 !== undefined);
    if (isDet) {
      const pick = (r) => (r['metrics/mAP50(B)'] !== undefined
        ? r['metrics/mAP50(B)'] : r.mAP50);
      el.innerHTML = MF.charts.line({
        series: [
          { name: 'mAP50', color: '#2563eb', data: h.map(pick), area: true },
          { name: 'mAP50-95', color: '#0f9d76',
            data: h.map((r) => (r['metrics/mAP50-95(B)'] !== undefined
              ? r['metrics/mAP50-95(B)'] : r['metrics/mAP50-95(B)'])) }
        ],
        labels: h.map((x) => 'E' + x.epoch), height: 240, yMin: 0, yMax: 1,
        markers: markers
      });
      return;
    }
    el.innerHTML = MF.charts.line({
      series: [
        { name: '训练损失', color: '#2563eb', data: h.map((x) => x.train_loss), area: true },
        { name: '验证损失', color: '#0f9d76', data: h.map((x) => x.val_loss) }
      ],
      labels: h.map((x) => 'E' + x.epoch), height: 240, yMin: 0,
      markers: markers
    });
  }

  /* ---------------- 最近任务列表 ---------------- */
  function renderList(view) {
    view.innerHTML = loadingHTML('正在加载训练任务…');
    MF.api.trainJobs().then((jobs) => {
      const rows = jobs.map((j) => {
        const pct = pctOf(j);
        return `<tr>
          <td><div class="cell-main mono">${esc(j.id)}</div><div class="cell-sub">${esc(j.created_at || '')}</div></td>
          <td class="small">${esc(j.dataset_name || j.dataset_id || '—')}</td>
          <td>${j.task_hint ? MF.taskPill(j.task_hint) : '<span class="small muted">—</span>'}</td>
          <td>${MF.expStatus(j.status)}</td>
          <td style="min-width:130px"><div class="flex items-center gap-8"><div class="bar" style="flex:1"><span style="width:${pct}%"></span></div><span class="tiny muted">${pct}%</span></div></td>
          <td class="small muted nowrap">${esc(j.updated_at || '')}</td>
          <td><a class="btn btn-sm btn-ghost" href="#/train?job=${esc(j.id)}">查看</a></td>
        </tr>`;
      }).join('');
      view.innerHTML = `
        ${MF.pageHead('训练监控', '服务端训练任务的实时进度、损失曲线与日志。',
          `<a class="btn btn-primary" href="#/build">${icon('sparkles')}新建训练任务</a>`)}
        <div class="card">
          <div class="table-wrap"><table class="tbl">
            <thead><tr><th>任务</th><th>数据集</th><th>任务类型</th><th>状态</th><th>进度</th><th>更新时间</th><th></th></tr></thead>
            <tbody>${rows}</tbody></table></div>
          ${jobs.length ? '' : `<div class="tbl-empty">${icon('cpu', 'icon-lg')}<div class="mt-8">还没有训练任务</div><div class="tiny muted mt-4">通过「新建训练任务」在引导式建模中提交第一个任务</div></div>`}
        </div>`;
    }).catch((e) => { view.innerHTML = errHTML(e); bindRetry(view); });
  }

  /* ---------------- 单任务监控 ---------------- */

  // 把准确率翻译成「这个模型到底行不行」的结论。
  // 小白不知道 55% 是好是坏，但他们知道「随机猜」是什么水平。
  function qualityVerdict(job) {
    const best = bestOf(job) || {};
    const m = job.metrics || {};
    const isDet = (m.task === 'detection') || (m.arch === 'yolo')
      || /mAP/i.test(JSON.stringify(m.best_val || {}));
    // 检测任务：看 mAP50
    if (isDet) {
      const map = (m.best_val || {})['metrics/mAP50(B)'];
      if (typeof map !== 'number') return null;
      const pct = map * 100;
      if (pct >= 70) return { level: 'success', title: '检测效果很好',
        text: `mAP50 达到 ${pct.toFixed(1)}%，目标基本都能认准、框也够准。` };
      if (pct >= 45) return { level: '', title: '检测效果尚可',
        text: `mAP50 为 ${pct.toFixed(1)}%，多数目标能识别，但小目标或遮挡场景可能不准。` };
      return { level: 'warn', title: '检测效果偏弱',
        text: `mAP50 仅 ${pct.toFixed(1)}%。可能是数据量偏少或目标过小，建议增加数据后再训。` };
    }
    const acc = best.val_acc;
    if (typeof acc !== 'number') return null;
    const pct = acc * 100;
    // 类别数决定随机猜的基线
    const ncls = (m.datacard && m.datacard.n_classes) || 0;
    const base = ncls > 1 ? 100 / ncls : 50;
    if (pct >= 90) return { level: 'success', title: '模型效果很好',
      text: `准确率 ${pct.toFixed(1)}%，在你的数据上判断得很准。` };
    if (pct >= base + 20) return { level: '', title: '模型效果不错',
      text: `准确率 ${pct.toFixed(1)}%，明显好于瞎猜的 ${base.toFixed(0)}%（共 ${ncls} 类），可用于实际判断。` };
    if (pct > base + 5) return { level: 'warn', title: '模型效果一般',
      text: `准确率 ${pct.toFixed(1)}%，只比瞎猜的 ${base.toFixed(0)}% 好一点。建议检查数据质量，或增加数据量再训一次。` };
    return { level: 'danger', title: '模型基本没学到东西',
      text: `准确率 ${pct.toFixed(1)}%，与瞎猜（${base.toFixed(0)}%）相当。多半是数据本身缺少可学习的规律，建议核对标签是否正确。` };
  }

  // 智能体的迭代过程：用户需要看到「系统试了几种方案、最终选了哪个」，
  // 否则训练结束就是几个数字凭空出现，无法建立信任。
  function roundsCard(job) {
    const m = job.metrics || {};
    const rounds = m.rounds || [];
    if (rounds.length < 2) return '';
    const isDet = (m.task === 'detection') || (m.arch === 'yolo');
    const rows = rounds.map((r) => {
      const p = r.plan || {};
      const v = r.val || {};
      const acc = isDet
        ? (v['metrics/mAP50(B)'] !== undefined ? (v['metrics/mAP50(B)'] * 100).toFixed(1) + '%' : '—')
        : (v.acc !== undefined ? (v.acc * 100).toFixed(1) + '%' : '—');
      const mode = p.mode === 'finetune' ? '微调主干' : '冻结主干';
      return `<tr>
        <td class="small">第 ${r.round} 轮</td>
        <td class="small">${esc(p.backbone || '—')} · ${p.img_size || '—'}px</td>
        <td class="small">${mode}</td>
        <td class="small">${esc(p.augment || '—')}</td>
        <td class="num">${acc}</td>
        <td class="small muted">${esc(r.verdict || '')}</td>
      </tr>`;
    }).join('');
    return sec('系统试过几种方案', '', `<div class="table-wrap"><table class="tbl">
        <thead><tr><th>轮次</th><th>方法</th><th>训练方式</th><th>数据增强</th><th>${isDet ? 'mAP50' : '准确率'}</th><th>诊断</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      <div class="hint mt-8">系统会自动尝试不同方案并保留效果最好的一轮。上面每一行都是一次真实尝试。</div>`, '智能体的自动尝试过程');
  }

  function metricCards(job) {
    const best = bestOf(job) || {};
    const reg = isReg(job);
    const m = job.metrics || {};
    const isDet = (m.task === 'detection') || (m.arch === 'yolo');
    const map50 = (m.best_val || {})['metrics/mAP50(B)'];
    const main = isDet
      ? ['检测准确度 mAP50', typeof map50 === 'number' ? (map50 * 100).toFixed(1) + '%' : '—']
      : reg
      ? ['预测误差 RMSE', best.val_rmse !== undefined ? String(round(best.val_rmse, 4)) : '—']
      : ['准确率', best.val_acc !== undefined ? (best.val_acc * 100).toFixed(1) + '%' : '—'];
    const cards = [main,
      ['训练轮次', `${job.epoch || 0} / ${job.total_epochs || 0}`],
      ['是否提前停止', m.stopped_early ? '是（已自动止损）' : '否'],
      ['最终检验', m.test_metrics && m.test_metrics.test_acc !== undefined
        ? (m.test_metrics.test_acc * 100).toFixed(1) + '%'
        : (m.test_metrics && m.test_metrics.test_rmse !== undefined
          ? String(round(m.test_metrics.test_rmse, 4)) : '—')]];
    // 训练损失这类数字对小白意义不大，收进技术细节
    const tech = `<details class="mt-12"><summary class="small muted" style="cursor:pointer">技术细节（可跳过）</summary>
      <div class="kv-list mt-8">
        <div class="kv-row"><span class="k">训练损失</span><span class="v">${best.train_loss !== undefined ? round(best.train_loss, 4) : '—'}</span></div>
        <div class="kv-row"><span class="k">验证损失</span><span class="v">${best.val_loss !== undefined ? round(best.val_loss, 4) : '—'}</span></div>
        <div class="kv-row"><span class="k">模型类型</span><span class="v">${esc(m.arch || '—')}</span></div>
      </div></details>`;
    return `<div class="grid grid-4">${cards.map(([l, v]) => `<div class="metric-card"><div class="l">${l}</div><div class="v">${v}</div><div class="d">${reg ? '预测数值' : '判断类别'}</div></div>`).join('')}</div>${tech}`;
  }

  function adviceCard(m) {
    const a = m.advice || {};
    // 优先用面向小白的文案；没有 plain_* 时退回技术文案
    const summary = a.plain_summary || a.summary;
    const tips = a.plain_tips && a.plain_tips.length ? a.plain_tips : (a.tips || []);
    if (!summary && !tips.length) return '';
    const map = { bad: ['danger', 'x-circle', '这个模型还需要改进'], warn: ['warn', 'alert', '可以再优化一下'], ok: ['success', 'check-circle', '训练诊断'] };
    const [cls, ic, title] = map[a.level] || map.ok;
    const tech = (a.plain_summary && a.summary && a.plain_summary !== a.summary)
      ? `<details class="mt-8"><summary class="tiny muted" style="cursor:pointer">技术说明</summary>
         <div class="tiny muted mt-4">${esc(a.summary)}</div>
         ${a.tips && a.tips.length ? `<ul class="advice-tips mt-4">${a.tips.map((t) => `<li>${esc(t)}</li>`).join('')}</ul>` : ''}
         </details>` : '';
    return `<div class="notice ${cls === 'success' ? 'success' : cls}" id="tr-advice">
      ${icon(ic, 'icon-sm')}<div style="flex:1;min-width:0">
        <b>${title}</b>${summary ? `<div class="mt-4">${esc(summary)}</div>` : ''}
        ${tips.length ? `<ul class="advice-tips">${tips.map((t) => `<li>${esc(t)}</li>`).join('')}</ul>` : ''}
        ${tech}
      </div>
      ${a.level !== 'ok' && m.best_epoch ? `<span class="tag" style="align-self:center">最佳第 ${m.best_epoch} 轮</span>` : ''}
    </div>`;
  }

  function renderMonitor(view, job) {
    const done = job.status === 'done';
    const failed = job.status === 'failed';
    const running = job.status === 'running' || job.status === 'queued';
    const last = lastOf(job);
    const pct = pctOf(job);
    const m = job.metrics || {};

    let result = '';
    if (done) {
      const verdict = qualityVerdict(job);
      const verdictCard = verdict
        ? `<div class="notice ${verdict.level}" id="tr-verdict">${icon(
            verdict.level === 'success' ? 'check-circle' : verdict.level === 'danger' ? 'x-circle' : 'info', 'icon-sm')}
          <div><b>${verdict.title}</b><br><span class="small">${esc(verdict.text)}</span></div></div>`
        : '';
      result = `${verdictCard}
        ${adviceCard(m)}
        ${(m.stopped_early ? `<div class="notice warn">${icon('clock', 'icon-sm')}<div><b>已提前停止</b>：${esc(m.stop_reason || '训练在效果不再提升时自动停止')}，并保留了效果最好的一版。</div></div>` : '')}
        ${metricCards(job)}
        ${roundsCard(job)}
        <div class="flex gap-10 mt-16" style="flex-wrap:wrap">
          <button class="btn btn-primary" id="tr-register">${icon('box', 'icon-sm')}保存这个模型</button>
          <a class="btn" href="#/services">${icon('rocket', 'icon-sm')}发布为服务</a>
          <a class="btn btn-ghost" href="#/build?dataset=${esc(job.dataset_id)}">${icon('refresh', 'icon-sm')}再训一个</a>
        </div>`;
    } else if (failed) {
      result = `<div class="notice danger">${icon('alert', 'icon-sm')}<div><b>训练没有成功</b><br>${esc(job.error || '未知错误')}</div></div>
        <div class="flex gap-10 mt-16">
          <a class="btn btn-primary" href="#/build?dataset=${esc(job.dataset_id)}">${icon('arrow-left', 'icon-sm')}回去改改再试</a>
          <a class="btn" href="#/train">看全部任务</a>
        </div>`;
    } else if (running && m.advice && m.advice.tips && m.advice.tips.length) {
      result = `<div id="tr-advice-slot">${adviceCard(m)}</div>`;   // 训练中实时给出预警
    }

    view.innerHTML = `
      ${MF.pageHead('训练监控', `${esc(job.dataset_name || job.dataset_id || '')} · 任务 ${esc(job.id)}`,
        `<a class="btn" href="#/train">${icon('cpu')}任务列表</a><a class="btn btn-primary" href="#/build">${icon('sparkles')}新建训练</a>`)}
      <div id="tr-root">
        ${sec('训练进度', '', `<div class="flex items-center gap-12" style="flex-wrap:wrap">
            ${MF.expStatus(job.status)}
            <span class="small muted" id="tr-stage">${esc((m && m.stage) || (running ? '正在准备…' : ''))}</span>
            <span class="small muted">本轮 <b id="tr-epoch">${job.epoch || 0} / ${job.total_epochs || '—'}</b> 轮</span>
            <span class="small muted">训练损失 <b id="tr-train-loss">${last ? round(last.train_loss, 4) : '—'}</b></span>
            <span class="small muted">验证损失 <b id="tr-val-loss">${last ? round(last.val_loss, 4) : '—'}</b></span>
            <span class="ml-auto tiny muted" id="tr-pct">${pct}%</span></div>
          <div class="bar progress-lg mt-10"><span id="tr-bar" style="width:${pct}%"></span></div>
          ${running ? `<div class="tiny muted mt-10">${icon('refresh', 'icon-sm spin')} 每 1.5 秒自动刷新 · 可离开本页，稍后从任务列表恢复查看 <span id="tr-poll-err"></span></div>` : ''}`, '')}
        ${result}
        ${sec('训练曲线', '', `<div class="chart" id="tr-loss-chart"></div>`, '')}
        ${sec('训练日志', '', `<div class="console" id="tr-log" style="height:220px">${logsHTML(job)}</div>`, '服务端日志 · 倒序展示最新记录')}
      </div>`;

    drawChart(job);
    const reg = MF.$('#tr-register', view);
    if (reg) reg.onclick = () => openRegister(job);
  }

  function tickUpdate(job) {
    const bar = MF.$('#tr-bar'); if (bar) bar.style.width = pctOf(job) + '%';
    const pct = MF.$('#tr-pct'); if (pct) pct.textContent = pctOf(job) + '%';
    const ep = MF.$('#tr-epoch'); if (ep) ep.textContent = `${job.epoch || 0} / ${job.total_epochs || '—'}`;
    // 当前阶段 + 预计剩余时间：让用户知道「在做什么、还要多久」
    const st = MF.$('#tr-stage');
    if (st) {
      const m = job.metrics || {};
      const stage = m.stage || '';
      const eta = m.eta_seconds;
      let txt = stage;
      if (stage && m.stage_progress !== undefined && m.stage_progress < 1) {
        txt += `（${Math.round((m.stage_progress || 0) * 100)}%）`;
      }
      if (eta && eta > 3) {
        const mins = Math.ceil(eta / 60);
        txt += txt ? ` · 预计还需 ${mins} 分钟` : `预计还需 ${mins} 分钟`;
      }
      st.textContent = txt || (job.status === 'running' ? '正在准备…' : '');
    }
    const last = lastOf(job);
    const tl = MF.$('#tr-train-loss'); if (tl) tl.textContent = last ? round(last.train_loss, 4) : '—';
    const vl = MF.$('#tr-val-loss'); if (vl) vl.textContent = last ? round(last.val_loss, 4) : '—';
    // 实时刷新诊断建议：训练中一旦发现过拟合/发散立刻提示
    const adv = MF.$('#tr-advice-slot');
    if (adv && job.metrics && job.metrics.advice) adv.innerHTML = adviceCard(job.metrics);
    drawChart(job);
    const log = MF.$('#tr-log');
    if (log) { log.innerHTML = logsHTML(job); log.scrollTop = 0; }
  }

  function startPoll(view, jobId) {
    stopPoll();
    pollTimer = setInterval(() => {
      if (!MF.$('#tr-root')) { stopPoll(); return; }
      MF.api.trainJob(jobId).then((job) => {
        if (!MF.$('#tr-root')) { stopPoll(); return; }
        if (job.status === 'done' || job.status === 'failed') {
          stopPoll();
          renderMonitor(view, job);
          if (job.status === 'done') {
            const best = bestOf(job) || {};
            const main = isReg(job) ? `RMSE ${round(best.val_rmse, 4)}` : `准确率 ${((best.val_acc || 0) * 100).toFixed(1)}%`;
            MF.toast('训练完成', `验证集 ${main}，可登记到模型仓库。`, 'success');
          } else {
            MF.toast('训练失败', job.error || '请返回重新配置', 'error');
          }
          return;
        }
        tickUpdate(job);
      }).catch((e) => {
        const el = MF.$('#tr-poll-err');
        if (el) el.textContent = `· 刷新失败：${e.message}`;
      });
    }, 1500);
  }

  /* ---------------- 登记到模型仓库 ---------------- */
  function openRegister(job) {
    const defName = (String(job.dataset_name || 'model').replace(/数据集$/, '').replace(/集$/, '') || 'model') + '模型';
    MF.modal({
      title: '登记到模型仓库',
      desc: `训练任务 ${job.id} · 同名模型将自动递增版本号`,
      body: `<div class="field"><label>模型名称<span class="req">*</span></label><input class="input" id="rg-name" value="${esc(defName)}"></div>`,
      foot: `<button class="btn" data-close>取消</button><button class="btn btn-primary" id="rg-ok">${icon('box', 'icon-sm')}登记模型</button>`,
      onMount(el, close) {
        MF.$('#rg-ok', el).onclick = async () => {
          const name = MF.$('#rg-name', el).value.trim();
          if (!name) { MF.toast('请填写模型名称', '', 'warn'); return; }
          const btn = MF.$('#rg-ok', el);
          btn.disabled = true; btn.innerHTML = `${icon('refresh', 'icon-sm spin')}登记中…`;
          try {
            const m = await MF.api.registerModel(job.id, name);
            close();
            MF.toast('模型已登记', `${m.name} · v${m.version} · ${m.architecture || ''}`, 'success');
            location.hash = '#/models';
          } catch (e) {
            btn.disabled = false; btn.innerHTML = `${icon('box', 'icon-sm')}登记模型`;
            MF.toast('登记失败', e.message, 'error');
          }
        };
      }
    });
  }

  /* ---------------- 路由入口 ---------------- */
  MF.views.train = function (r, view) {
    stopPoll();
    // 只认 job 查询参数；r.id 是路径段（#/train/<id>），兼容旧链接
    const jobId = r.query.get('job') || r.id;
    if (!jobId) { renderList(view); return; }
    view.innerHTML = loadingHTML('正在连接训练任务…');
    MF.api.trainJob(jobId).then((job) => {
      renderMonitor(view, job);
      if (job.status === 'queued' || job.status === 'running') startPoll(view, job.id);
    }).catch((e) => {
      // 任务可能刚被创建、或数据集/任务已删除：给出明确出口而不是停在报错页
      view.innerHTML = notFoundHTML(jobId, e);
      bindRetry(view);
      const back = MF.$('[data-back-list]', view);
      if (back) back.onclick = () => { location.hash = '#/train'; };
    });
  };

  function notFoundHTML(jobId, e) {
    const is404 = /不存在|HTTP 404/.test(e.message || '');
    return `<div class="card card-pad"><div class="notice danger">${icon('alert', 'icon-sm')}<div>
        <b>${is404 ? '找不到该训练任务' : '训练任务加载失败'}</b><br>
        ${is404 ? `任务 <span class="mono">${esc(jobId)}</span> 不存在，可能已被清理或链接有误。` : esc(e.message || e)}
      </div></div>
      <div class="flex gap-10 mt-12">
        <button class="btn btn-sm" data-retry>${icon('refresh', 'icon-sm')}重试</button>
        <button class="btn btn-sm btn-primary" data-back-list>${icon('cpu', 'icon-sm')}返回任务列表</button>
      </div></div>`;
  }
})(window);
