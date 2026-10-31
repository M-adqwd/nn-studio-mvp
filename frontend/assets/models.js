/* =========================================================
   ModelForge · 模型仓库与模型详情（真实 FastAPI 后端）
   列表 / 阶段流转 / 训练曲线 / 删除 / 推理试算 走 MF.api；
   推理表单与结果渲染封装为 MF.inferKit，供 ops.js 复用。
   ========================================================= */
(function (global) {
  const MF = global.MF;
  const { esc, fmt, icon } = MF;

  const ARCH_LABELS = { mlp: 'MLP', cnn: 'CNN', emb_mlp: 'EMB_MLP' };
  const STAGE_FLOW = [['candidate', '候选'], ['staging', '预发布'], ['production', '生产'], ['archived', '已归档']];
  const PROB_COLORS = ['#2563eb', '#60a5fa', '#0f9d76', '#f59e0b', '#7c3aed', '#ef4444'];

  const loadingHTML = (t) => `<div class="card card-pad"><div class="notice">${icon('refresh', 'icon-sm spin')}<div>${esc(t || '正在加载…')}</div></div></div>`;
  const errHTML = (e) => `<div class="card card-pad"><div class="notice danger">${icon('alert', 'icon-sm')}<div><b>请求失败</b><br>${esc(e.message || e)}</div></div></div>`;
  const numStr = (v) => (typeof v === 'number' && isFinite(v) ? (Math.abs(v) >= 100 ? v.toFixed(1) : String(+v.toFixed(4))) : '—');

  function stagePill(stage) {
    const map = { candidate: ['候选', 'pill-info'], staging: ['预发布', 'pill-warn'], production: ['生产', 'pill-success'], archived: ['已归档', 'pill-muted'] };
    const m = map[stage] || ['未知', 'pill-muted'];
    return `<span class="pill ${m[1]}"><span class="pdot"></span>${m[0]}</span>`;
  }

  function stageSelect(m, attr) {
    return `<select class="select" style="height:30px;font-size:12px;width:104px" ${attr} title="阶段流转">
      ${STAGE_FLOW.map(([k, l]) => `<option value="${k}" ${m.stage === k ? 'selected' : ''}>${l}</option>`).join('')}</select>`;
  }

  // 最终指标：分类取 best_val.val_acc，回归取 best_val.val_rmse
  function finalMetric(m) {
    const bv = ((m.metrics || {}).best_val) || {};
    if (m.task === 'classification') return ['val_acc', bv.val_acc !== undefined ? MF.pct(bv.val_acc) : '—'];
    return ['val_rmse', bv.val_rmse !== undefined ? numStr(bv.val_rmse) : '—'];
  }

  async function changeStage(id, stage, name) {
    try {
      await MF.api.updateModel(id, { stage });
      const label = (STAGE_FLOW.find(([k]) => k === stage) || [])[1] || stage;
      MF.toast('阶段已更新', `${name} → ${label}`, 'success');
      MF.route();
    } catch (e) { MF.toast('阶段更新失败', e.message, 'error'); MF.route(); }
  }

  /* ---------------- 推理套件（模型详情 / 在线推理 / 服务测试共用） ---------------- */
  // 加载模型对应数据集字段；mlp 时结合 preview 为类别列生成下拉选项
  async function loadSpec(model) {
    if (model.architecture !== 'mlp' || !model.dataset_id) return { cols: [] };
    const d = await MF.api.getDataset(model.dataset_id);
    const cols = (((d || {}).meta || {}).schema || []).filter((c) => c.role !== 'target');
    let preview = null;
    try { preview = await MF.api.preview(model.dataset_id, 40); } catch (e) { preview = null; }
    const colIdx = {};
    ((preview || {}).columns || []).forEach((c, i) => { colIdx[c] = i; });
    cols.forEach((c) => {
      if (c.type === 'number') {
        c.def = c.mean !== undefined ? +(+c.mean).toFixed(4) : (c.min !== undefined ? c.min : 0);
      } else {
        const opts = [];
        if (preview && colIdx[c.name] !== undefined) {
          (preview.rows || []).forEach((row) => {
            const v = row[colIdx[c.name]];
            if (v !== null && v !== undefined && opts.indexOf(String(v)) < 0) opts.push(String(v));
          });
        }
        c.options = opts;
      }
    });
    return { cols };
  }

  function formHTML(model, spec) {
    if (model.architecture === 'cnn') {
      return `<div class="field"><label>上传图像</label>
        <div class="upload-box" data-img-box><div class="up-ico">${icon('image', 'icon-lg')}</div>
          <h4 data-img-name>点击上传图像</h4><p>支持 JPG / PNG，将上传至后端实时推理</p></div></div>
        <input type="file" data-img-input accept="image/*" style="display:none">`;
    }
    if (model.architecture === 'emb_mlp') {
      return `<div class="field"><label>输入文本</label><textarea class="textarea" data-text style="min-height:130px" placeholder="输入需要分类的文本…"></textarea></div>`;
    }
    if (!spec || !spec.cols.length) {
      return `<div class="notice">${icon('info', 'icon-sm')}<div>未能获取数据集字段结构，请直接填写特征 JSON。</div></div>
        <div class="field"><label>特征 JSON（单条 dict）</label><textarea class="textarea mono" data-json style="min-height:120px">{}</textarea></div>`;
    }
    return `<div class="form-grid">${spec.cols.map((c) => {
      if (c.type === 'number') return `<div class="field"><label>${esc(c.name)}</label><input class="input" type="number" step="any" data-fname="${esc(c.name)}" data-ftype="number" value="${c.def}"></div>`;
      if (c.options && c.options.length) return `<div class="field"><label>${esc(c.name)}</label><select class="select" data-fname="${esc(c.name)}">${c.options.map((o) => `<option>${esc(o)}</option>`).join('')}</select></div>`;
      return `<div class="field"><label>${esc(c.name)}</label><input class="input" data-fname="${esc(c.name)}" placeholder="文本值"></div>`;
    }).join('')}</div>`;
  }

  function bindForm(model, root) {
    const box = MF.$('[data-img-box]', root);
    if (box) {
      const input = MF.$('[data-img-input]', root);
      box.onclick = () => input.click();
      input.onchange = () => {
        const f = input.files[0];
        if (f) { MF.$('[data-img-name]', root).textContent = f.name; box.style.borderColor = 'var(--primary)'; }
      };
    }
  }

  // 收集表单 → {kind:'json',payload} | {kind:'file',file} | {kind:'error',message}
  function collect(model, root) {
    if (model.architecture === 'cnn') {
      const input = MF.$('[data-img-input]', root);
      const f = input && input.files[0];
      if (!f) return { kind: 'error', message: '请先选择图像文件' };
      return { kind: 'file', file: f };
    }
    if (model.architecture === 'emb_mlp') {
      const t = ((MF.$('[data-text]', root) || {}).value || '').trim();
      if (!t) return { kind: 'error', message: '请输入文本' };
      return { kind: 'json', payload: { text: t } };
    }
    const jsonArea = MF.$('[data-json]', root);
    if (jsonArea) {
      try {
        const obj = JSON.parse(jsonArea.value || '{}');
        return { kind: 'json', payload: { rows: [obj] } };
      } catch (e) { return { kind: 'error', message: 'JSON 解析失败：' + e.message }; }
    }
    const row = {};
    MF.$$('[data-fname]', root).forEach((el) => {
      row[el.dataset.fname] = el.dataset.ftype === 'number' ? parseFloat(el.value || 0) : el.value;
    });
    return { kind: 'json', payload: { rows: [row] } };
  }

  // 结果渲染：分类=标签+置信度条（probabilities），回归=数值大字
  function resultHTML(res, model, extraRows) {
    const pred = (res.predictions || [])[0] || {};
    const meta = [
      ['模型版本', 'v' + model.version],
      ['任务类型', MF.TASK_LABELS[model.task_hint] || res.task || '—'],
      ['网络结构', ARCH_LABELS[res.arch || model.architecture] || res.arch || '—']
    ].concat(extraRows || []);
    const kv = meta.map(([k, v]) => `<div class="kv-row"><span class="k">${esc(k)}</span><span class="v mono">${esc(v)}</span></div>`).join('');
    if (pred.value !== undefined) {
      return `<div class="infer-result"><div class="infer-label">预测结果 · ${esc(MF.TASK_LABELS[model.task_hint] || '回归')}</div>
        <div class="infer-class"><span class="c">${esc(numStr(pred.value))}</span></div>
        <div class="kv-list mt-16">${kv}</div></div>`;
    }
    const probs = pred.probabilities || {};
    const rows = Object.keys(probs).map((k, i) => `<div class="prob-row"><span class="muted">${esc(k)}</span><div class="bar"><span style="width:${(probs[k] * 100).toFixed(1)}%;background:${PROB_COLORS[i % PROB_COLORS.length]}"></span></div><span class="p">${MF.pct(probs[k])}</span></div>`).join('');
    return `<div class="infer-result"><div class="infer-label">预测结果 · ${esc(MF.TASK_LABELS[model.task_hint] || '分类')}</div>
      <div class="infer-class"><span class="c">${esc(pred.label || '—')}</span><span class="conf">${MF.pct(pred.confidence || 0)}</span></div>
      <div class="mt-16">${rows}</div>
      <div class="kv-list mt-16">${kv}</div></div>`;
  }

  MF.inferKit = { loadSpec, formHTML, bindForm, collect, resultHTML, ARCH_LABELS, stagePill };

  /* ---------------- 模型列表 ---------------- */
  MF.views.models = function (r, view) {
    view.innerHTML = loadingHTML('正在从后端加载模型仓库…');
    MF.api.listModels()
      .then((list) => renderList(list))
      .catch((e) => { view.innerHTML = errHTML(e); });

    function renderList(list) {
      const prod = list.filter((m) => m.stage === 'production');
      const clsModels = list.filter((m) => m.task === 'classification' && (((m.metrics || {}).best_val) || {}).val_acc !== undefined);
      const avgAcc = clsModels.length ? clsModels.reduce((a, m) => a + m.metrics.best_val.val_acc, 0) / clsModels.length : null;
      const stats = [
        { label: '模型总数', value: list.length, suffix: '个', icon: 'box', note: '来自后端仓库' },
        { label: '生产模型', value: prod.length, suffix: '个', icon: 'rocket', tone: 'green', note: '已发布生产阶段' },
        { label: '平均准确率', value: avgAcc === null ? '—' : (avgAcc * 100).toFixed(1), suffix: avgAcc === null ? '' : '%', icon: 'target', tone: 'purple', note: '分类模型 val_acc 均值' },
        { label: '待上线', value: list.filter((m) => m.stage === 'candidate' || m.stage === 'staging').length, suffix: '个', icon: 'clock', tone: 'amber', note: '候选 / 预发布' }
      ];
      const cards = list.map((m) => {
        const [mLabel, mVal] = finalMetric(m);
        const hist = ((m.metrics || {}).history) || [];
        const lastLoss = hist.length ? hist[hist.length - 1].val_loss : null;
        return `<div class="card card-hover" style="padding:18px;cursor:pointer" data-model="${esc(m.id)}">
        <div class="flex items-start gap-10"><div class="ds-ico">${icon('box')}</div>
          <div style="flex:1;min-width:0"><div class="flex items-center gap-6" style="flex-wrap:wrap"><span class="svc-name">${esc(m.name)}</span>${stagePill(m.stage)}</div>
            <div class="tiny muted" style="margin-top:3px">${esc(ARCH_LABELS[m.architecture] || m.architecture || '—')} · v${esc(m.version)}</div></div>
          <button class="icon-btn" data-model-del="${esc(m.id)}" data-name="${esc(m.name)}" title="删除模型" style="color:#dc2626">${icon('trash')}</button></div>
        <div class="flex items-center gap-6 mt-12" style="flex-wrap:wrap">${MF.taskPill(m.task_hint)}<span class="tag">${esc(ARCH_LABELS[m.architecture] || m.architecture)}</span><span class="tiny muted">数据集 · ${esc(m.dataset_name || '—')}</span></div>
        <div class="grid grid-3 mt-12" style="gap:1px;background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden">
          <div style="background:#fff;padding:9px 10px"><div class="tiny muted">${esc(mLabel)}</div><div class="small" style="font-weight:700">${esc(mVal)}</div></div>
          <div style="background:#fff;padding:9px 10px"><div class="tiny muted">val_loss</div><div class="small" style="font-weight:700">${lastLoss === null ? '—' : esc(numStr(lastLoss))}</div></div>
          <div style="background:#fff;padding:9px 10px"><div class="tiny muted">训练轮次</div><div class="small" style="font-weight:700">${hist.length || '—'}</div></div>
        </div>
        <div class="flex items-center gap-8 mt-12" style="padding-top:11px;border-top:1px solid var(--line)">
          <span class="tiny muted">${esc(m.created_at || '')}</span>
          <span class="ml-auto" data-stage-stop>${stageSelect(m, `data-stage-for="${esc(m.id)}"`)}</span></div>
      </div>`;
      }).join('');

      view.innerHTML = `
        ${MF.pageHead('我的模型', '训练好的模型都存在这里，可以随时拿真实数据试试效果。', `<a class="btn btn-primary" href="#/build?new=1">${icon('plus')}新建模型</a>`)}
        <div class="grid grid-4">${stats.map(MF.statCard).join('')}</div>
        <div class="card mt-20" style="padding:14px 18px"><div class="flex items-center gap-12" style="flex-wrap:wrap">
          <div class="segmented" data-model-tabs><button class="active" data-stage="all">全部</button><button data-stage="production">生产</button><button data-stage="staging">预发布</button><button data-stage="candidate">候选</button><button data-stage="archived">已归档</button></div>
          <div class="ml-auto flex gap-8"><div class="search-box" style="width:220px;background:#fff">${icon('search', 'icon-sm')}<input placeholder="搜索模型名称 / 架构" data-model-search></div></div>
        </div></div>
        <div class="grid grid-3 mt-16" data-model-grid>${cards}</div>
        <div class="card card-pad mt-16" data-model-empty style="display:${list.length ? 'none' : ''}"><div class="empty"><div class="ico">${icon('box')}</div><h4>${list.length ? '没有符合条件的模型' : '模型仓库为空'}</h4><p>${list.length ? '调整筛选条件，或新建一个模型。' : '从数据开始，完成一次训练并登记模型。'}</p><a class="btn btn-primary" href="#/build?new=1">${icon('sparkles', 'icon-sm')}去训练模型</a></div></div>`;

      MF.$$('[data-model]', view).forEach((el) => {
        el.onclick = (e) => {
          if (e.target.closest('[data-model-del]') || e.target.closest('[data-stage-stop]')) return;
          location.hash = `#/models/${el.dataset.model}`;
        };
      });
      MF.$$('[data-stage-stop]', view).forEach((el) => (el.onclick = (e) => e.stopPropagation()));
      MF.$$('[data-stage-for]', view).forEach((sel) => (sel.onchange = () => {
        const m = list.find((x) => x.id === sel.dataset.stageFor);
        if (m && sel.value !== m.stage) changeStage(m.id, sel.value, m.name);
      }));
      MF.$$('[data-model-del]', view).forEach((b) => (b.onclick = async (e) => {
        e.stopPropagation();
        const ok = await MF.confirm({ title: '删除模型', message: `确认删除「${b.dataset.name}」？模型文件与指标记录将一并移除，且不可恢复。`, okText: '删除', danger: true });
        if (!ok) return;
        try {
          await MF.api.deleteModel(b.dataset.modelDel);
          MF.toast('模型已删除', b.dataset.name, 'success');
          MF.route();
        } catch (err) { MF.toast('删除失败', err.message, 'error'); }
      }));
      let stage = 'all';
      const apply = () => {
        const q = (MF.$('[data-model-search]', view).value || '').trim().toLowerCase();
        let n = 0;
        MF.$$('[data-model]', view).forEach((el) => {
          const m = list.find((x) => x.id === el.dataset.model);
          const hay = ((m.name || '') + (m.architecture || '') + (ARCH_LABELS[m.architecture] || '') + (m.dataset_name || '')).toLowerCase();
          const ok = (stage === 'all' || m.stage === stage) && (!q || hay.indexOf(q) >= 0);
          el.style.display = ok ? '' : 'none'; if (ok) n++;
        });
        MF.$('[data-model-empty]', view).style.display = n ? 'none' : '';
      };
      MF.$$('[data-model-tabs] button', view).forEach((b) => (b.onclick = () => {
        MF.$$('[data-model-tabs] button', view).forEach((x) => x.classList.remove('active'));
        b.classList.add('active'); stage = b.dataset.stage; apply();
      }));
      MF.$('[data-model-search]', view).oninput = apply;
    }
  };

  /* ---------------- 模型详情 ---------------- */
  MF.views.model = function (r, view) {
    view.innerHTML = loadingHTML('正在加载模型详情…');
    MF.api.getModel(r.id)
      .then((m) => renderDetail(m))
      .catch((e) => { view.innerHTML = `<a href="#/models" class="btn btn-sm btn-ghost" style="margin-bottom:12px">${icon('arrow-left', 'icon-sm')}模型仓库</a>` + errHTML(e); });

    function renderDetail(m) {
      const tab = r.query.get('tab') || 'overview';
      const tabs = [['overview', '概览'], ['predict', '推理试算']];
      const metrics = m.metrics || {};
      const hist = metrics.history || [];
      const bv = metrics.best_val || {};
      const cls = m.task === 'classification';
      const [mLabel, mVal] = finalMetric(m);
      const metricCards = [
        [mLabel, mVal],
        ['val_loss', bv.val_loss !== undefined ? numStr(bv.val_loss) : '—'],
        ['train_loss', bv.train_loss !== undefined ? numStr(bv.train_loss) : '—'],
        ['最佳轮次', bv.epoch !== undefined ? '#' + bv.epoch : '—']
      ];
      const epochs = hist.map((h) => String(h.epoch));
      const lossChart = hist.length ? MF.charts.line({
        series: [
          { name: 'train_loss', color: '#2563eb', data: hist.map((h) => h.train_loss) },
          { name: 'val_loss', color: '#f59e0b', data: hist.map((h) => h.val_loss) }
        ], labels: epochs, height: 220
      }) : `<div class="empty" style="padding:30px"><p>暂无训练曲线数据</p></div>`;
      const valKey = cls ? 'val_acc' : 'val_rmse';
      const valSeries = hist.map((h) => h[valKey]).filter((v) => v !== undefined && v !== null);
      const valChart = valSeries.length ? MF.charts.line({
        series: [{ name: valKey, color: '#0f9d76', data: valSeries, area: true }],
        labels: hist.filter((h) => h[valKey] !== undefined && h[valKey] !== null).map((h) => String(h.epoch)), height: 220
      }) : `<div class="empty" style="padding:30px"><p>暂无验证指标数据</p></div>`;

      const overview = `<div class="grid grid-4">${metricCards.map(([l, v]) => `<div class="metric-card"><div class="l">${esc(l)}</div><div class="v">${esc(v)}</div><div class="d">验证集最佳</div></div>`).join('')}</div>
        <div class="grid grid-2 mt-18">
          <div class="card card-pad"><div class="section-title">损失曲线</div><div class="section-sub" style="margin-bottom:10px">train_loss / val_loss 随 epoch 变化</div>
            ${lossChart}
            <div class="chart-legend"><span class="legend-item"><i style="background:#2563eb"></i>train_loss</span><span class="legend-item"><i style="background:#f59e0b"></i>val_loss</span></div></div>
          <div class="card card-pad"><div class="section-title">验证指标 · ${esc(valKey)}</div><div class="section-sub" style="margin-bottom:10px">最佳值 ${esc(mVal)}（epoch ${esc(bv.epoch !== undefined ? bv.epoch : '—')}）</div>
            ${valChart}</div>
        </div>
        <div class="grid grid-2-1 mt-18">
          <div class="card card-pad"><div class="section-title">模型信息</div>
            <div class="form-grid mt-16">
              <div class="kv-row"><span class="k">版本</span><span class="v mono">v${esc(m.version)}</span></div>
              <div class="kv-row"><span class="k">阶段</span><span class="v">${stagePill(m.stage)}</span></div>
              <div class="kv-row"><span class="k">任务类型</span><span class="v">${MF.taskPill(m.task_hint)}</span></div>
              <div class="kv-row"><span class="k">网络结构</span><span class="v">${esc(ARCH_LABELS[m.architecture] || m.architecture || '—')}</span></div>
              <div class="kv-row"><span class="k">训练数据</span><span class="v">${esc(m.dataset_name || '—')}</span></div>
              <div class="kv-row"><span class="k">训练任务</span><span class="v mono">${esc(m.train_job_id || '—')}</span></div>
              <div class="kv-row"><span class="k">创建时间</span><span class="v">${esc(m.created_at || '—')}</span></div>
              <div class="kv-row"><span class="k">更新时间</span><span class="v">${esc(m.updated_at || '—')}</span></div>
            </div></div>
          <div class="card card-pad"><div class="section-title">阶段流转</div><div class="section-sub" style="margin-bottom:14px">候选 → 预发布 → 生产 → 归档</div>
            <div class="flex items-center gap-8">${stageSelect(m, 'data-stage-detail')}</div>
            <div class="notice mt-16">${icon('info', 'icon-sm')}<div>阶段变更立即生效并写入后端；生产阶段模型建议先完成推理试算验证。</div></div>
            <div class="flex gap-8 mt-16"><a class="btn btn-sm btn-soft" href="#/models/${esc(m.id)}?tab=predict">${icon('cpu', 'icon-sm')}推理试算</a>
              <a class="btn btn-sm" href="#/services">${icon('rocket', 'icon-sm')}发布为服务</a></div></div>
        </div>`;

      view.innerHTML = `
        <div class="flex items-center gap-8 small" style="margin-bottom:12px"><a href="#/models" class="btn btn-sm btn-ghost">${icon('arrow-left', 'icon-sm')}模型仓库</a><span class="muted">/</span><span class="muted">${esc(ARCH_LABELS[m.architecture] || m.architecture || '')}</span></div>
        <div class="page-head"><div class="flex items-center gap-12"><div class="ds-ico" style="width:44px;height:44px;border-radius:11px">${icon('box', 'icon-lg')}</div>
          <div><h1 style="display:flex;align-items:center;gap:9px">${esc(m.name)} ${stagePill(m.stage)}</h1><p>v${esc(m.version)} · ${esc(ARCH_LABELS[m.architecture] || m.architecture || '—')} · ${esc(m.dataset_name || '')} · 更新于 ${esc(m.updated_at || '')}</p></div></div>
          <div class="actions"><button class="btn" data-del style="color:#dc2626">${icon('trash')}删除</button>
            <a class="btn" href="#/inference?model=${esc(m.id)}">${icon('cpu')}试试效果</a>
            <a class="btn btn-primary" href="#/services">${icon('rocket')}发布服务</a></div></div>
        <div class="tabs" style="margin-bottom:20px">${tabs.map(([k, l]) => `<a class="tab ${tab === k ? 'active' : ''}" href="#/models/${esc(m.id)}?tab=${k}">${l}</a>`).join('')}</div>
        <div data-tab-body>${tab === 'overview' ? overview : loadingHTML('正在准备推理表单…')}</div>`;

      MF.$('[data-del]', view).onclick = async () => {
        const ok = await MF.confirm({ title: '删除模型', message: `确认删除「${m.name} v${m.version}」？模型文件与指标记录将一并移除，且不可恢复。`, okText: '删除', danger: true });
        if (!ok) return;
        try {
          await MF.api.deleteModel(m.id);
          MF.toast('模型已删除', m.name, 'success');
          location.hash = '#/models';
        } catch (err) { MF.toast('删除失败', err.message, 'error'); }
      };
      const stageSel = MF.$('[data-stage-detail]', view);
      if (stageSel) stageSel.onchange = () => { if (stageSel.value !== m.stage) changeStage(m.id, stageSel.value, m.name); };

      if (tab === 'predict') renderPredict(m, MF.$('[data-tab-body]', view));
    }

    function renderPredict(m, body) {
      loadSpec(m).then((spec) => {
        body.innerHTML = `<div class="infer-layout">
          <div class="card"><div class="card-head" style="padding-bottom:12px"><div><div class="section-title">输入</div><div class="section-sub">按模型结构填写特征，提交后端实时推理</div></div></div>
            <div class="card-body" data-form>${formHTML(m, spec)}
              <button class="btn btn-primary btn-block mt-16" data-run>${icon('play', 'icon-sm')}运行推理</button></div></div>
          <div data-result><div class="card card-pad" style="min-height:300px;display:grid;place-items:center">
            <div class="empty"><div class="ico">${icon('cpu')}</div><h4>等待推理请求</h4><p>填写左侧输入并点击「运行推理」，结果将显示在这里。</p></div></div></div>
        </div>`;
        bindForm(m, body);
        MF.$('[data-run]', body).onclick = async () => {
          const btn = MF.$('[data-run]', body);
          const c = collect(m, body);
          if (c.kind === 'error') { MF.toast(c.message, '', 'warn'); return; }
          btn.disabled = true; btn.innerHTML = `${icon('refresh', 'icon-sm spin')}推理中…`;
          const t0 = Date.now();
          try {
            const res = c.kind === 'file' ? await MF.api.predictImage(m.id, c.file) : await MF.api.predict(m.id, c.payload);
            MF.$('[data-result]', body).innerHTML = resultHTML(res, m, [['推理耗时', (Date.now() - t0) + ' ms']]);
            MF.toast('推理完成', `${m.name} · ${Date.now() - t0} ms`, 'success', 1800);
          } catch (err) {
            MF.$('[data-result]', body).innerHTML = `<div class="card card-pad"><div class="notice danger">${icon('alert', 'icon-sm')}<div><b>推理失败</b><br>${esc(err.message)}</div></div></div>`;
          }
          btn.disabled = false; btn.innerHTML = `${icon('play', 'icon-sm')}运行推理`;
        };
      }).catch((e) => { body.innerHTML = errHTML(e); });
    }
  };
})(window);
