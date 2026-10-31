/* =========================================================
   ModelForge · 引导式建模向导（真实后端 API 版）
   7 步：选择数据集 → 确认任务 → 字段与目标 → 网络架构
        → 超参配置 → 确认配置 → 提交训练
   数据集来自 MF.api.listDatasets()，训练任务提交至 MF.api.createTrainJob()。
   ========================================================= */
(function (global) {
  const MF = global.MF;
  const { esc, fmt, icon } = MF;

  /* ---------------- 常量 ---------------- */
  // 步骤标题用大白话。用户不需要知道「超参」「架构」这些词，
  // 他只需要知道「现在在做什么」「我需要做什么」。
  const STEPS = [
    { no: 1, group: '准备数据', label: '选择数据', sub: '用哪份数据来训练' },
    { no: 2, group: '准备数据', label: '确认任务', sub: '模型要解决什么问题' },
    { no: 3, group: '准备数据', label: '字段与目标', sub: '数据结构、标签与文本列' },
    { no: 4, group: '构建模型', label: '用什么方法', sub: '已自动选定' },
    { no: 5, group: '构建模型', label: '训练设置', sub: '已自动配好' },
    { no: 6, group: '构建模型', label: '确认一下', sub: '核对后即可开始' },
    { no: 7, group: '开始训练', label: '开始训练', sub: '提交训练任务' }
  ];
  const TOTAL = STEPS.length;

  // 面向不懂 AI 的用户：先用大白话说「这个模型要做什么」，
  // 再补一句技术说明。术语一律翻译成人话。
  const TASK_DESC = {
    tabular_classification: '每一行是一条记录，模型要判断它属于哪一类（比如「是否会流失」）。评价标准是准确率——猜对的比例。',
    tabular_regression: '每一行是一条记录，模型要预测一个具体数字（比如「房价多少」）。评价标准是预测值与真实值的接近程度。',
    image_classification: '每张图片对应一个标签，模型要认出图片里是什么（比如「这是猫」）。评价标准是准确率。',
    text_classification: '每段文字对应一个标签，模型要判断这段文字表达了什么（比如「这条评论是好评」）。评价标准是准确率。',
    object_detection: '每张图片里可能有好几个目标，模型要同时说出「有什么」和「在哪里」，并用方框圈出来。评价标准是 mAP（框得准不准）。'
  };
  // 同一句话的技术补充，放在详情里给进阶用户看
  const TASK_TECH = {
    tabular_classification: '技术上：MLP 分类器，以准确率评估',
    tabular_regression: '技术上：MLP 回归器，以 RMSE 评估',
    image_classification: '技术上：预训练 CNN 迁移学习，以准确率评估',
    text_classification: '技术上：预训练语言模型 + 分类头，以准确率评估',
    object_detection: '技术上：YOLOv8 检测器，以 mAP 评估'
  };
  const ARCH = {
    tabular_classification: { key: 'mlp', name: 'MLP 多层感知机', desc: '两层全连接网络，隐藏层宽度可在超参中调整，收敛快、可解释，适合中小规模结构化数据。' },
    tabular_regression: { key: 'mlp', name: 'MLP 多层感知机（回归）', desc: '两层全连接回归网络，隐藏层宽度可在超参中调整，输出单个连续值。' },
    image_classification: { key: 'transfer_head', name: '预训练 CNN 迁移学习', desc: '冻结 ImageNet 预训练主干网络提取视觉特征，只训练分类头，小数据集上显著优于从零训练。' },
    text_classification: { key: 'text_transfer_head', name: '预训练语言模型 + 分类头', desc: '用预训练语言模型把文本编码成语义向量，再训练分类头，保留完整语言学知识。' },
    object_detection: { key: 'yolo', name: 'YOLO 目标检测', desc: '基于 COCO 预训练权重的实时检测器，输出每个目标的类别与边界框，指标为 mAP。' }
  };
  const TYPE_LABELS = { number: '数值', category: '类别', text: '文本', datetime: '日期', bool: '布尔' };
  const BATCHES = [16, 32, 64, 128];
  const SPLITS = [0.1, 0.15, 0.2, 0.25, 0.3];
  const OPTIMIZERS = [['adam', 'Adam（推荐）'], ['sgd', 'SGD + 动量'], ['rmsprop', 'RMSprop']];
  const SCHEDULERS = [['none', '不变'], ['cosine', '余弦退火'], ['step', '阶梯衰减']];

  const loadingHTML = (t) => `<div class="card card-pad"><div class="notice">${icon('refresh', 'icon-sm spin')}<div>${esc(t || '正在加载…')}</div></div></div>`;
  const errHTML = (e) => `<div class="card card-pad"><div class="notice danger">${icon('alert', 'icon-sm')}<div><b>请求失败</b><br>${esc(e.message || e)}</div></div>
    <div class="flex gap-10 mt-12"><button class="btn btn-sm" data-retry>${icon('refresh', 'icon-sm')}重试</button></div></div>`;
  const numStr = (v, p) => (typeof v !== 'number' || !isFinite(v)) ? '—' : (Math.abs(v) >= 100 ? v.toFixed(1) : String(+v.toFixed(p === undefined ? 2 : p)));

  /* ---------------- 状态 ---------------- */
  function freshState(datasetId) {
    return {
      v: 2, step: 1, maxStep: 1, datasetId: datasetId || null, task: null,
      hyper: { auto: true, epochs: 10, lr: 0.005, batch_size: 32, hidden: 64, val_split: 0.2, class_weight: true, dropout: 0, patience: 10, weight_decay: 0, optimizer: 'adam', scheduler: 'none', det_model: 'yolov8n.pt' },
      jobId: null, modelName: '', createdAt: MF.nowStamp()
    };
  }
  function save(s) { MF.wizardStore.save(s); }
  function datasetOf(s, list) { return list.filter((d) => d.id === s.datasetId)[0] || null; }

  function applyDataset(s, ds) {
    if (!ds) { s.datasetId = null; s.task = null; return s; }
    s.datasetId = ds.id;
    s.task = ds.task_hint || 'tabular_classification';
    s.hyper = { auto: true, epochs: 10, lr: 0.005, batch_size: 32, hidden: 64, val_split: 0.2, class_weight: true, dropout: 0, patience: 10, weight_decay: 0, optimizer: 'adam', scheduler: 'none', det_model: 'yolov8n.pt' };
    s.jobId = null;
    s.modelName = String(ds.name || 'model').replace(/数据集$/, '').replace(/集$/, '') + '模型';
    return s;
  }

  function ensureState(r, list) {
    let s = MF.wizardStore.load();
    if (!s || s.v !== 2) s = null;
    const qd = r.query.get('dataset');
    if (r.query.get('new') === '1') {
      s = freshState(null);
    } else if (qd && (!s || s.datasetId !== qd)) {
      s = applyDataset(freshState(qd), list.filter((d) => d.id === qd)[0]);
    } else if (!s) {
      s = freshState(null);
    }
    if (s.datasetId && !list.some((d) => d.id === s.datasetId)) {
      s = freshState(null); // 数据集已被删除
    }
    return s;
  }

  const isTabular = (task) => task === 'tabular_classification' || task === 'tabular_regression';
  const isCls = (task) => task !== 'tabular_regression';
  const isDetect = (task) => task === 'object_detection';
  // 图像类任务的单位是「张」，其余是「条」——别让用户看到「717 行图片」
  const rowUnit = (task) => (task === 'image_classification' || task === 'object_detection') ? '张' : '条';
  const schemaOf = (d) => ((d && d.meta) || {}).schema || [];
  const featureCols = (d) => schemaOf(d).filter((c) => c.role !== 'target');
  const targetCol = (d) => schemaOf(d).filter((c) => c.role === 'target')[0] || null;
  function outputDim(d) {
    const m = (d && d.meta) || {};
    if (m.classes && m.classes.length) return m.classes.length;
    const t = targetCol(d);
    return t && t.unique ? t.unique : 2;
  }

  function validation(s, d, step) {
    switch (step) {
      case 1: return s.datasetId && d ? null : '请选择一个数据集，或先上传数据。';
      case 2: return s.task ? null : '该数据集未识别任务类型，请重新选择。';
      case 3: return (d && (d.target_col || targetCol(d))) ? null : '该数据集缺少目标列，无法训练。';
      case 5: {
        const h = s.hyper;
        if (!(h.epochs >= 1 && h.epochs <= 100)) return '训练轮次需在 1–100 之间。';
        if (!(h.lr > 0 && h.lr <= 1)) return '学习率需在 0–1 之间。';
        if (BATCHES.indexOf(Number(h.batch_size)) < 0) return '请选择有效的批大小。';
        if (isTabular(s.task) && !(h.hidden >= 8 && h.hidden <= 512)) return '隐藏层宽度需在 8–512 之间。';
        if (!(h.val_split >= 0.1 && h.val_split <= 0.4)) return '验证集比例需在 10%–40% 之间。';
        if (!(h.dropout >= 0 && h.dropout <= 0.6)) return 'Dropout 需在 0–0.6 之间。';
        if (!(h.weight_decay >= 0 && h.weight_decay <= 0.1)) return '权重衰减需在 0–0.1 之间。';
        if (!(h.patience >= 0 && h.patience <= 50)) return '早停耐心值需在 0–50 之间。';
        return null;
      }
      default: return null;
    }
  }

  function go(s, n) {
    if (n > s.step) {
      const msg = validation(s, s._dataset, s.step);
      if (msg) { MF.toast('还不能继续', msg, 'warn'); return; }
    }
    if (n < 1 || n > TOTAL) return;
    s.step = n;
    s.maxStep = Math.max(s.maxStep || 1, n);
    save(s);
    MF.route();
  }

  /* ---------------- 公共片段 ---------------- */
  function sec(title, num, body, sub) {
    return `<div class="card card-pad" style="margin-bottom:14px">
      <div class="wiz-section-title">${num ? `<span class="num">${num}</span>` : ''}${title}</div>
      ${sub ? `<div class="section-sub" style="margin:-4px 0 12px ${num ? '28px' : '0'}">${sub}</div>` : ''}
      ${body}</div>`;
  }
  function metaTiles(items) {
    return `<div class="step-summary">${items.map(([l, v]) => `<div class="mini-stat"><div class="l">${l}</div><div class="v">${v}</div></div>`).join('')}</div>`;
  }
  const taskIcon = (t) => t === 'image_classification' ? 'image' : t === 'text_classification' ? 'text' : t === 'tabular_regression' ? 'trend' : 'table';

  function archCanvas(s, d) {
    const t = s.task;
    const layers = [];
    const push = (kind, text, dim) => layers.push({ kind, text, dim });

    if (t === 'image_classification') {
      push('in', '图像张量', '3 × 64 × 64');
      push('cnn', 'Conv 3→16 + Pool', '64 → 32');
      push('cnn', 'Conv 16→32 + Pool', '32 → 16');
      push('cnn', 'Conv 32→64 + Pool', '16 → 8');
      push('fc', 'Flatten + FC', '64·8·8 → 128');
      push('out', '分类输出', `${outputDim(d)} 类`);
    } else if (t === 'text_classification') {
      push('in', '文本', '哈希分词');
      push('emb', 'EmbeddingBag', '4096 → 64');
      push('pool', '均值池化', '64 维');
      push('fc', '线性分类头', `64 → ${outputDim(d)}`);
      push('out', '分类输出', `${outputDim(d)} 类`);
    } else {
      const dim = featureCols(d).length || '—';
      const hidden = s.hyper.hidden;
      const reg = t === 'tabular_regression';
      push('in', '输入特征', `${dim} 维`);
      push('fc', '全连接 + ReLU', `${dim} → ${hidden}`);
      push('fc', '全连接 + ReLU', `${hidden} → ${hidden}`);
      push('out', reg ? '回归输出' : '分类输出', reg ? '1 个连续值' : `${outputDim(d)} 类`);
    }

    const tip = { in: '接收数据', cnn: '卷积 + 池化', emb: '词嵌入', pool: '聚合', fc: '全连接', out: '预测' };
    const html = layers.map((L, i) => `${i ? '<span class="arch-arrow">→</span>' : ''}
      <div class="arch-block ${L.kind}"><div class="a">${esc(L.kind.toUpperCase())}</div><div class="n">${esc(L.text)}</div><div class="dim">${esc(L.dim || tip[L.kind] || '')}</div></div>`).join('');
    return `<div class="arch-canvas">${html}</div>
      <div class="tiny muted mt-8">结构与后端 ${esc((ARCH[t] || {}).key || '').toUpperCase()} 实现一致，维度由 ${esc(d ? d.name : '数据集')} 的实际 schema 推导。</div>`;
  }

  function trainCfg(s, d) {
    const h = s.hyper;
    const cfg = { dataset_id: d.id, auto: !!h.auto, epochs: h.epochs, lr: h.lr, batch_size: Number(h.batch_size), val_split: Number(h.val_split), optimizer: h.optimizer, scheduler: h.scheduler, weight_decay: Number(h.weight_decay) || 0, dropout: Number(h.dropout) || 0, patience: Number(h.patience) || 0 };
    if (isTabular(s.task)) cfg.hidden = h.hidden;
    if (isCls(s.task)) cfg.class_weight = !!h.class_weight;
    if (isDetect(s.task)) cfg.det_model = h.det_model || 'yolov8n.pt';
    return cfg;
  }

  /* ---------------- 第 1 步：选择数据集 ---------------- */
  const STEP_MOD = {};
  STEP_MOD[1] = {
    render(s, d, list) {
      const cards = list.map((x) => `<div class="choice-card ${s.datasetId === x.id ? 'selected' : ''}" data-pick-ds="${esc(x.id)}">
        <span class="tick">${icon('check', 'icon-sm')}</span>
        <div class="cc-ico">${icon(taskIcon(x.task_hint), 'icon-sm')}</div>
        <h4>${esc(x.name)}</h4>
        <p>${esc(MF.TASK_LABELS[x.task_hint] || x.task_hint)} · ${fmt(x.rows)} ${rowUnit(x.task_hint)} · v${x.version}</p>
        <div class="flex gap-6 mt-8" style="flex-wrap:wrap"><span class="tag">${x.cols ? x.cols + ' 列' : '图片文件夹'}</span><span class="tag">答案 ${esc(x.target_col || '—')}</span></div>
      </div>`).join('');
      const grid = list.length
        ? `<div class="choice-grid">${cards}</div>`
        : `<div class="empty" style="padding:34px 10px"><div class="ico">${icon('database')}</div><h4>还没有数据</h4><p>先到「我的数据」页上传一份数据，上传完就能回到这里开始训练模型了。</p>
            <a class="btn btn-primary mt-12" href="#/datasets">${icon('upload', 'icon-sm')}去上传数据</a></div>`;
      const summary = d ? `<div class="grid grid-2-1 mt-16">
        ${sec('这份数据的基本情况', '', `${metaTiles([
          ['数据量', fmt(d.rows) + ' ' + rowUnit(d.task_hint)],
          ['字段数', d.cols ? d.cols + ' 列' : '图片文件夹'],
          ['版本', 'v' + d.version],
          ['上传时间', esc(d.created_at || '—')]
        ])}
        <div class="kv-list mt-16">
          <div class="kv-row"><span class="k">模型要做的事</span><span class="v">${esc(MF.TASK_LABELS[d.task_hint] || d.task_hint)}</span></div>
          <div class="kv-row"><span class="k">答案列</span><span class="v">${esc(d.target_col || '—')}</span></div>
        </div>
        <a class="btn btn-sm btn-ghost mt-12" href="#/datasets/${esc(d.id)}?tab=overview">查看数据详情 ${icon('arrow-right', 'icon-sm')}</a>`, '')}
        ${sec('说明', '', `<div class="notice">${icon('shield', 'icon-sm')}<div>这里用的就是你上传的那份数据，训练在你自己的电脑上完成。训练好的模型可以保存下来，随时拿新数据试试效果。</div></div>
          <div class="kv-list mt-16">
            <div class="kv-row"><span class="k">数据来源</span><span class="v">我上传的数据</span></div>
            <div class="kv-row"><span class="k">数据用途</span><span class="v">只用于本次训练</span></div>
          </div>`, '')}
      </div>` : (list.length ? `<div class="notice info mt-16">${icon('info', 'icon-sm')}<div>请选择上方一个数据集作为本次建模的输入。</div></div>` : '');

      return `<div class="wiz-section-title"><span class="num">1</span>选一份数据</div>
        <div class="section-sub" style="margin:-4px 0 12px 28px">用你之前上传过的数据，或者现在上传一份新的。</div>
        ${grid}
        ${list.length ? `<div class="flex gap-10 mt-14"><button class="btn" data-wz-upload>${icon('upload', 'icon-sm')}上传新数据</button><span class="tiny muted" style="align-self:center">支持表格、图片、文本，单个文件最大 3 GB</span></div>` : ''}
        ${summary}`;
    },
    bind(s, host, d, list) {
      MF.$$('[data-pick-ds]', host).forEach((el) => (el.onclick = () => {
        const ds = list.filter((x) => x.id === el.dataset.pickDs)[0];
        if (!ds) return;
        if (s.datasetId && s.datasetId !== ds.id && s.maxStep > 1) {
          MF.confirm({ title: '换一份数据', message: '换了数据之后，后面几步的选择会重新来过。已经提交的训练任务不受影响。', okText: '换，并重置' }).then((ok) => {
            if (!ok) return;
            applyDataset(s, ds); s.step = 1; s.maxStep = 1; save(s); MF.route();
          });
          return;
        }
        applyDataset(s, ds); save(s); MF.route();
      }));
      const up = MF.$('[data-wz-upload]', host);
      if (up) up.onclick = () => MF.openUploadModal && MF.openUploadModal();
    }
  };

  /* ---------------- 第 2 步：确认任务 ---------------- */
  STEP_MOD[2] = {
    render(s, d) {
      const t = s.task;
      const card = `<div class="choice-card selected" style="cursor:default">
        <span class="tick">${icon('check', 'icon-sm')}</span>
        <div class="cc-ico">${icon(taskIcon(t), 'icon-sm')}</div>
        <h4>${esc(MF.TASK_LABELS[t] || t)}</h4><p>${esc(TASK_DESC[t] || '')}</p></div>`;
      // 目标列用大白话解释：模型要预测的是哪一列
      const targetExplain = d.target_col
        ? `模型要预测的是数据里的 <b>${esc(d.target_col)}</b> 这一列。`
        : '';
      const sc = (d.meta || {}).split_counts || null;
      const splitNote = sc && (sc.train || 0) > 0
        ? `数据已经自带划分：训练用 ${fmt(sc.train)} 条`
          + (sc.val ? `、验证用 ${fmt(sc.val)} 条` : '')
          + (sc.test ? `、测试用 ${fmt(sc.test)} 条` : '')
          + '。测试集只用于最后检验模型好坏，不参与训练。'
        : '数据没有自带划分，训练时会自动按比例切出一部分做验证。';
      return `${sec('模型要做的事', '1', `<div class="choice-grid">${card}</div>
          <div class="notice mt-14">${icon('info', 'icon-sm')}<div>系统已根据数据内容自动判断出任务类型，无需手动选择。${targetExplain}</div></div>`, '系统已经替你判断好了，确认一下即可')}
        ${sec('这份数据长什么样', '2', `${metaTiles([
          ['任务类型', esc(MF.TASK_LABELS[t] || t)],
          ['要预测的列', esc(d.target_col || '—')],
          ['数据量', fmt(d.rows) + ' ' + rowUnit(t)],
          ['字段数', d.cols ? d.cols + ' 个' : '图片文件夹']
        ])}
        <div class="notice mt-14">${icon('shield', 'icon-sm')}<div>${esc(splitNote)}</div></div>
        <details class="mt-12"><summary class="small muted" style="cursor:pointer">技术细节（可跳过）</summary>
          <div class="small muted mt-8">${esc(TASK_TECH[t] || '')}</div></details>`, '')}`;
    },
    bind() {}
  };

  /* ---------------- 第 3 步：字段与目标 ---------------- */
  STEP_MOD[3] = {
    render(s, d) {
      const m = d.meta || {};
      const schema = schemaOf(d);
      // 面向小白：先用人话说「哪一列是答案、哪些是线索」，
      // 完整字段表折叠起来，只给需要看细节的人。
      const rows = schema.map((c) => `<tr>
        <td class="small" style="font-weight:${c.role === 'target' ? '700' : '500'}">${esc(c.name)} ${c.role === 'target' ? '<span class="tag tag-blue">答案</span>' : ''}</td>
        <td class="small muted">${esc(TYPE_LABELS[c.type] || c.type)}</td>
        <td class="small">${c.role === 'target' ? '要预测的' : '用来参考的'}</td>
        <td class="num">${c.nulls || 0}</td>
        <td class="num">${c.unique !== undefined && c.unique !== null ? fmt(c.unique) : '—'}</td>
        <td class="small muted">${(c.min !== null && c.min !== undefined) ? `${numStr(c.min)} ~ ${numStr(c.max)}` : '—'}</td>
      </tr>`).join('');
      const table = `<div class="table-wrap" style="border:1px solid var(--line);border-radius:8px"><table class="tbl">
        <thead><tr><th>字段</th><th>类型</th><th>作用</th><th>缺失</th><th>不同取值</th><th>取值范围</th></tr></thead><tbody>${rows}</tbody></table></div>`;

      const nFeatures = schema.filter((c) => c.role !== 'target').length;
      // 让用户能自己纠正「答案列选错」——这是小白最高频的错误
      const targetOptions = schema.map((c) =>
        `<option value="${esc(c.name)}" ${c.name === d.target_col ? 'selected' : ''}>${esc(c.name)}</option>`
      ).join('');
      const editableTarget = (d.task_hint === 'tabular_classification'
        || d.task_hint === 'tabular_regression')
        ? `<div class="field full" style="margin-top:14px">
            <label>答案列选对了吗？</label>
            <select class="select" id="wz-target-col">${targetOptions}</select>
            <div class="hint">如果选错了，可以在这里改。改完点下一步即可生效，不用重新上传数据。</div>
          </div>` : '';
      const plain = `<div class="notice">${icon('info', 'icon-sm')}<div>
        这份数据有 <b>${schema.length}</b> 列：其中 <b>${esc(d.target_col || '—')}</b> 是模型要猜的「答案」，
        另外 <b>${nFeatures}</b> 列是模型做判断时参考的「线索」。
        ${m.classes && m.classes.length ? `答案一共有 <b>${m.classes.length}</b> 种可能。` : ''}
        ${m.text_col ? `其中 <b>${esc(m.text_col)}</b> 是正文内容。` : ''}
        </div></div>`;

      const classes = m.classes && m.classes.length ? `<div class="field full" style="margin-top:14px"><label>模型要区分的类别（${m.classes.length} 种）</label>
        <div class="flex gap-6" style="flex-wrap:wrap">${m.classes.map((c, i) => `<span class="tag tag-blue">${esc(c)}${m.class_counts && m.class_counts[i] !== undefined ? ' · ' + fmt(m.class_counts[i]) + ' 条' : ''}</span>`).join('')}</div></div>` : '';
      const textCol = m.text_col ? `<div class="notice mt-14">${icon('text', 'icon-sm')}<div>模型会读懂 <b>${esc(m.text_col)}</b> 这一列的文字内容来做判断，答案是 <b>${esc(d.target_col || '—')}</b>。</div></div>` : '';

      return `${sec('这份数据的结构', '1', `${plain}
          ${editableTarget}
          <details class="mt-12"><summary class="small muted" style="cursor:pointer">查看完整字段表</summary>
            <div class="mt-10">${table}</div></details>`, '')}
        ${classes || textCol ? sec('模型要区分的类别', '2', `${classes}${textCol}`, '') : ''}`;
    },
    bind(s, host, d) {
      const sel = MF.$('#wz-target-col', host);
      if (!sel || !d) return;
      sel.onchange = async () => {
        const v = sel.value;
        if (!v || v === d.target_col) return;
        try {
          const r = await MF.api.updateDatasetTarget(d.id, v);
          d.target_col = r.target_col;
          if (r.task_hint) d.task_hint = r.task_hint;
          s.task = d.task_hint;
          MF.toast('已更新', r.message || '答案列已修改', 'success');
          save(s);
          MF.route();
        } catch (e) {
          MF.toast('修改失败', e.message || '请稍后重试', 'error');
          sel.value = d.target_col || '';
        }
      };
    }
  };

  /* ---------------- 第 4 步：网络架构 ---------------- */
  // 面向小白：先说「用什么方法、为什么」，技术细节折叠起来
  const ARCH_PLAIN = {
    tabular_classification: {
      plain: '梯度提升树',
      why: '表格数据的公认强基线。它会自动学习各列之间的复杂关系，比普通神经网络更准、更不容易过拟合，而且不需要你调参。'
    },
    tabular_regression: {
      plain: '梯度提升树（预测数值）',
      why: '同一种强基线，只是输出一个数字而不是类别。'
    },
    image_classification: {
      plain: '预训练图像模型微调',
      why: '直接用「已经看过几十万张图、学会认物体」的现成模型，再针对你的图片做少量调整。数据量不大时，这比从零学快得多也准得多。'
    },
    text_classification: {
      plain: '预训练语言模型微调',
      why: '直接用「已经读过海量书籍、理解词义」的现成模型，再针对你的文本做少量调整。模型是真正理解语义在分类，不是靠词频猜。'
    },
    object_detection: {
      plain: 'YOLO 目标检测模型',
      why: '业界最常用的检测器，速度快、效果好。它基于「已经看过几十万个标注框」的预训练模型微调而来。'
    }
  };
  STEP_MOD[4] = {
    render(s, d) {
      const a = ARCH[s.task] || ARCH.tabular_classification;
      const p = ARCH_PLAIN[s.task] || { plain: a.name, why: a.desc };
      const card = `<div class="choice-card selected" style="cursor:default">
        <span class="tick">${icon('check', 'icon-sm')}</span>
        <div class="cc-ico">${icon('layers', 'icon-sm')}</div>
        <h4>${esc(p.plain)} <span class="tag tag-blue">已自动选定</span></h4><p>${esc(p.why)}</p></div>`;
      return `${sec('用什么方法', '1', `<div class="choice-grid">${card}</div>
          <div class="notice mt-14">${icon('info', 'icon-sm')}<div>你不需要自己选择模型。系统已经按数据类型选好了当前最合适的方法，直接往下走就行。</div></div>`, '已按你的数据自动选定最佳方案')}
        ${sec('模型内部是怎么工作的（可跳过）', '2', `<details><summary class="small muted" style="cursor:pointer">展开查看技术细节</summary>
          <div class="mt-10">${archCanvas(s, d)}</div>
          <div class="small muted mt-8">${esc(a.desc)}</div></details>`, '')}`;
    },
    bind() {}
  };

  /* ---------------- 第 5 步：超参配置 ---------------- */
  // 与后端 train.auto_params 规则保持一致：按任务类型与样本量推荐 epochs/lr/batch
  function autoHyper(task, rows) {
    rows = Number(rows) || 0;
    if (task === 'image_classification') {
      if (rows < 500) return { epochs: 15, lr: 0.005, batch_size: 16 };
      if (rows < 5000) return { epochs: 8, lr: 0.003, batch_size: 32 };
      return { epochs: 5, lr: 0.001, batch_size: 64 };
    }
    if (rows < 500) return { epochs: 30, lr: 0.01, batch_size: 16 };
    if (rows < 5000) return { epochs: 20, lr: 0.005, batch_size: 32 };
    if (rows < 50000) return { epochs: 10, lr: 0.003, batch_size: 64 };
    return { epochs: 5, lr: 0.001, batch_size: 128 };
  }
  // auto 模式下把推荐值同步进状态，保证确认页/提交值与后端实际一致
  function syncAuto(s, d) {
    if (s.hyper.auto && d) Object.assign(s.hyper, autoHyper(s.task, d.rows));
  }

  STEP_MOD[5] = {
    render(s, d) {
      const h = s.hyper;
      syncAuto(s, d);
      const tabular = isTabular(s.task);
      const dis = h.auto ? 'disabled' : '';
      // AI 解析阶段锁定的划分：存在时直接使用，不再让用户重设验证集比例
      const sc = (d.meta || {}).split_counts || null;
      const hasSplit = !!(sc && (sc.train || 0) > 0);
      const valRows = hasSplit ? (sc.val || 0) : Math.max(1, Math.round(d.rows * Number(h.val_split)));

      // 分支拆开写，避免嵌套模板字符串的反引号冲突
      let splitSection;
      if (isDetect(s.task)) {
        // 检测任务：展示边界框信息与检测专属方案
        const dc = (d.meta || {}).datacard || {};
        const plan = (d.meta || {}).ml_plan || {};
        const cls = (d.meta || {}).classes || [];
        const tiles = metaTiles([
          ['训练图', fmt(sc ? (sc.train || 0) : d.rows) + ' 张'],
          ['验证图', fmt((sc && sc.val) || 0) + ' 张'],
          ['测试图', fmt((sc && sc.test) || 0) + ' 张'],
          ['目标类别', cls.length + ' 类'],
          ['标注框', fmt(dc.n_boxes || 0) + ' 个'],
          ['平均目标数', (dc.avg_objects || 0) + ' 个/图']
        ]);
        const clsChips = cls.length
          ? '<div class="mt-16"><div class="small muted mb-8">检测类别</div><div class="flex gap-6" style="flex-wrap:wrap">' +
            cls.map((c) => '<span class="tag">' + esc(c) + '</span>').join('') + '</div></div>'
          : '';
        const notice = '<div class="notice mt-16">' + icon('target', 'icon-sm') +
          '<div><b>目标检测任务</b><br><span class="small">' +
          '平台已识别出数据中的边界框标注，将训练 YOLO 检测器，' +
          '输出每个目标的类别与位置。评估指标为 mAP（不是准确率）。' +
          '</span></div></div>';
        const planNote = plan.model
          ? '<div class="notice mt-16">' + icon('activity', 'icon-sm') +
            '<div><span class="small">' + esc(String(plan.reason || '')) +
            ' · ' + esc(String(plan.model)) + ' · ' + (plan.epochs || '—') + ' 轮 · ' +
            (plan.imgsz || '—') + 'px</span></div></div>'
          : '';
        splitSection = sec('数据怎么用', '2', tiles + clsChips + notice + planNote, '');
      } else if (hasSplit) {
        // 按数据实情展示：数据集有几个划分就显示几个，不硬塞空行
        const _u = rowUnit(s.task);
        const parts = [['训练集', fmt(sc.train || 0) + ' ' + _u]];
        if (sc.val) parts.push(['验证集', fmt(sc.val) + ' ' + _u]);
        if (sc.test) parts.push(['测试集', fmt(sc.test) + ' ' + _u]);
        const dimLabel = s.task === 'image_classification' ? '图像张量'
          : s.task === 'text_classification' ? 'Token 序列'
          : featureCols(d).length + ' 维';
        parts.push(['输入维度', dimLabel]);
        parts.push(['输出维度', s.task === 'tabular_regression' ? '1 个值' : outputDim(d) + ' 类']);
        const tiles = metaTiles(parts);

        let desc;
        if (sc.val && sc.test) {
          desc = '训练集用于训练、验证集用于早停与最优权重回滚、测试集全程不参与训练，' +
                 '训练结束后用于最终效果评估。';
        } else if (sc.val) {
          desc = '训练集用于训练、验证集用于早停与最优权重回滚。该划分由 AI 在解析阶段' +
                 '识别自你的原始数据目录，无需再调整。';
        } else if (sc.test) {
          desc = '训练集用于训练；测试集全程不参与训练，训练结束后用于最终效果评估。' +
                 '该数据集没有自带验证集，训练时会按比例从训练集中切出一份用于早停。';
        } else {
          desc = '你的数据只有训练集，没有自带验证集或测试集。训练时会按下方比例从训练集中' +
                 '切出一份做验证（用于早停与选优），这部分样本不参与梯度更新；' +
                 '样本不会被丢弃，但不做切分时全部用于训练。';
        }
        const notice = '<div class="notice mt-16">' + icon('shield', 'icon-sm') +
          '<div><b>沿用你原始数据的划分</b><br>' +
          '<span class="small">' + desc + '</span></div></div>';
        splitSection = sec('数据怎么用', '2', tiles + notice, '');
      } else {
        const tiles = metaTiles([
          ['训练集', fmt(d.rows - valRows) + ' ' + rowUnit(s.task)],
          ['验证集', fmt(valRows) + ' ' + rowUnit(s.task)],
          ['输入维度', s.task === 'image_classification' ? '图像张量' : s.task === 'text_classification' ? 'Token 序列' : featureCols(d).length + ' 维'],
          ['输出维度', s.task === 'tabular_regression' ? '1 个值' : outputDim(d) + ' 类']
        ]);
        const notice = '<div class="notice mt-16">' + icon('info', 'icon-sm') +
          '<div>该数据集未自带 train/val/test 划分，训练时将按下方比例自动切分验证集。</div></div>' +
          '<div class="notice mt-16">' + icon('clock', 'icon-sm') +
          '<div>训练在服务端执行，小型数据集通常数十秒内完成；进度与日志可在训练监控页实时查看。</div></div>';
        splitSection = sec('数据怎么用', '2', tiles + notice, '');
      }

      // 智能体建模方案：解析阶段就已定好，这里明明白白告诉用户打算怎么训
      let planSection = '';
      const plan = (d.meta || {}).ml_plan || null;
      const dc = (d.meta || {}).datacard || null;
      if (plan && !isDetect(s.task)) {
        let tiles = [];
        const notes = [];
        if (s.task === 'image_classification') {
          tiles = [
            ['用的现成模型', plan.backbone === 'resnet18' ? 'ResNet18（更强）' : 'MobileNetV3（更快）'],
            ['图片处理尺寸', (plan.img_size || '—') + ' 像素'],
            ['训练方式', plan.mode === 'finetune'
              ? '微调（现成模型也会跟着你的数据调整）'
              : '快速学习（现成模型保持不变，只学判断规则）'],
            ['防过拟合增强', plan.aug_copies > 1
              ? '已开启（每张图生成 ' + plan.aug_copies + ' 个变体）' : '未开启'],
            ['训练轮次', (plan.epochs || '—') + ' 轮']
          ];
          if (dc && (dc.bad_images || 0) > 0) {
            notes.push('抽查发现 ' + dc.bad_images + ' 张图像无法读取，已跳过');
          }
        } else if (s.task === 'text_classification') {
          tiles = [
            ['用的现成模型', esc(String(plan.encoder || '—'))],
            ['训练轮次', (plan.epochs || '—') + ' 轮'],
            ['学习速度', String(plan.lr || '—')],
            ['防过拟合强度', String(plan.dropout || '—')]
          ];
        } else {
          tiles = [
            ['模型类型', plan.model === 'hgb' ? '梯度提升树' : esc(String(plan.model || '—'))],
            ['最大迭代', '300 轮'],
            ['学习速度', String(plan.learning_rate || '—')],
            ['模型复杂度', String(plan.max_leaf_nodes || '—')]
          ];
          notes.push('表格任务默认使用梯度提升树，它对列与列之间复杂关系的建模能力比神经网络更强');
        }
        if (plan.class_weight) notes.push('各类别样本数差别较大，已自动调整各类权重');
        const extra = notes.length
          ? '<div class="notice mt-16">' + icon('info', 'icon-sm') +
            '<div><span class="small">' + esc(notes.join('；')) + '</span></div></div>'
          : '';
        planSection = sec('系统为你做的准备', '3', metaTiles(tiles) +
          '<div class="notice mt-16">' + icon('activity', 'icon-sm') +
          '<div><span class="small">' +
          esc(String(plan.reason || '按数据实情自动决定')) +
          '</span></div></div>' + extra, '');
      }

      // 检测任务用 YOLO 自己的超参体系，不展示 MLP 相关项
      const detectSection = isDetect(s.task) ? sec('训练设置', '1', `<div class="notice">${icon('sparkles', 'icon-sm')}<div><b>已为你配好训练方案</b><br><span class="small">系统根据你的数据量自动选择了合适的训练参数。直接点下一步开始训练即可；如果训练效果不理想，可以展开下方「高级调优」自行微调。</span></div></div>
          <label class="check-row" style="margin:14px 0">
            <input type="checkbox" id="wz-auto" ${h.auto ? 'checked' : ''}><span class="ct">自动匹配训练参数<small>${h.auto ? '当前已开启：每次训练都按数据量自动决定参数，推荐用这个。' : '已关闭，将使用你手动填写的参数。'}</small></span>
          </label>
          <details><summary class="small muted" style="cursor:pointer">高级调优（一般不需要改）</summary>
          <div class="form-grid cols-3 mt-12">
          <div class="field"><label>训练轮次 Epochs</label><input class="input" type="number" min="1" max="300" id="wz-epochs" value="${h.epochs}" ${dis}><div class="hint">${h.auto ? '自动模式，由样本量决定。' : '1–300，检测器收敛需要更多轮次。'}</div></div>
          <div class="field"><label>批大小 Batch Size</label><select class="select" id="wz-batch" ${dis}>${BATCHES.map((v) => `<option value="${v}" ${Number(h.batch_size) === v ? 'selected' : ''}>${v}</option>`).join('')}</select><div class="hint">CPU 上建议 8–16，太大容易内存不足。</div></div>
          <div class="field"><label>检测模型</label><select class="select" id="wz-det-model" ${dis}><option value="yolov8n.pt">YOLOv8n（最快）</option><option value="yolov8s.pt">YOLOv8s（更准）</option></select><div class="hint">基于 COCO 预训练权重微调，n 档在 CPU 上最实用。</div></div>
          ${(hasSplit && sc.val) ? '' : `<div class="field"><label>${hasSplit ? '从训练集切出验证集' : '验证集比例'}</label><select class="select" id="wz-valsplit">${SPLITS.map((v) => `<option value="${v}" ${Number(h.val_split) === v ? 'selected' : ''}>${Math.round(v * 100)}%</option>`).join('')}</select><div class="hint">${hasSplit ? '你的数据没有自带验证集，训练时按此比例从训练集中切出，用于早停与选优。' : '从全部样本中按此比例切出验证集，其余用于训练。'}</div></div>`}
        </div></details>`, '检测器以 mAP 作为评估指标，测试集全程不参与训练')
        : sec('训练设置', '1', `<div class="notice">${icon('sparkles', 'icon-sm')}<div><b>已为你配好训练方案</b><br><span class="small">系统根据你的数据量自动选择了合适的训练参数。直接点下一步开始训练即可；如果训练效果不理想，可以展开下方「高级调优」自行微调。</span></div></div>
          <label class="check-row" style="margin:14px 0">
            <input type="checkbox" id="wz-auto" ${h.auto ? 'checked' : ''}><span class="ct">自动匹配训练参数<small>${h.auto ? '当前已开启：每次训练都按数据量自动决定参数，推荐用这个。' : '已关闭，将使用你手动填写的参数。'}</small></span>
          </label>
          <details><summary class="small muted" style="cursor:pointer">高级调优（一般不需要改）</summary>
          <div class="form-grid cols-3 mt-12">
          <div class="field"><label>训练轮次 Epochs</label><input class="input" type="number" min="1" max="100" id="wz-epochs" value="${h.epochs}" ${dis}><div class="hint">${h.auto ? '自动模式，由样本量决定。' : '1–100，过多可能过拟合。'}</div></div>
          <div class="field"><label>学习率 Learning Rate</label><input class="input" type="number" step="0.001" min="0.00001" max="1" id="wz-lr" value="${h.lr}" ${dis}></div>
          <div class="field"><label>批大小 Batch Size</label><select class="select" id="wz-batch" ${dis}>${BATCHES.map((v) => `<option value="${v}" ${Number(h.batch_size) === v ? 'selected' : ''}>${v}</option>`).join('')}</select></div>
          <div class="field"><label>优化器 Optimizer</label><select class="select" id="wz-optimizer">${OPTIMIZERS.map((o) => `<option value="${o[0]}" ${h.optimizer === o[0] ? 'selected' : ''}>${o[1]}</option>`).join('')}</select><div class="hint">Adam 通用；SGD 需配合动量，小数据集可试。</div></div>
          <div class="field"><label>学习率调度</label><select class="select" id="wz-scheduler">${SCHEDULERS.map((o) => `<option value="${o[0]}" ${h.scheduler === o[0] ? 'selected' : ''}>${o[1]}</option>`).join('')}</select><div class="hint">余弦退火可帮助收敛到更优解。</div></div>
          <div class="field"><label>权重衰减 L2</label><input class="input" type="number" step="0.0001" min="0" max="0.1" id="wz-wd" value="${h.weight_decay}"><div class="hint">0–0.1，抑制过拟合。</div></div>
          <div class="field"><label>Dropout</label><input class="input" type="number" step="0.05" min="0" max="0.6" id="wz-dropout" value="${h.dropout}"><div class="hint">0–0.6，随机失活比例。</div></div>
          <div class="field"><label>早停耐心值</label><input class="input" type="number" min="0" max="50" id="wz-patience" value="${h.patience}"><div class="hint">${h.patience ? `验证指标连续 ${h.patience} 轮无改善即终止并回滚最佳权重。` : '填 0 关闭早停，将跑满全部轮次。'}</div></div>
          <div class="field"><label>隐藏层宽度 Hidden</label><input class="input" type="number" min="8" max="512" step="8" id="wz-hidden" value="${h.hidden}" ${tabular ? '' : 'disabled'}><div class="hint">${tabular ? '8–512，仅 MLP 架构可调。' : '当前架构由后端预置，无需设置。'}</div></div>
          ${(hasSplit && sc.val) ? '' : `<div class="field"><label>${hasSplit ? '从训练集切出验证集' : '验证集比例'}</label><select class="select" id="wz-valsplit">${SPLITS.map((v) => `<option value="${v}" ${Number(h.val_split) === v ? 'selected' : ''}>${Math.round(v * 100)}%</option>`).join('')}</select><div class="hint">${hasSplit ? '你的数据没有自带验证集，训练时按此比例从训练集中切出，用于早停与最优权重回滚。' : '从全部样本中按此比例切出验证集，其余用于训练。'}</div></div>`}
          ${isCls(s.task) ? `<div class="field full">
            <label class="check-row"><input type="checkbox" id="wz-cw" ${h.class_weight ? 'checked' : ''}><span class="ct">启用类别权重<small>根据类别频率自动加权，缓解类别不平衡影响。</small></span></label>
          </div>` : ''}
        </div></details>`, '保持默认设置即可开始训练');
      return `${detectSection}
        ${splitSection}
        ${planSection}`;
    },
    bind(s, host) {
      const h = s.hyper;
      const bindNum = (id, key, min, max, isInt) => {
        const el = MF.$(id, host);
        if (!el || el.disabled) return;
        el.onchange = () => {
          let v = parseFloat(el.value);
          if (isNaN(v)) v = h[key];
          v = Math.max(min, Math.min(max, v));
          if (isInt) v = Math.round(v);
          h[key] = v; el.value = v; save(s); MF.route();
        };
      };
      const auto = MF.$('#wz-auto', host);
      if (auto) auto.onchange = () => { h.auto = auto.checked; save(s); MF.route(); };
      bindNum('#wz-epochs', 'epochs', 1, isDetect(s.task) ? 300 : 100, true);
      bindNum('#wz-lr', 'lr', 0.00001, 1);
      bindNum('#wz-wd', 'weight_decay', 0, 0.1, false);
      bindNum('#wz-dropout', 'dropout', 0, 0.6, false);
      bindNum('#wz-patience', 'patience', 0, 50, true);
      bindNum('#wz-hidden', 'hidden', 8, 512, true);
      const optEl = MF.$('#wz-optimizer', host);
      if (optEl) optEl.onchange = () => { h.optimizer = optEl.value; save(s); MF.route(); };
      const schEl = MF.$('#wz-scheduler', host);
      if (schEl) schEl.onchange = () => { h.scheduler = schEl.value; save(s); MF.route(); };
      const batch = MF.$('#wz-batch', host);
      if (batch) batch.onchange = () => { h.batch_size = Number(batch.value); save(s); MF.route(); };
      const split = MF.$('#wz-valsplit', host);
      if (split) split.onchange = () => { h.val_split = Number(split.value); save(s); MF.route(); };
      const detModel = MF.$('#wz-det-model', host);
      if (detModel) detModel.onchange = () => { h.det_model = detModel.value; save(s); MF.route(); };
      const cw = MF.$('#wz-cw', host);
      if (cw) cw.onchange = () => { h.class_weight = cw.checked; save(s); };
    }
  };

  STEP_MOD[6] = {
    render(s, d) {
      const h = s.hyper;
      const a = ARCH[s.task] || ARCH.tabular_classification;
      const sc6 = (d.meta || {}).split_counts || null;
      const pct = Math.round(Number(h.val_split) * 100) + '%';
      const valLabel = !sc6 ? ('验证集比例 ' + pct)
        : ((sc6.val || 0) > 0 ? ('沿用原始验证集 ' + fmt(sc6.val) + ' ' + rowUnit(s.task))
          : ('从训练集切出 ' + pct));
      // 确认页：主视图只给小白看得懂的，技术细节折叠
      const hp = [
        ['用什么方法', (ARCH_PLAIN[s.task] || {}).plain || esc(a.name)],
        ['训练轮次', h.epochs + ' 轮' + (h.auto ? '（自动匹配）' : '')],
        ['数据划分', valLabel]
      ];
      if (sc6 && (sc6.test || 0) > 0) {
        hp.push(['最终检验', fmt(sc6.test) + ' 条 · 不参与训练']);
      }
      if (isDetect(s.task)) {
        hp.length = 0;
        hp.push(['用什么方法', (ARCH_PLAIN[s.task] || {}).plain || esc(a.name)],
          ['检测模型', h.det_model === 'yolov8s.pt' ? 'YOLOv8s（更准）' : 'YOLOv8n（更快）'],
          ['训练轮次', h.epochs + ' 轮' + (h.auto ? '（自动匹配）' : '')],
          ['数据划分', valLabel]);
        if (sc6 && (sc6.test || 0) > 0) {
          hp.push(['最终检验', fmt(sc6.test) + ' 张 · 不参与训练']);
        }
        const dc6 = (d.meta || {}).datacard || {};
        if (dc6.n_boxes) {
          hp.push(['标注框数量', fmt(dc6.n_boxes) + ' 个 · 平均每张 ' + (dc6.avg_objects || 0) + ' 个']);
        }
      } else if (isTabular(s.task)) {
        hp.push(['模型复杂度', '中等']);
      }
      const unit = (s.task === 'image_classification' || isDetect(s.task)) ? ' 张' : ' 条';
      return `${sec('你要用什么数据', '1', `${metaTiles([
          ['数据集', esc(d.name)], ['模型要做的事', esc(MF.TASK_LABELS[s.task] || s.task)],
          ['要预测的列', esc(d.target_col || '—')], ['数据量', fmt(d.rows) + unit]
        ])}`, '')}
        ${sec('训练方案', '2', `${metaTiles(hp)}
        <div class="notice mt-16">${icon('shield', 'icon-sm')}<div>确认无误后点下一步开始训练。训练在你自己的电脑上完成，全程可以随时离开，训练进度和结果会保留下来。</div></div>
        <details class="mt-12"><summary class="small muted" style="cursor:pointer">技术细节（可跳过）</summary>
          <div class="kv-list mt-8">
            <div class="kv-row"><span class="k">数据 ID</span><span class="v mono">${esc(d.id)}</span></div>
            <div class="kv-row"><span class="k">数据版本</span><span class="v">v${d.version}</span></div>
            <div class="kv-row"><span class="k">学习率</span><span class="v">${h.lr}</span></div>
            <div class="kv-row"><span class="k">批大小</span><span class="v">${h.batch_size}</span></div>
            <div class="kv-row"><span class="k">优化器</span><span class="v">${(OPTIMIZERS.find((o) => o[0] === h.optimizer) || OPTIMIZERS[0])[1]}</span></div>
            <div class="kv-row"><span class="k">早停耐心值</span><span class="v">${h.patience ? h.patience + ' 轮' : '关闭'}</span></div>
            <div class="kv-row"><span class="k">正则化</span><span class="v">L2 ${h.weight_decay} · Dropout ${h.dropout}</span></div>
            ${isTabular(s.task) ? `<div class="kv-row"><span class="k">隐藏层宽度</span><span class="v">${h.hidden}</span></div>` : ''}
            ${isCls(s.task) ? `<div class="kv-row"><span class="k">类别权重</span><span class="v">${h.class_weight ? '启用' : '关闭'}</span></div>` : ''}
          </div></details>`, '')}`;
    },
    bind() {}
  };

  /* ---------------- 第 7 步：提交训练 ---------------- */
  STEP_MOD[7] = {
    render(s, d) {
      if (s.jobId) {
        return `<div class="notice success">${icon('check-circle', 'icon-sm')}<div><b>训练已经开始</b><br><span class="small">你可以在训练监控页看到实时进度、曲线和最终结果。训练在后台进行，可以随时离开。</span></div></div>
          ${metaTiles([['数据集', esc(d ? d.name : '—')], ['用什么方法', (ARCH_PLAIN[s.task] || {}).plain || esc((ARCH[s.task] || {}).name || '—')], ['训练轮次', s.hyper.epochs + ' 轮']])}
          <div class="flex gap-10 mt-16">
            <a class="btn btn-primary" href="#/train?job=${esc(s.jobId)}">${icon('activity', 'icon-sm')}查看训练结果</a>
            <button class="btn" data-wz-again>${icon('plus', 'icon-sm')}再建一个任务</button>
          </div>`;
      }
      return `<div class="empty" style="padding:30px 10px"><div class="ico">${icon('rocket')}</div>
          <h4>准备就绪，提交训练任务</h4><p>后端将按第 6 步确认的配置立即开始训练，提交后可随时离开并在监控页恢复查看。</p></div>
        <div class="flex gap-10 mt-6" style="justify-content:center">
          <button class="btn btn-primary" data-wz-submit>${icon('play', 'icon-sm')}提交训练任务</button>
          <button class="btn" data-wz-back6>${icon('arrow-left', 'icon-sm')}返回确认配置</button>
        </div>`;
    },
    bind(s, host, d) {
      const again = MF.$('[data-wz-again]', host);
      if (again) again.onclick = () => { s.jobId = null; save(s); MF.route(); };
      const back6 = MF.$('[data-wz-back6]', host);
      if (back6) back6.onclick = () => go(s, 6);
      const btn = MF.$('[data-wz-submit]', host);
      if (!btn) return;
      btn.onclick = async () => {
        if (!d) { MF.toast('数据集不可用', '请返回第 1 步重新选择。', 'warn'); return; }
        btn.disabled = true; btn.innerHTML = `${icon('refresh', 'icon-sm spin')}正在创建任务…`;
        try {
          const job = await MF.api.createTrainJob(trainCfg(s, d));
          s.jobId = job.id; save(s);
          MF.toast('训练任务已创建', `任务 ${job.id} 已开始执行`, 'success');
          location.hash = `#/train?job=${encodeURIComponent(job.id)}`;
        } catch (e) {
          btn.disabled = false; btn.innerHTML = `${icon('play', 'icon-sm')}提交训练任务`;
          MF.toast('创建训练任务失败', e.message, 'error');
        }
      };
    }
  };

  /* ---------------- 向导外壳 ---------------- */
  function railHTML(s) {
    const groups = [];
    STEPS.forEach((st) => {
      let g = groups.filter((x) => x.label === st.group)[0];
      if (!g) { g = { label: st.group, items: [] }; groups.push(g); }
      g.items.push(st);
    });
    return groups.map((g) => `<div class="wiz-group"><div class="wiz-group-label">${g.label}</div>
      ${g.items.map((st) => {
        const cls = s.step === st.no ? 'active' : (s.maxStep || 1) >= st.no ? 'done' : 'locked';
        return `<div class="wiz-step ${cls}" data-wz-jump="${st.no}">
          <div class="dot">${(s.maxStep || 1) >= st.no && s.step !== st.no ? icon('check', 'icon-sm') : st.no}</div>
          <div>${st.label}<span class="sub">${st.sub}</span></div></div>`;
      }).join('')}</div>`).join('');
  }

  MF.views.build = function (r, view) {
    view.innerHTML = loadingHTML('正在从后端加载数据集…');
    MF.api.listDatasets()
      .then((list) => renderShell(r, view, list))
      .catch((e) => {
        view.innerHTML = errHTML(e);
        const retry = MF.$('[data-retry]', view);
        if (retry) retry.onclick = () => MF.route();
      });
  };

  function renderShell(r, view, list) {
    const s = ensureState(r, list);
    save(s);
    if (r.query.get('new') === '1') history.replaceState(null, '', '#/build');
    const d = datasetOf(s, list);
    s._dataset = d;
    const step = STEPS[s.step - 1];
    const mod = STEP_MOD[s.step];
    const body = mod ? mod.render(s, d, list) : '<div class="empty"><p>该步骤尚未实现。</p></div>';
    const nextLabel = s.step === 6 ? '下一步：提交训练' : '保存并继续';

    view.innerHTML = `
      ${MF.pageHead('新建模型', '选好数据，剩下的系统帮你搞定。进度会自动保存，随时可以离开。',
        `<a class="btn" href="#/datasets">${icon('database')}数据资产</a><button class="btn" data-wz-save>${icon('download')}保存草稿</button>`)}
      <div class="wizard">
        <div class="wizard-rail">
          <div class="wiz-top">
            <div class="t">建模向导</div>
            <div class="s">${esc(s.modelName || '未命名模型')} · ${d ? esc(d.name) : '未选择数据'}</div>
            <div class="wiz-progress"><div class="bar"><span style="width:${(s.step / TOTAL) * 100}%"></span></div><div class="n">${s.step}/${TOTAL}</div></div>
          </div>
          ${railHTML(s)}
        </div>
        <div class="wiz-main">
          <div class="wiz-head">
            <div style="flex:1"><div class="idx">第 ${s.step} 步 / 共 ${TOTAL} 步 · ${esc(step.group)}</div><h2>${esc(step.label)}</h2><p>${esc(step.sub)}</p></div>
            <div class="actions"><button class="btn btn-sm btn-soft" data-wz-ai>${icon('sparkles', 'icon-sm')}AI 建议</button><button class="btn btn-sm" data-wz-exit>退出</button></div>
          </div>
          <div class="wiz-body" id="wiz-body">${body}</div>
          <div class="wiz-foot">
            <div class="foot-note">${s.jobId ? `已创建训练任务 <span class="mono">${esc(s.jobId)}</span> · ` : ''}草稿自动保存在当前浏览器</div>
            <div class="actions">
              ${s.step > 1 ? `<button class="btn" data-wz-back>${icon('arrow-left', 'icon-sm')}上一步</button>` : ''}
              ${s.step < TOTAL ? `<button class="btn btn-primary" data-wz-next>${nextLabel}${icon('arrow-right', 'icon-sm')}</button>` : ''}
            </div>
          </div>
        </div>
      </div>`;

    if (mod && mod.bind) mod.bind(s, MF.$('#wiz-body', view), d, list);

    const saveBtn = MF.$('[data-wz-save]', view);
    if (saveBtn) saveBtn.onclick = () => { save(s); MF.toast('草稿已保存', `当前第 ${s.step} 步 · ${s.modelName || '未命名模型'}`, 'success', 2000); };
    const ai = MF.$('[data-wz-ai]', view);
    if (ai) ai.onclick = () => MF.openAssistant();
    MF.$('[data-wz-exit]', view).onclick = async () => {
      save(s);
      if (await MF.confirm({ title: '退出向导', message: '当前配置已保存为草稿，可在工作台继续。', okText: '退出向导' })) location.hash = '#/overview';
    };
    const back = MF.$('[data-wz-back]', view);
    if (back) back.onclick = () => go(s, s.step - 1);
    const next = MF.$('[data-wz-next]', view);
    if (next) next.onclick = () => go(s, s.step + 1);
    MF.$$('[data-wz-jump]', view).forEach((el) => (el.onclick = () => {
      const n = Number(el.dataset.wzJump);
      if (n > (s.maxStep || 1)) { MF.toast('步骤尚未解锁', '请按顺序完成前面的步骤。', 'warn'); return; }
      go(s, n);
    }));
  }
})(window);
