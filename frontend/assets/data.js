/* =========================================================
   ModelForge · 数据资产（真实 FastAPI 后端）
   列表 / 详情 / 上传 / 清洗 / EDA 走 MF.api。
   ========================================================= */
(function (global) {
  const MF = global.MF;
  const { esc, fmt, icon } = MF;

  const CATS = {
    套餐类型: ['基础', '畅享', '尊享'], 合约状态: ['是', '否'], 城市等级: ['A', 'B', 'C'],
    朝向: ['南北', '南', '东', '北'], 装修: ['精装', '简装', '毛坯'], 渠道类型: ['自然流量', '投放', '私域']
  };
  MF.CATS = CATS;

  /* ---------------- 适配工具 ---------------- */
  const TYPE_LABELS = { number: '数值', category: '类别', text: '文本', datetime: '日期', bool: '布尔' };
  const sizeStr = (b) => {
    b = Number(b) || 0;
    if (b >= 1073741824) return (b / 1073741824).toFixed(1) + ' GB';
    if (b >= 1048576) return (b / 1048576).toFixed(1) + ' MB';
    if (b >= 1024) return (b / 1024).toFixed(1) + ' KB';
    return b + ' B';
  };
  const numStr = (v, d) => {
    if (typeof v !== 'number' || !isFinite(v)) return '—';
    const p = d === undefined ? 2 : d;
    return Math.abs(v) >= 100 ? v.toFixed(1) : String(+v.toFixed(p));
  };
  const binLabel = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : String(+v.toFixed(2)));
  const loadingHTML = (t) => `<div class="card card-pad"><div class="notice">${icon('refresh', 'icon-sm spin')}<div>${esc(t || '正在加载…')}</div></div></div>`;
  const errHTML = (e) => `<div class="card card-pad"><div class="notice danger">${icon('alert', 'icon-sm')}<div><b>请求失败</b><br>${esc(e.message || e)}</div></div></div>`;
  const cleanState = (ds) => {
    const s = (ds.meta && ds.meta.stats) || {};
    return (s.missing_cells || 0) === 0 && (s.duplicate_rows || 0) === 0;
  };
  const qualityOf = (ds) => {
    const s = (ds.meta && ds.meta.stats) || {};
    let q = 100 - (s.missing_pct || 0) * 400;
    if ((s.duplicate_rows || 0) > 0) q -= 6;
    return Math.max(40, Math.min(99, Math.round(q)));
  };

  /* ---------------- 数据资产列表 ---------------- */
  // 路由对 #/datasets 与 #/datasets/:id 均调用 MF.views.datasets，这里按有无 id 分发
  MF.views.datasets = function (r, view) {
    if (r.id) return MF.views.dataset(r, view);
    return datasetsListView(r, view);
  };

  function datasetsListView(r, view) {
    view.innerHTML = loadingHTML('正在从后端加载数据集…');
    MF.api.listDatasets()
      .then((list) => renderList(list))
      .catch((e) => { view.innerHTML = errHTML(e); });

    function renderList(list) {
      const totalRows = list.reduce((a, d) => a + (d.rows || 0), 0);
      const avgMiss = list.length ? list.reduce((a, d) => a + (((d.meta || {}).stats || {}).missing_pct || 0), 0) / list.length : 0;
      const issues = list.filter((d) => !cleanState(d)).length;
      const stats = [
        { label: '数据集总数', value: list.length, suffix: '个', icon: 'database', note: '来自后端服务', spark: [2, 3, 3, 4, 4, 5, 5, 6, Math.max(6, list.length)] },
        { label: '总样本量', value: fmt(totalRows), suffix: '条', icon: 'layers', tone: 'green', note: '全部数据集合计', spark: [4, 6, 8, 10, 12, 14, 16, 18, 20], color: '#0f9d76' },
        { label: '平均缺失率', value: MF.pct(avgMiss), suffix: '', icon: 'shield', tone: 'purple', note: '按单元格统计', spark: [9, 8, 8, 7, 6, 6, 5, 5, 4], color: '#7c3aed' },
        { label: '待处理问题', value: issues, suffix: '项', icon: 'alert', tone: 'amber', note: '缺失 / 重复', spark: [7, 6, 6, 5, 5, 4, 4, 3, Math.max(1, issues)], color: '#c77a0a' }
      ];
      const rows = list.map((d) => {
        const st = (d.meta || {}).stats || {};
        const filename = (d.meta || {}).filename || d.source;
        return `<tr data-task="${esc(d.task_hint)}" data-search="${esc((d.name + filename).toLowerCase())}">
        <td><div class="dataset-title">${MF.charts.datasetIcon(d.task_hint)}<div><div class="cell-main">${esc(d.name)}</div><div class="cell-sub">${esc(filename)}</div></div></div></td>
        <td>${MF.taskPill(d.task_hint)}</td>
        <td class="num">${fmt(d.rows)}<span class="tiny muted"> 行</span></td>
        <td class="num">${d.cols ? d.cols + ' 列' : '—'}</td>
        <td class="num">${sizeStr(d.size_bytes)}</td>
        <td><span class="tag">v${d.version}</span></td>
        <td>${cleanState(d) ? '<span class="pill pill-success"><span class="pdot"></span>数据良好</span>' : `<span class="pill pill-warn"><span class="pdot"></span>待清洗</span>`}</td>
        <td class="small muted nowrap">${esc(d.created_at)}</td>
        <td><div class="tbl-actions">
          <a class="btn btn-sm btn-ghost" href="#/datasets/${esc(d.id)}">查看</a>
          <button class="btn btn-sm btn-ghost" data-build="${esc(d.id)}" title="开始建模">${icon('sparkles', 'icon-sm')}</button>
          <button class="btn btn-sm btn-ghost" data-del="${esc(d.id)}" data-name="${esc(d.name)}" title="删除" style="color:#dc2626">${icon('trash', 'icon-sm')}</button>
        </div></td>
      </tr>`;
      }).join('');

      view.innerHTML = `
        ${MF.pageHead('我的数据', '你上传过的所有数据都在这里，可以查看内容、清洗问题和查看分析结果。',
          `<button class="btn" data-upload>${icon('upload')}上传数据集</button><a class="btn btn-primary" href="#/build?new=1">${icon('sparkles')}从数据开始建模</a>`)}
        <div class="grid grid-4">${stats.map(MF.statCard).join('')}</div>
        <div class="notice mt-16">${icon('info', 'icon-sm')}<div>可以上传这几类数据：<b>表格</b>（CSV / Excel / JSON，每行一条记录）、<b>图片</b>（按类别分文件夹后打包成 ZIP）、<b>文字</b>（每行一段文本 + 一个标签）、<b>带框的图片</b>（YOLO/COCO 标注格式）。系统会自动判断类型，单个文件最大 3 GB。</div></div>
        <div class="card mt-16">
          <div class="card-head" style="padding-bottom:14px">
            <div class="segmented" data-filter-tabs><button class="active" data-task="all">全部</button><button data-task="tabular_classification">表格分类</button><button data-task="tabular_regression">表格回归</button><button data-task="image_classification">图像分类</button><button data-task="text_classification">文本分类</button></div>
            <div class="actions"><div class="search-box" style="width:210px;background:#fff">${icon('search', 'icon-sm')}<input placeholder="搜索数据集名称" data-ds-search></div></div>
          </div>
          <div class="table-wrap" style="border-top:1px solid var(--line)"><table class="tbl">
            <thead><tr><th>数据集</th><th>任务类型</th><th>样本量</th><th>字段数</th><th>大小</th><th>版本</th><th>清洗状态</th><th>上传时间</th><th>操作</th></tr></thead>
            <tbody data-ds-body>${rows}</tbody></table></div>
          <div class="tbl-empty" data-ds-empty style="display:${list.length ? 'none' : ''}">${icon('database', 'icon-lg')}<div class="mt-8">还没有上传的数据集</div><div class="tiny muted mt-4">点击右上角「上传数据集」导入 CSV 或 ZIP 文件</div></div>
        </div>`;

      MF.$('[data-upload]', view).onclick = () => MF.openUploadModal();
      MF.$$('[data-build]', view).forEach((b) => (b.onclick = () => { location.hash = `#/build?dataset=${b.dataset.build}`; }));
      MF.$$('[data-del]', view).forEach((b) => (b.onclick = async () => {
        const ok = await MF.confirm({ title: '删除数据集', message: `确认删除「${b.dataset.name}」？后端文件与全部数据版本将一并移除，且不可恢复。`, okText: '删除', danger: true });
        if (!ok) return;
        try {
          await MF.api.deleteDataset(b.dataset.del);
          MF.toast('数据集已删除', b.dataset.name, 'success');
          MF.route();
        } catch (e) { MF.toast('删除失败', e.message, 'error'); }
      }));
      let task = 'all';
      const applyFilter = () => {
        const q = (MF.$('[data-ds-search]', view).value || '').trim().toLowerCase();
        let shown = 0;
        MF.$$('[data-ds-body] tr', view).forEach((tr) => {
          const ok = (task === 'all' || tr.dataset.task === task) && (!q || tr.dataset.search.indexOf(q) >= 0);
          tr.style.display = ok ? '' : 'none'; if (ok) shown++;
        });
        MF.$('[data-ds-empty]', view).style.display = shown ? 'none' : '';
      };
      MF.$$('[data-filter-tabs] button', view).forEach((b) => (b.onclick = () => {
        MF.$$('[data-filter-tabs] button', view).forEach((x) => x.classList.remove('active'));
        b.classList.add('active'); task = b.dataset.task; applyFilter();
      }));
      MF.$('[data-ds-search]', view).oninput = applyFilter;
    }
  };

  /* ---------------- 上传数据集（AI 全自动解析） ---------------- */

  // 把后端的技术报错翻译成「发生了什么 + 你该怎么做」。
  // 小白看到 'Traceback' 或 'target_col' 是完全无从下手的。
  function friendlyError(raw) {
    const msg = String(raw || '');
    const rules = [
      [/Token Plan|429|用量上限/i,
        'AI 服务额度用完了。请联系管理员充值或更换模型服务，之后再试。'],
      [/尚未配置默认大模型|大模型服务/i,
        '还没有配置 AI 服务。请到「大模型配置」里添加一个服务并设为默认。'],
      [/类别少于 2 个|类别只有/i,
        '数据里能识别出的类别太少，训练至少需要 2 个类别。请检查数据是否完整。'],
      [/未找到可用的目标列|目标列/i,
        '没能认出哪一列是「答案」。请在上传时手动填写目标列名。'],
      [/未找到文本列|text_col|正文列/i,
        '没能认出哪一列是正文。请在上传时手动填写目标列名。'],
      [/路径.*不存在|不存在.*路径/i,
        '数据里的文件对不上（可能压缩包没解压完整）。请重新打包上传。'],
      [/写盘超过上限|12GB/i,
        '数据解压后太大了。请删掉不必要的大文件，或先分成几批上传。'],
      [/目录穿越|链接条目/i,
        '压缩包里有不安全的文件路径，已拒绝处理。请重新打包（不要包含软链接）。'],
      [/超时/i,
        '分析数据花了太久。请检查文件是否过大，或者网络是否通畅。'],
      [/解析脚本未产出有效|未产出 manifest/i,
        '系统没能读懂这份数据的结构。可以试着：把表格存成 CSV、或者把图片按类别分好文件夹再打包。'],
      [/图像文件不可用|无法读取|无法解码/i,
        '数据里的图片打不开（文件可能损坏或没解压完整）。请重新打包上传。']
    ];
    for (const [re, tip] of rules) {
      if (re.test(msg)) return tip;
    }
    // 没匹配到就保留原文，但去掉堆栈噪声
    const cleaned = msg.split('\n')[0].slice(0, 160);
    return cleaned || '分析数据时出了点问题，请重试。';
  }

  /* 解析过程的真实进度展示。
     之前只有一句「正在分析…」，大包解压要十几分钟，
     用户完全看不到平台到底在干什么，只能干等到失败。 */
  function parseProgressHTML(st, steps) {
    const pct = Math.max(0, Math.min(100, Math.round(Number(st.progress || 0) * 100)));
    const msg = st.message || '正在处理…';
    const stage = st.stage || '';

    // 阶段步骤条：让用户知道走到哪一步了
    // 顺序必须与后端实际执行顺序一致：
    // 解压文件 → 识别结构（平台直识）→ AI 解析（需要 AI 时）
    const stageOrder = ['extract', 'scan', 'ai_exec'];
    const curIdx = stageOrder.indexOf(stage);
    const dots = (steps || []).map((s, i) => {
      const idx = stageOrder.indexOf(s.key);
      let cls = '';
      if (curIdx > idx || (curIdx === idx && pct >= 100)) cls = 'done';
      else if (curIdx === idx) cls = 'active';
      return `<div class="pp-step ${cls}">
        <span class="dot">${cls === 'done' ? icon('check', 'icon-sm') : ''}</span>
        <span class="lb">${esc(s.label)}</span>
      </div>`;
    }).join('<span class="pp-sep"></span>');

    // 真实统计：文件数 / 已用时 / 预计剩余
    const stats = [];
    if (st.files_total) {
      stats.push(`文件 ${fmt(st.files_done || 0)} / ${fmt(st.files_total)}`);
    }
    if (st.elapsed) {
      const m = Math.floor(st.elapsed / 60);
      stats.push(m > 0 ? `已用 ${m} 分 ${st.elapsed % 60} 秒` : `已用 ${st.elapsed} 秒`);
    }
    if (st.eta_seconds) {
      const m = Math.ceil(st.eta_seconds / 60);
      stats.push(m > 1 ? `预计还需 ${m} 分钟` : `预计还需 ${st.eta_seconds} 秒`);
    }
    if (st.round && st.max_rounds) {
      stats.push(`第 ${st.round} / ${st.max_rounds} 轮尝试`);
    }

    return `<div class="notice info parse-progress">
      <div class="pp-head">${icon('sparkles', 'icon-sm spin')}
        <div><b>${esc(msg)}</b></div>
      </div>
      <div class="bar progress-lg mt-10"><span style="width:${pct}%"></span></div>
      <div class="flex items-center gap-12 mt-8" style="flex-wrap:wrap">
        <span class="small muted">${pct}%</span>
        ${stats.map((s) => `<span class="tiny muted">${esc(s)}</span>`).join('')}
      </div>
      ${dots ? `<div class="pp-steps mt-10">${dots}</div>` : ''}
      <div class="tiny muted mt-8">大文件解压可能要几分钟，进度会持续更新。同样的结构再来一次会快很多。</div>
    </div>`;
  }

  function openUploadModal(presetTask) {
    void presetTask; // 兼容旧调用签名（wizard / overview）
    let file = null;
    MF.modal({
      title: '上传你的数据',
      desc: '把数据传上来就行，不用整理格式。系统会自动看懂数据长什么样、哪些是训练用的、哪些是最后检验用的。',
      wide: true,
      body: `<div class="form-grid">
        <div class="field"><label>给这份数据起个名字<span class="req">*</span></label><input class="input" id="up-name" placeholder="例如：客户流失样本"></div>
        <div class="field"><label>你想让模型做什么</label><select class="select" id="up-task">
          <option value="">让系统自动判断</option>
          <option value="text_classification">判断一段文字属于哪一类</option>
        </select></div>
        <div class="field full" id="up-target-field"><label>哪一列是答案</label><input class="input" id="up-target" placeholder="不确定就留空，系统会帮你找"></div>
        <div class="field full"><label>选择数据文件<span class="req">*</span></label>
          <div class="upload-box" id="up-box"><div class="up-ico">${icon('upload', 'icon-lg')}</div>
            <h4 id="up-file">点击选择文件，或将文件拖拽到这里</h4><p id="up-hint">支持 CSV / Excel / JSON / Parquet，以及 ZIP / tar.gz 压缩包，单文件不超过 3 GB</p>
            <input type="file" id="up-input" style="display:none" accept=".csv,.tsv,.txt,.xlsx,.xls,.json,.jsonl,.parquet,.zip,.tar,.gz"></div></div>
      </div>
      <div class="notice mt-16">${icon('sparkles', 'icon-sm')}
        <div><b>你不需要懂这些术语</b><br>
        <span class="small">「训练集 / 验证集 / 测试集」是机器学习内部的叫法，你只要知道：
        <b>训练集</b>是模型学习用的，<b>验证集</b>是挑选最好一版的，<b>测试集</b>是最后用来检验的（它不会参与训练）。
        系统会自动帮你把它们分开。<br>
        首次分析一种新结构大约需要 30-120 秒；同样的结构再来一次，会直接复用结果，几秒就好。</span></div>
      </div>
      <div id="up-status" class="mt-12"></div>`,
      foot: `<button class="btn" data-close>取消</button><button class="btn btn-primary" id="up-submit">${icon('sparkles', 'icon-sm')}上传并自动分析</button>`,
      onMount(el, close) {
        const box = MF.$('#up-box', el), input = MF.$('#up-input', el), nameInput = MF.$('#up-name', el);
        const targetField = MF.$('#up-target-field', el), taskSel = MF.$('#up-task', el);
        const isArchive = () => file && /\.(zip|tar|gz|tgz)$/i.test(file.name);
        const syncForm = () => {
          // 压缩包可能是图像等文件型数据集：目标列由 AI 判定，隐藏手工输入
          targetField.style.display = isArchive() ? 'none' : '';
          taskSel.disabled = isArchive();
          if (isArchive()) taskSel.value = '';
          MF.$('#up-hint', el).textContent = isArchive()
            ? '压缩包内可含图像、表格或任意结构，AI 会自动判断并识别其中的训练/验证/测试划分'
            : '支持 CSV / TSV / TXT、Excel、JSON、Parquet，单文件不超过 3 GB · 文件型数据集请打包为 ZIP / tar.gz 上传';
        };
        const setFile = (f) => {
          if (!f) return;
          if (!/\.(csv|tsv|txt|xlsx|xls|json|jsonl|parquet|zip|tar|gz|tgz)$/i.test(f.name)) {
            MF.toast('不支持的文件格式', '支持 CSV、TSV、TXT、Excel、JSON、Parquet，或 ZIP / tar.gz 压缩包。', 'warn'); return;
          }
          file = f;
          MF.$('#up-file', el).textContent = `${f.name} · ${sizeStr(f.size)}`;
          if (!nameInput.value.trim()) nameInput.value = f.name.replace(/\.(csv|tsv|txt|xlsx|xls|json|jsonl|parquet|zip|tar|gz|tgz)$/i, '');
          syncForm();
        };
        box.onclick = () => input.click();
        ['dragenter', 'dragover'].forEach((ev) => box.addEventListener(ev, (e) => { e.preventDefault(); box.classList.add('drag'); }));
        ['dragleave', 'drop'].forEach((ev) => box.addEventListener(ev, (e) => { e.preventDefault(); box.classList.remove('drag'); }));
        box.addEventListener('drop', (e) => setFile(e.dataTransfer.files[0]));
        input.onchange = () => setFile(input.files[0]);

        MF.$('#up-submit', el).onclick = async () => {
          const name = nameInput.value.trim();
          if (!name) { nameInput.classList.add('error'); MF.toast('请填写数据集名称', '', 'warn'); return; }
          if (!file) { MF.toast('请选择数据文件', '支持 CSV / TSV / TXT / Excel / JSON / Parquet，或 ZIP / tar.gz 压缩包。', 'warn'); return; }
          const btn = MF.$('#up-submit', el);
          const statusBox = MF.$('#up-status', el);
          btn.disabled = true;
          btn.innerHTML = `${icon('sparkles', 'icon-sm spin')}上传中…`;
          const setStatus = (html) => { if (statusBox) statusBox.innerHTML = html; };
          setStatus(`<div class="notice info">${icon('sparkles', 'icon-sm spin')}
            <div><b>文件上传中…</b><br><span class="small">${esc(file.name)} · ${sizeStr(file.size)}</span></div></div>`);
          try {
            const up = await MF.api.uploadDataset(
              file, name,
              isArchive() ? '' : MF.$('#up-target', el).value.trim(),
              isArchive() ? '' : taskSel.value
            );
            // 后端立即返回任务 ID，解析在后台进行 —— 轮询进度
            if (up && up.parsing) {
              const taskId = up.task_id;
              let done = null;
              const STEPS = [
                { key: 'extract', label: '解压文件' },
                { key: 'scan', label: '识别结构' },
                { key: 'ai_exec', label: 'AI 解析' }
              ];              for (let i = 0; i < 720; i++) {   // 最多等 12 分钟
                await new Promise((r) => setTimeout(r, 1200));
                let st;
                try {
                  st = await MF.api.parseStatus(taskId);
                } catch (e) {
                  if (i === 0) throw e;
                  continue;
                }
                if (st.status === 'done') { done = st; break; }
                if (st.status === 'failed') {
                  throw new Error(friendlyError(st.message || 'AI 解析失败'));
                }
                setStatus(parseProgressHTML(st, STEPS));
              }
              if (!done) throw new Error(friendlyError('解析超时'));
              const ds = await MF.api.parseResult(taskId);
              close();
              if (MF.openParseReport) MF.openParseReport(ds);
              MF.toast('AI 解析完成', `${ds.name} · ${fmt(ds.rows)} 个样本`, 'success');
              location.hash = `#/datasets/${ds.id}?tab=overview`;
              MF.route();
              return;
            }
            // 兼容直接返回数据集的旧行为
            close();
            if (MF.openParseReport) MF.openParseReport(up);
            MF.toast('AI 解析完成', `${up.name} · ${fmt(up.rows)} 个样本`, 'success');
            location.hash = `#/datasets/${up.id}?tab=overview`;
            MF.route();
          } catch (e) {
            setStatus(`<div class="notice danger">${icon('alert', 'icon-sm')}
              <div><b>解析失败</b><br><span class="small">${esc(e.message)}</span></div></div>`);
            btn.disabled = false;
            btn.innerHTML = `${icon('sparkles', 'icon-sm')}上传并由 AI 解析`;
          }
        };
      }
    });
  }
  MF.openUploadModal = openUploadModal;

  /* ---------------- 数据集详情 ---------------- */
  MF.views.dataset = function (r, view) {
    view.innerHTML = loadingHTML('正在加载数据集详情…');
    MF.api.getDataset(r.id)
      .then((d) => renderDetail(d))
      .catch((e) => { view.innerHTML = `<a href="#/datasets" class="btn btn-sm btn-ghost" style="margin-bottom:12px">${icon('arrow-left', 'icon-sm')}数据资产</a>` + errHTML(e); });

    function renderDetail(d) {
      const tab = r.query.get('tab') || 'overview';
      const tabs = [['overview', '数据概览'], ['cleaning', '数据清洗'], ['eda', '探索分析 EDA'], ['schema', '字段信息']];
      const ok = cleanState(d);
      view.innerHTML = `
        <div class="flex items-center gap-8 small" style="margin-bottom:12px"><a href="#/datasets" class="btn btn-sm btn-ghost">${icon('arrow-left', 'icon-sm')}数据资产</a><span class="muted">/</span><span class="muted">${esc((d.meta || {}).filename || d.source)}</span></div>
        <div class="page-head">
          <div class="flex items-center gap-12">${MF.charts.datasetIcon(d.task_hint)}
            <div><h1 style="display:flex;align-items:center;gap:9px">${esc(d.name)}
              <span class="tag">v${d.version}</span>
              ${ok ? '<span class="pill pill-success"><span class="pdot"></span>数据良好</span>' : '<span class="pill pill-warn"><span class="pdot"></span>待清洗</span>'}</h1>
              <p>${esc(MF.TASK_LABELS[d.task_hint] || d.task_hint)} · 目标列「${esc(d.target_col || '—')}」 · 更新于 ${esc(d.updated_at)}</p></div></div>
          <div class="actions">
            <button class="btn" data-del style="color:#dc2626">${icon('trash')}删除</button>
            <a class="btn btn-primary" href="#/build?dataset=${esc(d.id)}">${icon('sparkles')}开始建模</a>
          </div>
        </div>
        <div class="tabs" style="margin-bottom:20px">${tabs.map(([k, l]) => `<a class="tab ${tab === k ? 'active' : ''}" href="#/datasets/${esc(d.id)}?tab=${k}">${l}</a>`).join('')}</div>
        <div data-tab-body>${loadingHTML('正在加载…')}</div>`;

      MF.$('[data-del]', view).onclick = async () => {
        const yes = await MF.confirm({ title: '删除数据集', message: `确认删除「${d.name}」？后端文件与全部数据版本将一并移除，且不可恢复。`, okText: '删除', danger: true });
        if (!yes) return;
        try {
          await MF.api.deleteDataset(d.id);
          MF.toast('数据集已删除', d.name, 'success');
          location.hash = '#/datasets';
        } catch (e) { MF.toast('删除失败', e.message, 'error'); }
      };

      const body = MF.$('[data-tab-body]', view);
      if (tab === 'overview') renderOverview(d, body);
      else if (tab === 'cleaning') renderCleaning(d, body);
      else if (tab === 'eda') renderEda(d, body);
      else renderSchema(d, body);
    }
  };

  /* ---------------- 概览 ---------------- */
  function renderOverview(d, body) {
    const s = (d.meta || {}).stats || {};
    const schema = (d.meta || {}).schema || [];
    const quality = qualityOf(d);
    const isImg = d.task_hint === 'image_classification';
    const classes = (d.meta || {}).classes || [];
    const stats = [
      { label: '样本量', value: fmt(d.rows) + (isImg ? ' 张' : ' 行'), icon: 'layers', note: isImg ? `${classes.length} 个类别` : `${d.cols} 个字段` },
      { label: '缺失单元格', value: fmt(s.missing_cells || 0), icon: 'alert', note: `缺失率 ${MF.pct(s.missing_pct || 0)}` },
      { label: '重复行', value: fmt(s.duplicate_rows || 0), icon: 'copy', note: (s.duplicate_rows || 0) === 0 ? '无重复' : '建议去重' },
      { label: '内存占用', value: sizeStr(s.memory_bytes), icon: 'table', note: `磁盘 ${sizeStr(d.size_bytes)}` }
    ];
    body.innerHTML = `<div class="grid grid-2-1">
      <div class="card card-pad">
        <div class="flex items-center gap-12" style="align-items:flex-start"><div><div class="section-title">数据质量总览</div><div class="section-sub">根据缺失、重复与字段完整性计算</div></div>
          <div class="ml-auto">${MF.charts.ring(quality, { size: 88, label: '质量分' })}</div></div>
        <div class="grid grid-2 mt-16" style="gap:10px">${stats.map((x) => `<div class="mini-stat"><div class="l">${icon(x.icon, 'icon-sm')} ${x.label}</div><div class="v">${x.value}</div><div class="tiny muted">${x.note}</div></div>`).join('')}</div>
        <div class="mt-16"><div class="wiz-section-title">字段构成</div><div class="flex gap-8" style="flex-wrap:wrap">
          ${schema.slice(0, 12).map((c) => `<span class="tag ${c.role === 'target' ? 'tag-blue' : ''}">${esc(c.name)}<span class="muted">· ${esc(TYPE_LABELS[c.type] || c.type)}</span></span>`).join('')}</div></div>
      </div>
      <div class="card card-pad">
        <div class="section-title">数据集信息</div><div class="section-sub" style="margin-bottom:14px">来源与版本信息</div>
        <div class="kv-list">
          <div class="kv-row"><span class="k">数据文件</span><span class="v mono">${esc((d.meta || {}).filename || '—')}</span></div>
          <div class="kv-row"><span class="k">模型要做的事</span><span class="v">${esc(MF.TASK_LABELS[d.task_hint] || d.task_hint)}</span></div>
          <div class="kv-row"><span class="k">答案列</span><span class="v">${esc(d.target_col || '—')}</span></div>
          <div class="kv-row"><span class="k">数据版本</span><span class="v">v${d.version}</span></div>
          <div class="kv-row"><span class="k">创建时间</span><span class="v">${esc(d.created_at)}</span></div>
          <div class="kv-row"><span class="k">更新时间</span><span class="v">${esc(d.updated_at)}</span></div>
          <div class="kv-row"><span class="k">存储大小</span><span class="v">${sizeStr(d.size_bytes)}</span></div>
        </div>
        <div class="notice mt-16">${icon('shield', 'icon-sm')}<div>原始数据会原样保存，不会被改动。清洗会生成新的版本，随时可以回到最初那份。</div></div>
      </div>
    </div>
    ${isImg ? '' : `<div class="card card-pad mt-20" data-ai-inspect>
      <div class="flex items-center gap-10" style="align-items:flex-start">
        <div style="width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#eef4ff,#e0e9ff);display:grid;place-items:center;color:var(--primary-700)">${icon('sparkles', 'icon-sm')}</div>
        <div style="flex:1"><div class="section-title">AI 数据洞察</div>
          <div class="section-sub">让大模型逐列判断哪些字段对建模有用、哪些建议丢弃，你确认后一键生成清洗版本</div></div>
        <button class="btn btn-primary btn-sm" data-ai-run>${icon('sparkles', 'icon-sm')}开始分析</button>
      </div>
      <div data-ai-body class="mt-16"></div>
    </div>`}
    <div class="mt-20">${MF.sectionCard(isImg ? '样本预览' : '数据预览', isImg ? '展示前 12 张图像' : '展示前 8 行', `<a class="btn btn-sm btn-ghost" href="#/datasets/${esc(d.id)}?tab=schema">全部字段 ${icon('arrow-right', 'icon-sm')}</a>`, `<div data-preview>${loadingHTML('正在加载预览…')}</div>`)}</div>`;

    MF.api.preview(d.id, isImg ? 12 : 8).then((p) => {
      const box = MF.$('[data-preview]', body);
      if (!box) return;
      if (isImg) {
        const fi = p.columns.indexOf('filepath'), li = p.columns.indexOf('label');
        box.innerHTML = `<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px">
          ${p.rows.map((row) => `<div style="border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#fff">
            <img src="${MF.api.imageUrl(d.id, row[fi])}" alt="${esc(row[li])}" loading="lazy" style="width:100%;aspect-ratio:1;object-fit:cover;display:block;background:#f1f5f9">
            <div class="tiny" style="padding:6px 8px;font-weight:600">${esc(row[li])}</div>
          </div>`).join('')}</div>`;
        return;
      }
      const target = d.target_col;
      box.innerHTML = `<div class="table-wrap" style="border:1px solid var(--line);border-radius:9px"><table class="tbl preview-table">
        <thead><tr>${p.columns.map((c) => `<th>${esc(c)}${c === target ? ' <span class="tag tag-blue">目标</span>' : ''}</th>`).join('')}</tr></thead>
        <tbody>${p.rows.map((row) => `<tr>${row.map((v) => `<td>${v === null || v === undefined || v === '' ? '<span class="null">—</span>' : esc(v)}</td>`).join('')}</tr>`).join('')}</tbody>
      </table></div>`;
    }).catch((e) => {
      const box = MF.$('[data-preview]', body);
      if (box) box.innerHTML = errHTML(e);
    });

    // ---- AI 数据洞察 ----
    const aiBox = MF.$('[data-ai-inspect]', body);
    if (aiBox) {
      const aiBody = MF.$('[data-ai-body]', aiBox);
      const runBtn = MF.$('[data-ai-run]', aiBox);
      runBtn.onclick = async () => {
        runBtn.disabled = true;
        runBtn.innerHTML = `${icon('refresh', 'icon-sm spin')}大模型分析中…`;
        aiBody.innerHTML = loadingHTML('正在将字段画像发送给大模型，请稍候…');
        try {
          const r = await MF.api.aiInspect(d.id);
          renderAiInspect(aiBody, d, r);
        } catch (e) {
          aiBody.innerHTML = errHTML(e);
        } finally {
          runBtn.disabled = false;
          runBtn.innerHTML = `${icon('sparkles', 'icon-sm')}重新分析`;
        }
      };
    }
  }

  function renderAiInspect(box, d, r) {
    const cols = r.columns || [];
    const target = d.target_col;
    box.innerHTML = `
      ${r.summary ? `<div class="notice" style="border-left:3px solid var(--primary);background:#f8faff">${icon('sparkles', 'icon-sm')}<div><b>整体分析</b><br>${esc(r.summary)}</div></div>` : ''}
      ${r.target_suggestion ? `<div class="tiny muted mt-10" style="padding:0 4px">目标列建议：${esc(r.target_suggestion)}</div>` : ''}
      <div class="table-wrap mt-14" style="border:1px solid var(--line);border-radius:9px">
        <table class="tbl">
          <thead><tr><th style="width:42px"><input type="checkbox" data-ai-all></th><th>字段</th><th style="width:90px">AI 判定</th><th>理由 / 建议</th></tr></thead>
          <tbody>${cols.map((c, i) => {
            const isTarget = c.name === target;
            const drop = !c.useful && !isTarget;
            return `<tr>
              <td><input type="checkbox" data-ai-drop="${esc(c.name)}" ${drop ? 'checked' : ''} ${isTarget ? 'disabled title="目标列不可丢弃"' : ''}></td>
              <td><div class="cell-main">${esc(c.name)}</div>${isTarget ? '<span class="tag tag-blue" style="margin-top:3px">目标</span>' : ''}</td>
              <td>${c.useful
                ? '<span class="pill pill-success"><span class="pdot"></span>保留</span>'
                : '<span class="pill pill-warn"><span class="pdot"></span>建议丢弃</span>'}</td>
              <td class="small">${esc(c.reason || '')}${c.suggestion && c.suggestion !== c.reason ? `<div class="tiny muted mt-2">建议：${esc(c.suggestion)}</div>` : ''}</td>
            </tr>`;
          }).join('')}</tbody>
        </table>
      </div>
      <div class="flex items-center gap-10 mt-14">
        <span class="tiny muted">已勾选的字段将被丢弃，生成新数据版本 v${d.version + 1}（原始数据保留）</span>
        <button class="btn btn-primary ml-auto" data-ai-apply>${icon('check', 'icon-sm')}应用所选丢弃</button>
      </div>`;

    // 全选/取消（只影响可勾选的行）
    const allCb = MF.$('[data-ai-all]', box);
    if (allCb) allCb.onchange = () => {
      MF.$$('[data-ai-drop]', box).forEach((cb) => { if (!cb.disabled) cb.checked = allCb.checked; });
    };
    MF.$('[data-ai-apply]', box).onclick = async () => {
      const drops = MF.$$('[data-ai-drop]', box).filter((cb) => cb.checked).map((cb) => cb.dataset.aiDrop);
      if (!drops.length) { MF.toast('未选择任何字段', '请至少勾选一个要丢弃的字段', 'warn'); return; }
      const ok = await MF.confirm({
        title: '应用 AI 建议',
        message: `将丢弃以下 ${drops.length} 个字段并生成新数据版本：\n${drops.join('、')}`,
        okText: '确认丢弃'
      });
      if (!ok) return;
      const btn = MF.$('[data-ai-apply]', box);
      btn.disabled = true; btn.innerHTML = `${icon('refresh', 'icon-sm spin')}正在清洗…`;
      try {
        const res = await MF.api.clean(d.id, { drop_columns: drops });
        MF.toast('已生成新版本', `${drops.length} 个字段已丢弃，数据版本 v${res.dataset.version}`, 'success');
        MF.route();
      } catch (e) {
        MF.toast('清洗失败', e.message, 'error');
        btn.disabled = false; btn.innerHTML = `${icon('check', 'icon-sm')}应用所选丢弃`;
      }
    };
  }

  /* ---------------- 清洗 ---------------- */
  const MISSING_OPTS = {
    number: [['mean', '均值填充'], ['median', '中位数填充'], ['zero', '填 0'], ['drop_row', '删除该行'], ['constant', '固定值']],
    category: [['mode', '众数填充'], ['drop_row', '删除该行'], ['constant', '固定值']],
    text: [['mode', '众数填充'], ['drop_row', '删除该行'], ['constant', '固定值']],
    bool: [['mode', '众数填充'], ['drop_row', '删除该行']],
    datetime: [['mode', '众数填充'], ['drop_row', '删除该行']]
  };
  const STRATEGY_LABELS = { mean: '均值填充', median: '中位数填充', zero: '填 0', mode: '众数填充', drop_row: '删除该行' };
  const stepText = (s) => {
    if (s.op === 'missing') {
      const st = s.strategy && s.strategy.indexOf('constant:') === 0 ? `固定值「${s.strategy.slice(9)}」` : (STRATEGY_LABELS[s.strategy] || s.strategy);
      return `缺失值：列「${s.col}」按${st}处理（${fmt(s.filled_or_dropped)} 个单元格）`;
    }
    if (s.op === 'outliers') return `异常值：列「${s.col}」按 ${s.method === 'iqr' ? 'IQR（1.5 倍）' : 'Z-Score（±3σ）'} 删除 ${fmt(s.dropped)} 行`;
    if (s.op === 'drop_duplicates') return `重复值：删除完全重复记录 ${fmt(s.dropped)} 行`;
    return esc(JSON.stringify(s));
  };

  function renderCleaning(d, body) {
    if (d.task_hint === 'image_classification') {
      const cc = (d.meta || {}).class_counts || {};
      const sc = (d.meta || {}).split_counts || null;
      const total = Object.values(cc).reduce((a, b) => a + b, 0) || 1;
      const SPLIT_LABELS = { train: '训练集', val: '验证集', test: '测试集（不参与训练）' };
      body.innerHTML = `<div class="card card-pad">
        <div class="section-title">数据清洗</div><div class="section-sub" style="margin-bottom:14px">图像数据集不适用表格清洗规则</div>
        <div class="notice">${icon('info', 'icon-sm')}<div>图像数据以像素为输入，无需缺失值 / 异常值处理。若类别不均衡，可在训练时启用类别权重（class weights）缓解。</div></div>
        <div class="mt-16"><div class="wiz-section-title">类别分布</div><div class="kv-list">
          ${Object.entries(cc).map(([k, v]) => `<div class="kv-row"><span class="k">${esc(k)}</span><span class="v" style="font-weight:600">${fmt(v)} 张 · ${(v / total * 100).toFixed(1)}%</span></div>`).join('')}
        </div></div>
        ${sc ? `<div class="mt-16"><div class="wiz-section-title">数据集划分（来自压缩包目录）</div><div class="kv-list">
          ${Object.entries(sc).map(([k, v]) => `<div class="kv-row"><span class="k">${esc(SPLIT_LABELS[k] || k)}</span><span class="v" style="font-weight:600">${fmt(v)} 张 · ${(v / total * 100).toFixed(1)}%</span></div>`).join('')}
        </div><div class="tiny muted" style="margin-top:8px">训练时将优先使用该自带划分：train 用于训练、val 用于验证。</div></div>` : ''}
      </div>`;
      return;
    }
    body.innerHTML = loadingHTML('正在加载字段结构…');
    Promise.all([MF.api.eda(d.id), MF.api.cleaningRuns(d.id).catch(() => [])])
      .then(([eda, runs]) => {
        const schema = eda.schema || [];
        const s = eda.stats || {};
        const runList = Array.isArray(runs) ? runs : [runs];
        const latest = runList[0];
        body.innerHTML = `<div class="grid grid-2-1">
          <div class="card">
            <div class="card-head"><div><div class="section-title">清洗方案</div><div class="section-sub">按列配置处理策略，原始数据不会被修改</div></div><span class="tag">v${d.version}</span></div>
            <div class="card-body">
              <div class="wiz-section-title" style="margin-bottom:10px">缺失值与异常值策略</div>
              <div class="table-wrap" style="border:1px solid var(--line);border-radius:9px"><table class="tbl">
                <thead><tr><th style="width:26%">字段</th><th>缺失值处理</th><th>异常值处理（数值列）</th></tr></thead>
                <tbody>${schema.map((c, i) => `<tr>
                  <td><div class="cell-main" style="font-size:12.5px">${esc(c.name)}</div><div class="cell-sub">${esc(TYPE_LABELS[c.type] || c.type)}${c.role === 'target' ? ' · 目标' : ''}${c.nulls ? ` · 缺失 ${fmt(c.nulls)}` : ''}</div></td>
                  <td><select class="select" style="height:32px;font-size:12px" data-mi="${i}">
                    <option value="">不处理</option>
                    ${(MISSING_OPTS[c.type] || MISSING_OPTS.text).map(([v, l]) => `<option value="${v}">${l}</option>`).join('')}
                  </select>
                  <input class="input mt-4" style="height:30px;font-size:12px;display:none" placeholder="固定值内容" data-mc="${i}"></td>
                  <td>${c.type === 'number' && c.role !== 'target'
                    ? `<select class="select" style="height:32px;font-size:12px" data-oi="${i}">
                        <option value="">不处理</option><option value="iqr">IQR 截断（1.5 倍）</option><option value="zscore">Z-Score 截断（±3σ）</option></select>`
                    : '<span class="tiny muted">—</span>'}</td>
                </tr>`).join('')}</tbody>
              </table></div>
              <label class="check-row mt-16"><input type="checkbox" id="cl-dup" checked><span class="ct">删除完全重复记录<small>当前重复行 ${fmt(s.duplicate_rows || 0)} 行</small></span></label>
              <div class="notice mt-16">${icon('info', 'icon-sm')}<div>执行后将生成新的数据版本（v${d.version + 1}），并记录完整清洗配方；建模与训练将引用该版本快照。</div></div>
              <div class="flex gap-10 mt-16"><button class="btn btn-primary" id="cl-run">${icon('play', 'icon-sm')}开始清洗</button><span class="tiny muted" style="align-self:center">提交至后端执行，通常数秒内完成</span></div>
            </div>
          </div>
          <div class="card card-pad">
            <div class="flex items-center gap-10"><div><div class="section-title">清洗结果</div><div class="section-sub">${latest ? '最近一次执行结果' : '尚未执行清洗'}</div></div>
              <div class="ml-auto">${MF.charts.ring(qualityOf(d), { size: 72, label: '质量分' })}</div></div>
            <div class="mt-16" id="cl-result">${latest ? runSummary(latest) : `<div class="empty" style="padding:28px 10px"><div class="ico">${icon('sparkles')}</div><h4>等待执行</h4><p>选择左侧方案并执行清洗，这里会显示前后对比与处理步骤。</p></div>`}</div>
            ${runList.length > 1 ? `<div class="mt-16"><div class="wiz-section-title">历史清洗记录</div><div class="kv-list">${runList.slice(1, 6).map((run) => `<div class="kv-row"><span class="k">v${run.version} · ${esc(run.created_at)}</span><span class="v" style="font-weight:500">${run.report ? `${fmt(run.report.before_rows)} → ${fmt(run.report.after_rows)} 行` : '—'}</span></div>`).join('')}</div></div>` : ''}
          </div>
        </div>`;

        // 固定值输入框跟随策略下拉显隐
        schema.forEach((c, i) => {
          const sel = MF.$(`[data-mi="${i}"]`, body), inp = MF.$(`[data-mc="${i}"]`, body);
          if (sel && inp) sel.onchange = () => { inp.style.display = sel.value === 'constant' ? '' : 'none'; };
        });

        MF.$('#cl-run', body).onclick = async () => {
          const recipe = {};
          const missing = {}, outliers = {};
          schema.forEach((c, i) => {
            const sel = MF.$(`[data-mi="${i}"]`, body);
            if (sel && sel.value) {
              missing[c.name] = sel.value === 'constant'
                ? `constant:${(MF.$(`[data-mc="${i}"]`, body).value || '').trim()}`
                : sel.value;
            }
            const os = MF.$(`[data-oi="${i}"]`, body);
            if (os && os.value) outliers[c.name] = os.value;
          });
          if (MF.$('#cl-dup', body).checked) recipe.drop_duplicates = true;
          if (Object.keys(missing).length) recipe.missing = missing;
          if (Object.keys(outliers).length) recipe.outliers = outliers;
          if (!Object.keys(recipe).length) { MF.toast('请至少选择一项清洗操作', '当前方案不会改变数据。', 'warn'); return; }

          const btn = MF.$('#cl-run', body);
          btn.disabled = true; btn.innerHTML = `${icon('refresh', 'icon-sm spin')}正在清洗…`;
          try {
            const res = await MF.api.clean(d.id, recipe);
            const rep = res.run.report;
            MF.toast('清洗完成', `${fmt(rep.before_rows)} → ${fmt(rep.after_rows)} 行，数据版本已更新为 v${res.dataset.version}`, 'success');
            MF.route(); // 重新拉取详情，版本 +1
          } catch (e) {
            MF.toast('清洗失败', e.message, 'error');
            btn.disabled = false; btn.innerHTML = `${icon('play', 'icon-sm')}开始清洗`;
          }
        };
      })
      .catch((e) => { body.innerHTML = errHTML(e); });
  }

  function runSummary(run) {
    const rep = run.report || {};
    return `<div class="notice success">${icon('check-circle', 'icon-sm')}<div><b>清洗完成</b> · ${esc(run.created_at)}<br>共执行 ${(rep.steps || []).length} 项处理，生成数据版本 v${run.version}。</div></div>
      <div class="grid grid-2 mt-12" style="gap:10px">
        <div class="mini-stat"><div class="l">处理前行数</div><div class="v">${fmt(rep.before_rows)}</div></div>
        <div class="mini-stat"><div class="l">处理后行数</div><div class="v">${fmt(rep.after_rows)}</div></div>
      </div>
      <div class="mt-16"><div class="wiz-section-title">已执行的处理步骤</div><div class="kv-list">
        ${(rep.steps || []).map((s, i) => `<div class="kv-row"><span class="k">${String(i + 1).padStart(2, '0')}</span><span class="v" style="font-weight:500;text-align:left;flex:1">${esc(stepText(s))}</span>${icon('check', 'icon-sm')}</div>`).join('')}</div></div>`;
  }

  /* ---------------- EDA ---------------- */
  function renderEda(d, body) {
    body.innerHTML = loadingHTML('正在生成探索性分析…');
    MF.api.eda(d.id)
      .then((eda) => {
        const colors = ['#2563eb', '#60a5fa', '#0f9d76', '#f59e0b', '#7c3aed', '#0891b2'];
        const numericCols = Object.keys(eda.numeric || {});
        const categoryCols = Object.keys(eda.category || {});

        // 目标分布
        let targetCard = '';
        if (eda.target && eda.target.kind === 'classification') {
          const total = eda.target.counts.reduce((a, b) => a + b, 0) || 1;
          const items = eda.target.labels.map((l, i) => ({ label: l, value: eda.target.counts[i], color: colors[i % colors.length] }));
          targetCard = MF.sectionCard('目标分布', `目标列「${esc(d.target_col)}」类别占比`, '',
            `<div class="flex items-center gap-16">${MF.charts.donut(items, 148, { value: fmt(total), label: '样本' })}
             <div style="flex:1">${items.map((c) => `<div class="flex items-center gap-8" style="margin-bottom:9px"><i style="width:8px;height:8px;border-radius:2px;background:${c.color}"></i><span class="small muted" style="flex:1">${esc(c.label)}</span><span class="small" style="font-weight:600">${fmt(c.value)}</span><span class="tiny muted">${(c.value / total * 100).toFixed(1)}%</span></div>`).join('')}</div></div>`);
        } else if (eda.target && eda.target.kind === 'regression') {
          const t = eda.target;
          targetCard = MF.sectionCard('目标分布', `目标列「${esc(d.target_col)}」数值分布 · 均值 ${numStr(t.mean)} / 中位 ${numStr(t.median)}`, '',
            MF.charts.bars({ labels: (t.bins || []).map(binLabel), values: t.counts || [], color: '#3b82f6' }));
        } else {
          targetCard = MF.sectionCard('目标分布', '未识别到目标列', '', '<div class="chart-empty">该数据集无目标分布</div>');
        }

        // 数值列直方图
        const numericCards = numericCols.slice(0, 4).map((col) => {
          const h = eda.numeric[col];
          return MF.sectionCard(`数值分布 · ${esc(col)}`, `均值 ${numStr(h.mean)} · 标准差 ${numStr(h.std)} · 范围 ${numStr(h.min)} ~ ${numStr(h.max)}`, '',
            MF.charts.bars({ labels: (h.bins || []).map(binLabel), values: h.counts || [], color: '#3b82f6' }));
        }).join('');

        // 类别列 Top 频次
        const categoryCards = categoryCols.slice(0, 4).map((col) => {
          const c = eda.category[col];
          const items = c.labels.map((l, i) => ({ label: l, value: c.counts[i], color: colors[i % colors.length] }));
          return MF.sectionCard(`类别分布 · ${esc(col)}`, `共 ${c.labels.length} 个类别`, '', MF.charts.hbars(items));
        }).join('');

        // 相关性热力图
        const corrCard = eda.correlation && eda.correlation.labels && eda.correlation.labels.length
          ? MF.sectionCard('相关性矩阵', '数值特征 Pearson 相关系数', '', MF.charts.heatmap(eda.correlation.labels, eda.correlation.matrix))
          : '';

        // 图像数据集：尺寸分布 + 样本网格
        const isImg = d.task_hint === 'image_classification';
        let imgCards = '', sampleCard = '';
        if (isImg && eda.images) {
          const sz = eda.images.sizes || {};
          imgCards = ['width', 'height'].map((k) => {
            const h = sz[k];
            if (!h || !h.counts || !h.counts.length) return '';
            return MF.sectionCard(`图像${k === 'width' ? '宽度' : '高度'}分布`, `采样 ${fmt((h.counts || []).reduce((a, b) => a + b, 0))} 张 · 均值 ${numStr(h.mean)} px · 范围 ${numStr(h.min)} ~ ${numStr(h.max)}`, '',
              MF.charts.bars({ labels: (h.bins || []).map(binLabel), values: h.counts || [], color: '#0f9d76' }));
          }).join('');
          sampleCard = MF.sectionCard('样本抽检', `随机 ${eda.images.samples.length} 张（含类别标签）`, '',
            `<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px">
              ${eda.images.samples.map((s) => `<div style="border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#fff">
                <img src="${MF.api.imageUrl(d.id, s.path)}" alt="${esc(s.label)}" loading="lazy" style="width:100%;aspect-ratio:1;object-fit:cover;display:block;background:#f1f5f9">
                <div class="tiny" style="padding:6px 8px;font-weight:600">${esc(s.label)}</div>
              </div>`).join('')}</div>`);
        }

        // 文本数据集：文本长度分布
        const isText = d.task_hint === 'text_classification';
        let textCards = '';
        if (isText && eda.text) {
          textCards = Object.keys(eda.text).slice(0, 2).map((col) => {
            const h = eda.text[col];
            return MF.sectionCard(`文本长度分布 · ${esc(col)}`, `均值 ${numStr(h.mean)} 字符 · 中位 ${numStr(h.median)} · 最长 ${numStr(h.max)}`, '',
              MF.charts.bars({ labels: (h.bins || []).map(binLabel), values: h.counts || [], color: '#7c3aed' }));
          }).join('');
        }

        body.innerHTML = `
          <div class="flex items-center gap-10" style="margin-bottom:14px">
            <div class="section-title" style="margin:0">探索性分析</div>
            <span class="tiny muted">基于当前数据版本生成</span>
            <span class="tag ml-auto">v${d.version}</span>
          </div>
          <div class="grid grid-2">${targetCard}${corrCard}</div>
          ${imgCards ? `<div class="grid grid-2 mt-18">${imgCards}</div>` : ''}
          ${sampleCard ? `<div class="mt-18">${sampleCard}</div>` : ''}
          ${textCards ? `<div class="grid grid-2 mt-18">${textCards}</div>` : ''}
          ${numericCards ? `<div class="grid grid-2 mt-18">${numericCards}</div>` : ''}
          ${categoryCards ? `<div class="grid grid-2 mt-18">${categoryCards}</div>` : ''}`;
      })
      .catch((e) => { body.innerHTML = errHTML(e); });
  }

  /* ---------------- 字段信息 ---------------- */
  function renderSchema(d, body) {
    const schema = (d.meta || {}).schema || [];
    const rowsHtml = schema.map((c) => `<tr data-col="${esc(c.name.toLowerCase())}">
      <td><div class="cell-main">${esc(c.name)}</div><div class="cell-sub">${c.role === 'target' ? '目标变量' : '特征列'}</div></td>
      <td><span class="tag">${esc(TYPE_LABELS[c.type] || c.type)}</span></td>
      <td>${c.role === 'target' ? '<span class="pill pill-info">目标</span>' : '<span class="pill pill-muted">特征</span>'}</td>
      <td class="num">${c.nulls ? fmt(c.nulls) : '0'}</td>
      <td class="num">${fmt(c.unique)}</td>
      <td class="num">${c.min !== null && c.min !== undefined ? numStr(c.min) + ' ~ ' + numStr(c.max) : '—'}</td>
      <td class="num">${c.mean !== null && c.mean !== undefined ? numStr(c.mean) : '—'}</td>
      <td class="num">${c.std !== null && c.std !== undefined ? numStr(c.std) : '—'}</td>
    </tr>`).join('');
    body.innerHTML = `<div class="card">
      <div class="card-head" style="padding-bottom:14px"><div><div class="section-title">字段信息</div><div class="section-sub">共 ${schema.length} 个字段，目标列已自动标记</div></div>
        <div class="actions"><div class="search-box" style="width:210px;background:#fff">${icon('search', 'icon-sm')}<input placeholder="搜索字段" data-col-search></div></div></div>
      <div class="table-wrap"><table class="tbl"><thead><tr><th>字段名</th><th>类型</th><th>角色</th><th>缺失</th><th>唯一值</th><th>取值范围</th><th>均值</th><th>标准差</th></tr></thead><tbody data-col-body>${rowsHtml}</tbody></table></div>
    </div>`;
    const input = MF.$('[data-col-search]', body);
    input.oninput = () => {
      const q = input.value.trim().toLowerCase();
      MF.$$('[data-col-body] tr', body).forEach((tr) => { tr.style.display = !q || tr.dataset.col.indexOf(q) >= 0 ? '' : 'none'; });
    };
  }

})(window);
