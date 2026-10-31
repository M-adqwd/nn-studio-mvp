/* =========================================================
   ModelForge · 在线推理 / 服务管理 / 大模型配置（真实 FastAPI 后端）
   推理、服务启停与发布、Provider 连通性测试均走 MF.api；
   推理表单与结果渲染复用 models.js 的 MF.inferKit。
   ========================================================= */
(function (global) {
  const MF = global.MF;
  const { esc, fmt, icon } = MF;
  const K = () => MF.inferKit;

  const loadingHTML = (t) => `<div class="card card-pad"><div class="notice">${icon('refresh', 'icon-sm spin')}<div>${esc(t || '正在加载…')}</div></div></div>`;
  const errHTML = (e) => `<div class="card card-pad"><div class="notice danger">${icon('alert', 'icon-sm')}<div><b>请求失败</b><br>${esc(e.message || e)}</div></div></div>`;

  /* ---------------- 在线推理 ---------------- */
  MF.views.inference = function (r, view) {
    view.innerHTML = loadingHTML('正在从后端加载模型…');
    MF.api.listModels()
      .then((list) => {
        if (!list.length) {
          view.innerHTML = `${MF.pageHead('试一下', '拿真实数据试试模型的效果，不用写代码。')}
            <div class="card card-pad"><div class="empty"><div class="ico">${icon('box')}</div><h4>还没有模型可以试</h4><p>先训练出一个模型，保存之后就能在这里试效果了。</p><a class="btn btn-primary" href="#/build?new=1">${icon('sparkles', 'icon-sm')}去训练模型</a></div></div>`;
          return;
        }
        const modelId = r.query.get('model');
        const model = list.find((m) => m.id === modelId) || list[0];
        K().loadSpec(model).then((spec) => render(list, model, spec)).catch((e) => { view.innerHTML = errHTML(e); });
      })
      .catch((e) => { view.innerHTML = errHTML(e); });

    function render(list, model, spec) {
      const options = list.map((m) => `<option value="${esc(m.id)}" ${m.id === model.id ? 'selected' : ''}>${esc(m.name)} · v${esc(m.version)}</option>`).join('');
      const isCnn = model.architecture === 'cnn';
      const batchHint = model.architecture === 'emb_mlp'
        ? JSON.stringify(['物流很快，包装完整。', '用了三天就发热，客服响应慢。'], null, 2)
        : JSON.stringify([Object.fromEntries((spec.cols || []).map((c) => [c.name, c.type === 'number' ? c.def : (c.options && c.options[0]) || '']))], null, 2);

      view.innerHTML = `
        ${MF.pageHead('试一下', '拿真实数据试试模型的效果，不用写代码。',
          `<button class="btn" id="infer-clear">${icon('refresh')}重置输入</button><a class="btn btn-primary" href="#/services">${icon('rocket')}发布为服务</a>`)}
        <div class="grid grid-3" style="margin-bottom:18px">
          <div class="card card-pad"><div class="tiny muted">当前模型</div><div class="flex items-center gap-8 mt-8">${K().stagePill(model.stage)}<span class="small" style="font-weight:600">v${esc(model.version)}</span></div></div>
          <div class="card card-pad"><div class="tiny muted">任务类型</div><div class="small mt-8" style="font-weight:600">${esc(MF.TASK_LABELS[model.task_hint] || model.task_hint || '—')} · ${esc(K().ARCH_LABELS[model.architecture] || model.architecture || '—')}</div></div>
          <div class="card card-pad"><div class="tiny muted">训练数据</div><div class="small mt-8" style="font-weight:600">${esc(model.dataset_name || '—')}</div></div>
        </div>
        <div class="infer-layout">
          <div class="card">
            <div class="card-head" style="padding-bottom:12px"><div><div class="section-title">输入</div><div class="section-sub">选择模型并填写特征，运行推理</div></div>
              <div class="actions"><select class="select" id="infer-model" style="width:230px;height:34px;font-size:12px">${options}</select></div></div>
            <div class="card-body">
              ${isCnn ? '' : `<div class="tabs" style="margin-bottom:14px"><div class="tab active" data-infer-tab="single">单条推理</div><div class="tab" data-infer-tab="batch">批量推理</div></div>`}
              <div data-infer-single data-form>${K().formHTML(model, spec)}
                <button class="btn btn-primary btn-block mt-16" id="run-infer">${icon('play', 'icon-sm')}运行推理</button></div>
              ${isCnn ? '' : `<div data-infer-batch style="display:none">
                <div class="field"><label>批量输入（JSON 数组，${model.architecture === 'emb_mlp' ? '每项为文本' : '每项为特征 dict'}，最多 64 条）</label><textarea class="textarea mono" style="min-height:170px" id="batch-input">${esc(batchHint)}</textarea></div>
                <button class="btn btn-primary btn-block mt-12" id="run-batch">${icon('play', 'icon-sm')}批量推理</button></div>`}
            </div>
          </div>
          <div data-result>
            <div class="card card-pad" style="min-height:320px;display:grid;place-items:center" id="infer-placeholder">
              <div class="empty"><div class="ico">${icon('cpu')}</div><h4>等待推理请求</h4><p>填写左侧输入并点击「运行推理」，结果将显示在这里。</p></div>
            </div>
          </div>
        </div>
        <div class="grid grid-2 mt-18">
          ${MF.sectionCard('API 调用示例', '将模型集成到业务系统', '', MF.charts.code(`curl -X POST ${MF.API_BASE}/api/models/${model.id}/predict \\
  -H "Authorization: Bearer <登录后获取的 token>" \\
  -H "Content-Type: application/json" \\
  -d '${isCnn ? '图像模型请使用 multipart 上传（字段名 file）' : JSON.stringify(model.architecture === 'emb_mlp' ? { text: '示例文本' } : { rows: [{ 特征列: '值' }] })}'`))}
          ${MF.sectionCard('推理说明', '当前版本的限制', '', `<div class="kv-list">
            <div class="kv-row"><span class="k">模型版本</span><span class="v mono">v${esc(model.version)}</span></div>
            <div class="kv-row"><span class="k">输入约束</span><span class="v">${model.architecture === 'emb_mlp' ? '文本，单次最多 64 条' : isCnn ? '单张图像，multipart 上传' : '特征需与训练时一致'}</span></div>
            <div class="kv-row"><span class="k">单次上限</span><span class="v">${isCnn ? '1 张 / 请求' : '64 条 / 请求'}</span></div>
            <div class="kv-row"><span class="k">数据留存</span><span class="v">推理输入不用于训练</span></div>
          </div><div class="notice mt-16">${icon('shield', 'icon-sm')}<div>在线推理仅用于验证；生产流量请通过「服务管理」发布的 API 调用。</div></div>`)}
        </div>`;

      MF.$('#infer-model', view).onchange = (e) => { location.hash = `#/inference?model=${e.target.value}`; };
      MF.$('#infer-clear', view).onclick = () => MF.route();
      MF.$$('[data-infer-tab]', view).forEach((t) => (t.onclick = () => {
        MF.$$('[data-infer-tab]', view).forEach((x) => x.classList.remove('active')); t.classList.add('active');
        const single = t.dataset.inferTab === 'single';
        MF.$('[data-infer-single]', view).style.display = single ? '' : 'none';
        MF.$('[data-infer-batch]', view).style.display = single ? 'none' : '';
      }));
      const formRoot = MF.$('[data-form]', view);
      K().bindForm(model, formRoot);

      MF.$('#run-infer', view).onclick = async () => {
        const btn = MF.$('#run-infer', view);
        const c = K().collect(model, formRoot);
        if (c.kind === 'error') { MF.toast(c.message, '', 'warn'); return; }
        btn.disabled = true; btn.innerHTML = `${icon('refresh', 'icon-sm spin')}推理中…`;
        const t0 = Date.now();
        try {
          const res = c.kind === 'file' ? await MF.api.predictImage(model.id, c.file) : await MF.api.predict(model.id, c.payload);
          MF.$('[data-result]', view).innerHTML = K().resultHTML(res, model, [['推理耗时', (Date.now() - t0) + ' ms']]);
          MF.toast('推理完成', `${model.name} · ${Date.now() - t0} ms`, 'success', 1800);
        } catch (err) {
          MF.$('[data-result]', view).innerHTML = `<div class="card card-pad"><div class="notice danger">${icon('alert', 'icon-sm')}<div><b>推理失败</b><br>${esc(err.message)}</div></div></div>`;
        }
        btn.disabled = false; btn.innerHTML = `${icon('play', 'icon-sm')}运行推理`;
      };

      const batchBtn = MF.$('#run-batch', view);
      if (batchBtn) batchBtn.onclick = async () => {
        const raw = MF.$('#batch-input', view).value.trim();
        let arr;
        try {
          arr = JSON.parse(raw);
          if (!Array.isArray(arr) || !arr.length) throw new Error('not array');
        } catch (e) {
          MF.toast('输入格式有误', '请提交非空 JSON 数组。', 'warn');
          return;
        }
        arr = arr.slice(0, 64);
        batchBtn.disabled = true; batchBtn.innerHTML = `${icon('refresh', 'icon-sm spin')}批量推理中…`;
        const t0 = Date.now();
        try {
          const payload = model.architecture === 'emb_mlp' ? { texts: arr.map((x) => String(x)) } : { rows: arr };
          const res = await MF.api.predict(model.id, payload);
          const preds = res.predictions || [];
          MF.$('[data-result]', view).innerHTML = `<div class="card"><div class="card-head" style="padding-bottom:12px"><div><div class="section-title">批量推理结果</div><div class="section-sub">共 ${preds.length} 条 · 模型 v${esc(model.version)}</div></div><span class="pill pill-success"><span class="pdot"></span>完成</span></div>
            <div class="table-wrap"><table class="tbl"><thead><tr><th>#</th><th>输入摘要</th><th>预测</th><th>置信度</th></tr></thead><tbody>
              ${preds.map((p, i) => {
                const input = typeof arr[i] === 'string' ? arr[i] : JSON.stringify(arr[i]);
                const out = p.value !== undefined ? String(p.value) : p.label;
                const conf = p.confidence !== undefined ? MF.pct(p.confidence) : '—';
                return `<tr><td>${i + 1}</td><td class="small muted" style="max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(String(input).slice(0, 80))}</td><td><span class="pill pill-info">${esc(out)}</span></td><td class="small">${conf}</td></tr>`;
              }).join('')}
            </tbody></table></div></div>`;
          MF.toast('批量推理完成', `${preds.length} 条 · ${Date.now() - t0} ms`, 'success', 1800);
        } catch (err) {
          MF.toast('批量推理失败', err.message, 'error');
        }
        batchBtn.disabled = false; batchBtn.innerHTML = `${icon('play', 'icon-sm')}批量推理`;
      };
    }
  };

  /* ---------------- 服务管理 ---------------- */
  function svcStatus(s) {
    const map = { running: ['运行中', 'pill-success'], stopped: ['已停止', 'pill-muted'] };
    const m = map[s.status] || ['未知', 'pill-muted'];
    return `<span class="pill ${m[1]}"><span class="pdot"></span>${m[0]}</span>`;
  }

  MF.views.services = function (r, view) {
    view.innerHTML = loadingHTML('正在从后端加载服务列表…');
    MF.api.listServices()
      .then((list) => renderList(list))
      .catch((e) => { view.innerHTML = errHTML(e); });

    function renderList(list) {
      const running = list.filter((s) => s.status === 'running');
      const totalReq = list.reduce((a, s) => a + (s.requests_total || 0), 0);
      const stats = [
        { label: '运行中服务', value: running.length, suffix: '个', icon: 'rocket', note: `${list.length} 个已登记` },
        { label: '累计调用量', value: fmt(totalReq), suffix: '次', icon: 'activity', tone: 'green', note: '全部服务合计' },
        { label: '已停止', value: list.filter((s) => s.status === 'stopped').length, suffix: '个', icon: 'stop', tone: 'purple', note: '启动后即可调用' },
        { label: '接入模型', value: new Set(list.map((s) => s.model_id)).size, suffix: '个', icon: 'box', tone: 'amber', note: '来自模型仓库' }
      ];
      const cards = list.map((s) => `<div class="card svc-card card-hover">
        <div class="svc-head"><div class="ds-ico" style="width:36px;height:36px">${icon('server')}</div>
          <div style="flex:1;min-width:0"><div class="flex items-center gap-7"><span class="svc-name">${esc(s.name)}</span>${svcStatus(s)}</div>
            <div class="svc-env">${esc(s.model_name || '—')} v${esc(s.model_version || '—')} · 创建于 ${esc(s.created_at || '')}</div></div></div>
        <div class="svc-metrics">
          <div class="svc-metric"><div class="l">累计调用</div><div class="v">${fmt(s.requests_total || 0)}</div></div>
          <div class="svc-metric"><div class="l">任务类型</div><div class="v" style="font-size:13px">${esc(MF.TASK_LABELS[s.task_hint] || s.task_hint || '—')}</div></div>
          <div class="svc-metric"><div class="l">创建人</div><div class="v" style="font-size:13px">${esc(s.created_by || '—')}</div></div>
        </div>
        <div class="endpoint mt-12"><span>${esc(s.endpoint)}</span><button class="icon-btn" style="width:24px;height:24px" data-copy-ep="${esc(s.id)}" title="复制端点">${icon('copy', 'icon-sm')}</button></div>
        <div class="flex items-center gap-8 mt-14" style="flex-wrap:wrap">
          ${MF.taskPill(s.task_hint)}
          <div class="ml-auto flex gap-6">
            <button class="btn btn-sm btn-ghost" data-svc-test="${esc(s.id)}">${icon('cpu', 'icon-sm')}调用测试</button>
            ${s.status === 'running'
              ? `<button class="btn btn-sm btn-danger" data-svc-stop="${esc(s.id)}">${icon('stop', 'icon-sm')}停止</button>`
              : `<button class="btn btn-sm btn-soft" data-svc-start="${esc(s.id)}">${icon('play', 'icon-sm')}启动</button>`}
            <button class="btn btn-sm btn-ghost" data-svc-del="${esc(s.id)}" title="删除服务" style="color:#dc2626">${icon('trash', 'icon-sm')}</button>
          </div>
        </div>
      </div>`).join('');

      view.innerHTML = `
        ${MF.pageHead('服务管理', '把模型变成一个可以随时调用的接口，供其他系统使用。',
          `<button class="btn" id="svc-refresh">${icon('refresh')}刷新</button><button class="btn btn-primary" data-deploy-new>${icon('rocket')}发布新服务</button>`)}
        <div class="grid grid-4">${stats.map(MF.statCard).join('')}</div>
        <div class="grid grid-2 mt-20" id="svc-grid">${cards}</div>
        <div class="card card-pad mt-16" data-svc-empty style="display:${list.length ? 'none' : ''}"><div class="empty"><div class="ico">${icon('rocket')}</div><h4>还没有发布的服务</h4><p>从模型仓库选择一个版本，发布为可调用的 API 服务。</p><button class="btn btn-primary" data-deploy-new-2>${icon('rocket', 'icon-sm')}发布新服务</button></div></div>
        <div class="notice mt-18">${icon('shield', 'icon-sm')}<div>服务端点形如 /api/svc/{name}/predict，调用需携带登录 token；停止服务后调用将返回 409。</div></div>`;

      const byId = (id) => list.find((s) => s.id === id);
      MF.$$('[data-copy-ep]', view).forEach((b) => (b.onclick = () => MF.copy(`${MF.API_BASE}${byId(b.dataset.copyEp).endpoint}`, '端点已复制')));
      MF.$('#svc-refresh', view).onclick = () => MF.route();
      MF.$$('[data-deploy-new],[data-deploy-new-2]', view).forEach((b) => (b.onclick = openDeployServiceModal));
      MF.$$('[data-svc-stop]', view).forEach((b) => (b.onclick = async () => {
        const s = byId(b.dataset.svcStop);
        if (await MF.confirm({ title: '停止服务', message: `停止 ${s.name} 后，所有调用将立即失败（409）。`, okText: '停止服务', danger: true })) {
          try {
            await MF.api.stopService(s.id);
            MF.toast('服务已停止', `${s.name} 已下线`, 'success');
            MF.route();
          } catch (e) { MF.toast('操作失败', e.message, 'error'); }
        }
      }));
      MF.$$('[data-svc-start]', view).forEach((b) => (b.onclick = async () => {
        const s = byId(b.dataset.svcStart);
        try {
          await MF.api.startService(s.id);
          MF.toast('服务已启动', `${s.name} 正在恢复流量`, 'success');
          MF.route();
        } catch (e) { MF.toast('操作失败', e.message, 'error'); }
      }));
      MF.$$('[data-svc-del]', view).forEach((b) => (b.onclick = async () => {
        const s = byId(b.dataset.svcDel);
        if (await MF.confirm({ title: '删除服务', message: `删除 ${s.name} 后其端点将立即失效，且不可恢复。`, okText: '删除', danger: true })) {
          try {
            await MF.api.deleteService(s.id);
            MF.toast('服务已删除', s.name, 'success');
            MF.route();
          } catch (e) { MF.toast('删除失败', e.message, 'error'); }
        }
      }));
      MF.$$('[data-svc-test]', view).forEach((b) => (b.onclick = () => openSvcTest(byId(b.dataset.svcTest))));
    }
  };

  // 调用测试：仅 JSON 任务；图像服务引导至模型页上传
  function openSvcTest(s) {
    if (s.task_hint === 'image_classification') {
      MF.modal({
        title: `调用测试 · ${s.name}`, desc: s.endpoint, slim: true,
        body: `<div class="notice">${icon('info', 'icon-sm')}<div>图像分类服务需以 multipart 上传图像，暂不支持 JSON 调用测试；请前往模型详情页的「推理试算」上传图像验证。</div></div>`,
        foot: `<a class="btn btn-primary" href="#/models/${esc(s.model_id)}?tab=predict" data-close>去模型页试算</a>`
      });
      return;
    }
    MF.modal({
      title: `调用测试 · ${s.name}`, desc: `POST ${s.endpoint}`, wide: true,
      body: `<div data-svc-form>${loadingHTML('正在加载模型字段结构…')}</div>
        <div data-svc-result style="margin-top:14px"></div>`,
      foot: `<button class="btn" data-close>关闭</button><button class="btn btn-primary" id="st-run">${icon('play', 'icon-sm')}发送请求</button>`,
      onMount(el, close) {
        let model = null;
        MF.api.getModel(s.model_id)
          .then((m) => { model = m; return K().loadSpec(m); })
          .then((spec) => {
            const form = MF.$('[data-svc-form]', el);
            form.innerHTML = K().formHTML(model, spec);
            K().bindForm(model, form);
          })
          .catch((e) => { MF.$('[data-svc-form]', el).innerHTML = errHTML(e); });
        MF.$('#st-run', el).onclick = async () => {
          if (!model) { MF.toast('模型信息尚未加载完成', '', 'warn'); return; }
          const btn = MF.$('#st-run', el);
          const c = K().collect(model, MF.$('[data-svc-form]', el));
          if (c.kind === 'error') { MF.toast(c.message, '', 'warn'); return; }
          if (c.kind === 'file') { MF.toast('该服务仅支持 JSON 调用', '', 'warn'); return; }
          btn.disabled = true; btn.innerHTML = `${icon('refresh', 'icon-sm spin')}请求中…`;
          const t0 = Date.now();
          try {
            const res = await MF.api.svcPredict(s.name, c.payload);
            MF.$('[data-svc-result]', el).innerHTML = K().resultHTML(res, model, [['服务端点', s.endpoint], ['耗时', (Date.now() - t0) + ' ms']]);
          } catch (err) {
            MF.$('[data-svc-result]', el).innerHTML = `<div class="notice danger">${icon('alert', 'icon-sm')}<div><b>调用失败</b><br>${esc(err.message)}</div></div>`;
          }
          btn.disabled = false; btn.innerHTML = `${icon('play', 'icon-sm')}发送请求`;
        };
      }
    });
  }

  function openDeployServiceModal() {
    MF.modal({
      title: '发布新服务', desc: '从模型仓库选择一个版本，部署为可调用的 API 服务。',
      body: `<div data-nd-body>${loadingHTML('正在加载模型列表…')}</div>`,
      foot: `<button class="btn" data-close>取消</button><button class="btn btn-primary" id="nd-submit">${icon('rocket', 'icon-sm')}发布</button>`,
      onMount(el, close) {
        MF.api.listModels()
          .then((models) => {
            if (!models.length) {
              MF.$('[data-nd-body]', el).innerHTML = `<div class="notice">${icon('info', 'icon-sm')}<div>模型仓库为空，请先完成训练并登记模型。</div></div>`;
              MF.$('#nd-submit', el).style.display = 'none';
              return;
            }
            MF.$('[data-nd-body]', el).innerHTML = `<div class="form-grid">
              <div class="field full"><label>选择模型<span class="req">*</span></label><select class="select" id="nd-model">${models.map((m) => `<option value="${esc(m.id)}">${esc(m.name)} · v${esc(m.version)} · ${esc(K().ARCH_LABELS[m.architecture] || m.architecture || '')}</option>`).join('')}</select></div>
              <div class="field full"><label>服务名称<span class="req">*</span></label><input class="input mono" id="nd-name" placeholder="例如 churn-api">
                <div class="hint">小写字母开头的 slug（a-z0-9-，2~41 字符），端点为 /api/svc/{name}/predict</div></div>
            </div>`;
            const nameInput = MF.$('#nd-name', el), modelSel = MF.$('#nd-model', el);
            const slugOf = (m) => (m.name || 'model').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').replace(/^[^a-z]+/, '') || 'model';
            const syncName = () => {
              const m = models.find((x) => x.id === modelSel.value);
              if (m) nameInput.value = `${slugOf(m)}-api`.slice(0, 41);
            };
            modelSel.onchange = syncName;
            syncName();
          })
          .catch((e) => { MF.$('[data-nd-body]', el).innerHTML = errHTML(e); });
        MF.$('#nd-submit', el).onclick = async () => {
          const modelSel = MF.$('#nd-model', el);
          if (!modelSel) return;
          const name = (MF.$('#nd-name', el).value || '').trim().toLowerCase();
          const btn = MF.$('#nd-submit', el);
          btn.disabled = true; btn.innerHTML = `${icon('refresh', 'icon-sm spin')}发布中…`;
          try {
            const svc = await MF.api.createService(name, modelSel.value);
            close();
            MF.toast('服务已发布', `${svc.name} 已运行，端点 ${svc.endpoint}`, 'success');
            MF.route();
          } catch (e) {
            MF.toast('发布失败', e.message, 'error', 5000);
            btn.disabled = false; btn.innerHTML = `${icon('rocket', 'icon-sm')}发布`;
          }
        };
      }
    });
  }

  /* ---------------- 大模型配置 ---------------- */
  const PROVIDER_COLORS = ['#1d4ed8', '#0891b2', '#0f766e', '#7c3aed', '#334155', '#c77a0a'];

  function providerStatus(p) {
    const map = {
      active: ['已连接', 'pill-success'],
      auth_failed: ['鉴权失败', 'pill-warn'],
      unreachable: ['无法连接', 'pill-danger'],
      unknown: ['未验证', 'pill-muted']
    };
    const m = map[p.status] || map.unknown;
    return `<span class="pill ${m[1]}"><span class="pdot"></span>${m[0]}</span>`;
  }

  MF.views.settings = function (r, view) {
    view.innerHTML = loadingHTML('正在加载大模型服务配置…');
    MF.api.listProviders()
      .then((list) => renderList(list))
      .catch((e) => { view.innerHTML = errHTML(e); });

    function renderList(list) {
      const cards = list.map((p, i) => {
        const letter = (p.name || 'LL').replace(/[^A-Za-z0-9一-龥]/g, '').slice(0, 2).toUpperCase() || 'LL';
        const color = PROVIDER_COLORS[i % PROVIDER_COLORS.length];
        const host = (p.base_url || '').replace(/^https?:\/\//, '').split('/')[0];
        return `<div class="card provider-card ${p.is_default ? 'card-accent' : ''}">
        ${p.is_default ? '<span class="default-badge">默认服务</span>' : ''}
        <div class="provider-head"><div class="provider-logo" style="background:${color}">${esc(letter)}</div>
          <div style="flex:1;min-width:0"><div class="flex items-center gap-7"><span class="provider-name">${esc(p.name)}</span>${providerStatus(p)}</div>
            <div class="provider-model">${esc(p.model)}</div></div></div>
        <div class="key-row"><span>${icon('key', 'icon-sm')}</span><span style="flex:1">${p.has_key ? esc(p.api_key) : '未配置 API Key'}</span>${p.latency_ms !== null && p.latency_ms !== undefined ? `<span class="tiny">${esc(p.latency_ms)} ms</span>` : ''}</div>
        <div class="provider-meta"><span class="tag">${esc(p.provider_type || 'openai_compat')}</span><span class="tag mono" style="font-size:10px">${esc(host)}</span></div>
        <div class="provider-foot">
          <button class="btn btn-sm btn-ghost" data-provider-test="${esc(p.id)}" ${p.has_key ? '' : 'disabled title="请先编辑并填入 API Key"'}>${icon('refresh', 'icon-sm')}测试连接</button>
          <button class="btn btn-sm btn-ghost" data-provider-edit="${esc(p.id)}">${icon('edit', 'icon-sm')}编辑</button>
          ${p.is_default ? '' : `<button class="btn btn-sm btn-ghost" data-provider-default="${esc(p.id)}">设为默认</button>`}
          <span class="ml-auto tiny muted">${esc((p.created_at || '').slice(0, 10))}</span></div>
      </div>`;
      }).join('');

      view.innerHTML = `
        ${MF.pageHead('AI 服务配置', '设置平台用哪个 AI 服务。只有在「AI 助手」等智能功能里才会用到。',
          `<button class="btn btn-primary" id="add-provider">${icon('plus')}新增服务</button>`)}
        <div class="notice">${icon('sparkles', 'icon-sm')}<div><b>大模型在本平台中的作用：</b>作为流程内助手提供清洗建议、模型推荐、训练诊断与评估解读；神经网络的实际训练与推理由平台训练引擎完成，不依赖大模型。</div></div>
        <div class="grid grid-2 mt-18">${cards}</div>
        <div class="card card-pad mt-16" data-provider-empty style="display:${list.length ? 'none' : ''}"><div class="empty"><div class="ico">${icon('settings')}</div><h4>还没有配置大模型服务</h4><p>新增一个 OpenAI 兼容服务，测试连接后即可作为平台助手。</p></div></div>
        <div class="notice warn mt-18">${icon('lock', 'icon-sm')}<div>API Key 采用服务端存储，前端仅展示末四位掩码；仅平台管理员可新增、编辑或删除服务配置。</div></div>`;

      const byId = (id) => list.find((p) => p.id === id);
      MF.$('#add-provider', view).onclick = () => openProviderModal(null);
      MF.$$('[data-provider-test]', view).forEach((b) => (b.onclick = async () => {
        const p = byId(b.dataset.providerTest);
        if (!p.has_key) { MF.toast('请先配置 API Key', `${p.name} 当前没有可用的密钥。`, 'warn'); return; }
        b.disabled = true; b.innerHTML = `${icon('refresh', 'icon-sm spin')}测试中…`;
        try {
          const res = await MF.api.testProvider(p.id);
          if (res.ok) MF.toast('连接成功', `${p.name} · ${p.model} · ${res.latency_ms} ms`, 'success');
          else MF.toast('连接失败', res.detail || '测试未通过', 'error', 5000);
        } catch (e) {
          MF.toast('连接失败', e.message, 'error', 5000);
        }
        MF.route();
      }));
      MF.$$('[data-provider-edit]', view).forEach((b) => (b.onclick = () => openProviderModal(byId(b.dataset.providerEdit))));
      MF.$$('[data-provider-default]', view).forEach((b) => (b.onclick = async () => {
        const p = byId(b.dataset.providerDefault);
        try {
          await MF.api.updateProvider(p.id, { is_default: true });
          MF.toast('默认服务已切换', `${p.name} 将用于平台助手与 AI 分析。`, 'success');
          MF.route();
        } catch (e) { MF.toast('设置失败', e.message, 'error'); }
      }));
    }
  };

  function openProviderModal(provider) {
    const editing = !!provider;
    MF.modal({
      title: editing ? '编辑大模型服务' : '新增大模型服务',
      desc: '支持 OpenAI 兼容协议。保存后请先测试连接验证可用性。',
      body: `${editing ? '' : `<div class="flex gap-8 mb-12" style="flex-wrap:wrap">
          <span class="tiny muted" style="align-self:center">快捷预设：</span>
          <button type="button" class="btn btn-sm btn-soft" data-preset="minimax">MiniMax</button>
          <button type="button" class="btn btn-sm btn-soft" data-preset="openai">OpenAI</button>
        </div>`}
        <div class="form-grid">
        <div class="field"><label>服务名称<span class="req">*</span></label><input class="input" id="pv-name" value="${editing ? esc(provider.name) : ''}" placeholder="例如：MiniMax"></div>
        <div class="field"><label>模型名称<span class="req">*</span></label><input class="input" id="pv-model" value="${editing ? esc(provider.model) : ''}" placeholder="MiniMax-M3"></div>
        <div class="field full"><label>Base URL<span class="req">*</span></label><input class="input mono" id="pv-base" value="${editing ? esc(provider.base_url) : ''}" placeholder="https://api.minimax.cn/v1"></div>
        <div class="field full"><label>API Key${editing ? '' : '<span class="req">*</span>'}</label><div class="input-affix"><input class="input mono" type="password" id="pv-key" value="" placeholder="${editing ? '留空表示不修改现有 Key' : 'sk-...'}">
          <button class="affix-btn" type="button" id="pv-eye">${icon('eye', 'icon-sm')}</button></div><div class="hint">密钥仅保存于服务端，前端不展示完整内容。</div></div>
        <div class="field full"><label class="check-row"><input type="checkbox" id="pv-default" ${editing && provider.is_default ? 'checked' : ''}><span class="ct">设为默认服务<small>平台助手默认使用该服务；同一时间仅允许一个默认服务</small></span></label></div>
      </div>
      ${editing ? `<div class="notice danger mt-16">${icon('alert', 'icon-sm')}<div>删除服务后，引用该服务的助手会话将不可用。</div></div>` : ''}`,
      foot: `${editing ? '<button class="btn btn-danger" id="pv-delete" style="margin-right:auto">删除服务</button>' : ''}<button class="btn" data-close>取消</button><button class="btn btn-primary" id="pv-save">${editing ? '保存配置' : '添加服务'}</button>`,
      onMount(el, close) {
        const eye = MF.$('#pv-eye', el), key = MF.$('#pv-key', el);
        eye.onclick = () => { key.type = key.type === 'password' ? 'text' : 'password'; eye.innerHTML = icon(key.type === 'password' ? 'eye' : 'eye-off', 'icon-sm'); };
        // 快捷预设
        const PRESETS = {
          minimax: { name: 'MiniMax', model: 'MiniMax-M3', base: 'https://api.minimax.cn/v1' },
          openai: { name: 'OpenAI', model: 'gpt-4o-mini', base: 'https://api.openai.com/v1' }
        };
        MF.$$('[data-preset]', el).forEach((b) => (b.onclick = () => {
          const p = PRESETS[b.dataset.preset];
          if (!p) return;
          MF.$('#pv-name', el).value = p.name;
          MF.$('#pv-model', el).value = p.model;
          MF.$('#pv-base', el).value = p.base;
        }));
        const del = MF.$('#pv-delete', el);
        if (del) del.onclick = async () => {
          if (await MF.confirm({ title: '删除大模型服务', message: `删除 ${provider.name} 后不可恢复。`, okText: '删除', danger: true })) {
            try {
              await MF.api.deleteProvider(provider.id);
              close(); MF.toast('服务已删除', provider.name, 'success'); MF.route();
            } catch (e) { MF.toast('删除失败', e.message, 'error'); }
          }
        };
        MF.$('#pv-save', el).onclick = async () => {
          const name = MF.$('#pv-name', el).value.trim(), model = MF.$('#pv-model', el).value.trim();
          const baseUrl = MF.$('#pv-base', el).value.trim(), apiKey = key.value.trim();
          if (!name || !model || !baseUrl) { MF.toast('请填写完整信息', '服务名称、模型与 Base URL 均为必填。', 'warn'); return; }
          if (!editing && !apiKey) { MF.toast('请填写 API Key', '新增服务需要有效的密钥。', 'warn'); return; }
          const btn = MF.$('#pv-save', el);
          btn.disabled = true; btn.innerHTML = `${icon('refresh', 'icon-sm spin')}保存中…`;
          try {
            if (editing) {
              const fields = { name, model, base_url: baseUrl, is_default: MF.$('#pv-default', el).checked };
              if (apiKey) fields.api_key = apiKey;
              await MF.api.updateProvider(provider.id, fields);
              MF.toast('配置已保存', apiKey ? '密钥已更新，请重新测试连接。' : '请测试连接验证可用性。', 'success');
            } else {
              await MF.api.createProvider({ name, model, base_url: baseUrl, api_key: apiKey, is_default: MF.$('#pv-default', el).checked });
              MF.toast('服务已添加', '请点击「测试连接」验证可用性。', 'success');
            }
            close(); MF.route();
          } catch (e) {
            MF.toast('保存失败', e.message, 'error', 5000);
            btn.disabled = false; btn.innerHTML = editing ? '保存配置' : '添加服务';
          }
        };
      }
    });
  }
})(window);
