/* =========================================================
   ModelForge · AI 解析报告
   上传完成后展示：AI 如何理解这份数据、锁定了哪些样本参与训练、
   是否命中结构记忆（命中则说明未调用 AI）。
   ========================================================= */
(function (global) {
  const MF = global.MF;
  const { esc, icon, fmt } = MF;

  const SPLIT_LABELS = { train: '训练集（学习用）', val: '验证集（挑选用）', test: '测试集（最后检验用）' };
  const SPLIT_CLASS = { train: 'tag-blue', val: 'tag-green', test: 'tag-amber' };

  /* ---------------- 解析报告弹窗 ---------------- */
  function openParseReport(ds) {
    const r = (ds && ds._parse_report) || {};
    const meta = (ds && ds.meta) || {};
    const counts = r.split_counts || meta.split_counts || {};

    const splitRows = ['train', 'val', 'test'].map((k) => {
      const n = counts[k] || 0;
      const pct = ds && ds.rows ? Math.round((n / ds.rows) * 100) : 0;
      if (!n) return '';
      return `<div class="mini-stat">
        <div class="l">${esc(SPLIT_LABELS[k])}</div>
        <div class="v">${fmt(n)} <small>${pct}%</small></div>
      </div>`;
    }).join('');

    // 记忆命中：说明本次没有为生成脚本调用 AI
    const fixCalls = r.ai_fix_calls || 0;
    const path = r.memory_hit
      ? `<div class="notice success">${icon('check-circle', 'icon-sm')}
           <div><b>分析完成（很快）</b><br>
           <span class="small">这份数据的结构和之前见过的「${esc(r.memory_source || '同类数据')}」一样，
           系统直接复用了上次验证过的处理方法${r.seconds ? `，只用了 ${r.seconds} 秒` : ''}。</span></div>
         </div>`
      : `<div class="notice info">${icon('sparkles', 'icon-sm')}
           <div><b>分析完成</b><br>
           <span class="small">系统已经看懂这份数据的结构${r.seconds ? `，用了 ${r.seconds} 秒` : ''}。
           同样的结构下次会快很多。</span></div>
         </div>`;

    const structure = r.structure
      ? `<div class="kv-row"><span class="k">结构摘要</span><span class="v small">${esc(r.structure)}</span></div>`
      : '';
    const aiNote = meta.ai_notes || meta.summary
      ? `<div class="kv-row"><span class="k">系统说明</span><span class="v small">${esc(meta.ai_notes || meta.summary)}</span></div>`
      : '';

    const excluded = counts.test || 0;
    const noValTest = !excluded && !(counts.val || 0);

    const splitNote = noValTest
      ? `<div class="notice">${icon('info', 'icon-sm')}
          <div><b>你的数据没有自带验证集和测试集</b><br>
          <span class="small">${fmt(counts.train || 0)} 条样本会全部用来训练，
          训练时系统会自动按比例切出一小部分做验证。你原本的数据是什么样就保留什么样，
          不会被重新打乱。</span></div>
        </div>`
      : (excluded ? `<div class="notice mt-16">${icon('shield', 'icon-sm')}
          <div><b>测试集已单独留出，不会参与训练</b><br>
          <span class="small">${fmt(excluded)} 条样本只留到最后检验模型用。
          这样得到的分数才可信 —— 如果拿它来反复调整模型，分数会虚高、实际用起来却不准。</span></div>
        </div>` : '');

    MF.modal({
      title: '数据准备好了',
      desc: '系统已经看懂这份数据，并分好了哪些用来学、哪些用来检验。',
      wide: true,
      body: `
        ${path}
        <div class="ai-verdict mt-16">
          <div class="flex items-center gap-8 mb-8">
            ${icon('database', 'icon-sm')}
            <b>${esc((ds && ds.name) || '数据集')}</b>
          </div>
          <div class="step-summary">${splitRows}</div>
          <div class="kv-list mt-16">
            <div class="kv-row"><span class="k">数据总量</span><span class="v">${fmt((ds && ds.rows) || 0)} 条</span></div>
            <div class="kv-row"><span class="k">要预测的列</span><span class="v">${esc((ds && ds.target_col) || '—')}</span></div>
            ${aiNote}
          </div>
          <details class="mt-12"><summary class="small muted" style="cursor:pointer">技术细节（可跳过）</summary>
            <div class="kv-list mt-8">
              ${structure}
              ${r.fingerprint ? `<div class="kv-row"><span class="k">结构指纹</span><span class="v mono tiny">${esc(r.fingerprint)}</span></div>` : ''}
              ${fixCalls ? `<div class="kv-row"><span class="k">自动修正次数</span><span class="v">${fixCalls}</span></div>` : ''}
            </div></details>
        </div>
        ${splitNote}`,
      foot: `<button class="btn" data-close>关闭</button>
             <button class="btn btn-primary" id="pr-build">${icon('sparkles', 'icon-sm')}开始训练模型</button>`,
      onMount(el, close) {
        const btn = MF.$('#pr-build', el);
        if (btn) btn.onclick = () => {
          close();
          location.hash = `#/build?dataset=${encodeURIComponent(ds.id)}`;
          MF.route();
        };
      }
    });
  }

  MF.openParseReport = openParseReport;
})(window);
