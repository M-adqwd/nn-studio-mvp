/* =========================================================
   ModelForge · 前端基础层 / 图表层 / 通用组件
   说明：仅提供 UI 工具与图表绘制，业务数据全部来自 MF.api（真实后端）。
   ========================================================= */
(function (global) {
  const MF = global.MF || (global.MF = {});

  /* ---------------- 基础工具 ---------------- */
  MF.$ = (sel, root) => (root || document).querySelector(sel);
  MF.$$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  MF.esc = (v) => String(v === null || v === undefined ? '' : v).replace(/[&<>"']/g, (m) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[m]));
  MF.fmt = (n) => (typeof n === 'number' ? n.toLocaleString('zh-CN') : MF.esc(n));
  MF.pct = (v, digits) => `${(Number(v) * 100).toFixed(digits === undefined ? 1 : digits)}%`;
  MF.icon = (name, cls) => `<svg class="icon ${cls || ''}" aria-hidden="true"><use href="#i-${name}"></use></svg>`;
  MF.uid = (prefix) => `${prefix || 'id'}_${Math.random().toString(36).slice(2, 8)}`;
  MF.delay = (ms) => new Promise((r) => setTimeout(r, ms));

  MF.toast = function (title, msg, type, duration) {
    let wrap = MF.$('.toast-wrap');
    if (!wrap) {
      wrap = document.createElement('div');
      wrap.className = 'toast-wrap';
      document.body.appendChild(wrap);
    }
    const icons = { success: 'check-circle', warn: 'alert', error: 'x-circle' };
    const el = document.createElement('div');
    el.className = `toast ${type || ''}`;
    el.innerHTML = `
      <div class="ico">${MF.icon(icons[type] || 'info', 'icon-sm')}</div>
      <div class="tx"><b>${MF.esc(title)}</b>${msg ? MF.esc(msg) : ''}</div>
      <button class="btn-ghost x" aria-label="关闭">${MF.icon('x', 'icon-sm')}</button>`;
    wrap.appendChild(el);
    const close = () => { el.style.opacity = '0'; el.style.transform = 'translateX(12px)'; setTimeout(() => el.remove(), 180); };
    el.querySelector('.x').onclick = close;
    setTimeout(close, duration || 3600);
  };

  MF.modal = function (opts) {
    const mask = document.createElement('div');
    mask.className = 'modal-mask';
    mask.innerHTML = `
      <div class="modal ${opts.wide ? 'wide' : ''} ${opts.slim ? 'slim' : ''}" role="dialog" aria-modal="true">
        <div class="modal-head">
          <div><h3>${opts.title || ''}</h3>${opts.desc ? `<p>${opts.desc}</p>` : ''}</div>
          <button class="icon-btn x" aria-label="关闭">${MF.icon('x', 'icon-sm')}</button>
        </div>
        <div class="modal-body">${opts.body || ''}</div>
        ${opts.foot === null ? '' : `<div class="modal-foot">${opts.foot || `<button class="btn" data-close>取消</button>`}</div>`}
      </div>`;
    document.body.appendChild(mask);
    const close = () => { mask.remove(); document.removeEventListener('keydown', onKey); };
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    document.addEventListener('keydown', onKey);
    mask.querySelectorAll('[data-close], .x').forEach((b) => (b.onclick = close));
    if (!opts.static) mask.addEventListener('click', (e) => { if (e.target === mask) close(); });
    if (opts.onMount) opts.onMount(mask, close);
    return { el: mask, close };
  };

  MF.confirm = function (opts) {
    return new Promise((resolve) => {
      const m = MF.modal({
        title: opts.title || '确认操作',
        desc: opts.desc || '',
        slim: true,
        body: `<div class="notice ${opts.danger ? 'danger' : ''}">${MF.icon(opts.danger ? 'alert' : 'info', 'icon-sm')}<div>${opts.message || '该操作执行后将立即生效。'}</div></div>`,
        foot: `<button class="btn" data-cancel>取消</button><button class="btn ${opts.danger ? 'btn-danger' : 'btn-primary'}" data-ok>${opts.okText || '确认'}</button>`,
        onMount(el, close) {
          el.querySelector('[data-cancel]').onclick = () => { close(); resolve(false); };
          el.querySelector('[data-ok]').onclick = () => { close(); resolve(true); };
          el.querySelector('.x').onclick = () => { close(); resolve(false); };
        }
      });
      void m;
    });
  };

  MF.copy = async function (text, label) {
    try {
      if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
      else {
        const ta = document.createElement('textarea');
        ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove();
      }
      MF.toast('已复制', label || '内容已复制到剪贴板', 'success', 1800);
    } catch (e) {
      MF.toast('复制失败', '请手动选择文本复制', 'warn');
    }
  };

  /* ---------------- 会话 ---------------- */
  const SESSION_KEY = 'mf.session.v1';
  MF.session = {
    get() {
      try { return JSON.parse(localStorage.getItem(SESSION_KEY) || 'null'); } catch (e) { return null; }
    },
    set(user) { localStorage.setItem(SESSION_KEY, JSON.stringify(user)); },
    clear() { localStorage.removeItem(SESSION_KEY); },
    guard() { if (!MF.session.get()) { location.replace('login.html'); return false; } return true; }
  };

  /* ---------------- 状态 ---------------- */
  MF.state = {
    get user() { return MF.session.get() || { name: '访客', role: '未登录', account: 'guest' }; },
    get isAdmin() { const u = MF.session.get(); return !!u && u.role === '平台管理员'; }
  };

  MF.TASK_LABELS = {
    tabular_classification: '表格分类', tabular_regression: '表格回归',
    image_classification: '图像分类', text_classification: '文本分类',
    object_detection: '目标检测', time_series: '时序预测'
  };

  const WIZ_KEY = 'mf.wizard.v1';
  MF.wizardStore = {
    load() { try { return JSON.parse(localStorage.getItem(WIZ_KEY) || 'null'); } catch (e) { return null; } },
    save(s) { localStorage.setItem(WIZ_KEY, JSON.stringify(s)); },
    clear() { localStorage.removeItem(WIZ_KEY); }
  };

  /* ---------------- 图表 ---------------- */
  const C = MF.charts = {};
  let seq = 0;

  C.spark = function (values, color) {
    const w = 92, h = 36, max = Math.max.apply(null, values), min = Math.min.apply(null, values);
    const span = max - min || 1;
    const pts = values.map((v, i) => [4 + i * ((w - 8) / (values.length - 1)), h - 4 - ((v - min) / span) * (h - 10)]);
    const d = pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
    const area = `${d} L${w - 4} ${h} L4 ${h} Z`;
    const id = `sp${++seq}`;
    return `<svg class="sparkline" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      <defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="${color || '#2563eb'}" stop-opacity=".18"/>
        <stop offset="1" stop-color="${color || '#2563eb'}" stop-opacity="0"/>
      </linearGradient></defs>
      <path d="${area}" fill="url(#${id})"/>
      <path d="${d}" fill="none" stroke="${color || '#2563eb'}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="${pts[pts.length - 1][0]}" cy="${pts[pts.length - 1][1]}" r="2.2" fill="${color || '#2563eb'}"/>
    </svg>`;
  };

  C.line = function (opts) {
    const series = opts.series || [];
    const labels = opts.labels || [];
    const w = 640, h = opts.height || 220;
    const pad = { l: 42, r: 16, t: 14, b: 30 };
    const all = series.reduce((a, s) => a.concat(s.data), []);
    let max = opts.yMax !== undefined ? opts.yMax : Math.max.apply(null, all);
    let min = opts.yMin !== undefined ? opts.yMin : Math.min.apply(null, all);
    if (max === min) { max += 1; min -= 1; }
    if (opts.yMin === undefined) min = Math.max(0, min - (max - min) * 0.12);
    if (opts.yMax === undefined) max = max + (max - min) * 0.1;
    const span = max - min || 1;
    const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    const px = (i, n) => pad.l + (n <= 1 ? 0 : i * (iw / (n - 1)));
    const py = (v) => pad.t + ih - ((v - min) / span) * ih;
    const fmt = opts.fmtY || ((v) => (v >= 1000 ? (v / 1000).toFixed(1) + 'k' : (Math.round(v * 100) / 100).toString()));
    let grid = '';
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (ih / 4) * i;
      const val = max - (span / 4) * i;
      grid += `<line x1="${pad.l}" y1="${y}" x2="${w - pad.r}" y2="${y}" stroke="#e8eef7" stroke-width="1" stroke-dasharray="${i === 4 ? '0' : '3 4'}"/>
        <text x="${pad.l - 7}" y="${y + 3.5}" text-anchor="end" font-size="9" fill="#94a3b8">${fmt(val)}</text>`;
    }
    let xlabels = '';
    const tickCount = Math.min(labels.length, 6);
    for (let i = 0; i < tickCount; i++) {
      const idx = Math.round((labels.length - 1) * (i / Math.max(1, tickCount - 1)));
      const x = px(idx, labels.length);
      xlabels += `<text x="${x}" y="${h - 9}" text-anchor="${i === 0 ? 'start' : i === tickCount - 1 ? 'end' : 'middle'}" font-size="9" fill="#94a3b8">${MF.esc(labels[idx])}</text>`;
    }
    let paths = '';
    series.forEach((s) => {
      if (!s.data || !s.data.length) return;
      const n = s.data.length;
      const pts = s.data.map((v, i) => [px(i, n), py(v)]);
      const d = pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
      if (s.area) {
        const id = `la${++seq}`;
        paths += `<defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="${s.color}" stop-opacity=".16"/><stop offset="1" stop-color="${s.color}" stop-opacity="0"/>
        </linearGradient></defs>
        <path d="${d} L${px(n - 1, n)} ${pad.t + ih} L${px(0, n)} ${pad.t + ih} Z" fill="url(#${id})"/>`;
      }
      paths += `<path d="${d}" fill="none" stroke="${s.color || '#2563eb'}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`;
      const last = pts[pts.length - 1];
      paths += `<circle cx="${last[0]}" cy="${last[1]}" r="3" fill="#fff" stroke="${s.color || '#2563eb'}" stroke-width="2"/>`;
      if (opts.showLast !== false) {
        paths += `<text x="${Math.min(last[0] + 6, w - 8)}" y="${last[1] - 7}" text-anchor="end" font-size="9" font-weight="600" fill="${s.color}">${fmt(last[1] === undefined ? 0 : s.data[s.data.length - 1])}</text>`;
      }
    });
    // 竖向分隔标记：用于标出「换方案」的轮次分界
    let marks = '';
    (opts.markers || []).forEach((mi) => {
      const x = px(mi.index, labels.length);
      marks += `<line x1="${x.toFixed(1)}" y1="${pad.t}" x2="${x.toFixed(1)}" y2="${pad.t + ih}"
        stroke="${mi.color || '#94a3b8'}" stroke-width="1" stroke-dasharray="4 3"/>
        ${mi.label ? `<text x="${(x + 3).toFixed(1)}" y="${pad.t + 9}" font-size="8.5" fill="${mi.color || '#94a3b8'}">${MF.esc(mi.label)}</text>` : ''}`;
    });
    return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet" role="img">
      ${grid}${xlabels}${marks}${paths}</svg>`;
  };

  C.bars = function (opts) {
    const values = opts.values || [];
    const labels = opts.labels || [];
    const w = 640, h = opts.height || 190, pad = { l: 36, r: 10, t: 12, b: 30 };
    const max = Math.max.apply(null, values) || 1;
    const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    const bw = iw / values.length;
    let bars = '';
    values.forEach((v, i) => {
      const bh = Math.max(2, (v / max) * ih);
      const x = pad.l + i * bw + bw * 0.16;
      const width = bw * 0.68;
      const y = pad.t + ih - bh;
      bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${width.toFixed(1)}" height="${bh.toFixed(1)}" rx="3" fill="${opts.color || '#3b82f6'}" opacity="${i === values.length - 1 ? 1 : 0.82}">
        <title>${MF.esc(labels[i])}: ${MF.fmt(v)}</title></rect>
        <text x="${(x + width / 2).toFixed(1)}" y="${h - 10}" text-anchor="middle" font-size="8.5" fill="#94a3b8">${MF.esc(labels[i] || '')}</text>`;
    });
    let grid = '';
    for (let i = 0; i <= 3; i++) {
      const y = pad.t + (ih / 3) * i;
      grid += `<line x1="${pad.l}" y1="${y}" x2="${w - pad.r}" y2="${y}" stroke="#eef2f8"/>
        <text x="${pad.l - 6}" y="${y + 3}" text-anchor="end" font-size="8.5" fill="#94a3b8">${Math.round(max - (max / 3) * i)}</text>`;
    }
    return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet">${grid}${bars}</svg>`;
  };

  C.donut = function (items, size, center) {
    const s = size || 150, r = 54, cx = s / 2, cy = s / 2, circ = 2 * Math.PI * r;
    const total = items.reduce((a, b) => a + b.value, 0) || 1;
    let off = 0, segs = '';
    items.forEach((it) => {
      const len = (it.value / total) * circ;
      segs += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${it.color}" stroke-width="17"
        stroke-dasharray="${len.toFixed(2)} ${(circ - len).toFixed(2)}" stroke-dashoffset="${(-off).toFixed(2)}"
        transform="rotate(-90 ${cx} ${cy})" stroke-linecap="butt"><title>${MF.esc(it.label)} ${MF.fmt(it.value)}</title></circle>`;
      off += len;
    });
    return `<svg viewBox="0 0 ${s} ${s}" width="${s}" height="${s}" style="flex-shrink:0">
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#eef2f8" stroke-width="17"/>
      ${segs}
      <text x="${cx}" y="${cy + 1}" text-anchor="middle" font-size="18" font-weight="700" fill="#0f172a">${MF.esc(center ? center.value : MF.fmt(total))}</text>
      <text x="${cx}" y="${cy + 16}" text-anchor="middle" font-size="8" fill="#94a3b8">${MF.esc(center ? center.label : '总样本')}</text></svg>`;
  };

  C.legend = function (items) {
    return `<div class="chart-legend">${items.map((i) => `<span class="legend-item"><i style="background:${i.color}"></i>${MF.esc(i.label)}${i.value !== undefined ? ` · ${MF.fmt(i.value)}` : ''}</span>`).join('')}</div>`;
  };

  C.ring = function (value, opts) {
    const o = opts || {};
    const size = o.size || 78, stroke = o.stroke || 7, r = (size - stroke) / 2, c = 2 * Math.PI * r;
    const pctv = Math.max(0, Math.min(100, value));
    const color = o.color || (pctv >= 85 ? '#0f9d76' : pctv >= 70 ? '#2563eb' : '#f59e0b');
    return `<div class="ring-wrap" style="width:${size}px;height:${size}px">
      <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
        <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="#e8eef7" stroke-width="${stroke}"/>
        <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="${color}" stroke-width="${stroke}"
          stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${c - (c * pctv) / 100}"/>
      </svg>
      <div class="ring-center"><div class="v" style="font-size:${size / 4.6}px">${Math.round(pctv)}</div><div class="l">${MF.esc(o.label || '质量分')}</div></div>
    </div>`;
  };

  C.heatmap = function (labels, matrix) {
    const n = labels.length;
    let html = `<div style="overflow-x:auto"><div style="display:grid;grid-template-columns:66px repeat(${n},minmax(42px,1fr));gap:2px;min-width:340px">`;
    html += '<div></div>' + labels.map((l) => `<div class="cm-head" style="font-size:9px;padding:2px">${MF.esc(l)}</div>`).join('');
    matrix.forEach((row, ri) => {
      html += `<div style="font-size:9.5px;color:#64748b;display:grid;align-items:center;padding-right:4px;white-space:nowrap">${MF.esc(labels[ri])}</div>`;
      row.forEach((v, ci) => {
        const a = Math.min(1, Math.abs(v));
        const bg = v >= 0 ? `rgba(37,99,235,${0.08 + a * 0.82})` : `rgba(239,68,68,${0.08 + a * 0.7})`;
        const color = a > 0.55 ? '#fff' : '#334155';
        html += `<div class="heat-cell" style="background:${bg};color:${color}" title="${MF.esc(labels[ri])} × ${MF.esc(labels[ci])}: ${v}">${v.toFixed(2)}</div>`;
      });
    });
    html += '</div></div>';
    return html;
  };

  C.confusion = function (matrix, labels) {
    const n = labels.length;
    const total = matrix.reduce((a, row) => a + row.reduce((x, y) => x + y, 0), 0) || 1;
    let html = `<div class="cm-grid" style="grid-template-columns:88px repeat(${n},1fr)">`;
    html += '<div class="cm-head">实际 \\ 预测</div>' + labels.map((l) => `<div class="cm-head">${MF.esc(l)}</div>`).join('');
    matrix.forEach((row, ri) => {
      html += `<div class="cm-head" style="justify-content:flex-start;font-weight:600;color:#334155">${MF.esc(labels[ri])}</div>`;
      row.forEach((v, ci) => {
        const p = v / total;
        const diag = ri === ci;
        html += `<div class="cm-cell" style="background:${diag ? `rgba(37,99,235,${0.1 + p * 5})` : p > 0.02 ? 'rgba(239,68,68,.1)' : '#f8fafd'};color:${diag ? '#1e40af' : '#475569'}">
          <div style="font-size:15px">${MF.fmt(v)}</div><div style="font-size:9px;font-weight:400;opacity:.7">${(p * 100).toFixed(1)}%</div></div>`;
      });
    });
    html += '</div>';
    return html;
  };

  C.hbars = function (items, color) {
    const max = Math.max.apply(null, items.map((i) => i.value)) || 1;
    return items.map((it) => `<div style="margin-bottom:11px">
      <div style="display:flex;justify-content:space-between;font-size:11.5px;margin-bottom:4px">
        <span style="color:#475569">${MF.esc(it.label)}</span><span style="color:#64748b;font-weight:600">${it.display || MF.fmt(it.value)}</span>
      </div>
      <div class="bar" style="height:6px"><span style="width:${(it.value / max) * 100}%;background:${it.color || color || '#3b82f6'}"></span></div>
    </div>`).join('');
  };

  C.roc = function (points) {
    const w = 300, h = 260, pad = 30;
    const px = (v) => pad + v * (w - pad - 12), py = (v) => h - pad - v * (h - pad - 12);
    const d = points.map((p, i) => `${i ? 'L' : 'M'}${px(p[0])} ${py(p[1])}`).join(' ');
    let grid = '';
    for (let i = 0; i <= 4; i++) {
      const x = pad + (w - pad - 12) * (i / 4), y = h - pad - (h - pad - 12) * (i / 4);
      grid += `<line x1="${pad}" y1="${y}" x2="${w - 12}" y2="${y}" stroke="#eef2f8"/>
        <line x1="${x}" y1="${pad}" x2="${x}" y2="${h - pad}" stroke="#eef2f8"/>
        <text x="${pad - 6}" y="${y + 3}" text-anchor="end" font-size="8" fill="#94a3b8">${i / 4}</text>
        <text x="${x}" y="${h - pad + 12}" text-anchor="middle" font-size="8" fill="#94a3b8">${i / 4}</text>`;
    }
    return `<svg viewBox="0 0 ${w} ${h}" style="max-width:300px;margin:0 auto">
      ${grid}
      <line x1="${px(0)}" y1="${py(0)}" x2="${px(1)}" y2="${py(1)}" stroke="#cbd5e1" stroke-dasharray="4 4"/>
      <path d="${d}" fill="none" stroke="#2563eb" stroke-width="2.2" stroke-linecap="round"/>
      <text x="${pad + 46}" y="${pad + 18}" font-size="9" fill="#64748b">AUC = 0.936</text>
      <text x="${w - 30}" y="${h - 14}" text-anchor="middle" font-size="8" fill="#94a3b8">FPR</text>
      <text x="${pad - 12}" y="${pad - 6}" font-size="8" fill="#94a3b8">TPR</text>
    </svg>`;
  };

  C.code = function (text) {
    const esc = MF.esc(text);
    return `<div class="code-block"><button class="copy-btn" data-copy-code>${MF.icon('copy', 'icon-sm')}复制</button><pre style="margin:0;white-space:pre-wrap">${esc}</pre></div>`;
  };

  C.previewTable = function (dataset) {
    if (!dataset.preview || !dataset.preview.length) {
      return `<div class="empty"><div class="ico">${MF.icon('image')}</div><h4>该数据集为文件型数据</h4><p>原型中仅展示数据统计信息；图像/文本样本预览以缩略图或原文形式提供。</p></div>`;
    }
    const cols = dataset.schema.slice(0, 10);
    return `<div class="table-wrap" style="border:1px solid var(--line);border-radius:9px"><table class="tbl preview-table">
      <thead><tr>${cols.map((c) => `<th>${MF.esc(c.name)}${c.role === 'target' ? ' <span class="tag tag-blue">目标</span>' : ''}</th>`).join('')}</tr></thead>
      <tbody>${dataset.preview.map((row) => `<tr>${row.map((v) => `<td>${v === null || v === undefined || v === '' ? '<span class="null">—</span>' : MF.esc(v)}</td>`).join('')}</tr>`).join('')}</tbody>
    </table></div>`;
  };

  C.datasetIcon = function (task) {
    const map = {
      tabular_classification: ['table', ''],
      tabular_regression: ['trend', 'reg'],
      image_classification: ['image', 'image'],
      text_classification: ['text', 'text'],
      object_detection: ['target', 'det']
    };
    const m = map[task] || ['table', ''];
    return `<div class="ds-ico ${m[1]}">${MF.icon(m[0])}</div>`;
  };
})(window);
