/* =========================================================
   ModelForge · 真实后端 API 封装
   - 统一注入 Bearer token
   - 401 时清理会话并跳回登录页
   兼容 file:// 与 http://，不依赖 ES Module。
   ========================================================= */
(function (global) {
  const MF = global.MF || (global.MF = {});
  const API_BASE = 'http://127.0.0.1:8000';
  MF.API_BASE = API_BASE;

  function token() {
    const s = MF.session && MF.session.get();
    return s && s.token ? s.token : null;
  }

  function handle401() {
    if (MF.session) MF.session.clear();
    if (!/login\.html$/.test(location.pathname)) {
      location.replace('login.html');
    }
  }

  // 统一请求：非 2xx 抛错，后端 detail 带进 message；401 跳转登录
  async function request(path, opts) {
    opts = opts || {};
    const headers = Object.assign({}, opts.headers || {});
    const tk = token();
    if (tk) headers['Authorization'] = `Bearer ${tk}`;
    opts.headers = headers;

    let res;
    try {
      res = await fetch(API_BASE + path, opts);
    } catch (e) {
      throw new Error(`无法连接后端服务（${API_BASE}）：${e.message || '网络错误'}`);
    }
    if (res.status === 401) {
      handle401();
      throw new Error('登录已过期，请重新登录');
    }
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (e) { data = null; }
    if (!res.ok) {
      const detail = data && data.detail;
      throw new Error(typeof detail === 'string' ? detail : (detail ? JSON.stringify(detail) : `HTTP ${res.status}`));
    }
    return data;
  }

  const get = (path) => request(path);
  const postJSON = (path, body) => request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {})
  });
  const putJSON = (path, body) => request(path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {})
  });
  const postEmpty = (path) => request(path, { method: 'POST' });
  const del = (path) => request(path, { method: 'DELETE' });
  const ds = (id, suffix) => `/api/datasets/${encodeURIComponent(id)}${suffix || ''}`;

  MF.api = {
    // ---- 认证 ----
    login: (account, password) => postJSON('/api/auth/login', { account, password }),
    me: () => get('/api/auth/me'),

    // ---- 数据集 ----
    health: () => get('/api/health'),
    listDatasets: () => get('/api/datasets'),
    getDataset: (id) => get(ds(id)),
    uploadDataset(file, name, targetCol, taskHint) {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('name', name);
      fd.append('target_col', targetCol || '');
      fd.append('task_hint', taskHint || '');
      return request('/api/datasets/upload', { method: 'POST', body: fd });
    },
    // ---------- 后台解析任务（大数据集解析可能很久，接口立即返回任务ID） ----------
    parseStatus: (taskId) => get(`/api/datasets/parse-status/${encodeURIComponent(taskId)}`),
    parseResult: (taskId) => get(`/api/datasets/parse-status/${encodeURIComponent(taskId)}/result`),
    // ---------- 解析记忆库（结构 → 已验证脚本） ----------
    parseMemory: () => get('/api/parse-memory'),
    deleteParseMemory: (id) => request(`/api/parse-memory/${encodeURIComponent(id)}`, { method: 'DELETE' }),
    // 修改「答案列」：用户自己纠正 AI 的判断，不必重传数据
    updateDatasetTarget: (id, targetCol) =>
      request(`/api/datasets/${encodeURIComponent(id)}/target`, {
        method: 'PATCH',
        body: JSON.stringify({ target_col: targetCol }),
      }),
    // 图像数据集内单张图片的 URL（<img> 无法带 header，token 走 query）
    imageUrl(id, path) {
      return `${API_BASE}${ds(id, `/image?path=${encodeURIComponent(path)}&token=${encodeURIComponent(token() || '')}`)}`;
    },
    preview: (id, n) => get(ds(id, `/preview?n=${n || 20}`)),
    deleteDataset: (id) => request(ds(id), { method: 'DELETE' }),
    eda: (id) => get(ds(id, '/eda')),
    aiInspect: (id) => get(ds(id, '/ai-inspect')),
    clean: (id, recipe) => postJSON(ds(id, '/clean'), recipe),
    cleaningRuns: (id) => get(ds(id, '/cleaning-runs')),

    // ---- 训练任务 ----
    createTrainJob: (cfg) => postJSON('/api/train', cfg),
    trainJobs: () => get('/api/train'),
    trainJob: (id) => get(`/api/train/${encodeURIComponent(id)}`),

    // ---- 模型仓库 ----
    listModels: () => get('/api/models'),
    getModel: (id) => get(`/api/models/${encodeURIComponent(id)}`),
    registerModel: (trainJobId, name) => postJSON('/api/models', { train_job_id: trainJobId, name: name || '' }),
    updateModel: (id, fields) => putJSON(`/api/models/${encodeURIComponent(id)}`, fields),
    deleteModel: (id) => del(`/api/models/${encodeURIComponent(id)}`),
    predict: (id, payload) => postJSON(`/api/models/${encodeURIComponent(id)}/predict`, payload),
    predictImage(id, file) {
      const fd = new FormData();
      fd.append('file', file);
      return request(`/api/models/${encodeURIComponent(id)}/predict`, { method: 'POST', body: fd });
    },

    // ---- 服务发布 ----
    listServices: () => get('/api/services'),
    createService: (name, modelId) => postJSON('/api/services', { name, model_id: modelId }),
    startService: (id) => postEmpty(`/api/services/${encodeURIComponent(id)}/start`),
    stopService: (id) => postEmpty(`/api/services/${encodeURIComponent(id)}/stop`),
    deleteService: (id) => del(`/api/services/${encodeURIComponent(id)}`),
    svcPredict: (name, payload) => postJSON(`/api/svc/${encodeURIComponent(name)}/predict`, payload),

    // ---- 大模型 Provider ----
    listProviders: () => get('/api/providers'),
    createProvider: (p) => postJSON('/api/providers', p),
    updateProvider: (id, fields) => putJSON(`/api/providers/${encodeURIComponent(id)}`, fields),
    deleteProvider: (id) => del(`/api/providers/${encodeURIComponent(id)}`),
    testProvider: (id) => postEmpty(`/api/providers/${encodeURIComponent(id)}/test`),

    // ---- 大模型对话 ----
    llmChat: (messages, system) => postJSON('/api/llm/chat', { messages, system })
  };
})(window);
