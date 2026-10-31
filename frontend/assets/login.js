/* =========================================================
   ModelForge · 登录页（对接后端真实鉴权 MF.api.login）
   ========================================================= */
(function (global) {
  const MF = global.MF;
  const REMEMBER_KEY = 'mf.remember.v1';

  const account = MF.$('#login-account');
  const password = MF.$('#login-password');
  const remember = MF.$('#login-remember');
  const errorBox = MF.$('#login-error');
  const errorText = MF.$('#login-error-text');
  const submit = MF.$('#login-submit');
  const eye = MF.$('#login-eye');

  if (MF.session.get()) {
    location.replace('index.html');
    return;
  }

  const saved = localStorage.getItem(REMEMBER_KEY);
  if (saved) { account.value = saved; remember.checked = true; setTimeout(() => password.focus(), 60); }

  function showError(msg) {
    errorText.textContent = msg;
    errorBox.classList.add('show');
    account.classList.add('error');
    password.classList.add('error');
    password.focus();
  }
  function clearError() {
    errorBox.classList.remove('show');
    account.classList.remove('error');
    password.classList.remove('error');
  }

  eye.onclick = () => {
    const hidden = password.type === 'password';
    password.type = hidden ? 'text' : 'password';
    eye.innerHTML = `<svg class="icon icon-sm" aria-hidden="true"><use href="#i-${hidden ? 'eye-off' : 'eye'}"></use></svg>`;
  };

  MF.$('#login-forgot').onclick = () => {
    MF.toast('演示环境', '请联系平台管理员重置密码，或使用下方演示账号登录。', 'info', 4200);
  };

  [account, password].forEach((el) => (el.oninput = clearError));

  MF.$('#login-form').onsubmit = async (e) => {
    e.preventDefault();
    clearError();
    const acc = account.value.trim();
    const pwd = password.value;
    if (!acc) { showError('请输入账号。'); return; }
    if (!pwd) { showError('请输入密码。'); return; }

    submit.disabled = true;
    submit.innerHTML = `<svg class="icon icon-sm spin" aria-hidden="true"><use href="#i-refresh"></use></svg>正在验证身份…`;

    try {
      const res = await MF.api.login(acc, pwd);
      if (remember.checked) localStorage.setItem(REMEMBER_KEY, acc);
      else localStorage.removeItem(REMEMBER_KEY);

      MF.session.set({
        token: res.token,
        account: res.user.account,
        name: res.user.name,
        role: res.user.role,
        org: res.user.org || '',
        loginAt: new Date().toISOString()
      });
      submit.innerHTML = `<svg class="icon icon-sm" aria-hidden="true"><use href="#i-check-circle"></use></svg>登录成功，正在进入…`;
      await MF.delay(320);
      location.replace('index.html');
    } catch (err) {
      submit.disabled = false;
      submit.textContent = '登录';
      showError(err.message || '登录失败，请稍后重试。');
    }
  };
})(window);
