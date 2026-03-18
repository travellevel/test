const { createApp, ref, computed } = Vue;

createApp({
  setup() {
    const email = ref('');
    const password = ref('');
    const remember = ref(true);
    const submitting = ref(false);
    const message = ref('');

    const canSubmit = computed(() => email.value.trim() && password.value.trim());

    const submit = () => {
      if (!canSubmit.value) {
        message.value = '请输入邮箱和密码后再继续。';
        return;
      }

      submitting.value = true;
      message.value = '';

      setTimeout(() => {
        submitting.value = false;
        message.value = `欢迎回来，${email.value}。已为你准备好今日工作台。`;
      }, 900);
    };

    return {
      email,
      password,
      remember,
      submitting,
      message,
      canSubmit,
      submit,
    };
  },
  template: `
    <main class="page-shell">
      <section class="ambient ambient-left"></section>
      <section class="ambient ambient-right"></section>

      <div class="login-layout">
        <aside class="brand-panel">
          <div class="brand-chip">Nebula OS</div>
          <h1>连接你的团队，开启更高效的一天</h1>
          <p>
            在统一工作台中处理任务、查看指标并同步协作，让登录成为高效工作的开始。
          </p>

          <div class="feature-list">
            <article>
              <span>✦</span>
              <div>
                <strong>实时协作</strong>
                <p>会议纪要、项目进度和消息通知保持同步。</p>
              </div>
            </article>
            <article>
              <span>✦</span>
              <div>
                <strong>数据可视化</strong>
                <p>核心指标概览清晰呈现，决策更加从容。</p>
              </div>
            </article>
            <article>
              <span>✦</span>
              <div>
                <strong>安全可信</strong>
                <p>双层防护与设备识别机制守护账户安全。</p>
              </div>
            </article>
          </div>
        </aside>

        <section class="form-panel">
          <div class="panel-header">
            <div>
              <p class="eyebrow">欢迎回来</p>
              <h2>登录你的账号</h2>
            </div>
            <button class="ghost-button" type="button">切换语言</button>
          </div>

          <div class="social-grid">
            <button type="button">Google</button>
            <button type="button">GitHub</button>
          </div>

          <div class="divider"><span>或使用邮箱登录</span></div>

          <form class="login-form" @submit.prevent="submit">
            <label>
              邮箱地址
              <input v-model="email" type="email" placeholder="name@company.com" />
            </label>

            <label>
              登录密码
              <input v-model="password" type="password" placeholder="请输入至少 8 位密码" />
            </label>

            <div class="form-meta">
              <label class="checkbox-row">
                <input v-model="remember" type="checkbox" />
                <span>记住本设备</span>
              </label>
              <a href="#">忘记密码？</a>
            </div>

            <button class="primary-button" :disabled="!canSubmit || submitting" type="submit">
              <span v-if="submitting">登录中...</span>
              <span v-else>立即登录</span>
            </button>

            <p v-if="message" class="message">{{ message }}</p>
          </form>

          <footer class="panel-footer">
            还没有账号？<a href="#">立即创建</a>
          </footer>
        </section>
      </div>
    </main>
  `,
}).mount('#app');
