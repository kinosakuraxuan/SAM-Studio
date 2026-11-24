export class LivePreview {
  constructor(status) {
    this.status = status;
    this.canvas = document.getElementById("live-canvas");
    this.context = this.canvas.getContext("2d");
    this.button = document.getElementById("live-toggle");
    this.position = document.getElementById("live-position");
    this.countLabel = document.getElementById("live-count");
    this.button.onclick = () => {
      if (this.playing) this.pause();
      else {
        this.playing = true;
        this.render(this.index + 1 < this.count ? this.index + 1 : 0);
      }
    };
    this.position.onchange = () => {
      const index = Math.min(Number(this.position.value), this.ready - 1);
      if (index >= 0) this.render(index);
    };
    this.reset();
  }

  reset() {
    this.close();
    this.key = null;
    this.job = null;
    this.ready = 0;
    this.index = -1;
    this.canvas.width = 1;
    this.canvas.height = 1;
  }

  open(path, body, count, style, strength) {
    const key = path + body;
    if (this.active && this.key === key) return;
    if (this.key !== key) this.reset();
    Object.assign(this, {
      key,
      path,
      body,
      count,
      style,
      strength,
      active: true,
    });
    this.playing = true;
    this.render(this.index >= 0 && this.index + 1 < count ? this.index : 0);
  }

  close() {
    this.active = false;
    this.pause();
  }

  pause() {
    this.playing = false;
    this.request?.abort();
    if (this.active) {
      this.drawControls();
      this.status("跟随预览已暂停");
    }
  }

  effect(style, strength) {
    if (style === this.style && strength === this.strength) return;
    this.style = style;
    this.strength = strength;
    if (this.active) this.render(Math.max(0, this.index));
  }

  drawControls() {
    this.button.textContent = this.playing ? "暂停" : "播放";
    this.position.disabled = this.ready < 2;
    this.position.max = this.count - 1;
    this.position.value = Math.max(0, this.index);
    this.position.dataset.ready = this.ready;
    this.position.setAttribute(
      "aria-valuetext",
      `第 ${this.index + 1} 帧，共 ${this.count} 帧，已处理 ${this.ready} 帧`,
    );
    this.countLabel.textContent = `${this.index + 1} / ${this.count} 帧`;
  }

  async render(index, due = 0) {
    this.request?.abort();
    this.request = new AbortController();
    const { signal } = this.request;
    this.drawControls();
    try {
      if (!this.job) {
        this.status("正在启动选区跟踪…");
        const response = await fetch(this.path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: this.body,
          signal,
        });
        const job = await response.json();
        if (!response.ok) throw Error(job.detail || "无法启动选区跟踪");
        signal.throwIfAborted();
        this.job = job;
        this.ready = job.ready;
      }
      const query = new URLSearchParams({
        effect: this.style === "original" ? "mask" : this.style,
        strength: this.strength,
      });
      const response = await fetch(
        `${this.path}/${this.job.id}/${index}?${query}`,
        { signal },
      );
      if (!response.ok) {
        const error = await response.json().catch(() => null);
        throw Error(error?.detail || "无法显示跟随预览");
      }
      const bitmap = await createImageBitmap(await response.blob());
      try {
        const delay = due - performance.now();
        if (delay > 0 && !signal.aborted)
          await new Promise((resolve) => {
            const timer = setTimeout(resolve, delay);
            signal.addEventListener(
              "abort",
              () => {
                clearTimeout(timer);
                resolve();
              },
              { once: true },
            );
          });
        signal.throwIfAborted();
        this.canvas.width = bitmap.width;
        this.canvas.height = bitmap.height;
        this.context.drawImage(bitmap, 0, 0);
      } finally {
        bitmap.close();
      }
      this.index = index;
      this.ready = Number(response.headers.get("X-Preview-Ready"));
      const next = index + 1;
      if (next >= this.count) this.playing = false;
      this.drawControls();
      this.status(
        this.playing
          ? "正在跟随选区"
          : next >= this.count
            ? "跟随预览已到片尾"
            : "跟随预览已暂停",
      );
      if (this.playing) {
        const seconds = this.job.times[next] - this.job.times[index];
        this.render(
          next,
          performance.now() +
            1000 * (seconds > 0 ? seconds : this.job.interval),
        );
      }
    } catch (error) {
      if (signal.aborted) return;
      this.job = null;
      this.ready = 0;
      this.index = -1;
      this.playing = false;
      this.drawControls();
      this.status(error.message);
    }
  }
}
