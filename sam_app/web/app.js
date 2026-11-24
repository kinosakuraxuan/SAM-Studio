import { LivePreview } from "./live-preview.js?v=5";

const $ = (id) => document.getElementById(id);
const canvas = $("canvas"),
  ctx = canvas.getContext("2d");
const player = $("preview-player");
let media,
  base,
  overlay,
  styled,
  videoVariant,
  mode = "pos",
  at = 0,
  history = {},
  revision,
  exported,
  boxes = [],
  drag,
  pendingPoint,
  busy = false;
let staleAt = Infinity;
let originalVideoUrl = "";
let webPlayback = false;
let liveAvailable = false;
const msg = (text) => {
  if ($("status").textContent !== text) $("status").textContent = text;
};
const live = new LivePreview(msg);
const url = (path) => "/api/" + media.id + "/" + path;
async function api(path, json) {
  const opts =
    json === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(json),
        };
  const r = await fetch(path, opts);
  if (!r.ok) {
    const data = await r.json();
    throw Error(data.detail || r.statusText);
  }
  return r.json();
}
async function image(src) {
  const i = new Image();
  i.src = src;
  await i.decode();
  return i;
}
async function run(fn) {
  if (busy) return;
  busy = true;
  if (live.active) live.pause();
  const inputs = [
    "controls",
    "prompt-controls",
    "frame",
    "prev-frame",
    "next-frame",
    "file",
    "video-file",
    "examples-toggle",
    "edit-mode",
    "play-mode",
    "playback-source",
    "live-toggle",
    "live-effect",
  ];
  for (const id of inputs) $(id).disabled = true;
  try {
    await fn();
  } catch (e) {
    msg(e.message);
  } finally {
    busy = false;
    for (const id of inputs) $(id).disabled = false;
    $("controls").disabled = !media;
    $("prompt-controls").disabled = !media;
    $("prev-frame").disabled = !media?.video || at === 0;
    $("next-frame").disabled = !media?.video || at >= media.count - 1;
  }
}
const ops = () => history[at] || [];
function prompt(list) {
  const p = { points: [], labels: [], box: null };
  for (const op of list) {
    if (op.mode === "box") p.box = op.v;
    else {
      p.points.push(op.v);
      p.labels.push(op.mode === "pos" ? 1 : 0);
    }
  }
  return p;
}
function trackingRequest() {
  const keyframes = {};
  for (const [index, list] of Object.entries(history))
    if (list.length) keyframes[index] = prompt(list);
  return { version: $("version").value, keyframes };
}
function updateLiveOption() {
  $("playback-live").disabled = $("playback-live").hidden =
    !liveAvailable || !media?.video || !history[0]?.length;
}
function displayUnit() {
  return canvas.width / (canvas.getBoundingClientRect().width || canvas.width);
}
function fitCanvas() {
  if (!media || canvas.hidden) return;
  const surface = $("surface");
  if (!surface.clientWidth || !surface.clientHeight) return;
  const bounds = surface.getBoundingClientRect();
  const scales = {
    width: surface.clientWidth / media.w,
    height: surface.clientHeight / media.h,
    fit: Math.min(bounds.width / media.w, bounds.height / media.h),
  };
  const scale = scales[$("view").value];
  canvas.style.width = media.w * scale + "px";
  canvas.style.height = media.h * scale + "px";
}
function outline(b, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 * displayUnit();
  ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
}
function paint() {
  if (!base) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(styled || base, 0, 0);
  if (styled && !drag && !boxes.length) return;
  if (overlay) ctx.drawImage(overlay, 0, 0);
  for (const b of boxes) outline(b.box, "#ffc261");
  const p = prompt(ops());
  if (p.box) outline(p.box, "#74e7c4");
  if (drag) outline(drag, "white");
  p.points.forEach(([x, y], i) => {
    ctx.beginPath();
    ctx.arc(x, y, 5 * displayUnit(), 0, Math.PI * 2);
    ctx.fillStyle = p.labels[i] ? "#49e7aa" : "#ff746e";
    ctx.fill();
    ctx.strokeStyle = "white";
    ctx.lineWidth = 1.5 * displayUnit();
    ctx.stroke();
  });
}
function selectionToken() {
  return revision || (exported && at < staleAt ? exported : null);
}
function stylePath(kind) {
  return (
    "style/" +
    selectionToken() +
    "/" +
    at +
    "/" +
    $("style").value +
    "/" +
    kind +
    "?strength=" +
    $("strength").value
  );
}
function variantKey() {
  return [exported, $("style").value, $("strength").value].join("/");
}
function loadPlayback() {
  const kind = $("playback-source").value;
  const following = kind === "live";
  player.hidden = following;
  $("live-canvas").hidden = !following;
  $("live-controls").hidden = !following;
  if (following) {
    player.pause();
    $("stage").style.removeProperty("--playback-ratio");
    $("live-effect").value = $("style").value;
    live.open(
      url("live-preview"),
      JSON.stringify(trackingRequest()),
      media.count,
      $("style").value,
      $("strength").value,
    );
    return;
  }
  live.close();
  const source =
    kind === "tracked"
      ? url("export/" + exported + "/preview.mp4")
      : kind === "styled"
        ? url("video-style/" + variantKey() + "/styled.mp4")
        : originalVideoUrl;
  if (player.dataset.source === source) {
    if (player.error) playbackError();
    else if (player.readyState > 0) playbackReady();
    return;
  }
  player.dataset.source = source;
  delete player.dataset.compatible;
  $("stage").style.removeProperty("--playback-ratio");
  const query = new URLSearchParams({
    token: exported || "",
    style: $("style").value,
    strength: $("strength").value,
  });
  if (kind !== "styled") query.delete("style");
  player.dataset.fallback = url("playback/" + kind + "?" + query);
  if (!busy) msg("正在载入网页播放器…");
  player.src = source;
}
function showPlayback(playing) {
  webPlayback = playing;
  $("surface").hidden = playing;
  $("stage").classList.toggle("playing-video", playing);
  $("view-control").hidden = playing;
  $("timeline").hidden = playing || !media.video;
  $("prompt-controls").hidden = playing;
  $("playback-source").hidden = !playing;
  $("edit-mode").setAttribute("aria-pressed", !playing);
  $("play-mode").setAttribute("aria-pressed", playing);
  if (playing) {
    player.poster = url("frame/" + at);
    loadPlayback();
  } else {
    player.hidden = true;
    player.pause();
    live.close();
    $("live-canvas").hidden = true;
    $("live-controls").hidden = true;
    fitCanvas();
    paint();
  }
}
function playbackReady() {
  if (player.hidden || !player.videoWidth || !player.videoHeight) return;
  $("stage").style.setProperty(
    "--playback-ratio",
    String(player.videoWidth / player.videoHeight),
  );
  const label = $("playback-source").selectedOptions[0].textContent;
  if (!busy)
    msg(player.dataset.compatible ? "兼容播放（无声）" : "可播放" + label);
}
$("edit-mode").onclick = () =>
  run(async () => {
    if (!webPlayback) return;
    const wasLive = live.active;
    showPlayback(false);
    if (wasLive) await appearance();
    msg("可在当前帧添加提示");
  });
$("play-mode").onclick = () => {
  if (!webPlayback && !$("playback-live").disabled)
    $("playback-source").value = "live";
  showPlayback(true);
};
$("playback-source").onchange = loadPlayback;
player.onloadedmetadata = playbackReady;
function playbackError() {
  if (!player.hasAttribute("src") || player.hidden) return;
  if (player.dataset.compatible) {
    msg("视频预览未能载入，可切回逐帧编辑。");
    return;
  }
  player.dataset.compatible = "true";
  msg("正在准备兼容播放版本…");
  player.src = player.dataset.fallback;
}
player.onerror = playbackError;
async function appearance() {
  const token = selectionToken();
  styled = null;
  overlay = null;
  if (token) {
    if ($("style").value !== "original")
      styled = await image(url(stylePath("image")));
    else
      await tint(
        url(
          revision
            ? "result/" + revision + "/mask"
            : "tracked/" + exported + "/" + at,
        ),
      );
  }
  paint();
  links();
}
async function tint(src) {
  const im = await image(src),
    c = document.createElement("canvas");
  c.width = im.width;
  c.height = im.height;
  const co = c.getContext("2d");
  co.drawImage(im, 0, 0);
  const pix = co.getImageData(0, 0, c.width, c.height);
  for (let i = 0; i < pix.data.length; i += 4) {
    const on = pix.data[i] > 127;
    pix.data[i] = 44;
    pix.data[i + 1] = 218;
    pix.data[i + 2] = 155;
    pix.data[i + 3] = on ? 105 : 0;
  }
  co.putImageData(pix, 0, 0);
  overlay = c;
}
function links() {
  const el = $("links");
  el.replaceChildren();
  const link = (title, href, name) => {
    const a = document.createElement("a");
    a.href = url(href);
    a.textContent = title;
    a.download = name;
    el.append(a);
  };
  if (revision) {
    link("透明 PNG", "result/" + revision + "/cutout", "foreground.png");
    link("二值掩膜", "result/" + revision + "/mask", "mask.png");
  }
  const useStyle = $("style").value !== "original";
  const tracked = Boolean(exported && staleAt === Infinity);
  const styledVideo = tracked && useStyle && videoVariant === variantKey();
  updateLiveOption();
  $("playback-overlay").disabled = $("playback-overlay").hidden = !tracked;
  $("playback-styled").disabled = $("playback-styled").hidden = !styledVideo;
  if ($("playback-source").selectedOptions[0]?.disabled)
    $("playback-source").value = "original";
  if (webPlayback) loadPlayback();
  $("strength-controls").hidden = !useStyle;
  $("style-video").hidden = !(media && media.video && useStyle && tracked);
  if (useStyle && selectionToken() && styled) {
    link("风格合成 PNG", stylePath("image"), $("style").value + ".png");
    link(
      "风格透明前景",
      stylePath("cutout"),
      $("style").value + "-foreground.png",
    );
  }
  if (tracked) {
    link(
      "前景序列 ZIP",
      "export/" + exported + "/foreground.zip",
      "foreground.zip",
    );
    link("视频预览 MP4", "export/" + exported + "/preview.mp4", "preview.mp4");
    link("提示记录", "export/" + exported + "/prompts.json", "prompts.json");
    if (styledVideo) {
      const prefix = "video-style/" + variantKey() + "/";
      link("风格 MP4", prefix + "styled.mp4", "styled.mp4");
      link("无损风格帧 ZIP", prefix + "styled-frames.zip", "styled-frames.zip");
    }
  }
  if (!el.children.length) el.textContent = "选择目标后可导出。";
}
async function frame() {
  base = await image(url("frame/" + at));
  canvas.width = media.w;
  canvas.height = media.h;
  fitCanvas();
  overlay = null;
  styled = null;
  revision = null;
  boxes = [];
  $("choices").replaceChildren();
  $("framenum").textContent = at + 1 + " / " + media.count;
  if (!selectionToken() && ops().length) await segment();
  else await appearance();
}
async function segment() {
  overlay = null;
  styled = null;
  revision = null;
  links();
  paint();
  if (!ops().length) {
    msg("当前帧没有提示");
    return;
  }
  msg("正在计算分割…");
  const r = await api(url("segment"), {
    frame: at,
    version: $("version").value,
    prompt: prompt(ops()),
  });
  revision = r.revision;
  await appearance();
  msg(
    liveAvailable && media.video && at === 0
      ? "轮廓已更新，点「网页播放」查看跟随效果。"
      : "轮廓已更新",
  );
}
async function edit(op) {
  if (op) (history[at] ||= []).push(op);
  staleAt = Math.min(staleAt, at);
  boxes = [];
  $("choices").replaceChildren();
  await segment();
}
async function openMedia(f, video = false) {
  if (!f) return;
  msg(video ? "正在逐帧读取视频，完成后会显示时间轴…" : "正在读取素材…");
  const form = new FormData();
  form.append("file", f);
  const r = await fetch("/api/open", { method: "POST", body: form });
  const data = await r.json();
  if (!r.ok) throw Error(data.detail);
  live.reset();
  player.hidden = true;
  player.removeAttribute("src");
  player.removeAttribute("poster");
  player.load();
  delete player.dataset.source;
  if (originalVideoUrl) URL.revokeObjectURL(originalVideoUrl);
  originalVideoUrl = data.video ? URL.createObjectURL(f) : "";
  $("playback-source").value = "original";
  media = data;
  base = null;
  at = 0;
  history = {};
  exported = null;
  videoVariant = null;
  $("style").value = "original";
  $("strength").value = "100";
  $("strength-value").textContent = "100%";
  staleAt = Infinity;
  $("filename").textContent = f.name;
  $("filename").title = f.name;
  $("dimensions").textContent =
    (media.video ? "视频" : "图片") + " · " + media.w + " × " + media.h;
  $("stage").style.setProperty("--media-ratio", String(media.w / media.h));
  $("stage").style.removeProperty("--playback-ratio");
  $("stage").classList.add("has-media");
  document.body.classList.add("editing");
  $("view").value = "fit";
  $("stage").dataset.view = "fit";
  $("surface").removeAttribute("tabindex");
  $("surface").scrollTo(0, 0);
  drag = pendingPoint = null;
  canvas.width = media.w;
  canvas.height = media.h;
  canvas.hidden = false;
  $("empty").hidden = true;
  $("video-toolbar").hidden = !media.video;
  showPlayback(false);
  $("track").hidden = !media.video;
  $("frame").max = media.count - 1;
  $("frame").value = 0;
  await frame();
  msg("请选择目标");
}
function handleFileSelection(event) {
  const input = event.currentTarget;
  const file = input.files[0];
  input.value = "";
  if (!file) return;
  if (
    input.id === "video-file" &&
    !input.accept
      .split(",")
      .some((ext) => file.name.toLowerCase().endsWith(ext))
  ) {
    msg("请选择 MP4、MOV、AVI、MKV、WebM 或 M4V 视频。");
    return;
  }
  return run(() => openMedia(file, input.id === "video-file"));
}
$("file").onchange = handleFileSelection;
$("video-file").onchange = handleFileSelection;
async function openSample(video) {
  const label = video ? "示例视频" : "示例图";
  msg("正在载入" + label + "…");
  const response = await fetch(
    video ? "/assets/car-turn.mp4" : "/assets/camel.jpg",
  );
  if (!response.ok) throw new Error(label + "读取失败，请重试。");
  const file = new File(
    [await response.blob()],
    video ? "汽车转弯示例.mp4" : "骆驼示例.jpg",
    {
      type: video ? "video/mp4" : "image/jpeg",
    },
  );
  await openMedia(file, video);
  if (video)
    msg(
      liveAvailable
        ? "在首帧框选汽车，再点击「网页播放」查看跟随效果。"
        : "在首帧框选汽车，再点击「传播并导出视频」。",
    );
}
function toggleExamples(open) {
  $("example-options").hidden = !open;
  $("examples-toggle").setAttribute("aria-expanded", open);
}
$("examples-toggle").onclick = () =>
  toggleExamples($("example-options").hidden);
document.addEventListener("pointerdown", (event) => {
  if (!$("examples").contains(event.target)) toggleExamples(false);
});
$("examples").addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    toggleExamples(false);
    $("examples-toggle").focus();
  }
});
$("sample").onclick = () => {
  toggleExamples(false);
  return run(() => openSample(false));
};
$("sample-video").onclick = () => {
  toggleExamples(false);
  return run(() => openSample(true));
};
if (new URLSearchParams(location.search).get("sample") === "video") {
  $("sample-video").click();
}
new ResizeObserver(() => {
  if (!base || canvas.hidden) return;
  fitCanvas();
  paint();
}).observe($("surface"));
$("view").onchange = () => {
  const surface = $("surface"),
    rect = canvas.getBoundingClientRect(),
    viewport = surface.getBoundingClientRect();
  const x = (viewport.left + surface.clientWidth / 2 - rect.left) / rect.width;
  const y = (viewport.top + surface.clientHeight / 2 - rect.top) / rect.height;
  const view = $("view").value;
  const zoomed = view !== "fit";
  $("stage").dataset.view = view;
  if (zoomed) surface.tabIndex = 0;
  else surface.removeAttribute("tabindex");
  fitCanvas();
  const updated = canvas.getBoundingClientRect();
  surface.scrollTo(
    zoomed ? updated.width * x - surface.clientWidth / 2 : 0,
    zoomed ? updated.height * y - surface.clientHeight / 2 : 0,
  );
  paint();
};
$("surface").addEventListener(
  "wheel",
  (e) => {
    if ($("view").value !== "height" || e.deltaX || e.ctrlKey || e.shiftKey)
      return;
    const surface = e.currentTarget;
    const scale = [1, 16, surface.clientWidth][e.deltaMode];
    const before = surface.scrollLeft;
    surface.scrollLeft += e.deltaY * scale;
    if (surface.scrollLeft !== before) e.preventDefault();
  },
  { passive: false },
);
const seek = (index) =>
  run(async () => {
    at = index;
    $("frame").value = at;
    await frame();
    msg("可在当前帧添加提示");
  });
$("frame").onchange = (e) => seek(Number(e.currentTarget.value));
$("prev-frame").onclick = () => seek(at - 1);
$("next-frame").onclick = () => seek(at + 1);
$("modes").onclick = (e) => {
  const tool = e.target.closest("button[data-mode]");
  if (!tool) return;
  mode = tool.dataset.mode;
  $("stage").dataset.tool = mode;
  for (const b of $("modes").children)
    b.setAttribute("aria-pressed", b === tool);
};
function xy(e) {
  const r = canvas.getBoundingClientRect();
  return [
    Math.min(
      canvas.width - 1,
      Math.max(0, ((e.clientX - r.left) * canvas.width) / r.width),
    ),
    Math.min(
      canvas.height - 1,
      Math.max(0, ((e.clientY - r.top) * canvas.height) / r.height),
    ),
  ];
}
canvas.onpointerdown = (e) => {
  if (busy || !base || !e.isPrimary || e.button !== 0) return;
  canvas.setPointerCapture(e.pointerId);
  const p = xy(e);
  if (mode === "box") drag = [...p, ...p];
  else pendingPoint = { mode, v: p, x: e.clientX, y: e.clientY };
};
canvas.onpointermove = (e) => {
  if (!e.isPrimary) return;
  if (
    pendingPoint &&
    Math.hypot(e.clientX - pendingPoint.x, e.clientY - pendingPoint.y) > 6
  )
    pendingPoint = null;
  if (drag) {
    drag = [drag[0], drag[1], ...xy(e)];
    paint();
  }
};
canvas.onpointercancel = (e) => {
  if (!e.isPrimary) return;
  drag = pendingPoint = null;
  paint();
};
canvas.onpointerup = (e) => {
  if (!e.isPrimary) return;
  if (pendingPoint) {
    const { mode, v } = pendingPoint;
    pendingPoint = null;
    run(() => edit({ mode, v }));
  }
  if (!drag) return;
  const [x, y] = drag,
    [r, b] = xy(e);
  drag = null;
  paint();
  if (Math.abs(r - x) > 1 && Math.abs(b - y) > 1)
    run(() =>
      edit({
        mode: "box",
        v: [Math.min(x, r), Math.min(y, b), Math.max(x, r), Math.max(y, b)],
      }),
    );
};
$("undo").onclick = () =>
  run(async () => {
    ops().pop();
    await edit();
  });
$("clear").onclick = () =>
  run(async () => {
    history[at] = [];
    await edit();
  });
$("version").onchange = () =>
  run(async () => {
    if (webPlayback) showPlayback(false);
    exported = null;
    await segment();
  });
function changeEffect() {
  if (live.active) {
    $("live-effect").value = $("style").value;
    live.effect($("style").value, $("strength").value);
    links();
    return;
  }
  return run(async () => {
    await appearance();
    msg(selectionToken() ? "选区预览已更新" : "请先选择目标");
  });
}
$("style").onchange = changeEffect;
$("live-effect").onchange = () => {
  $("style").value = $("live-effect").value;
  changeEffect();
};
$("strength").oninput = () => {
  $("strength-value").textContent = $("strength").value + "%";
};
$("strength").onchange = () => (live.active ? changeEffect() : run(appearance));
$("style-video").onclick = () =>
  run(async () => {
    msg("正在逐帧编辑所选目标…");
    await api(url("video-style/" + exported), {
      style: $("style").value,
      strength: Number($("strength").value),
    });
    videoVariant = variantKey();
    links();
    msg("风格视频和无损帧序列已生成");
  });
$("detect").onclick = () =>
  run(async () => {
    if (webPlayback) showPlayback(false);
    msg("正在定位文字描述的目标…");
    boxes = await api(url("detect"), { frame: at, text: $("text").value });
    $("choices").replaceChildren();
    boxes.forEach((b, i) => {
      const btn = document.createElement("button");
      btn.textContent =
        "候选 " + (i + 1) + " · " + Math.round(b.score * 100) + "%";
      btn.onclick = () => run(() => edit({ mode: "box", v: b.box }));
      $("choices").append(btn);
    });
    paint();
    msg(boxes.length ? "请选择候选目标" : "没有匹配，试试其他词或手动框选。");
  });
$("text").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.isComposing) $("detect").click();
});
$("track").onclick = () =>
  run(async () => {
    const request = trackingRequest();
    if (!request.keyframes[0]) throw Error("请先在首帧标记目标。");
    msg("正在传播掩膜并编码视频…");
    const r = await api(url("propagate"), request);
    exported = r.export;
    staleAt = Infinity;
    await frame();
    msg("视频已生成，可浏览各帧或下载。");
  });
api("/api/status")
  .then((s) => {
    liveAvailable = s.live_preview === true;
    updateLiveOption();
    $("hardware").textContent = s.gpu;
    $("hardware").parentElement.dataset.ready = "true";
    $("tuned").disabled = !s.tuned;
    $("refined").disabled = !s.refined;
  })
  .catch((e) => {
    $("hardware").textContent = "服务未连接";
    msg(e.message);
  });
