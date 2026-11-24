"use strict";

const assets = "./assets/";
const subjects = {
  car: { label: "汽车", sequence: "car-turn" },
  person: { label: "人物", sequence: "tennis" },
};
const styleLabels = { cartoon: "卡通", sketch: "线稿" };
const videoLabels = { original: "原片", tracking: "传播掩膜", cartoon: "卡通视频", sketch: "线稿视频" };

const player = document.querySelector("#player");
const compare = document.querySelector("#compare");
const split = document.querySelector("#split");
const original = document.querySelector("#original");
const edited = document.querySelector("#edited");

let subject = "car";
let style = "cartoon";
let videoVersion = "original";
let videoChange = 0;
let resumePosition = 0;
let resumePlayback = false;

function selectButtons(attribute, value, handler) {
  const buttons = Array.from(document.querySelectorAll("[data-" + attribute + "]"));
  for (const button of buttons) {
    button.addEventListener("click", () => {
      const choice = button.dataset[attribute];
      if (choice === value()) return;
      for (const item of buttons) {
        item.setAttribute("aria-pressed", String(item === button));
      }
      handler(choice);
    });
  }
}

let activePanel = "image";
selectButtons("panel", () => activePanel, (panel) => {
  document.querySelector("#" + activePanel + "-panel").hidden = true;
  if (activePanel === "video") {
    player.pause();
    resumePlayback = false;
  }
  document.querySelector("#" + panel + "-panel").hidden = false;
  activePanel = panel;
});

function renderImage() {
  const { label, sequence } = subjects[subject];
  const file = assets + subject + "-" + style + ".webp";
  original.src = assets + subject + "-original.webp";
  original.alt = label + "示例的原图";
  edited.src = file;
  edited.alt = label + "选区" + styleLabels[style] + "结果";
  document.querySelector("#view-label").textContent = styleLabels[style];
  document.querySelector("#image-caption").textContent = "DAVIS · " + sequence;
  document.querySelector("#image-download").href = file;
}

selectButtons("subject", () => subject, (choice) => {
  subject = choice;
  renderImage();
});
selectButtons("style", () => style, (choice) => {
  style = choice;
  renderImage();
});
split.addEventListener("input", () => {
  compare.style.setProperty("--split", split.value + "%");
});

selectButtons("video", () => videoVersion, (choice) => {
  if (player.readyState >= HTMLMediaElement.HAVE_METADATA) {
    resumePosition = player.currentTime;
    resumePlayback = !player.paused && !player.ended;
  }
  const change = ++videoChange;
  const path = assets + "video-" + choice + ".mp4";
  videoVersion = choice;
  player.addEventListener("loadedmetadata", () => {
    if (change !== videoChange) return;
    player.currentTime = Math.min(resumePosition, player.duration);
    if (resumePlayback) player.play().catch(() => {});
  }, { once: true });
  player.src = path;
  document.querySelector("#video-caption").textContent = "DAVIS · car-turn · " + videoLabels[choice];
  document.querySelector("#video-download").href = path;
});
