const summaryGrid = document.getElementById("summary-grid");
const summaryNote = document.getElementById("summary-note");
const actionResult = document.getElementById("action-result");
const cameraThumb = document.getElementById("camera-thumb");
const cameraMeta = document.getElementById("camera-meta");
const cameraFocusButton = document.getElementById("camera-focus-btn");
const controlsHint = document.getElementById("controls-hint");
const ratingHint = document.getElementById("rating-hint");

const controls = {
  start: document.getElementById("start-btn"),
  stop: document.getElementById("stop-btn"),
  discard: document.getElementById("discard-btn"),
  estop: document.getElementById("estop-btn"),
  clearFault: document.getElementById("clear-fault-btn"),
  quit: document.getElementById("quit-btn"),
};

const ratingButtons = {
  "1": document.getElementById("rating-1-btn"),
  "2": document.getElementById("rating-2-btn"),
  "3": document.getElementById("rating-3-btn"),
  "4": document.getElementById("rating-4-btn"),
};

const ratingPresets = {
  "1": {
    segment_status: "clean",
    success: "success",
    termination_reason: "goal_reached",
  },
  "2": {
    segment_status: "usable",
    success: "partial",
    termination_reason: "near_goal_stop",
  },
  "3": {
    segment_status: "usable",
    success: "fail",
    termination_reason: "operator_stop",
  },
  "4": {
    segment_status: "discard",
    success: "",
    termination_reason: "",
  },
};

const configFields = {
  sceneId: document.getElementById("cfg-scene-id"),
  operatorId: document.getElementById("cfg-operator-id"),
  instruction: document.getElementById("cfg-instruction"),
  taskFamily: document.getElementById("cfg-task-family"),
  targetType: document.getElementById("cfg-target-type"),
  targetDescription: document.getElementById("cfg-target-description"),
  collectorNotes: document.getElementById("cfg-collector-notes"),
  cmdVxMax: document.getElementById("cfg-cmd-vx-max"),
  cmdVyMax: document.getElementById("cfg-cmd-vy-max"),
  cmdWzMax: document.getElementById("cfg-cmd-wz-max"),
};

let statusCache = null;
let cameraStreamUrl = "";
let cameraReconnectTimer = null;
let cameraFocusMode = false;

const STATUS_POLL_INTERVAL_MS = 250;
const CAMERA_RECONNECT_DELAY_MS = 1000;

const inputHints = {
  wireless_controller: {
    controlsHint: "wireless controller + web buttons",
    ratingHint: "D-pad or 1-4",
    buttons: {
      start: ["Start", "A"],
      stop: ["Stop", "B"],
      discard: ["Discard", "X"],
      estop: ["E-Stop", "R2"],
      clearFault: ["Clear Fault", "Y"],
      quit: ["Quit", "UI"],
    },
  },
  evdev: {
    controlsHint: "keyboard + web buttons",
    ratingHint: "1-4 shortcuts",
    buttons: {
      start: ["Start", "R"],
      stop: ["Stop", "T"],
      discard: ["Discard", "ESC"],
      estop: ["E-Stop", "Space"],
      clearFault: ["Clear Fault", "C"],
      quit: ["Quit", "X"],
    },
  },
  tty: {
    controlsHint: "keyboard + web buttons",
    ratingHint: "1-4 shortcuts",
    buttons: {
      start: ["Start", "R"],
      stop: ["Stop", "T"],
      discard: ["Discard", "ESC"],
      estop: ["E-Stop", "Space"],
      clearFault: ["Clear Fault", "C"],
      quit: ["Quit", "X"],
    },
  },
};

function setLog(value) {
  if (actionResult) {
    actionResult.textContent = value;
  }
}

function setCameraFocusMode(enabled) {
  cameraFocusMode = enabled;
  document.body.classList.toggle("camera-focus-mode", enabled);
  cameraFocusButton.textContent = enabled ? "Exit Focus" : "Focus View";
  cameraFocusButton.setAttribute("aria-pressed", enabled ? "true" : "false");
}

function clearCameraReconnectTimer() {
  if (cameraReconnectTimer !== null) {
    window.clearTimeout(cameraReconnectTimer);
    cameraReconnectTimer = null;
  }
}

function stopCameraStream() {
  clearCameraReconnectTimer();
  cameraThumb.removeAttribute("src");
  cameraThumb.hidden = true;
  cameraStreamUrl = "";
}

function scheduleCameraReconnect() {
  if (cameraReconnectTimer !== null || !statusCache || !statusCache.robot.image_valid) {
    return;
  }
  cameraReconnectTimer = window.setTimeout(() => {
    cameraReconnectTimer = null;
    ensureCameraStream();
  }, CAMERA_RECONNECT_DELAY_MS);
}

function ensureCameraStream() {
  if (!statusCache || !statusCache.robot.image_valid) {
    stopCameraStream();
    return;
  }

  if (cameraStreamUrl) {
    return;
  }

  clearCameraReconnectTimer();
  const nextUrl = `/api/image/stream.mjpeg?t=${Date.now()}`;
  cameraStreamUrl = nextUrl;
  cameraThumb.src = nextUrl;
  cameraThumb.hidden = false;
}

function setButtonHint(button, label, shortcut) {
  const labelNode = button.querySelector(".btn-label");
  const shortcutNode = button.querySelector(".kbd");
  if (labelNode) {
    labelNode.textContent = label;
  }
  if (shortcutNode) {
    shortcutNode.textContent = shortcut;
  }
}

function renderItems(target, items) {
  target.innerHTML = items.map(([label, value]) => (
    `<div class="item"><span class="label">${label}</span><span class="value">${value ?? "-"}</span></div>`
  )).join("");
}

async function postJson(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return response.json();
}

async function postAction(url, body = {}) {
  const result = await postJson(url, body);
  setLog(JSON.stringify(result, null, 2));
  await refreshStatus();
  return result;
}

function syncConfigForm(status) {
  const cfg = status.run_context;
  configFields.sceneId.value = cfg.scene_id || "";
  configFields.operatorId.value = cfg.operator_id || "";
  configFields.instruction.value = cfg.instruction || "";
  configFields.taskFamily.value = cfg.task_family || "";
  configFields.targetType.value = cfg.target_type || "";
  configFields.targetDescription.value = cfg.target_description || "";
  configFields.collectorNotes.value = cfg.collector_notes || "";
  configFields.cmdVxMax.value = cfg.cmd_vx_max ?? "";
  configFields.cmdVyMax.value = cfg.cmd_vy_max ?? "";
  configFields.cmdWzMax.value = cfg.cmd_wz_max ?? "";
}

function setRatingButtonsDisabled(disabled) {
  Object.values(ratingButtons).forEach((button) => {
    button.disabled = disabled;
  });
}

function describeStopPhase(stopPhase) {
  switch (stopPhase) {
    case "waiting_release":
      return "waiting_release / release motion keys";
    case "finalizing":
      return "finalizing / settling before label";
    case "label_ready":
      return "label_ready / press 1-4";
    default:
      return stopPhase || "idle";
  }
}

function applyInputHints(inputBackend) {
  const hints = inputHints[inputBackend] || inputHints.tty;
  controlsHint.textContent = hints.controlsHint;
  ratingHint.textContent = hints.ratingHint;
  Object.entries(hints.buttons).forEach(([name, [label, shortcut]]) => {
    setButtonHint(controls[name], label, shortcut);
  });
}

async function submitRatingPreset(key) {
  if (!statusCache || !statusCache.actions.can_submit_label) {
    return;
  }
  const preset = ratingPresets[key];
  if (!preset) {
    return;
  }
  await postAction("/api/label/submit", preset);
}

function renderStatus(status) {
  statusCache = status;
  applyInputHints(status.process_config.input_backend);
  const robotState = status.robot.connected ? "online" : "offline";
  const cameraState = status.robot.image_valid
    ? `${Number(status.robot.image_age_s).toFixed(2)}s`
    : "waiting";
  renderItems(summaryGrid, [
    ["capture_state", status.collector.capture_state],
    ["safety_state", status.collector.safety_state],
    ["robot", robotState],
    ["camera", cameraState],
    ["recording", status.collector.recording ? "on" : "off"],
    ["cmd", `vx ${Number(status.command.vx).toFixed(2)} vy ${Number(status.command.vy).toFixed(2)} wz ${Number(status.command.wz).toFixed(2)}`],
  ]);
  const summaryHints = [];
  if (status.collector.fault_reason) {
    summaryHints.push(`fault: ${status.collector.fault_reason}`);
  } else if (status.collector.stop_phase && status.collector.stop_phase !== "idle") {
    summaryHints.push(`label flow: ${describeStopPhase(status.collector.stop_phase)}`);
  } else {
    const previewTag = status.process_config.preview_mode ? " / preview" : "";
    summaryHints.push(`${status.process_config.input_backend}${previewTag} / video ${String(status.process_config.video_poll_hz)}Hz`);
  }
  summaryNote.textContent = summaryHints.join("  |  ");

  syncConfigForm(status);

  controls.start.disabled = !status.actions.can_start_recording;
  controls.stop.disabled = !status.actions.can_stop_recording;
  controls.discard.disabled = !status.actions.can_discard_segment;
  controls.estop.disabled = !status.actions.can_estop;
  controls.clearFault.disabled = !status.actions.can_clear_fault;
  setRatingButtonsDisabled(!status.actions.can_submit_label);

  if (status.robot.image_valid) {
    if (cameraReconnectTimer !== null) {
      cameraThumb.hidden = true;
      cameraMeta.textContent = "stream reconnecting...";
    } else if (Number(status.robot.image_age_s) > 1.5) {
      stopCameraStream();
      cameraMeta.textContent = "stream stalled, reconnecting...";
      scheduleCameraReconnect();
    } else {
      ensureCameraStream();
      cameraThumb.hidden = false;
      cameraMeta.textContent = status.process_config.preview_mode
        ? `image age ${Number(status.robot.image_age_s).toFixed(2)}s / preview stream`
        : `image age ${Number(status.robot.image_age_s).toFixed(2)}s / live stream`;
    }
  } else {
    stopCameraStream();
    cameraMeta.textContent = "waiting for image...";
  }
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    renderStatus(status);
  } catch (error) {
    setLog(String(error));
  }
}

cameraThumb.addEventListener("load", () => {
  clearCameraReconnectTimer();
});

cameraThumb.addEventListener("error", () => {
  cameraMeta.textContent = "stream reconnecting...";
  cameraStreamUrl = "";
  scheduleCameraReconnect();
});

controls.start.addEventListener("click", async () => {
  await postAction("/api/control/start");
});
controls.stop.addEventListener("click", async () => {
  await postAction("/api/control/stop");
});
controls.discard.addEventListener("click", async () => {
  await postAction("/api/control/discard");
});
controls.estop.addEventListener("click", async () => {
  await postAction("/api/control/estop");
});
controls.clearFault.addEventListener("click", async () => {
  await postAction("/api/control/clear-fault");
});
controls.quit.addEventListener("click", async () => {
  const result = await postJson("/api/control/quit");
  setLog(JSON.stringify(result, null, 2));
});

Object.entries(ratingButtons).forEach(([key, button]) => {
  button.addEventListener("click", async () => {
    await submitRatingPreset(key);
  });
});

document.addEventListener("keydown", async (event) => {
  if (event.repeat) {
    return;
  }

  if (event.key === "Escape" && cameraFocusMode) {
    event.preventDefault();
    setCameraFocusMode(false);
    return;
  }

  const activeTag = document.activeElement ? document.activeElement.tagName : "";
  if (activeTag === "INPUT" || activeTag === "TEXTAREA" || activeTag === "SELECT") {
    return;
  }

  if (statusCache && statusCache.actions.can_submit_label && ["1", "2", "3", "4"].includes(event.key)) {
    event.preventDefault();
    await submitRatingPreset(event.key);
  }
});

cameraFocusButton.addEventListener("click", () => {
  setCameraFocusMode(!cameraFocusMode);
});

cameraThumb.addEventListener("dblclick", () => {
  setCameraFocusMode(!cameraFocusMode);
});

setInterval(refreshStatus, STATUS_POLL_INTERVAL_MS);
refreshStatus();
