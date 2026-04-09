const summaryGrid = document.getElementById("summary-grid");
const robotGrid = document.getElementById("robot-grid");
const processGrid = document.getElementById("process-grid");
const actionResult = document.getElementById("action-result");
const configForm = document.getElementById("config-form");
const cameraThumb = document.getElementById("camera-thumb");
const cameraMeta = document.getElementById("camera-meta");

const controls = {
  start: document.getElementById("start-btn"),
  stop: document.getElementById("stop-btn"),
  discard: document.getElementById("discard-btn"),
  estop: document.getElementById("estop-btn"),
  clearFault: document.getElementById("clear-fault-btn"),
  quit: document.getElementById("quit-btn"),
  applyConfig: document.getElementById("apply-config-btn"),
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
  captureMode: document.getElementById("cfg-capture-mode"),
  taskFamily: document.getElementById("cfg-task-family"),
  targetType: document.getElementById("cfg-target-type"),
  targetDescription: document.getElementById("cfg-target-description"),
  collectorNotes: document.getElementById("cfg-collector-notes"),
  cmdVxMax: document.getElementById("cfg-cmd-vx-max"),
  cmdVyMax: document.getElementById("cfg-cmd-vy-max"),
  cmdWzMax: document.getElementById("cfg-cmd-wz-max"),
};

let statusCache = null;
let lastImageRefreshAt = 0;
let configDirty = false;
let applyInFlight = false;

function setLog(value) {
  actionResult.textContent = value;
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

function currentConfigPayload() {
  return {
    scene_id: configFields.sceneId.value,
    operator_id: configFields.operatorId.value,
    instruction: configFields.instruction.value,
    capture_mode: configFields.captureMode.value,
    task_family: configFields.taskFamily.value,
    target_type: configFields.targetType.value,
    target_description: configFields.targetDescription.value,
    collector_notes: configFields.collectorNotes.value,
    cmd_vx_max: Number(configFields.cmdVxMax.value),
    cmd_vy_max: Number(configFields.cmdVyMax.value),
    cmd_wz_max: Number(configFields.cmdWzMax.value),
  };
}

async function applyConfigFromForm() {
  applyInFlight = true;
  try {
    const result = await postJson("/api/config/update", currentConfigPayload());
    if (result.ok) {
      configDirty = false;
    }
    setLog(JSON.stringify(result, null, 2));
    await refreshStatus();
    return result;
  } finally {
    applyInFlight = false;
  }
}

function syncConfigForm(status) {
  if (configDirty || applyInFlight || configForm.contains(document.activeElement)) {
    return;
  }
  const cfg = status.run_context;
  configFields.sceneId.value = cfg.scene_id || "";
  configFields.operatorId.value = cfg.operator_id || "";
  configFields.instruction.value = cfg.instruction || "";
  configFields.captureMode.value = cfg.capture_mode || "single_action";
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

async function submitRatingPreset(key) {
  if (!statusCache || !statusCache.actions.can_submit_label) {
    return;
  }
  const preset = ratingPresets[key];
  if (!preset) {
    return;
  }
  const result = await postJson("/api/label/submit", preset);
  setLog(JSON.stringify(result, null, 2));
  refreshStatus();
}

function renderStatus(status) {
  statusCache = status;
  renderItems(summaryGrid, [
    ["capture_state", status.collector.capture_state],
    ["startup_gate", status.startup_gate.active ? "locked / apply config first" : "ready"],
    ["label_flow", describeStopPhase(status.collector.stop_phase)],
    ["safety_state", status.collector.safety_state],
    ["recording", String(status.collector.recording)],
    ["seg_s", String(status.collector.segment_duration_s)],
    ["fault_reason", status.collector.fault_reason || "-"],
  ]);

  renderItems(robotGrid, [
    ["robot_connected", String(status.robot.connected)],
    ["state_valid", String(status.robot.state_valid)],
    ["cmd", `vx ${Number(status.command.vx).toFixed(2)}  vy ${Number(status.command.vy).toFixed(2)}  wz ${Number(status.command.wz).toFixed(2)}`],
    ["body_height", String(status.robot.body_height)],
    ["roll", String(status.robot.roll)],
    ["pitch", String(status.robot.pitch)],
    ["yaw", String(status.robot.yaw)],
  ]);

  renderItems(processGrid, [
    ["network_interface", status.process_config.network_interface],
    ["input_backend", status.process_config.input_backend],
    ["capture_mode", status.run_context.capture_mode],
    ["output_dir", status.process_config.output_dir],
    ["defaults_path", status.process_config.defaults_path || "-"],
  ]);

  syncConfigForm(status);

  controls.start.disabled = status.startup_gate.active || !status.actions.can_start_recording;
  controls.stop.disabled = !status.actions.can_stop_recording;
  controls.discard.disabled = !status.actions.can_discard_segment;
  controls.estop.disabled = !status.actions.can_estop;
  controls.clearFault.disabled = !status.actions.can_clear_fault;
  setRatingButtonsDisabled(!status.actions.can_submit_label);

  if (status.robot.image_valid) {
    const now = Date.now();
    if (now - lastImageRefreshAt > 700) {
      cameraThumb.src = `/api/image/latest.jpg?t=${now}`;
      lastImageRefreshAt = now;
    }
    cameraThumb.hidden = false;
    cameraMeta.textContent = `image age ${Number(status.robot.image_age_s).toFixed(2)}s`;
  } else {
    cameraThumb.hidden = true;
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

document.getElementById("config-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await applyConfigFromForm();
});

Object.values(configFields).forEach((field) => {
  const markDirty = () => {
    configDirty = true;
  };
  field.addEventListener("input", markDirty);
  field.addEventListener("change", markDirty);
});

controls.start.addEventListener("click", async () => {
  setLog(JSON.stringify(await postJson("/api/control/start"), null, 2));
  refreshStatus();
});
controls.stop.addEventListener("click", async () => {
  setLog(JSON.stringify(await postJson("/api/control/stop"), null, 2));
  refreshStatus();
});
controls.discard.addEventListener("click", async () => {
  setLog(JSON.stringify(await postJson("/api/control/discard"), null, 2));
  refreshStatus();
});
controls.estop.addEventListener("click", async () => {
  setLog(JSON.stringify(await postJson("/api/control/estop"), null, 2));
  refreshStatus();
});
controls.clearFault.addEventListener("click", async () => {
  setLog(JSON.stringify(await postJson("/api/control/clear-fault"), null, 2));
  refreshStatus();
});
controls.quit.addEventListener("click", async () => {
  setLog(JSON.stringify(await postJson("/api/control/quit"), null, 2));
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
  const activeTag = document.activeElement ? document.activeElement.tagName : "";
  if (activeTag === "INPUT" || activeTag === "TEXTAREA" || activeTag === "SELECT") {
    return;
  }

  if (event.key === "o" || event.key === "O") {
    event.preventDefault();
    await applyConfigFromForm();
    return;
  }

  if (statusCache && statusCache.actions.can_submit_label && ["1", "2", "3", "4"].includes(event.key)) {
    event.preventDefault();
    await submitRatingPreset(event.key);
  }
});

setInterval(refreshStatus, 500);
refreshStatus();
