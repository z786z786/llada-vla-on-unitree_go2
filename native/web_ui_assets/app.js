const summaryGrid = document.getElementById("summary-grid");
const robotGrid = document.getElementById("robot-grid");
const processGrid = document.getElementById("process-grid");
const pendingMeta = document.getElementById("pending-meta");
const actionResult = document.getElementById("action-result");
const configForm = document.getElementById("config-form");

const controls = {
  start: document.getElementById("start-btn"),
  stop: document.getElementById("stop-btn"),
  discard: document.getElementById("discard-btn"),
  estop: document.getElementById("estop-btn"),
  clearFault: document.getElementById("clear-fault-btn"),
  submitLabel: document.getElementById("submit-label-btn"),
  applyConfig: document.getElementById("apply-config-btn"),
  saveDefaults: document.getElementById("save-defaults-btn"),
};

const labelFields = {
  segmentStatus: document.querySelectorAll('input[name="segment-status"]'),
  success: document.querySelectorAll('input[name="success"]'),
  terminationReason: document.querySelectorAll('input[name="termination-reason"]'),
};

const configFields = {
  sceneId: document.getElementById("cfg-scene-id"),
  operatorId: document.getElementById("cfg-operator-id"),
  instruction: document.getElementById("cfg-instruction"),
  captureMode: document.getElementById("cfg-capture-mode"),
  taskFamily: document.getElementById("cfg-task-family"),
  targetType: document.getElementById("cfg-target-type"),
  targetDescription: document.getElementById("cfg-target-description"),
  targetInstanceId: document.getElementById("cfg-target-instance-id"),
  taskTagsCsv: document.getElementById("cfg-task-tags-csv"),
  collectorNotes: document.getElementById("cfg-collector-notes"),
  cmdVxMax: document.getElementById("cfg-cmd-vx-max"),
  cmdVyMax: document.getElementById("cfg-cmd-vy-max"),
  cmdWzMax: document.getElementById("cfg-cmd-wz-max"),
};

let statusCache = null;

function setLog(value) {
  actionResult.textContent = value;
}

function renderItems(target, items) {
  target.innerHTML = items.map(([label, value]) => (
    `<div class="item"><span class="label">${label}</span><span class="value">${value ?? "-"}</span></div>`
  )).join("");
}

function getCheckedValue(nodeList) {
  const node = Array.from(nodeList).find((item) => item.checked);
  return node ? node.value : "";
}

function setCheckedValue(nodeList, value) {
  Array.from(nodeList).forEach((item) => {
    item.checked = item.value === value;
  });
}

function setRadioDisabled(nodeList, disabled) {
  Array.from(nodeList).forEach((item) => {
    item.disabled = disabled;
  });
}

async function postJson(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return response.json();
}

function syncConfigForm(status) {
  if (configForm.contains(document.activeElement)) {
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
  configFields.targetInstanceId.value = cfg.target_instance_id || "";
  configFields.taskTagsCsv.value = cfg.task_tags_csv || "";
  configFields.collectorNotes.value = cfg.collector_notes || "";
  configFields.cmdVxMax.value = cfg.cmd_vx_max ?? "";
  configFields.cmdVyMax.value = cfg.cmd_vy_max ?? "";
  configFields.cmdWzMax.value = cfg.cmd_wz_max ?? "";
}

function renderStatus(status) {
  statusCache = status;
  renderItems(summaryGrid, [
    ["capture_state", status.collector.capture_state],
    ["safety_state", status.collector.safety_state],
    ["recording", String(status.collector.recording)],
    ["seg_s", String(status.collector.segment_duration_s)],
    ["frames", String(status.collector.buffered_frames)],
    ["wait_label", status.pending_label.active ? "yes" : "no"],
    ["pending", String(status.pending_label.buffered_frames)],
    ["fault_reason", status.collector.fault_reason || "-"],
  ]);

  renderItems(robotGrid, [
    ["robot_connected", String(status.robot.connected)],
    ["state_valid", String(status.robot.state_valid)],
    ["image_valid", String(status.robot.image_valid)],
    ["state_age_s", String(status.robot.state_age_s)],
    ["image_age_s", String(status.robot.image_age_s)],
    ["body_height", String(status.robot.body_height)],
    ["roll", String(status.robot.roll)],
    ["pitch", String(status.robot.pitch)],
    ["yaw", String(status.robot.yaw)],
    ["command_vx", String(status.command.vx)],
    ["command_vy", String(status.command.vy)],
    ["command_wz", String(status.command.wz)],
  ]);

  renderItems(processGrid, [
    ["network_interface", status.process_config.network_interface],
    ["output_dir", status.process_config.output_dir],
    ["loop_hz", String(status.process_config.loop_hz)],
    ["video_poll_hz", String(status.process_config.video_poll_hz)],
    ["input_backend", status.process_config.input_backend],
    ["input_device", status.process_config.input_device || "-"],
    ["defaults_path", status.process_config.defaults_path || "-"],
  ]);

  renderItems(pendingMeta, [
    ["waiting_for_label", status.pending_label.active ? "yes" : "no"],
    ["frames_in_pending_segment", String(status.pending_label.buffered_frames)],
  ]);

  syncConfigForm(status);

  controls.start.disabled = !status.actions.can_start_recording;
  controls.stop.disabled = !status.actions.can_stop_recording;
  controls.discard.disabled = !status.actions.can_discard_segment;
  controls.estop.disabled = !status.actions.can_estop;
  controls.clearFault.disabled = !status.actions.can_clear_fault;
  controls.submitLabel.disabled = !status.actions.can_submit_label;

  const discardSelected = getCheckedValue(labelFields.segmentStatus) === "discard";
  setRadioDisabled(labelFields.segmentStatus, !status.actions.can_submit_label);
  setRadioDisabled(labelFields.success, discardSelected || !status.actions.can_submit_label);
  setRadioDisabled(labelFields.terminationReason, discardSelected || !status.actions.can_submit_label);
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
  const result = await postJson("/api/config/update", {
    scene_id: configFields.sceneId.value,
    operator_id: configFields.operatorId.value,
    instruction: configFields.instruction.value,
    capture_mode: configFields.captureMode.value,
    task_family: configFields.taskFamily.value,
    target_type: configFields.targetType.value,
    target_description: configFields.targetDescription.value,
    target_instance_id: configFields.targetInstanceId.value,
    task_tags_csv: configFields.taskTagsCsv.value,
    collector_notes: configFields.collectorNotes.value,
    cmd_vx_max: Number(configFields.cmdVxMax.value),
    cmd_vy_max: Number(configFields.cmdVyMax.value),
    cmd_wz_max: Number(configFields.cmdWzMax.value),
  });
  setLog(JSON.stringify(result, null, 2));
  refreshStatus();
});

controls.saveDefaults.addEventListener("click", async () => {
  const result = await postJson("/api/config/save-defaults");
  setLog(JSON.stringify(result, null, 2));
  refreshStatus();
});

document.getElementById("label-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const result = await postJson("/api/label/submit", {
    segment_status: getCheckedValue(labelFields.segmentStatus),
    success: getCheckedValue(labelFields.success),
    termination_reason: getCheckedValue(labelFields.terminationReason),
  });
  setLog(JSON.stringify(result, null, 2));
  if (result.ok) {
    setCheckedValue(labelFields.segmentStatus, "clean");
    setCheckedValue(labelFields.success, "success");
    setCheckedValue(labelFields.terminationReason, "goal_reached");
  }
  refreshStatus();
});

Array.from(labelFields.segmentStatus).forEach((item) => {
  item.addEventListener("change", () => {
    const discardSelected = getCheckedValue(labelFields.segmentStatus) === "discard";
    setRadioDisabled(labelFields.success, discardSelected);
    setRadioDisabled(labelFields.terminationReason, discardSelected);
  });
});

setCheckedValue(labelFields.segmentStatus, "clean");
setCheckedValue(labelFields.success, "success");
setCheckedValue(labelFields.terminationReason, "goal_reached");

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

document.addEventListener("keydown", async (event) => {
  if (event.repeat) {
    return;
  }
  const activeTag = document.activeElement ? document.activeElement.tagName : "";
  if (activeTag === "INPUT" || activeTag === "TEXTAREA" || activeTag === "SELECT") {
    return;
  }
  if (event.key === "f" || event.key === "F") {
    if (controls.submitLabel.disabled) {
      return;
    }
    event.preventDefault();
    const result = await postJson("/api/label/submit", {
      segment_status: getCheckedValue(labelFields.segmentStatus),
      success: getCheckedValue(labelFields.success),
      termination_reason: getCheckedValue(labelFields.terminationReason),
    });
    setLog(JSON.stringify(result, null, 2));
    if (result.ok) {
      setCheckedValue(labelFields.segmentStatus, "clean");
      setCheckedValue(labelFields.success, "success");
      setCheckedValue(labelFields.terminationReason, "goal_reached");
    }
    refreshStatus();
  }
});

setInterval(refreshStatus, 500);
refreshStatus();
