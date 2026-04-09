from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import io
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODALITIES = {"text": True, "image": True, "state": False}
DEFAULT_TARGET_FIELDS = ("vx", "vy", "wz")
_BASELINE_V2_API: Optional[Tuple[Any, Any, Sequence[str], Any, Any, Any, Any]] = None
DEFAULT_MODE = "observe"
EXECUTE_MODES = {"step", "continuous"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="加载 Go2 VLA baseline v2 checkpoint，执行单帧或桥接式推理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="训练输出的 checkpoint 文件，例如 best.pt")
    parser.add_argument("--instruction", type=str, required=True, help="语言指令，例如 'go to the red object'")
    parser.add_argument("--image-path", type=Path, help="离线单帧推理时的输入图像路径")
    parser.add_argument("--device", type=str, help="推理设备，例如 cuda、cuda:0、cpu")
    parser.add_argument("--image-size", type=int, default=224, help="缺省图像尺寸，仅在 checkpoint 未记录时使用")
    parser.add_argument("--bridge-bin", type=Path, help="go2_bridge 可执行文件路径；不传则仅支持 --image-path")
    parser.add_argument(
        "--bridge-command",
        type=str,
        help="完整 bridge 启动命令；适合通过 ssh 在远端启动 go2_bridge，例如 ssh user@host /path/to/go2_bridge --network-interface eth0",
    )
    parser.add_argument("--network-interface", type=str, help="桥接 Go2 时使用的网卡，例如 eth0")
    parser.add_argument("--control-hz", type=float, default=50.0, help="桥接控制循环频率")
    parser.add_argument("--video-poll-hz", type=float, default=10.0, help="桥接视频抓取频率")
    parser.add_argument("--mode", choices=["observe", "step", "continuous"], help="执行模式，默认 observe")
    parser.add_argument("--loop-interval", type=float, default=0.2, help="桥接模式下每次推理的间隔秒数")
    parser.add_argument("--max-steps", type=int, default=1, help="桥接模式下推理步数；0 表示持续运行")
    parser.add_argument("--snapshot-retries", type=int, default=10, help="等待远端视频首帧时，连续重试 snapshot 的次数")
    parser.add_argument("--snapshot-retry-interval", type=float, default=0.2, help="snapshot 重试间隔秒数")
    parser.add_argument("--image-stale-timeout-sec", type=float, default=0.6, help="图像最大可接受延迟，超出视为 stale")
    parser.add_argument("--max-image-miss-cycles", type=int, default=2, help="连续图像失效周期阈值，触发 safe stop")
    parser.add_argument("--inference-timeout-sec", type=float, default=0.8, help="单次推理超时阈值")
    parser.add_argument("--loop-timeout-sec", type=float, default=1.2, help="主循环单周期超时阈值")
    parser.add_argument("--command-ttl-sec", type=float, default=0.3, help="非零命令 TTL，到期自动 stop")
    parser.add_argument("--step-duration-sec", type=float, default=0.15, help="step 模式脉冲持续时长")
    parser.add_argument("--vx-deadband", type=float, default=0.03, help="vx 死区")
    parser.add_argument("--vy-deadband", type=float, default=0.03, help="vy 死区")
    parser.add_argument("--wz-deadband", type=float, default=0.05, help="wz 死区")
    parser.add_argument("--deploy-vx-max", type=float, default=0.20, help="部署侧 vx 限幅")
    parser.add_argument("--deploy-vy-max", type=float, default=0.10, help="部署侧 vy 限幅")
    parser.add_argument("--deploy-wz-max", type=float, default=0.40, help="部署侧 wz 限幅")
    parser.add_argument("--allow-vy", action="store_true", help="允许横移；默认禁用 vy")
    parser.add_argument("--observe-start-stop", action="store_true", help="observe 模式启动时主动发送一次 STOP")
    parser.add_argument("--execute", action="store_true", help="把推理出的速度命令下发给 Go2；默认只打印结果")
    parser.add_argument("--stand-up-first", action="store_true", help="执行模式启动前先下发 STAND_UP")
    parser.add_argument("--max-vx", type=float, default=0.4, help="兼容参数：旧前进/后退速度裁剪上限（将映射到 deploy-vx-max）")
    parser.add_argument("--max-vy", type=float, default=0.3, help="兼容参数：旧横移速度裁剪上限（将映射到 deploy-vy-max）")
    parser.add_argument("--max-wz", type=float, default=0.8, help="兼容参数：旧转向角速度裁剪上限（将映射到 deploy-wz-max）")
    return parser


def _device_from_arg(device_arg: Optional[str]) -> Any:
    _, _, _, _, _, _, torch_module = _baseline_v2_api()
    if device_arg:
        return torch_module.device(device_arg)
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def _require_pillow() -> None:
    if Image is None:
        raise ModuleNotFoundError("推理脚本依赖 Pillow，请先安装后再运行。")


def _baseline_v2_api() -> Tuple[Any, Any, Sequence[str], Any, Any, Any, Any]:
    global _BASELINE_V2_API
    if _BASELINE_V2_API is None:
        from llada_vla_baseline_v2 import (
            Go2VLABaselineV2,
            InstructionTokenizer,
            TARGET_FIELDS,
            _image_transform,
            _load_checkpoint,
            _require_torch,
            torch,
        )

        _BASELINE_V2_API = (
            Go2VLABaselineV2,
            InstructionTokenizer,
            TARGET_FIELDS,
            _image_transform,
            _load_checkpoint,
            _require_torch,
            torch,
        )
    return _BASELINE_V2_API


def _float_dict(values: Sequence[float]) -> Dict[str, float]:
    return {axis: float(values[index]) for index, axis in enumerate(DEFAULT_TARGET_FIELDS)}


@dataclass
class SafetyRuntimeState:
    last_image_timestamp: Optional[float] = None
    last_step_image_timestamp: Optional[float] = None
    consecutive_image_failures: int = 0
    last_infer_success_mono: Optional[float] = None
    last_nonzero_command_mono: Optional[float] = None
    stop_sent: bool = False
    safe_stop_count: int = 0
    last_safe_stop_reason: str = ""


def _clip_action(action: Dict[str, float], args: argparse.Namespace) -> Dict[str, float]:
    return {
        "vx": max(-args.max_vx, min(args.max_vx, float(action.get("vx", 0.0)))),
        "vy": max(-args.max_vy, min(args.max_vy, float(action.get("vy", 0.0)))),
        "wz": max(-args.max_wz, min(args.max_wz, float(action.get("wz", 0.0)))),
    }


def _resolve_mode(args: argparse.Namespace) -> str:
    if args.mode:
        return args.mode
    if args.execute:
        return "continuous"
    return DEFAULT_MODE


def _is_execute_mode(mode: str) -> bool:
    return mode in EXECUTE_MODES


def _zero_action() -> Dict[str, float]:
    return {"vx": 0.0, "vy": 0.0, "wz": 0.0}


def _is_nonzero_action(action: Dict[str, float], eps: float = 1e-6) -> bool:
    return abs(float(action["vx"])) > eps or abs(float(action["vy"])) > eps or abs(float(action["wz"])) > eps


def _apply_deadband(action: Dict[str, float], args: argparse.Namespace) -> Tuple[Dict[str, float], Dict[str, bool]]:
    out = dict(action)
    deadband_triggered = {"vx": False, "vy": False, "wz": False}
    if abs(out["vx"]) < args.vx_deadband:
        out["vx"] = 0.0
        deadband_triggered["vx"] = True
    if abs(out["vy"]) < args.vy_deadband:
        out["vy"] = 0.0
        deadband_triggered["vy"] = True
    if abs(out["wz"]) < args.wz_deadband:
        out["wz"] = 0.0
        deadband_triggered["wz"] = True
    return out, deadband_triggered


def _clamp_deploy_action(action: Dict[str, float], args: argparse.Namespace) -> Dict[str, float]:
    return {
        "vx": max(-args.deploy_vx_max, min(args.deploy_vx_max, float(action["vx"]))),
        "vy": max(-args.deploy_vy_max, min(args.deploy_vy_max, float(action["vy"]))),
        "wz": max(-args.deploy_wz_max, min(args.deploy_wz_max, float(action["wz"]))),
    }


def _apply_safety_shell(raw_action: Dict[str, float], args: argparse.Namespace) -> Tuple[Dict[str, float], Dict[str, Any]]:
    deadbanded, deadband_flags = _apply_deadband(raw_action, args)
    clamped = _clamp_deploy_action(deadbanded, args)
    vy_forced_zero = False
    if not args.allow_vy:
        clamped["vy"] = 0.0
        vy_forced_zero = True
    details = {
        "deadband_triggered": deadband_flags,
        "vy_forced_zero": vy_forced_zero,
    }
    return clamped, details


def _send_stop_command(client: Go2BridgeClient, reason: str, mode: str, state: SafetyRuntimeState) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "event": "safe_stop",
        "mode": mode,
        "reason": reason,
        "safe_stop_count": state.safe_stop_count + 1,
    }
    try:
        payload["bridge_response"] = client.stop()
    except Exception as error:
        payload["stop_error"] = str(error)
    state.stop_sent = True
    state.last_nonzero_command_mono = None
    state.safe_stop_count += 1
    state.last_safe_stop_reason = reason
    return payload


def _refresh_command_watchdog(client: Go2BridgeClient, args: argparse.Namespace, mode: str, state: SafetyRuntimeState, now_mono: float) -> Optional[Dict[str, Any]]:
    if not _is_execute_mode(mode):
        return None
    if state.last_nonzero_command_mono is None:
        return None
    if now_mono - state.last_nonzero_command_mono <= args.command_ttl_sec:
        return None
    payload = _send_stop_command(client, "command_ttl_expired", mode=mode, state=state)
    payload["watchdog"] = True
    return payload


def _is_image_fresh(snapshot: Dict[str, Any], args: argparse.Namespace, state: SafetyRuntimeState) -> Tuple[bool, Optional[Image.Image], str]:
    image_block = snapshot.get("image")
    if not isinstance(image_block, dict):
        state.consecutive_image_failures += 1
        return False, None, "image_missing"
    image_timestamp = image_block.get("timestamp")
    if not isinstance(image_timestamp, (int, float)):
        state.consecutive_image_failures += 1
        return False, None, "image_timestamp_missing"
    host_time = snapshot.get("host_time")
    if isinstance(host_time, (int, float)) and (float(host_time) - float(image_timestamp)) > args.image_stale_timeout_sec:
        state.consecutive_image_failures += 1
        return False, None, "image_stale"
    if state.last_image_timestamp is not None and float(image_timestamp) <= float(state.last_image_timestamp):
        state.consecutive_image_failures += 1
        return False, None, "image_not_updated"
    try:
        image = _image_from_snapshot(snapshot)
    except Exception:
        state.consecutive_image_failures += 1
        return False, None, "image_decode_failed"
    state.last_image_timestamp = float(image_timestamp)
    state.consecutive_image_failures = 0
    return True, image, "image_ok"


class Go2VLAInferenceEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        (
            go2_vla_baseline_cls,
            tokenizer_cls,
            _target_fields,
            image_transform_fn,
            load_checkpoint_fn,
            require_torch_fn,
            torch_module,
        ) = _baseline_v2_api()
        require_torch_fn()
        _require_pillow()
        checkpoint = load_checkpoint_fn(args.checkpoint_path.resolve(), device=torch_module.device("cpu"))
        tokenizer_payload = checkpoint.get("tokenizer")
        if not isinstance(tokenizer_payload, dict):
            raise RuntimeError("checkpoint 缺少 tokenizer，无法进行文本推理")

        model_config = checkpoint.get("model_config") or {}
        self.modalities = dict(checkpoint.get("modalities") or DEFAULT_MODALITIES)
        self.state_fields = list(checkpoint.get("state_fields") or [])
        self.tokenizer = tokenizer_cls.from_payload(tokenizer_payload)
        self.image_size = int(model_config.get("image_size", args.image_size))
        self.device = _device_from_arg(args.device)
        self.transform = image_transform_fn(self.image_size)
        self.torch = torch_module

        # Inference only needs the architecture shape; the checkpoint restores weights.
        self.model = go2_vla_baseline_cls(
            vocab_size=self.tokenizer.vocab_size,
            pad_id=self.tokenizer.pad_id,
            hidden_dim=int(model_config.get("hidden_dim", 256)),
            text_embed_dim=int(model_config.get("text_embed_dim", 64)),
            fusion_type=str(model_config.get("fusion_type") or "mlp"),
            modalities=self.modalities,
            state_dim=len(self.state_fields),
            pretrained_vision=False,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, instruction: str, image: Image.Image) -> Dict[str, float]:
        instruction_ids = self.tokenizer.encode(instruction)
        batch: Dict[str, Any] = {
            "instruction_ids": self.torch.tensor([instruction_ids], dtype=self.torch.long, device=self.device),
        }
        if self.modalities.get("image", False):
            image_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            batch["image"] = image_tensor
        if self.modalities.get("state", False):
            batch["state"] = self.torch.zeros((1, len(self.state_fields)), dtype=self.torch.float32, device=self.device)

        with self.torch.no_grad():
            output = self.model(batch)[0].detach().cpu().tolist()
        return _float_dict(output)


class Go2BridgeClient:
    def __init__(
        self,
        bridge_bin: Optional[Path],
        bridge_command: Optional[str],
        network_interface: str,
        control_hz: float,
        video_poll_hz: float,
    ) -> None:
        if bridge_command:
            command = shlex.split(bridge_command)
        else:
            if bridge_bin is None:
                raise RuntimeError("缺少 bridge 启动参数：请传入 --bridge-bin 或 --bridge-command")
            command = [str(bridge_bin.resolve())]
            if network_interface:
                command.extend(["--network-interface", network_interface])
            command.extend(["--control-hz", str(control_hz), "--video-poll-hz", str(video_poll_hz)])
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._read_json_line()
        if not ready.get("ok") or ready.get("type") != "ready":
            raise RuntimeError(f"go2_bridge 未正常启动: {ready}")

    def _read_json_line(self) -> Dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("go2_bridge stdout 不可用")
        line = self.process.stdout.readline()
        if not line:
            stderr_text = ""
            if self.process.stderr is not None:
                stderr_text = self.process.stderr.read().strip()
            raise RuntimeError(f"go2_bridge 已退出: {stderr_text}")
        return json.loads(line.strip())

    def send(self, command: str) -> Dict[str, Any]:
        if self.process.stdin is None:
            raise RuntimeError("go2_bridge stdin 不可用")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        return self._read_json_line()

    def snapshot(self) -> Dict[str, Any]:
        return self.send("SNAPSHOT")

    def stand_up(self) -> Dict[str, Any]:
        return self.send("STAND_UP")

    def stop(self) -> Dict[str, Any]:
        return self.send("STOP")

    def set_velocity(self, vx: float, vy: float, wz: float) -> Dict[str, Any]:
        return self.send(f"SET_VELOCITY {vx:.6f} {vy:.6f} {wz:.6f}")

    def shutdown(self) -> None:
        try:
            self.send("SHUTDOWN")
        except Exception:
            pass
        finally:
            try:
                self.process.wait(timeout=3.0)
            except Exception:
                self.process.kill()


def _image_from_snapshot(snapshot: Dict[str, Any]) -> Image.Image:
    _require_pillow()
    image_block = snapshot.get("image")
    if not isinstance(image_block, dict):
        raise RuntimeError("snapshot 中缺少图像")
    jpeg_b64 = image_block.get("jpeg_b64")
    if not isinstance(jpeg_b64, str) or not jpeg_b64:
        raise RuntimeError("snapshot 图像字段为空")
    image_bytes = base64.b64decode(jpeg_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _print_result(result: Dict[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


def _run_offline(args: argparse.Namespace, engine: Go2VLAInferenceEngine) -> int:
    if args.image_path is None:
        raise RuntimeError("离线模式必须传入 --image-path")
    _require_pillow()
    with Image.open(args.image_path) as image:
        raw_action = engine.predict(args.instruction, image.convert("RGB"))
    clipped_action = _clip_action(raw_action, args)
    _print_result(
        {
            "mode": "offline_image",
            "instruction": args.instruction,
            "image_path": str(args.image_path.resolve()),
            "raw_action": raw_action,
            "clipped_action": clipped_action,
            "device": str(engine.device),
        }
    )
    return 0


def _run_bridge_loop(args: argparse.Namespace, engine: Go2VLAInferenceEngine) -> int:
    if args.bridge_bin is None and not args.bridge_command:
        raise RuntimeError("桥接模式必须传入 --bridge-bin 或 --bridge-command")
    mode = _resolve_mode(args)

    client = Go2BridgeClient(
        bridge_bin=args.bridge_bin,
        bridge_command=args.bridge_command,
        network_interface=args.network_interface or "",
        control_hz=args.control_hz,
        video_poll_hz=args.video_poll_hz,
    )
    state = SafetyRuntimeState()

    try:
        _print_result({"event": "mode_selected", "mode": mode, "legacy_execute_flag": bool(args.execute), "allow_vy": bool(args.allow_vy)})
        if mode == "observe" and args.observe_start_stop:
            _print_result({"mode": mode, "event": "observe_start_stop", "response": client.stop()})
        if _is_execute_mode(mode) and args.stand_up_first:
            _print_result({"mode": mode, "event": "stand_up", "response": client.stand_up()})
            time.sleep(1.0)

        step = 0
        while args.max_steps == 0 or step < args.max_steps:
            loop_start = time.monotonic()
            watchdog_event = _refresh_command_watchdog(client, args, mode=mode, state=state, now_mono=loop_start)
            if watchdog_event is not None:
                _print_result(watchdog_event)

            snapshot: Optional[Dict[str, Any]] = None
            image: Optional[Image.Image] = None
            image_reason = "snapshot_not_ready"
            for _ in range(max(1, args.snapshot_retries)):
                try:
                    snapshot = client.snapshot()
                except Exception as error:
                    payload = _send_stop_command(client, f"snapshot_error:{error}", mode=mode, state=state)
                    payload["fatal"] = True
                    _print_result(payload)
                    return 1
                is_fresh, maybe_image, image_reason = _is_image_fresh(snapshot, args=args, state=state)
                if is_fresh:
                    image = maybe_image
                    break
                time.sleep(max(0.0, args.snapshot_retry_interval))

            if snapshot is None:
                payload = _send_stop_command(client, "snapshot_missing", mode=mode, state=state)
                payload["fatal"] = True
                _print_result(payload)
                return 1

            if image is None:
                safe_reason = f"image_missing_or_stale:{image_reason}"
                payload = _send_stop_command(client, safe_reason, mode=mode, state=state)
                payload["consecutive_image_failures"] = state.consecutive_image_failures
                _print_result(payload)
                if state.consecutive_image_failures >= args.max_image_miss_cycles:
                    _print_result({"event": "exit_on_image_failure", "mode": mode, "reason": safe_reason})
                    return 1
                time.sleep(max(0.0, args.loop_interval))
                continue

            image_timestamp = float((snapshot.get("image") or {}).get("timestamp", 0.0))
            infer_start = time.monotonic()
            try:
                raw_action = engine.predict(args.instruction, image)
            except Exception as error:
                payload = _send_stop_command(client, f"inference_exception:{error}", mode=mode, state=state)
                payload["fatal"] = True
                _print_result(payload)
                return 1
            infer_elapsed = time.monotonic() - infer_start
            if infer_elapsed > args.inference_timeout_sec:
                payload = _send_stop_command(client, "inference_timeout", mode=mode, state=state)
                payload["inference_sec"] = infer_elapsed
                _print_result(payload)
                time.sleep(max(0.0, args.loop_interval))
                continue
            state.last_infer_success_mono = time.monotonic()

            deploy_action, shell_details = _apply_safety_shell(raw_action, args)
            result = {
                "mode": mode,
                "step": step,
                "instruction": args.instruction,
                "image_fresh": True,
                "image_reason": image_reason,
                "last_infer_success_sec_ago": 0.0 if state.last_infer_success_mono is None else (time.monotonic() - state.last_infer_success_mono),
                "raw_model_action": raw_action,
                "final_action": deploy_action,
                "raw_action": raw_action,
                "state": snapshot.get("state"),
                "host_time": snapshot.get("host_time"),
                "execute": _is_execute_mode(mode),
                "shell": shell_details,
                "last_safe_stop_reason": state.last_safe_stop_reason,
            }
            try:
                if mode == "observe":
                    result["execution"] = "observe_only"
                elif mode == "step":
                    step_image_gate_ok = True
                    if state.last_step_image_timestamp is not None and image_timestamp <= state.last_step_image_timestamp:
                        step_image_gate_ok = False
                    result["step_image_gate_ok"] = step_image_gate_ok
                    result["step_image_timestamp"] = image_timestamp
                    result["last_step_image_timestamp"] = state.last_step_image_timestamp
                    if not step_image_gate_ok:
                        payload = _send_stop_command(client, "step_requires_new_image", mode=mode, state=state)
                        payload["step"] = step
                        payload["image_timestamp"] = image_timestamp
                        payload["last_step_image_timestamp"] = state.last_step_image_timestamp
                        _print_result(payload)
                        time.sleep(max(0.0, args.loop_interval))
                        continue
                    if _is_nonzero_action(deploy_action):
                        result["step_pulse"] = client.set_velocity(deploy_action["vx"], deploy_action["vy"], deploy_action["wz"])
                        state.last_nonzero_command_mono = time.monotonic()
                        state.stop_sent = False
                        time.sleep(max(0.0, args.step_duration_sec))
                    stop_payload = client.stop()
                    state.stop_sent = True
                    state.last_nonzero_command_mono = None
                    state.last_step_image_timestamp = image_timestamp
                    result["step_stop"] = stop_payload
                    result["execution"] = "step_pulse_then_stop"
                elif mode == "continuous":
                    if _is_nonzero_action(deploy_action):
                        result["bridge_response"] = client.set_velocity(deploy_action["vx"], deploy_action["vy"], deploy_action["wz"])
                        state.last_nonzero_command_mono = time.monotonic()
                        state.stop_sent = False
                    else:
                        result["bridge_response"] = client.stop()
                        state.stop_sent = True
                        state.last_nonzero_command_mono = None
                    result["execution"] = "continuous"
                else:
                    raise RuntimeError(f"unsupported mode: {mode}")
            except Exception as error:
                payload = _send_stop_command(client, f"bridge_write_exception:{error}", mode=mode, state=state)
                payload["fatal"] = True
                _print_result(payload)
                return 1
            _print_result(result)
            step += 1
            loop_elapsed = time.monotonic() - loop_start
            if loop_elapsed > args.loop_timeout_sec:
                payload = _send_stop_command(client, "loop_timeout", mode=mode, state=state)
                payload["loop_elapsed_sec"] = loop_elapsed
                _print_result(payload)
            time.sleep(max(0.0, args.loop_interval))
    except KeyboardInterrupt:
        payload = _send_stop_command(client, "keyboard_interrupt", mode=mode, state=state)
        _print_result(payload)
    finally:
        if _is_execute_mode(mode):
            try:
                _print_result({"mode": mode, "event": "exit_stop", "response": client.stop()})
            except Exception as error:
                _print_result({"mode": mode, "event": "exit_stop_failed", "error": str(error)})
        client.shutdown()
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    # Keep backward compatibility for old max-* flags while making deployment defaults conservative.
    args.deploy_vx_max = min(float(args.deploy_vx_max), float(args.max_vx))
    args.deploy_vy_max = min(float(args.deploy_vy_max), float(args.max_vy))
    args.deploy_wz_max = min(float(args.deploy_wz_max), float(args.max_wz))
    engine = Go2VLAInferenceEngine(args)
    if args.image_path is not None:
        return _run_offline(args, engine)
    return _run_bridge_loop(args, engine)


if __name__ == "__main__":
    raise SystemExit(main())
