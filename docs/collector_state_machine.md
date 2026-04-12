# Collector State Machine

## Scope

This document describes the current native collector state machine in
`native/go2_collector.cpp`.

The collector now supports only the `trajectory` capture flow.

## Top-Level States

The runtime state is tracked by two orthogonal state groups:

- `SafetyState`
  - `SafeReady`: motion and recording are allowed
  - `FaultLatched`: a safety fault was detected and must be cleared
  - `EstopLatched`: operator-triggered emergency stop; must be cleared
- `CaptureState`
  - `Idle`: no active segment and no fault-owned capture state
  - `Capturing`: a trajectory segment is active
  - `Fault`: capture flow is blocked by a latched fault

In addition, the label flow is tracked by the logger:

- `pendingLabel = false`: no segment is waiting for scoring
- `pendingLabel = true`: a finished segment is waiting for `1/2/3/4` or Web UI submission

## State Transitions

### Startup Gate

On `wireless_controller`, the collector starts with a startup gate:

- gate active -> wait for one native `Start`
- gate cleared -> `can_start_recording = true`

The startup gate does not create a capture state on its own; it only blocks
`Idle -> Capturing`.

In `--preview-ui` mode, the startup gate is disabled and the collector seeds
mock state/image data so the Web UI can be exercised without a Go2
connection.

### Normal Capture Flow

Trajectory capture uses the following path:

1. `Idle` + `SafeReady`
2. operator presses `R` / `A`
3. `BeginSegmentInternal()`
4. logger starts a trajectory-gated segment
5. `CaptureState` becomes `Capturing`
6. video/state/control samples are buffered only after effective motion starts
7. operator presses `T` / `B`
8. stop flow waits for input release and grace period when needed
9. `FinalizeCaptureForLabelInternal()`
10. `CaptureState` returns to `Idle`
11. logger enters `pendingLabel = true`
12. operator submits label
13. episode is written to disk and logger returns to `pendingLabel = false`

### Discard Flow

Discard is valid in either of these conditions:

- active capture: `CaptureState == Capturing`
- pending label exists

Discard does:

1. reset capture progress
2. clear any pending segment from the logger
3. move `CaptureState` to `Idle` or `Fault` depending on `SafetyState`

### Fault Flow

Faults can be triggered by:

- invalid or stale robot state
- roll/pitch safety threshold violations
- operator emergency stop

When a fault is latched:

1. `SafetyState` becomes `FaultLatched` or `EstopLatched`
2. `CaptureState` becomes `Fault`
3. capture progress is reset
4. pending segment is discarded
5. teleop output is forced to zero
6. native joystick passthrough is disabled on wireless mode

After the fault condition is gone and the operator clears it:

- `SafetyState -> SafeReady`
- `CaptureState: Fault -> Idle`

## Terminal vs Web UI Control

Both terminal input and Web UI buttons call the same internal handlers:

- start: `BeginSegmentInternal()`
- stop: `StopSegmentForLabelInternal()`
- discard: `DiscardSegmentInternal()`
- label submit: `FinalizePendingLabelInternal()`
- estop: `RequestEmergencyStop()`

Terminal-only auto prompt behavior:

- after a successful stop, terminal modes schedule a deferred prompt
- the prompt runs on the keyboard/controller input thread, not the control loop
- this keeps the control loop free to publish zero velocity before waiting for stdin

## Important Invariants

- `Capturing` implies exactly one active logger segment
- `pendingLabel = true` implies start is blocked until label submit or discard
- `SafetyState != SafeReady` implies motion commands are zeroed
- trajectory stop uses a release wait and grace period before final label-ready state
- Web UI and terminal paths must not fork the state machine; they should share handlers
