# Pilot Collection Plan

This note is now aligned with the keyboard-only Go2 collection loop.

## Goal

Collect enough short, clean episodes to answer three questions:

- does the camera stream line up with the operator intent?
- do `vx`, `vy`, `wz` describe the behavior we care about?
- is the dataset learnable with the current baseline interface?

## Required Metadata Per Session

Before recording, set at startup:

- `scene_id`
- `operator_id`

Each episode gets its `instruction` only after the segment ends.

## Instruction Set

Use only the controlled set:

- `go forward`
- `move backward`
- `strafe left`
- `strafe right`
- `stand up`
- `lie down`
- `turn left`
- `turn right`
- `stay still`

## Collection Flow

1. start `go2_collector`
2. use keyboard teleop to position the robot
3. press `R` to start a capture segment
4. execute exactly one intended behavior
5. press `T` to end the segment
6. choose the instruction number and press Enter
7. if the segment is bad, press `ESC` during labeling to discard it

Frames outside the explicit segment are not written to disk.

## Minimum Coverage Goal

Aim for:

- multiple sessions
- multiple scenes
- all core motion instructions represented
- both easy and slightly messy examples

## Stop Conditions

Pause collection and fix the pipeline if you see:

- missing images
- wrong instruction labels
- persistent flat action traces
- obvious image lag or stale action timestamps
- repeated safety faults during normal teleop
- heavy bias toward a single instruction
