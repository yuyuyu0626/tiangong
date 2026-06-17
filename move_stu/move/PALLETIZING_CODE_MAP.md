# Online Robot Palletizing Code Map

This task is implemented as an online execution pipeline:

`current item -> PCT ranked leaf proposal -> actual-aware feasibility check -> robot execution candidate -> trace execution -> success commit`

## Main Entry

- `move/tasks/online_robot_palletizing_trace_task.py`
  Thin CLI wrapper used for headless trace runs.

- `move/tasks/online_robot_palletizing_task.py`
  Main execution state machine.  It creates one Isaac Gym episode, spawns items
  one by one, executes pick/attach/transport/place, validates placement, commits
  PCT only after success, and writes trace/metrics.

## Supporting Modules

- `move/palletizing_runtime.py`
  Constants, thresholds, dataclasses, dense-fill sampling, and completion reason
  groups.

- `move/palletizing_geometry.py`
  Actual packing state, AABB computation, support/obstacle overlap separation,
  support-top calculation, inside-cube checks, local repair vectors, and
  actual-vs-PCT deviation helpers.

- `move/palletizing_candidate_selection.py`
  Actual-aware PCT leaf selection.  It scans ranked valid leaves, rejects only
  unsupported or actually infeasible leaves, and generates robot execution
  candidates using approach side and stand-off options.

- `move/palletizing_sim_helpers.py`
  Isaac Gym utility layer: static scene creation, source box creation, robot/box
  collision handling, trace writing, timeline expansion, and hand-frame
  diagnostics.

- `move/robot_placement.py`
  Converts PCT placements into robot placement targets.  It keeps
  `original_size_m` for the real source actor and `placed_size_m` for the PCT
  target AABB, and supports same/yaw-only orientation classes.

- `move/grab_test.py`
  Local grasp/place planner reused by the online task.  It provides the
  single-item IK plan with source yaw, target yaw, target AABB, and release
  height.

- `move/tasks/grab_test_task.py`
  Original Isaac Gym grasp execution utilities reused by the online task:
  attach to hand frame, actor pose helpers, DOF utilities, colors/friction, and
  timeline operations.

## Rendering

- `move/render_palletizing_trace.py`
  Final video renderer.  It consumes trace JSONL and metrics only; it does not
  call PCT, IK, or planned-box replay.  It creates Isaac Gym scenes and camera
  frames for robot/table/pallet/boxes/1m boundary/target ghost.

- `move/scripts/render_isaac_keyframe_list.sh`
  Conservative full-asset render driver.  It renders selected trace frames in
  independent Isaac Gym processes to avoid full-URDF camera crashes on this
  server, then assembles the successful original-asset frames into mp4.

## Metrics

The default `metrics.json` is compact for delivery.  Full per-item debug state
is written only through `--debug-metrics`.
