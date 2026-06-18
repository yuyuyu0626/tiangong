# Online 3D Bin Packing Robot Palletizing Report

## 1. Task Target

The target is an online robot palletizing task in Isaac Gym:

- Container / palletizing volume: `1m x 1m x 1m`.
- Item dimensions: each axis sampled from `[0.1, 0.2, 0.3, 0.4, 0.5]` meters.
- Placement decision: based on the prior PCT online 3D-BPP policy model from
  `Online-3D-BPP-PCT`.
- Robot execution: Tianyi robot picks one online-arriving item at a time and
  places it into the stack.

## 2. What Was Tried

### 2.1 Initial keyframe rendering path

The first implementation connected PCT placements and IK plans, then used a
keyframe renderer to directly replay robot and box poses.  This produced videos
quickly, but it was not a valid robot execution path:

- Future boxes could appear too early.
- Box motion was driven by planned centers or replayed states.
- The hand-box relationship was visually weak.
- It could not prove pick, carry, place, and commit behavior.

That path is now treated as debug-only and is not used for final delivery.

### 2.2 Continuous trace execution path

The implementation was moved back to the real task flow:

`current item -> PCT propose -> robot execution -> validation -> PCT commit`

The task now creates one Isaac Gym episode, processes items one by one, and
writes a trace of the execution state.  Future items are not spawned at the
start.

### 2.3 Grasp and attach corrections

The original grasp flow was extended instead of bypassed:

- Source actor uses `original_size_m`.
- PCT target AABB uses `placed_size_m`.
- Same and yaw-only orientation are supported.
- Unsupported height flips are not treated as a normal first-version motion.
- Attach happens only after contact/gap/drift checks.
- After attach, the box follows the actual hand frame, not a planned box-center
  replay.

### 2.4 Release and settle corrections

The release phase was a major failure source.  We observed that a box could be
valid in center error but already penetrating the support surface.  The release
logic was changed to use:

- `support_top_z` based on actual frozen boxes.
- bottom-gap checks before release.
- support overlap vs obstacle overlap separation.
- `verified_kinematic_place` fallback: freeze at actual pre-release pose only
  after geometric validation.
- `snap_to_pct_target=false`.

This avoids PhysX contact explosions caused by releasing an already-penetrating
dynamic object.

### 2.5 Multi-object execution and actual-aware PCT selection

Pure rank-1 PCT placement is not always executable by the current robot.  The
final path keeps PCT ranking but adds actual-scene feasibility:

- PCT valid leaves are checked in policy-rank order.
- Rank 1 is always inspected first and rejected only with a recorded reason.
- Actual frozen AABBs are used for support, obstacle, and inside-cube checks.
- Small local execution adjustment is allowed, but the box is never snapped to
  the PCT target.
- If a leaf is geometrically plausible, robot execution candidates are generated
  by approach side and stand-off.

This is not replacing PCT with a hand-written packing rule.  PCT still proposes
the placement order and leaf ranking; the execution layer filters what the robot
can physically realize in the actual accumulated stack.

### 2.6 Rendering attempts

Live Isaac Gym camera recording was unstable on the current server because the
NVIDIA Vulkan graphics path is not available.  The final rendering method is:

1. Execute the robot task headlessly and save `trace.jsonl`.
2. Reconstruct selected trace frames as Isaac Gym scenes.
3. Render robot/table/pallet/boxes/1m frame with Isaac Gym camera sensors.
4. Assemble these frames into mp4 videos.

This is not PIL drawing and not the old planned-keyframe renderer.  It is a
trace visualization backend.  Some camera frames can still fail with Lavapipe,
so videos are assembled from successfully rendered Isaac Gym frames, using the
nearest successful frame only when a selected frame fails.

For the final five videos, `render_palletizing_trace.py` loads the original
Tianyi robot asset.  Because the full URDF camera path is unstable when many
frames are rendered in one process, `render_isaac_keyframe_list.sh` renders
selected trace frames in independent Isaac Gym processes and then assembles the
successful original-asset frames into mp4 files.

## 3. Current Method

### 3.1 Online control loop

For each item:

1. Sample current online item.
2. Query PCT ranked valid leaves.
3. Convert PCT placement to a robot target.
4. Check orientation support and actual-scene feasibility.
5. Generate robot execution candidates.
6. Execute pick, attach, transport, place, and validated freeze.
7. Commit PCT only after execution success.

### 3.2 Placement policy and robot execution split

PCT decides where the item should go.  The robot layer decides how to approach
that target and whether it can execute the target without violating the actual
scene.

Important constraints:

- `original_size_m` is used for the real source actor.
- `placed_size_m` is used for the target AABB.
- target yaw and robot approach yaw are decoupled.
- actual frozen boxes are the source of truth for support and collision checks.
- future boxes are not spawned at the beginning.

### 3.3 Dense filling

For long runs with `max_items=0`, the task continues sampling until a stop
condition.  Once the stack is dense, optional fine-fill sampling biases later
items toward smaller legal sizes.  This is a pragmatic online testing mode:
the item sizes still come from the allowed set, but it avoids wasting many
arrivals on large items that cannot fit late in the episode.

## 4. Result Snapshot

Five different seeds have been frozen as delivery cases:

| Case | Seed | Placed | Utilization |
| --- | ---: | ---: | ---: |
| case_00 | 20260614 | 28 | 0.776 |
| case_01 | 20260615 | 37 | 0.740 |
| case_02 | 20260616 | 32 | 0.756 |
| case_03 | 20260617 | 36 | 0.733 |
| case_04 | 20260618 | 33 | 0.764 |

All five cases report `all_boxes_inside_1m_cube=true`.

## 5. Code Organization

The implementation has been split for delivery readability:

- `move/tasks/online_robot_palletizing_trace_task.py`
  Thin CLI wrapper for no-camera trace execution.
- `move/tasks/online_robot_palletizing_task.py`
  Main online execution state machine.
- `move/palletizing_runtime.py`
  Runtime constants, dataclasses, dense-fill sampling.
- `move/palletizing_geometry.py`
  Actual AABB/support/overlap/inside-cube logic.
- `move/palletizing_candidate_selection.py`
  Actual-aware PCT leaf filtering and robot candidate planning.
- `move/palletizing_sim_helpers.py`
  Isaac Gym scene, source actor, trace, hand-frame, and timeline helpers.
- `move/robot_placement.py`
  PCT placement to robot target conversion and yaw-only classification.
- `move/render_palletizing_trace.py`
  Isaac Gym trace renderer.
- `move/scripts/render_isaac_keyframe_list.sh`
  Conservative full-asset frame renderer.

See also `move/PALLETIZING_CODE_MAP.md`.

## 6. Known Limitations

- The current environment cannot provide stable NVIDIA Vulkan live rendering.
  Videos are therefore Isaac Gym scene reconstructions from execution traces.
- Dense final filling is limited by the current robot reachability and actual
  scene feasibility; the current random-seed cases reach about 73% to 78%
  utilization rather than 100%.
- Height-flip orientation is not fully executed in-air.  First-version support
  is same/yaw-only; unsupported flips are filtered or handled through feasible
  source-side setup rather than arbitrary mid-air flipping.
- `verified_kinematic_place` is used to avoid PhysX release explosions after
  the pre-release geometry is validated.  The freeze pose is the actual
  pre-release pose, not the PCT target.

## 7. Delivery Files

Delivery output directory:

`move/outputs/final_online_palletizing_ge70_delivery`

Each case keeps:

- `case_xx_trace.jsonl`
- `case_xx_metrics.json`
- `case_xx_isaac_stride15_keyframed.mp4`

The compact metrics file contains only delivery-level indicators.  Full debug
diagnostics are not included by default.

Debug frame caches, render logs, and raw run folders are stored under
`debug_render/` and `debug_runs/` so the top-level delivery directory stays
readable.
