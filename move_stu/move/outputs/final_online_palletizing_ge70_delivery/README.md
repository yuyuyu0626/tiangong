Final online palletizing delivery workspace.

This directory keeps the five selected delivery traces, metrics, videos, and
the report.  Old PIL schematic videos and one-off debug outputs are not kept at
the top level.

Current policy:
- Keep cases with utilization >= 0.70.
- Continue full-fill runs with target_utilization=1.0 where possible.
- Final videos are rendered by `move.render_palletizing_trace`, which uses
  Isaac Gym camera scenes with the original Tianyi robot asset, table, pallet,
  1m container frame, target ghost, and trace boxes.  The conservative full-URDF
  render driver renders selected trace frames in independent Isaac Gym
  processes to avoid camera crashes on this server.

Main run command:

```bash
cd /2024233240

/2024233240/move/run_move_bpp_env.sh -m move.tasks.online_robot_palletizing_trace_task \
  --model-path /2024233240/external/Online-3D-BPP-PCT/pretrained/setting2_discrete.pt \
  --pct-root /2024233240/external/Online-3D-BPP-PCT \
  --seed 20260614 \
  --max-items 0 \
  --target-utilization 1.0 \
  --trace-every 12 \
  --headless \
  --out-trace /tmp/case_trace.jsonl \
  --metrics /tmp/case_metrics.json
```

Dependency note:
- The delivery zip includes `move/` and `external/Online-3D-BPP-PCT/`.
- Isaac Gym itself is not bundled because it is large and environment-specific;
  run in an Isaac Gym-enabled environment matching `run_move_bpp_env.sh`.

Core code layout:
- `move/tasks/online_robot_palletizing_trace_task.py`
  CLI wrapper for the no-camera continuous execution trace task.
- `move/tasks/online_robot_palletizing_task.py`
  Main online execution state machine: item arrival, PCT propose, robot execute,
  validation, commit, trace and metrics.
- `move/palletizing_runtime.py`
  Runtime constants, small data models, dense-fill sampling helpers.
- `move/palletizing_geometry.py`
  Actual-stack AABB, support surface, overlap, inside-cube, and deviation
  helpers.
- `move/palletizing_candidate_selection.py`
  Actual-aware PCT leaf filtering plus robot approach_side/stand_off candidate
  generation and IK prefiltering.
- `move/palletizing_sim_helpers.py`
  Isaac Gym scene creation, source box creation, timeline expansion, hand-frame
  diagnostics, and trace frame writing.
- `move/robot_placement.py`
  Conversion from PCT placement to robot target, including original_size vs
  placed_size and yaw-only orientation classification.
- `move/grab_test.py` and `move/tasks/grab_test_task.py`
  Existing single-item grasp/attach/release logic reused by the online task.
- `move/render_palletizing_trace.py`
  Isaac Gym trace visualizer for final videos.
- `move/scripts/render_isaac_keyframe_list.sh`
  Conservative full-asset frame renderer used for the final mp4 files.

Metrics policy:
- `metrics.json` is compact and contains only delivery-level indicators such as
  seed, arrived_count, placed_count, utilization, max_height, inside-cube status,
  placement_mode_counts, mean/max placement error, adjustment totals, completion
  reason, and core method flags.
- Full per-item results and actual packing state are written only when
  `--debug-metrics <path>` is explicitly passed.

Delivery files:
- `case_00` to `case_04` each include trace, metrics, and original-asset mp4.
- `summary.csv` summarizes the five selected seeds.
- `REPORT.md` describes the implementation attempts, current method, and known
  limitations.
