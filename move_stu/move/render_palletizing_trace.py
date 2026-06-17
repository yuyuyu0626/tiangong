#!/usr/bin/env python3
"""Render an online palletizing execution trace.

This is a visualization backend only.  It consumes trace JSONL produced by the
continuous Isaac Gym execution task and never plans placements, interpolates box
centers, or creates boxes that are absent from a trace frame.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

try:
    from isaacgym import gymapi  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Isaac Gym Python package is not importable. Use run_move_bpp_env.sh.") from exc

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MOVE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MOVE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from move.planning import PALLET_SIZE, PALLET_SURFACE_Z, PALLET_THICKNESS, TABLE_POSE, TABLE_SIZE, pallet_center_near_table
from move.tasks.move_test1 import _make_transform, _set_color, _set_shape_friction, load_robot_asset


def _thin_box(gym, sim, env, center, size, name, color):
    opts = gymapi.AssetOptions()
    opts.fix_base_link = True
    asset = gym.create_box(sim, float(size[0]), float(size[1]), float(size[2]), opts)
    actor = gym.create_actor(env, asset, _make_transform(float(center[0]), float(center[1]), float(center[2]), 0.0), name, 0, 0)
    _set_color(gym, env, actor, color)
    return actor


def _add_box_wireframe(gym, sim, env, center, size, name, color, thickness=0.008):
    cx, cy, cz = [float(v) for v in center]
    sx, sy, sz = [float(v) for v in size]
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    t = float(thickness)
    # X edges
    for yi, y in enumerate((cy - hy, cy + hy)):
        for zi, z in enumerate((cz - hz, cz + hz)):
            _thin_box(gym, sim, env, (cx, y, z), (sx, t, t), f"{name}_x_{yi}_{zi}", color)
    # Y edges
    for xi, x in enumerate((cx - hx, cx + hx)):
        for zi, z in enumerate((cz - hz, cz + hz)):
            _thin_box(gym, sim, env, (x, cy, z), (t, sy, t), f"{name}_y_{xi}_{zi}", color)
    # Z edges
    for xi, x in enumerate((cx - hx, cx + hx)):
        for yi, y in enumerate((cy - hy, cy + hy)):
            _thin_box(gym, sim, env, (x, y, cz), (t, t, sz), f"{name}_z_{xi}_{yi}", color)


def _overlay_frame(image: np.ndarray, frame: dict, metrics: dict | None) -> np.ndarray:
    boxes = frame.get("boxes", [])
    placed = [b for b in boxes if b.get("state") == "PLACED_FROZEN"]
    volume = 0.0
    for box in placed:
        sx, sy, sz = [float(v) for v in box.get("size_m", (0.0, 0.0, 0.0))]
        volume += sx * sy * sz
    pct = frame.get("pct") or {}
    item_size = pct.get("actor_size_m") or []
    placement_mode_counts = (metrics or {}).get("placement_mode_counts", {})
    placement_mode = "verified_kinematic_place" if placement_mode_counts else ""
    fail_reason = (metrics or {}).get("failure_reason", "")
    lines = [
        f"frame {frame.get('frame')}  item {frame.get('item_index')}  phase {frame.get('phase')}",
        f"placed {len(placed)}  utilization {volume:.3f}  item_size {item_size}",
        f"placement_mode {placement_mode}  fail {fail_reason}",
    ]
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    pad = 6
    line_h = 18
    width = max(draw.textlength(line, font=font) for line in lines) + pad * 2
    height = line_h * len(lines) + pad * 2
    draw.rectangle((8, 8, 8 + width, 8 + height), fill=(0, 0, 0, 150))
    for i, line in enumerate(lines):
        draw.text((8 + pad, 8 + pad + i * line_h), line, font=font, fill=(255, 255, 255, 255))
    return np.asarray(img)


def _create_render_sim(gym, graphics_device_id: int):
    """Create a sim with graphics enabled for camera-sensor trace rendering."""
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 120.0
    sim_params.substeps = 4
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 12
    sim_params.physx.num_velocity_iterations = 4
    sim_params.physx.contact_collection = gymapi.CC_ALL_SUBSTEPS
    sim_params.physx.contact_offset = 0.006
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.use_gpu = False
    sim_params.use_gpu_pipeline = False
    return gym.create_sim(0, int(graphics_device_id), gymapi.SIM_PHYSX, sim_params)


def _load_trace(path: Path, max_frames: int = 0, frame_stride: int = 1, start_index: int = 0) -> list[dict]:
    frames = []
    stride = max(1, int(frame_stride))
    start_index = max(0, int(start_index))
    with path.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            if not line.strip():
                continue
            if index < start_index:
                continue
            if index % stride != 0:
                continue
            frames.append(json.loads(line))
            if max_frames and len(frames) >= max_frames:
                break
    return frames


def _camera(gym, env, width: int, height: int):
    props = gymapi.CameraProperties()
    props.width = width
    props.height = height
    props.horizontal_fov = 104.0
    cam = gym.create_camera_sensor(env, props)
    pallet_center = pallet_center_near_table()
    mid_x = 0.5 * (TABLE_POSE[0] + pallet_center[0])
    mid_y = 0.5 * (TABLE_POSE[1] + pallet_center[1])
    gym.set_camera_location(
        cam,
        env,
        gymapi.Vec3(mid_x - 3.9, mid_y + 5.2, 4.35),
        gymapi.Vec3(mid_x + 0.45, mid_y + 0.35, 0.55),
    )
    return cam


def _create_frame_env(gym, sim, robot_asset, frame: dict, width: int, height: int):
    env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(6.5, 6.5, 2.5), 1)
    fixed_opts = gymapi.AssetOptions()
    fixed_opts.fix_base_link = True
    table_asset = gym.create_box(sim, TABLE_SIZE[0], TABLE_SIZE[1], TABLE_SIZE[2], fixed_opts)
    pallet_asset = gym.create_box(sim, PALLET_SIZE[0], PALLET_SIZE[1], PALLET_THICKNESS, fixed_opts)
    root = frame.get("robot", {}).get("root_pose", [0.0, 0.0, 0.0, 0.0])
    if robot_asset is None:
        raise RuntimeError("robot_asset is required for final trace rendering")
    robot = gym.create_actor(env, robot_asset, _make_transform(root[0], root[1], root[2], root[3]), "robot", 0, 1)
    table = gym.create_actor(env, table_asset, _make_transform(*TABLE_POSE), "table", 0, 0)
    pallet_center = pallet_center_near_table()
    pallet = gym.create_actor(env, pallet_asset, _make_transform(pallet_center[0], pallet_center[1], PALLET_SURFACE_Z - PALLET_THICKNESS * 0.5), "pallet", 0, 0)
    _set_color(gym, env, table, (1.0, 1.0, 1.0))
    _set_color(gym, env, pallet, (0.15, 0.55, 0.45))
    _set_shape_friction(gym, env, robot, 6.0)
    _add_box_wireframe(
        gym,
        sim,
        env,
        (pallet_center[0], pallet_center[1], PALLET_SURFACE_Z + 0.5),
        (1.0, 1.0, 1.0),
        "stack_1m_frame",
        (0.05, 0.9, 0.95),
        thickness=0.010,
    )

    pct = frame.get("pct") or {}
    target_world = pct.get("target_world_center_m")
    target_size = pct.get("placed_size_m")
    if target_world and target_size:
        _add_box_wireframe(gym, sim, env, target_world, target_size, "target_ghost", (1.0, 0.95, 0.05), thickness=0.012)

    trace_names = frame.get("robot", {}).get("dof_names", [])
    trace_values = frame.get("robot", {}).get("dof_values", [])
    if trace_names and trace_values:
        name_to_value = {name: float(value) for name, value in zip(trace_names, trace_values)}
        asset_names = list(gym.get_asset_dof_names(robot_asset))
        state = gym.get_actor_dof_states(env, robot, gymapi.STATE_ALL)
        for index, name in enumerate(asset_names):
            state["pos"][index] = name_to_value.get(name, 0.0)
            state["vel"][index] = 0.0
        gym.set_actor_dof_states(env, robot, state, gymapi.STATE_ALL)
        gym.set_actor_dof_position_targets(env, robot, state["pos"])

    palette = [(0.9, 0.05, 0.02), (0.2, 0.45, 0.95), (0.95, 0.72, 0.12), (0.25, 0.75, 0.35)]
    for box in frame.get("boxes", []):
        if not box.get("visible", True):
            continue
        state = box.get("state")
        if state == "NOT_SPAWNED":
            continue
        size = [float(v) for v in box["size_m"]]
        pose = [float(v) for v in box["pose"]]
        opts = gymapi.AssetOptions()
        opts.fix_base_link = state == "PLACED_FROZEN"
        asset = gym.create_box(sim, size[0], size[1], size[2], opts)
        actor = gym.create_actor(env, asset, _make_transform(pose[0], pose[1], pose[2], pose[3]), f"box_{box['item_index']}", 0, 0)
        color = palette[int(box["item_index"]) % len(palette)]
        if state == "HELD_ATTACHED":
            color = (0.95, 0.18, 0.08)
        _set_color(gym, env, actor, color)
        _set_shape_friction(gym, env, actor, 6.0)
    cam = _camera(gym, env, width, height)
    return env, cam


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a palletizing trace as an mp4.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=368)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0, help="Start from this raw trace line index.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Render every Nth trace frame for accelerated videos.")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--frames-only", action="store_true", help="Write PNG frames but do not assemble mp4.")
    parser.add_argument("--keep-frame-dir", action="store_true", help="Do not delete an existing output frame directory.")
    parser.add_argument(
        "--graphics-device-id",
        type=int,
        default=int(os.environ.get("MOVE_RENDER_GRAPHICS_DEVICE_ID", os.environ.get("MOVE_GRAPHICS_DEVICE_ID", "0"))),
        help="Graphics device for Isaac Gym camera sensors. Use -1 only for no-camera execution, not rendering.",
    )
    args = parser.parse_args()

    frames = _load_trace(args.trace, args.max_frames, args.frame_stride, args.start_index)
    metrics = None
    if args.metrics is not None and args.metrics.exists():
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if not frames:
        raise RuntimeError(f"Trace is empty: {args.trace}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = args.out.with_suffix("")
    if frame_dir.exists() and not args.keep_frame_dir:
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    chunk_size = max(1, int(args.chunk_size))
    for start in range(0, len(frames), chunk_size):
        chunk = frames[start:start + chunk_size]
        gym = gymapi.acquire_gym()
        sim = _create_render_sim(gym, args.graphics_device_id)
        if sim is None:
            raise RuntimeError(f"Failed to create Isaac Gym simulation with graphics_device_id={args.graphics_device_id}")
        try:
            plane = gymapi.PlaneParams()
            plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            gym.add_ground(sim, plane)
            robot_asset = load_robot_asset(gym, sim)
            env_cameras = []
            for offset, frame in enumerate(chunk):
                env, cam = _create_frame_env(gym, sim, robot_asset, frame, args.width, args.height)
                if int(cam) < 0:
                    raise RuntimeError(f"Failed to create camera for trace frame {start + offset}")
                env_cameras.append((env, cam))
            gym.prepare_sim(sim)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            for offset, (env, cam) in enumerate(env_cameras):
                path = frame_dir / f"frame_{args.start_index + start + offset:05d}.png"
                gym.write_camera_image_to_file(sim, env, cam, gymapi.IMAGE_COLOR, str(path))
                written.append(path)
        finally:
            gym.destroy_sim(sim)
        print(f"rendered_trace_frames {len(written)}/{len(frames)}", flush=True)

    if args.frames_only:
        print(f"wrote_frames={frame_dir} frames={len(written)}")
        return
    images = [_overlay_frame(imageio.imread(path)[:, :, :3], frames[index], metrics) for index, path in enumerate(written)]
    imageio.mimsave(args.out, images, fps=args.fps)
    print(f"wrote_video={args.out} frames={len(images)}")


if __name__ == "__main__":
    main()
