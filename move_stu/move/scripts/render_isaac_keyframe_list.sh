#!/usr/bin/env bash
set -u

if [[ $# -lt 5 ]]; then
  echo "usage: $0 TRACE METRICS KEYFRAME_LIST FRAME_DIR OUT_MP4 [FPS] [WIDTH] [HEIGHT]" >&2
  exit 2
fi

TRACE="$1"
METRICS="$2"
KEYFRAMES="$3"
FRAME_DIR="$4"
OUT_MP4="$5"
FPS="${6:-8}"
WIDTH="${7:-640}"
HEIGHT="${8:-368}"
RENDER_OUT="${FRAME_DIR%/}.mp4"

mkdir -p "$FRAME_DIR"

while read -r idx; do
  [[ -z "$idx" ]] && continue
  frame_name="$(printf 'frame_%05d.png' "$idx")"
  [[ -f "$FRAME_DIR/$frame_name" ]] && continue
  env VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/2024233240/local/nvidia-vulkan-550/usr/share/vulkan/icd.d/lvp_local_icd.json}" \
    MOVE_GRAPHICS_DEVICE_ID="${MOVE_GRAPHICS_DEVICE_ID:-0}" \
    MOVE_RENDER_GRAPHICS_DEVICE_ID="${MOVE_RENDER_GRAPHICS_DEVICE_ID:-0}" \
    /2024233240/move/run_move_bpp_env.sh -m move.render_palletizing_trace \
    --trace "$TRACE" \
    --metrics "$METRICS" \
    --out "$RENDER_OUT" \
    --width "$WIDTH" --height "$HEIGHT" --fps "$FPS" \
    --start-index "$idx" --max-frames 1 --frame-stride 1 --chunk-size 1 \
    --frames-only --keep-frame-dir --graphics-device-id 0 \
    > "$FRAME_DIR/render_${idx}.log" 2>&1 || true
done < "$KEYFRAMES"

/2024233240/move/run_move_bpp_env.sh - "$FRAME_DIR" "$OUT_MP4" "$FPS" <<'PY'
from pathlib import Path
import sys

import imageio.v2 as imageio

frame_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
fps = int(sys.argv[3])

frames = sorted(frame_dir.glob("frame_*.png"))
if not frames:
    raise RuntimeError(f"No frames found in {frame_dir}")
images = [imageio.imread(path)[:, :, :3] for path in frames]
out.parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(out, images, fps=fps)
print(f"wrote_video={out} frames={len(images)} fps={fps}")
PY
