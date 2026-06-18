#!/usr/bin/env bash
# 稳定的逐帧渲染脚本：每个关键帧在独立 Isaac Gym 进程中渲染，避免 Lavapipe 多帧崩溃
#
# 用法:
#   ./scripts/render_trace_robust.sh TRACE.jsonl METRICS.json OUT.mp4 [FPS] [FRAME_STRIDE]
#
# 示例:
#   ./scripts/render_trace_robust.sh \
#     outputs/my_run/trace.jsonl \
#     outputs/my_run/metrics.json \
#     outputs/my_run/render.mp4 12 15

set -euo pipefail

TRACE="$1"
METRICS="$2"
OUT_MP4="$3"
FPS="${4:-12}"
FRAME_STRIDE="${5:-15}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-368}"

BPP_PYTHON="/home/ubuntu/anaconda3/envs/bpp/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRAME_DIR="${OUT_MP4%.mp4}_frames"
RENDER_SCRIPT="${REPO_ROOT}/render_palletizing_trace.py"

# ---- 计算总帧数 ----
TRACE_LINES=$(wc -l < "$TRACE")
TOTAL_FRAMES=$(( (TRACE_LINES + FRAME_STRIDE - 1) / FRAME_STRIDE ))
echo "trace_lines=${TRACE_LINES} total_keyframes=${TOTAL_FRAMES} stride=${FRAME_STRIDE}"

mkdir -p "$FRAME_DIR"

# ---- 逐帧渲染 ----
RENDERED=0
SKIPPED=0

for ((i=0; i<TRACE_LINES; i+=FRAME_STRIDE)); do
    FRAME_NAME="$(printf 'frame_%05d.png' "$i")"
    if [[ -f "$FRAME_DIR/$FRAME_NAME" ]]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo -n "rendering frame $i/$TRACE_LINES ... "
    if MOVE_GRAPHICS_DEVICE_ID=0 MOVE_RENDER_GRAPHICS_DEVICE_ID=0 \
       "$BPP_PYTHON" -m move.render_palletizing_trace \
         --trace "$TRACE" \
         --metrics "$METRICS" \
         --out "$OUT_MP4" \
         --width "$WIDTH" --height "$HEIGHT" --fps "$FPS" \
         --start-index "$i" --max-frames 1 --frame-stride 1 --chunk-size 1 \
         --frames-only --keep-frame-dir \
         --graphics-device-id 0 \
         > "$FRAME_DIR/render_${i}.log" 2>&1; then
        # 检查是否真的生成了帧
        if [[ -f "$FRAME_DIR/$FRAME_NAME" ]]; then
            echo "ok (${RENDERED}+1)"
            RENDERED=$((RENDERED + 1))
        else
            echo "no_frame"
            SKIPPED=$((SKIPPED + 1))
        fi
    else
        echo "crashed (skipped)"
        SKIPPED=$((SKIPPED + 1))
    fi
done

echo "done rendered=${RENDERED} skipped=${SKIPPED}"

# ---- 组装 mp4 ----
if [[ $RENDERED -eq 0 ]]; then
    echo "No frames rendered, aborting."
    exit 1
fi

echo "assembling mp4 ..."
"$BPP_PYTHON" -c "
from pathlib import Path
import imageio.v2 as imageio

frame_dir = Path('$FRAME_DIR')
frames = sorted(frame_dir.glob('frame_*.png'))
images = [imageio.imread(str(p))[:, :, :3] for p in frames]
out = Path('$OUT_MP4')
imageio.mimsave(str(out), images, fps=$FPS)
print(f'wrote_video={out} frames={len(images)} fps=$FPS')
"

echo "final_video=$OUT_MP4"
ls -lh "$OUT_MP4"
