#!/usr/bin/env bash
# move_pro 仿真启动脚本
#
# 作用：自动注入 rlgpu 环境的 PATH / LD_LIBRARY_PATH，再运行 move_pro.run。
#   - PATH 加 rlgpu/bin：否则 gymtorch JIT 编译报 "Ninja is required"
#   - LD_LIBRARY_PATH 加 rlgpu/lib：否则报 "libpython3.7m.so.1.0: cannot open ..."
#
# 用法：
#   ./run_sim.sh                                  # 默认 sim 模式，带 viewer 窗口
#   ./run_sim.sh --mode sim --num-boxes 6 --fast  # 透传任意 move_pro.run 参数
#   ./run_sim.sh --mode sim --headless --fast --num-boxes 3
#   ./run_sim.sh --mode plan --num-boxes 15 --verbose   # 仅 BPP 决策，不开 Isaac Gym
#
# 所有参数原样透传给 `python -m move_pro.run`，详见 `./run_sim.sh -h`。

set -euo pipefail

# ---- conda env 路径（如换机器，改这一行即可） ----
RLGPU_ENV="/home/u2004/miniconda3/envs/rlgpu"

# ---- 定位仓库根目录（脚本所在目录的上一级） ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ ! -x "$RLGPU_ENV/bin/python" ]]; then
    echo "错误：找不到 rlgpu 环境的 python: $RLGPU_ENV/bin/python" >&2
    echo "请确认 conda env 'rlgpu' 已安装，或修改脚本顶部的 RLGPU_ENV。" >&2
    exit 1
fi

export PATH="$RLGPU_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$RLGPU_ENV/lib:${LD_LIBRARY_PATH:-}"

# ---- 默认参数：无参数调用时跑一个带窗口的 6 箱仿真 ----
if [[ $# -eq 0 ]]; then
    set -- --mode sim --method LSAH --num-boxes 6 --seed 42 --fast
fi

cd "$REPO_ROOT"
exec python -m move_pro.run "$@"
