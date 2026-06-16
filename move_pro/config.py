"""move_pro 共享配置与常量。"""

from __future__ import annotations

import math
from pathlib import Path

# ---- 路径 ----
MOVE_PRO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MOVE_PRO_ROOT.parent
MOVE_ROOT = REPO_ROOT / "move"
PCT_ROOT = REPO_ROOT / "Online-3D-BPP-PCT"

MOVE_URDF = MOVE_ROOT / "assets" / "integrated" / "tianyi_xhand_move.urdf"

# ---- 托盘与场景几何（与 move/planning.py 对齐） ----
TABLE_SIZE = (0.80, 0.60, 0.06)
TABLE_POSE = (0.60, 0.0, 0.60)
PALLET_SIZE = (1.0, 1.0)
PALLET_SURFACE_Z = 0.10
PALLET_DIAGONAL_DISTANCE = 5.0
PALLET_THICKNESS = 0.02
STACK_HEIGHT_LIMIT = 1.6           # 比 move 默认更高，容纳更大的堆垛
DEFAULT_ROOT_Z = 0.0
DEFAULT_HEIGHT_CLEARANCE = 0.18
SAFE_CORRIDOR_MARGIN = 1.20
MOVE_STAND_OFF = 0.60
TABLE_RETREAT = 0.90
STACK_TMP_RETREAT = 0.80

# ---- 仿真物理 ----
BOX_MASS = 0.35
PRESET_BOX_MASS = 0.40
SIM_DT = 1.0 / 60.0

# ---- 托盘世界坐标 ----
def pallet_center_world() -> tuple[float, float, float]:
    """托盘中心世界坐标，与 move/planning.py 一致。"""
    offset = PALLET_DIAGONAL_DISTANCE / math.sqrt(2.0)
    return (
        TABLE_POSE[0] + offset,
        TABLE_POSE[1] + offset,
        PALLET_SURFACE_Z,
    )

# ---- BPP 决策器配置 ----
# 物品尺寸集：不同大小箱子的尺寸组合
# 离散模式从给定列表中选取，连续模式从均匀分布采样
DEFAULT_ITEM_SET = [
    (1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1),
    (1, 2, 2), (2, 1, 2), (2, 2, 1),
    (2, 2, 2), (2, 2, 3), (2, 3, 2), (3, 2, 2),
    (2, 3, 3), (3, 2, 3), (3, 3, 2),
    (3, 3, 3), (3, 3, 4), (3, 4, 3), (4, 3, 3),
    (4, 4, 4),
]

# 箱子缩放因子：PCT 使用抽象坐标 [0, container_size]，
# 需要映射到真实尺寸
# container_size = (10, 10, 10) 对应 1.0m × 1.0m × 1.6m 的托盘空间
BIN_TO_WORLD_SCALE = (
    PALLET_SIZE[0],                    # 10 units → 1.0 m
    PALLET_SIZE[1],                    # 10 units → 1.0 m
    STACK_HEIGHT_LIMIT,                # 10 units → 1.6 m (高度上限)
)

# ---- 机器人可达性约束 ----
# 机器人是移动底盘，正对 +x 站在托盘近端边缘，靠手臂前伸放置。
# 经 tools/probe_reach.py 实测：放置可达上限约为托盘近端起 0.70m 深
# （世界 x≈4.34），超过则 place IK 误差 > PLACE_ERROR_LIMIT(0.06)。
# 换算成 PCT bin 坐标（容器 x=10 对应托盘 1.0m）：箱子远端边缘 lx+x_size 不得超过 7。
# BPPDecider 用它过滤够不到的候选格，避免 LSAH 把箱子放到机器人够不着的地方。
MAX_REACH_X_BIN = 7

# ---- IK 参数 ----
DEFAULT_IK_ITERATIONS = 80
PLACE_IK_ITERATIONS = 80
PICK_IK_ITERATIONS = 16
MOVE_APPROACH_HEIGHT = 0.15
MOVE_PREPLACE_HEIGHT = 0.16
PICK_LIFT_HEIGHT = 0.20
PICK_READY_Z_OFFSET = 0.12
PLACE_RELEASE_HEIGHT = 0.030
