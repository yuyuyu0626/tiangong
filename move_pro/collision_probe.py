"""碰撞度量：几何 AABB 重叠检测（阶段二的"测量手段"）。

仿真是 kinematic（箱子被强制锁定、机器人被 teleport），PhysX 接触力不可靠，
故用几何 AABB 重叠来量化"可视化里看到的穿透"。检测对象：
- 搬运中的箱子（主要碰撞体，AABB 精确——PCT 放置 yaw=0）；
- 机器人关键连杆（手掌/手指/前臂/躯体），用连杆世界位置 + 保守半径近似 AABB。
对照基准：已放好的箱子（kinematic 固定在各自 PCT 目标位姿）。

只读不改放置行为，纯观测。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from isaacgym import gymapi  # type: ignore


# 机器人连杆的保守碰撞半径（米）。连杆精确网格尺寸难取，用位置中心套球/立方近似。
# 手掌和手指是抓取时最贴近邻箱的部位，给略大半径；手臂连杆较粗，给中等半径。
_LINK_RADII = {
    "palm": 0.06,     # xhand_*_hand_link：手掌主体
    "finger": 0.025,  # 各指节 tip/link
    "wrist": 0.05,
    "elbow": 0.06,
    "shoulder": 0.07,
    "forearm": 0.05,
    "body": 0.10,     # base/waist/leg/body_yaw 躯体
    "default": 0.04,
}

# 默认参与检测的连杆类别（手 + 前臂；躯体一般离托盘远，默认也纳入以防底盘蹭箱）。
_DEFAULT_LINK_KEYWORDS = (
    "hand_link", "tip", "thumb", "index", "mid", "ring", "pinky",  # 手 + 手指
    "wrist", "elbow",                                              # 前臂
)


@dataclass(frozen=True)
class AABB:
    """轴对齐包围盒，min/max 各 (x,y,z)。"""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]


def box_aabb(center, size) -> AABB:
    """箱子轴对齐包围盒（PCT 放置 yaw=0，AABB 精确）。center/size 为世界 (x,y,z)。"""
    hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    return AABB(
        (center[0] - hx, center[1] - hy, center[2] - hz),
        (center[0] + hx, center[1] + hy, center[2] + hz),
    )


def _sphere_aabb(center, radius) -> AABB:
    return AABB(
        (center[0] - radius, center[1] - radius, center[2] - radius),
        (center[0] + radius, center[1] + radius, center[2] + radius),
    )


def overlap_extent(a: AABB, b: AABB) -> tuple[float, float, float]:
    """三轴重叠长度（负值表示该轴分离）。"""
    return (
        min(a.hi[0], b.hi[0]) - max(a.lo[0], b.lo[0]),
        min(a.hi[1], b.hi[1]) - max(a.lo[1], b.lo[1]),
        min(a.hi[2], b.hi[2]) - max(a.lo[2], b.lo[2]),
    )


def is_overlapping(a: AABB, b: AABB, eps: float = 1e-6) -> bool:
    ox, oy, oz = overlap_extent(a, b)
    return ox > eps and oy > eps and oz > eps


def overlap_volume(a: AABB, b: AABB) -> float:
    ox, oy, oz = overlap_extent(a, b)
    if ox <= 0 or oy <= 0 or oz <= 0:
        return 0.0
    return ox * oy * oz


def penetration_depth(a: AABB, b: AABB) -> float:
    """重叠时的最小穿透深度（= 三轴重叠中的最小正值；最小轴是脱离碰撞的最短方向）。

    不重叠返回 0。
    """
    ox, oy, oz = overlap_extent(a, b)
    if ox <= 0 or oy <= 0 or oz <= 0:
        return 0.0
    return min(ox, oy, oz)


def _link_radius(name: str) -> float:
    low = name.lower()
    if "hand_link" in low:
        return _LINK_RADII["palm"]
    if any(k in low for k in ("tip", "thumb", "index", "mid", "ring", "pinky")):
        return _LINK_RADII["finger"]
    if "wrist" in low:
        return _LINK_RADII["wrist"]
    if "elbow" in low:
        return _LINK_RADII["elbow"]
    if "shoulder" in low:
        return _LINK_RADII["shoulder"]
    if any(k in low for k in ("base", "waist", "leg", "body_yaw", "torso")):
        return _LINK_RADII["body"]
    return _LINK_RADII["default"]


def robot_link_aabbs(gym, env, robot, keywords=_DEFAULT_LINK_KEYWORDS) -> list[tuple[str, AABB]]:
    """机器人关键连杆的近似 AABB（位置中心 + 保守半径）。

    keywords=None 表示所有连杆；否则只取名字含任一关键字的连杆。
    """
    states = gym.get_actor_rigid_body_states(env, robot, gymapi.STATE_POS)
    names = gym.get_actor_rigid_body_names(env, robot)
    result: list[tuple[str, AABB]] = []
    for name, state in zip(names, states):
        if keywords is not None and not any(k in name.lower() for k in keywords):
            continue
        p = state["pose"]["p"]
        center = (float(p["x"]), float(p["y"]), float(p["z"]))
        result.append((name, _sphere_aabb(center, _link_radius(name))))
    return result


@dataclass
class CollisionStats:
    """单个箱子放置过程的碰撞累计。"""

    box_index: int
    frames_checked: int = 0
    frames_in_collision: int = 0
    max_penetration: float = 0.0
    max_pen_source: str = ""        # 触发最大穿透的部位（连杆名或 "carried_box"）
    max_pen_victim: int = -1        # 被穿透的已放箱 index

    @property
    def collision_ratio(self) -> float:
        if self.frames_checked == 0:
            return 0.0
        return self.frames_in_collision / self.frames_checked


def probe_frame(
    gym,
    env,
    robot,
    carried_box_actor,
    carried_box_size,
    placed_boxes: list[tuple[int, tuple[float, float, float], tuple[float, float, float]]],
    stats: CollisionStats,
    include_carried: bool = True,
    include_links: bool = True,
) -> None:
    """单帧检测：搬运箱 + 机器人连杆 vs 所有已放箱，更新 stats。

    placed_boxes: [(box_index, center, size), ...] 已放好的箱（kinematic 固定）。
    carried_box_actor: 当前搬运中的箱 actor（None 表示无，如悬空回退或纯检测连杆）。
    """
    from move.tasks.grab_test_task import _actor_center

    placed_aabbs = [(idx, box_aabb(c, s)) for idx, c, s in placed_boxes]
    if not placed_aabbs:
        stats.frames_checked += 1
        return

    probes: list[tuple[str, AABB]] = []
    if include_carried and carried_box_actor is not None:
        cc = _actor_center(gym, env, carried_box_actor)
        probes.append(("carried_box", box_aabb(cc, carried_box_size)))
    if include_links:
        probes.extend(robot_link_aabbs(gym, env, robot))

    frame_has_collision = False
    for source_name, probe_box in probes:
        for victim_idx, placed in placed_aabbs:
            pen = penetration_depth(probe_box, placed)
            if pen > 0.0:
                frame_has_collision = True
                if pen > stats.max_penetration:
                    stats.max_penetration = pen
                    stats.max_pen_source = source_name
                    stats.max_pen_victim = victim_idx

    stats.frames_checked += 1
    if frame_has_collision:
        stats.frames_in_collision += 1


@dataclass
class CollisionReport:
    """整批仿真的碰撞汇总。"""

    per_box: list[CollisionStats] = field(default_factory=list)

    def add(self, stats: CollisionStats) -> None:
        self.per_box.append(stats)

    def summary_line(self) -> str:
        n_colliding = sum(1 for s in self.per_box if s.frames_in_collision > 0)
        worst = max(self.per_box, key=lambda s: s.max_penetration, default=None)
        worst_str = (
            f"worst box={worst.box_index} pen={worst.max_penetration:.4f} "
            f"src={worst.max_pen_source} victim={worst.max_pen_victim}"
            if worst and worst.max_penetration > 0
            else "none"
        )
        return (
            f"move_pro_collision_summary boxes_with_collision={n_colliding}/{len(self.per_box)} "
            f"{worst_str}"
        )
