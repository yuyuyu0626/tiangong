"""PCT 可学习模型放置决策器（阶段一）。

替换 bpp_decider 的启发式（LASH/OnlineBPH/DBL/BR），改用 PCT 训练好的 GAT 策略网络
从 EMS 候选里挑放置点。逻辑完全复刻 PCT 的推理路径：
- 观测构造  = PCT/pct_envs/PctDiscrete0/bin3D.py: cur_observation / get_possible_position
- 推理循环  = PCT/evaluation_tools.py: evaluate（deterministic=True）
- 动作转换  = bin3D.py: LeafNode2Action + space.drop_box / GENEMS

决策在 PCT 原生 bin 坐标系 (10,10,10) 完成，再映射到 move 世界坐标（统一 0.1 m/unit）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from move_pro.config import (
    PALLET_SURFACE_Z,
    PCT_CONTAINER_SIZE,
    PCT_EMBEDDING_SIZE,
    PCT_GAT_LAYER_NUM,
    PCT_HIDDEN_SIZE,
    PCT_INTERNAL_NODE_HOLDER,
    PCT_INTERNAL_NODE_LENGTH,
    PCT_ITEM_SET,
    PCT_LEAF_NODE_HOLDER,
    PCT_MODEL_PATH,
    PCT_NEXT_HOLDER,
    PCT_NORM_FACTOR,
    PCT_BIN_TO_WORLD_SCALE,
    PCT_SETTING,
    PCT_SHUFFLE,
    pallet_center_world,
)
from move_pro.bpp_decider import Placement
from move_pro.pct_core import Space
from move_pro.pct_core.policy import DRL_GAT, load_policy


class _PolicyArgs:
    """DRL_GAT 构造所需的最小参数集（替代 PCT argparse 的 args）。"""

    def __init__(self) -> None:
        self.embedding_size = PCT_EMBEDDING_SIZE
        self.hidden_size = PCT_HIDDEN_SIZE
        self.gat_layer_num = PCT_GAT_LAYER_NUM
        self.internal_node_holder = PCT_INTERNAL_NODE_HOLDER
        self.internal_node_length = PCT_INTERNAL_NODE_LENGTH
        self.leaf_node_holder = PCT_LEAF_NODE_HOLDER


@dataclass(frozen=True)
class PctPlacement:
    """一个 PCT 决策的放置（bin 坐标 + 世界中心）。"""

    lx: int
    ly: int
    lz: int
    x: int
    y: int
    z: int
    orientation: int
    world_x: float
    world_y: float
    world_z: float

    @property
    def world_center(self) -> tuple[float, float, float]:
        return (self.world_x, self.world_y, self.world_z)


class PctModelPlanner:
    """用 PCT 训练好的模型对一串箱子做在线放置决策。

    单容器单 episode：依次喂箱子，模型每步从 EMS 候选里挑一个放置点，直到放不下
    （无可行候选）或达到 num_boxes 上限。
    """

    def __init__(
        self,
        model_path=PCT_MODEL_PATH,
        container_size: tuple[int, int, int] = PCT_CONTAINER_SIZE,
        setting: int = PCT_SETTING,
        device: str = "cpu",
    ) -> None:
        self.container_size = tuple(int(v) for v in container_size)
        self.setting = int(setting)
        self.orientation = 6 if self.setting == 2 else 2
        self.internal_node_holder = PCT_INTERNAL_NODE_HOLDER
        self.leaf_node_holder = PCT_LEAF_NODE_HOLDER
        self.next_holder = PCT_NEXT_HOLDER
        self.norm_factor = PCT_NORM_FACTOR
        self.shuffle = PCT_SHUFFLE
        self.item_set = [tuple(int(v) for v in s) for s in PCT_ITEM_SET]
        self.size_minimum = int(np.min(np.array(self.item_set)))
        self.device = torch.device(device)

        # 策略网络（与 PCT evaluation 完全一致的加载方式）。
        self.policy = DRL_GAT(_PolicyArgs()).to(self.device)
        self.policy = load_policy(str(model_path), self.policy)
        self.policy.eval()

        self._build_space()

    def _build_space(self) -> None:
        self.space = Space(
            *self.container_size,
            self.size_minimum,
            self.internal_node_holder,
        )
        self.next_box_vec = np.zeros((self.next_holder, 9))
        self._placements: list[PctPlacement] = []

    def reset(self) -> None:
        self.space.reset()
        self.next_box_vec[:] = 0
        self._placements = []

    # ---- 观测构造（复刻 bin3D.cur_observation / get_possible_position）----

    def _get_possible_position(self, next_box) -> np.ndarray:
        """生成 EMS 候选叶子节点并过滤可行性。等价 bin3D.get_possible_position（EMS+setting）。"""
        all_position = self.space.EMSPoint(next_box, self.setting)
        if self.shuffle:
            np.random.shuffle(all_position)

        leaf_node_vec = np.zeros((self.leaf_node_holder, 9))
        tmp_list = []
        for position in all_position:
            xs, ys, zs, xe, ye, ze = position
            x = xe - xs
            y = ye - ys
            z = ze - zs
            if self.space.drop_box_virtual([x, y, z], (xs, ys), False, 1.0, self.setting):
                # 注意：与 PCT 一致，ze 写容器高度而非真实顶面（模型如此训练）。
                tmp_list.append([xs, ys, zs, xe, ye, self.container_size[2], 0, 0, 1])
                if len(tmp_list) >= self.leaf_node_holder:
                    break

        if tmp_list:
            leaf_node_vec[0:len(tmp_list)] = np.array(tmp_list)
        return leaf_node_vec

    def _observation(self, next_box) -> tuple[np.ndarray, np.ndarray]:
        """构造模型观测张量并返回 (obs_1d, leaf_node_vec)。等价 bin3D.cur_observation。"""
        leaf_nodes = self._get_possible_position(next_box)
        next_box_sorted = sorted(list(next_box))
        self.next_box_vec[:] = 0
        self.next_box_vec[:, 3:6] = next_box_sorted
        self.next_box_vec[:, 0] = 1.0  # density (setting1 恒为 1)
        self.next_box_vec[:, -1] = 1
        obs = np.reshape(
            np.concatenate((self.space.box_vec, leaf_nodes, self.next_box_vec)),
            (-1,),
        )
        return obs, leaf_nodes

    # ---- 动作转换（复刻 bin3D.LeafNode2Action）----

    def _leaf_node_to_action(self, leaf_node, next_box):
        if np.sum(leaf_node[0:6]) == 0:
            return None  # 无效叶子，episode 结束
        x = int(leaf_node[3] - leaf_node[0])
        y = int(leaf_node[4] - leaf_node[1])
        remaining = list(next_box)
        remaining.remove(x)
        remaining.remove(y)
        z = remaining[0]
        action = (0, int(leaf_node[0]), int(leaf_node[1]))  # (rotation_flag, lx, ly)
        oriented = (x, y, int(z))
        return action, oriented

    def _to_world(self, lx, ly, lz, x, y, z) -> tuple[float, float, float]:
        """bin 坐标 → move 世界坐标（统一 0.1 尺度，托盘中心对齐）。"""
        center_x, center_y, _ = pallet_center_world()
        sx, sy, sz = PCT_BIN_TO_WORLD_SCALE
        bx, by, _bz = self.container_size
        return (
            center_x - sx * bx / 2.0 + (lx + x / 2.0) * sx,
            center_y - sy * by / 2.0 + (ly + y / 2.0) * sy,
            PALLET_SURFACE_Z + (lz + z / 2.0) * sz,
        )

    # ---- 推理循环（复刻 evaluation_tools.evaluate，deterministic）----

    @torch.no_grad()
    def plan(
        self,
        box_sequence: Optional[Sequence[tuple[int, int, int]]] = None,
        num_boxes: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> list[PctPlacement]:
        """对一串箱子跑 PCT 决策，返回成功放置的 placement 列表。

        box_sequence 给定则按序放（达 num_boxes 或放不下即停）；否则从 PCT item_set
        随机采样，跑到放不下或达到 num_boxes。
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.reset()
        sequence = list(box_sequence) if box_sequence is not None else None
        limit = num_boxes if num_boxes is not None else (len(sequence) if sequence else 10_000)

        step = 0
        while step < limit:
            next_box = self._next_box(sequence, step)
            if next_box is None:
                break
            next_box = [int(v) for v in next_box]

            obs, leaf_nodes = self._observation(next_box)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            all_nodes = obs_t.reshape((1, -1, 9))

            _logp, selected_idx, _entropy, _value = self.policy(
                all_nodes, True, normFactor=self.norm_factor
            )
            idx = int(selected_idx.squeeze().item())
            selected_leaf = leaf_nodes[idx]

            converted = self._leaf_node_to_action(selected_leaf[0:6], next_box)
            if converted is None:
                break  # 模型选了无效（全 0）叶子 → 放不下，episode 结束
            action, oriented = converted
            (_rot, lx, ly) = action

            if not self.space.drop_box(oriented, (lx, ly), False, 1.0, self.setting):
                break  # 提交失败（理论上不应发生）→ 结束
            packed = self.space.boxes[-1]
            self.space.GENEMS([
                packed.lx, packed.ly, packed.lz,
                packed.lx + packed.x, packed.ly + packed.y, packed.lz + packed.z,
            ])
            world = self._to_world(packed.lx, packed.ly, packed.lz, packed.x, packed.y, packed.z)
            self._placements.append(
                PctPlacement(
                    lx=int(packed.lx), ly=int(packed.ly), lz=int(packed.lz),
                    x=int(packed.x), y=int(packed.y), z=int(packed.z),
                    orientation=0, world_x=world[0], world_y=world[1], world_z=world[2],
                )
            )
            step += 1

        return list(self._placements)

    def _next_box(self, sequence, step):
        if sequence is not None:
            if step >= len(sequence):
                return None
            return sequence[step]
        idx = np.random.randint(0, len(self.item_set))
        return self.item_set[idx]

    @property
    def placements(self) -> tuple[PctPlacement, ...]:
        return tuple(self._placements)

    def utilization(self) -> float:
        return float(self.space.get_ratio())

    # ---- 与 bpp_decider 兼容的 Placement（供 integrator 复用 BoxTask）----

    def as_placement(self, p: PctPlacement) -> Placement:
        return Placement(
            lx=p.lx, ly=p.ly, lz=p.lz, x=p.x, y=p.y, z=p.z,
            orientation=p.orientation,
            world_x=p.world_x, world_y=p.world_y, world_z=p.world_z,
            feasible=True, score=0.0,
        )
