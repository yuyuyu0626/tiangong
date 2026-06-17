#!/usr/bin/env python3
"""Bridge Online-3D-BPP-PCT policy outputs to metric pallet placements."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch

MOVE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MOVE_ROOT.parent
DEFAULT_PCT_ROOT = REPO_ROOT / "external" / "Online-3D-BPP-PCT"




@dataclass(frozen=True)
class PctCandidate:
    item_index: int
    original_size_m: tuple[float, float, float]
    placed_size_m: tuple[float, float, float]
    min_corner_m: tuple[float, float, float]
    max_corner_m: tuple[float, float, float]
    action_leaf_index: int
    leaf_node: tuple[float, ...]
    utilization_if_committed: float
    policy_score: float = 0.0
    policy_rank: int = 0

@dataclass(frozen=True)
class PctPlacement:
    item_index: int
    original_size_m: tuple[float, float, float]
    placed_size_m: tuple[float, float, float]
    min_corner_m: tuple[float, float, float]
    max_corner_m: tuple[float, float, float]
    action_leaf_index: int
    utilization: float


class FixedSequenceCreator:
    """Minimal PCT BoxCreator-compatible online sequence provider."""

    def __init__(self, sequence_grid: Iterable[tuple[int, int, int]]):
        self.sequence = [tuple(int(v) for v in item) for item in sequence_grid]
        self.cursor = 0
        self.box_list: list[tuple[int, int, int]] = []

    def reset(self) -> None:
        self.cursor = 0
        self.box_list.clear()

    def generate_box_size(self, **_kwargs) -> None:
        if self.cursor < len(self.sequence):
            self.box_list.append(self.sequence[self.cursor])
            self.cursor += 1
        else:
            self.box_list.append((100, 100, 100))

    def preview(self, length: int):
        while len(self.box_list) < length:
            self.generate_box_size()
        return [tuple(v) for v in self.box_list[:length]]

    def drop_box(self) -> None:
        self.box_list.pop(0)


def _meters_to_grid(size_m: tuple[float, float, float], grid_scale: float) -> tuple[int, int, int]:
    grid = tuple(int(round(v / grid_scale)) for v in size_m)
    if any(v < 1 or v > 5 for v in grid):
        raise ValueError(f"Item size {size_m} maps to invalid PCT grid size {grid}; expected 1..5.")
    if any(abs(size_m[i] - grid[i] * grid_scale) > 1e-6 for i in range(3)):
        raise ValueError(f"Item size {size_m} is not aligned to {grid_scale} m PCT grid.")
    return grid


def _pct_args(device: str, model_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        setting=2,
        lnes="EMS",
        internal_node_holder=80,
        internal_node_length=6,
        leaf_node_holder=50,
        next_holder=1,
        shuffle=False,
        continuous=False,
        no_cuda=device == "cpu",
        device=device,
        seed=4,
        embedding_size=64,
        hidden_size=128,
        gat_layer_num=1,
        evaluate=True,
        load_model=True,
        model_path=str(model_path),
        container_size=[10, 10, 10],
        item_size_set=[(i, j, k) for i in range(1, 6) for j in range(1, 6) for k in range(1, 6)],
        id="PctDiscrete-v0",
        normFactor=0.1,
    )


def load_pct_modules(pct_root: Path):
    if not pct_root.exists():
        raise FileNotFoundError(f"PCT repo not found: {pct_root}")
    sys.path.insert(0, str(pct_root))
    from model import DRL_GAT  # type: ignore
    from tools import load_policy, registration_envs  # type: ignore
    from pct_envs.PctDiscrete0 import PackingDiscrete  # type: ignore

    return DRL_GAT, load_policy, registration_envs, PackingDiscrete



class PCTOnlineController:
    """Online propose/commit wrapper around the discrete PCT policy.

    ``propose`` evaluates one currently arrived box without mutating the real
    PCT packing state. ``commit`` performs the corresponding env.step only
    after robot execution succeeds.
    """

    def __init__(
        self,
        model_path: Path,
        pct_root: Path = DEFAULT_PCT_ROOT,
        device: str | None = None,
        grid_scale: float = 0.1,
    ) -> None:
        self.model_path = model_path.expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"PCT pretrained model not found: {self.model_path}")
        self.pct_root = pct_root
        self.grid_scale = grid_scale
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(device)
        self.args = _pct_args("cpu" if self.torch_device.type == "cpu" else self.torch_device.index or 0, self.model_path)
        DRL_GAT, load_policy, _registration_envs, PackingDiscrete = load_pct_modules(pct_root)
        self.policy = DRL_GAT(self.args).to(self.torch_device)
        self.policy = load_policy(str(self.model_path), self.policy).to(self.torch_device)
        self.policy.eval()
        self.PackingDiscrete = PackingDiscrete
        self.env = None
        self.obs = None
        self._pending: PctCandidate | None = None
        self._pending_leaf_node = None
        self._pending_grid: tuple[int, int, int] | None = None
        self._item_index = 0
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self.env = self.PackingDiscrete(
            setting=self.args.setting,
            container_size=self.args.container_size,
            item_set=self.args.item_size_set,
            internal_node_holder=self.args.internal_node_holder,
            leaf_node_holder=self.args.leaf_node_holder,
            LNES=self.args.lnes,
            shuffle=False,
        )
        if seed is not None:
            self.env.seed(seed)
        self.obs = self.env.reset()
        self._pending = None
        self._pending_leaf_node = None
        self._pending_grid = None
        self._item_index = 0

    def _set_current_box(self, grid: tuple[int, int, int]):
        assert self.env is not None
        self.env.box_creator.box_list = [tuple(int(v) for v in grid)]
        self.env.box_creator.cursor = getattr(self.env.box_creator, "cursor", 0)
        self.obs = self.env.cur_observation()

    def propose(self, box_size_m: tuple[float, float, float]) -> PctCandidate:
        assert self.env is not None and self.obs is not None
        grid = _meters_to_grid(tuple(float(v) for v in box_size_m), self.grid_scale)
        self._set_current_box(grid)
        obs_tensor = torch.as_tensor(self.obs, dtype=torch.float32, device=self.torch_device).view(1, -1, 9)
        with torch.no_grad():
            _logp, pointer, _entropy, _value = self.policy(obs_tensor, deterministic=True, normFactor=self.args.normFactor, evaluate=True)
        leaf_index = int(pointer.detach().cpu().view(-1)[0].item())
        leaf_nodes = self.obs.reshape((-1, 9))[self.args.internal_node_holder : self.args.internal_node_holder + self.args.leaf_node_holder]
        if leaf_index < 0 or leaf_index >= len(leaf_nodes) or leaf_nodes[leaf_index, 8] <= 0:
            raise RuntimeError(f"PCT selected invalid leaf index {leaf_index} for item {self._item_index}.")
        leaf_node = leaf_nodes[leaf_index].copy()

        virtual_env = copy.deepcopy(self.env)
        _obs, _reward, done, info = virtual_env.step(leaf_node)
        if done:
            raise RuntimeError(f"PCT failed to virtually place item {self._item_index} size={box_size_m}; info={info}")
        packed = virtual_env.space.boxes[-1]
        min_grid = (int(packed.lx), int(packed.ly), int(packed.lz))
        size_grid = (int(packed.x), int(packed.y), int(packed.z))
        max_grid = tuple(min_grid[i] + size_grid[i] for i in range(3))
        candidate = PctCandidate(
            item_index=self._item_index,
            original_size_m=tuple(float(v) for v in box_size_m),
            placed_size_m=tuple(v * self.grid_scale for v in size_grid),
            min_corner_m=tuple(v * self.grid_scale for v in min_grid),
            max_corner_m=tuple(v * self.grid_scale for v in max_grid),
            action_leaf_index=leaf_index,
            leaf_node=tuple(float(v) for v in leaf_node),
            utilization_if_committed=float(virtual_env.space.get_ratio()),
            policy_score=float(torch.exp(_logp.detach().cpu().view(-1)[0]).item()),
            policy_rank=0,
        )
        self._pending = candidate
        self._pending_leaf_node = leaf_node
        self._pending_grid = grid
        return candidate

    def _candidate_from_leaf(
        self,
        box_size_m: tuple[float, float, float],
        leaf_index: int,
        leaf_node,
        *,
        policy_score: float = 0.0,
        policy_rank: int = 0,
    ) -> PctCandidate:
        assert self.env is not None
        virtual_env = copy.deepcopy(self.env)
        _obs, _reward, done, info = virtual_env.step(leaf_node)
        if done:
            raise RuntimeError(f"PCT virtual step failed for leaf {leaf_index}; info={info}")
        packed = virtual_env.space.boxes[-1]
        min_grid = (int(packed.lx), int(packed.ly), int(packed.lz))
        size_grid = (int(packed.x), int(packed.y), int(packed.z))
        max_grid = tuple(min_grid[i] + size_grid[i] for i in range(3))
        return PctCandidate(
            item_index=self._item_index,
            original_size_m=tuple(float(v) for v in box_size_m),
            placed_size_m=tuple(v * self.grid_scale for v in size_grid),
            min_corner_m=tuple(v * self.grid_scale for v in min_grid),
            max_corner_m=tuple(v * self.grid_scale for v in max_grid),
            action_leaf_index=leaf_index,
            leaf_node=tuple(float(v) for v in leaf_node),
            utilization_if_committed=float(virtual_env.space.get_ratio()),
            policy_score=float(policy_score),
            policy_rank=int(policy_rank),
        )

    def ranked_leaf_candidates(self, box_size_m: tuple[float, float, float]) -> list[PctCandidate]:
        """Return all valid PCT leaves sorted by current policy preference.

        This does not mutate the real PCT environment. The first returned
        candidate is the same leaf the deterministic policy would choose.
        """

        assert self.env is not None and self.obs is not None
        grid = _meters_to_grid(tuple(float(v) for v in box_size_m), self.grid_scale)
        self._set_current_box(grid)
        obs_tensor = torch.as_tensor(self.obs, dtype=torch.float32, device=self.torch_device).view(1, -1, 9)
        with torch.no_grad():
            _action_log_prob, pointer, _dist_entropy, _hidden, dist = self.policy.actor(
                obs_tensor,
                deterministic=True,
                normFactor=self.args.normFactor,
                evaluate=True,
            )
        selected_leaf = int(pointer.detach().cpu().view(-1)[0].item())
        scores = dist.probs.detach().cpu().view(-1).numpy()
        leaf_nodes = self.obs.reshape((-1, 9))[self.args.internal_node_holder : self.args.internal_node_holder + self.args.leaf_node_holder]

        ranked_indices = [
            int(i)
            for i in np.argsort(-scores)
            if i >= 0 and i < len(leaf_nodes) and leaf_nodes[i, 8] > 0
        ]
        if selected_leaf in ranked_indices:
            ranked_indices.remove(selected_leaf)
        ranked_indices.insert(0, selected_leaf)

        candidates: list[PctCandidate] = []
        for rank, leaf_index in enumerate(ranked_indices):
            if leaf_index < 0 or leaf_index >= len(leaf_nodes) or leaf_nodes[leaf_index, 8] <= 0:
                continue
            try:
                candidate = self._candidate_from_leaf(
                    box_size_m,
                    leaf_index,
                    leaf_nodes[leaf_index].copy(),
                    policy_score=float(scores[leaf_index]),
                    policy_rank=int(rank),
                )
            except RuntimeError:
                continue
            candidates.append(candidate)

        if candidates:
            self._pending = candidates[0]
            self._pending_leaf_node = np.asarray(candidates[0].leaf_node, dtype=np.float32)
            self._pending_grid = grid
        return candidates

    def propose_robot_compatible(self, box_size_m: tuple[float, float, float], allow_yaw: bool = False) -> PctCandidate:
        first = self.propose(box_size_m)

        def compatible(candidate: PctCandidate) -> bool:
            o = candidate.original_size_m
            p = candidate.placed_size_m
            eps = 1e-6
            if all(abs(o[i] - p[i]) < eps for i in range(3)):
                return True
            return bool(allow_yaw and abs(o[2] - p[2]) < eps and abs(o[0] - p[1]) < eps and abs(o[1] - p[0]) < eps)

        if compatible(first):
            return first
        assert self.obs is not None
        leaf_nodes = self.obs.reshape((-1, 9))[self.args.internal_node_holder : self.args.internal_node_holder + self.args.leaf_node_holder]
        order = [i for i in range(len(leaf_nodes)) if i != first.action_leaf_index]
        for leaf_index in order:
            leaf_node = leaf_nodes[leaf_index]
            if leaf_node[8] <= 0:
                continue
            try:
                candidate = self._candidate_from_leaf(box_size_m, leaf_index, leaf_node.copy())
            except RuntimeError:
                continue
            if compatible(candidate):
                self._pending = candidate
                self._pending_leaf_node = leaf_node.copy()
                self._pending_grid = _meters_to_grid(candidate.original_size_m, self.grid_scale)
                return candidate
        self._pending = first
        raise RuntimeError(f"No robot-compatible PCT leaf for item {self._item_index} size={box_size_m}; policy candidate placed_size={first.placed_size_m}")

    def propose_alternatives(
        self,
        box_size_m: tuple[float, float, float],
        max_candidates: int = 8,
        allow_yaw: bool = True,
    ) -> list[PctCandidate]:
        """Return PCT leaf alternatives for the current item without committing.

        The deterministic policy leaf is always first. The remaining candidates
        are valid EMS leaves in observation order, filtered to orientations the
        first robot execution layer can support. This is only a robot-execution
        fallback; the caller should try the first policy candidate before using
        later entries.
        """

        first = self.propose(box_size_m)

        def compatible(candidate: PctCandidate) -> bool:
            o = candidate.original_size_m
            p = candidate.placed_size_m
            eps = 1e-6
            if all(abs(o[i] - p[i]) < eps for i in range(3)):
                return True
            return bool(allow_yaw and abs(o[2] - p[2]) < eps and abs(o[0] - p[1]) < eps and abs(o[1] - p[0]) < eps)

        candidates: list[PctCandidate] = [first]
        assert self.obs is not None
        leaf_nodes = self.obs.reshape((-1, 9))[self.args.internal_node_holder : self.args.internal_node_holder + self.args.leaf_node_holder]
        for leaf_index, leaf_node in enumerate(leaf_nodes):
            if len(candidates) >= max(1, int(max_candidates)):
                break
            if leaf_index == first.action_leaf_index or leaf_node[8] <= 0:
                continue
            try:
                candidate = self._candidate_from_leaf(box_size_m, leaf_index, leaf_node.copy())
            except RuntimeError:
                continue
            if compatible(candidate):
                candidates.append(candidate)
        self._pending = first
        self._pending_leaf_node = np.asarray(first.leaf_node, dtype=np.float32)
        self._pending_grid = _meters_to_grid(first.original_size_m, self.grid_scale)
        return candidates

    def commit(self, candidate: PctCandidate | None = None) -> PctPlacement:
        assert self.env is not None
        if candidate is None:
            candidate = self._pending
        if candidate is None:
            raise RuntimeError("No pending PCT candidate to commit.")
        leaf_node = self._pending_leaf_node
        if leaf_node is None or self._pending is None or self._pending.action_leaf_index != candidate.action_leaf_index:
            leaf_node = np.asarray(candidate.leaf_node, dtype=np.float32)
        self._set_current_box(self._pending_grid or _meters_to_grid(candidate.original_size_m, self.grid_scale))
        self.obs, _reward, done, info = self.env.step(np.asarray(leaf_node, dtype=np.float32))
        if done:
            raise RuntimeError(f"PCT commit failed for item {candidate.item_index}; info={info}")
        packed = self.env.space.boxes[-1]
        min_grid = (int(packed.lx), int(packed.ly), int(packed.lz))
        size_grid = (int(packed.x), int(packed.y), int(packed.z))
        max_grid = tuple(min_grid[i] + size_grid[i] for i in range(3))
        placement = PctPlacement(
            item_index=candidate.item_index,
            original_size_m=candidate.original_size_m,
            placed_size_m=tuple(v * self.grid_scale for v in size_grid),
            min_corner_m=tuple(v * self.grid_scale for v in min_grid),
            max_corner_m=tuple(v * self.grid_scale for v in max_grid),
            action_leaf_index=candidate.action_leaf_index,
            utilization=float(self.env.space.get_ratio()),
        )
        self._item_index += 1
        self._pending = None
        self._pending_leaf_node = None
        self._pending_grid = None
        return placement

    def reject(self, candidate: PctCandidate | None = None) -> None:
        del candidate
        self._pending = None
        self._pending_leaf_node = None
        self._pending_grid = None


def run_pct_policy(
    item_sizes_m: Iterable[tuple[float, float, float]],
    model_path: Path,
    pct_root: Path = DEFAULT_PCT_ROOT,
    device: str | None = None,
    grid_scale: float = 0.1,
) -> list[PctPlacement]:
    """Run the discrete EMS PCT policy and return metric placements.

    The model is mandatory. This function intentionally fails if the PCT
    repository, gym dependency, or trained weight file is missing.
    """

    model_path = model_path.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"PCT pretrained model is required but not found: {model_path}. "
            "Download the EMS discrete 10x10x10 model from the PCT Google Drive folder."
        )
    item_sizes_m = [tuple(float(v) for v in item) for item in item_sizes_m]
    item_grid = [_meters_to_grid(item, grid_scale) for item in item_sizes_m]

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    args = _pct_args("cpu" if torch_device.type == "cpu" else torch_device.index or 0, model_path)

    DRL_GAT, load_policy, registration_envs, PackingDiscrete = load_pct_modules(pct_root)
    policy = DRL_GAT(args).to(torch_device)
    policy = load_policy(str(model_path), policy).to(torch_device)
    policy.eval()

    env = PackingDiscrete(
        setting=args.setting,
        container_size=args.container_size,
        item_set=args.item_size_set,
        internal_node_holder=args.internal_node_holder,
        leaf_node_holder=args.leaf_node_holder,
        LNES=args.lnes,
        shuffle=False,
    )
    env.box_creator = FixedSequenceCreator(item_grid)
    obs = env.reset()

    placements: list[PctPlacement] = []
    for item_index, original_m in enumerate(item_sizes_m):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=torch_device).view(1, -1, 9)
        with torch.no_grad():
            _logp, pointer, _entropy, _value = policy(obs_tensor, deterministic=True, normFactor=args.normFactor, evaluate=True)
        leaf_index = int(pointer.detach().cpu().view(-1)[0].item())
        leaf_nodes = obs.reshape((-1, 9))[args.internal_node_holder : args.internal_node_holder + args.leaf_node_holder]
        if leaf_index < 0 or leaf_index >= len(leaf_nodes) or leaf_nodes[leaf_index, 8] <= 0:
            raise RuntimeError(f"PCT selected invalid leaf index {leaf_index} for item {item_index}.")

        obs, _reward, done, info = env.step(leaf_nodes[leaf_index])
        if done:
            raise RuntimeError(f"PCT failed to place item {item_index} size={original_m}; info={info}")

        packed = env.space.boxes[-1]
        min_grid = (int(packed.lx), int(packed.ly), int(packed.lz))
        size_grid = (int(packed.x), int(packed.y), int(packed.z))
        max_grid = tuple(min_grid[i] + size_grid[i] for i in range(3))
        placements.append(
            PctPlacement(
                item_index=item_index,
                original_size_m=original_m,
                placed_size_m=tuple(v * grid_scale for v in size_grid),
                min_corner_m=tuple(v * grid_scale for v in min_grid),
                max_corner_m=tuple(v * grid_scale for v in max_grid),
                action_leaf_index=leaf_index,
                utilization=float(env.space.get_ratio()),
            )
        )
    return placements


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCT model on an online 1m pallet sequence.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--pct-root", type=Path, default=DEFAULT_PCT_ROOT)
    parser.add_argument("--items-json", type=Path, required=True, help="JSON list of item sizes in meters.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    items = [tuple(item) for item in json.loads(args.items_json.read_text())]
    placements = run_pct_policy(items, args.model_path, args.pct_root, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([asdict(p) for p in placements], indent=2), encoding="utf-8")
    print(f"wrote {len(placements)} PCT placements to {args.out}")


if __name__ == "__main__":
    main()
