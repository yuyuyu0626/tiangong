"""PCT 可学习放置策略（vendored 自 tiangong/PCT，纯 torch 推理，无 Isaac Gym 依赖）。

对外暴露 DRL_GAT 策略网络与模型加载/观测解析工具，供 move_pro.pct_policy 使用。
"""

from .model import DRL_GAT
from .attention_model import AttentionModel
from .loader import (
    init,
    load_policy,
    observation_decode_leaf_node,
    get_leaf_nodes_with_factor,
)

__all__ = [
    "DRL_GAT",
    "AttentionModel",
    "init",
    "load_policy",
    "observation_decode_leaf_node",
    "get_leaf_nodes_with_factor",
]
