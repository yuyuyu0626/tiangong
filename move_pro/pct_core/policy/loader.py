"""PCT 推理所需的辅助函数（从 PCT/tools.py 抽取，去掉 argparse/gym 依赖）。

保持与 PCT 原实现逐行一致，仅删去训练/注册相关代码，使 move_pro 自包含。
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        elif x.dim() == 1:
            bias = self._bias.t().view(1, -1)
        elif x.dim() == 3:
            bias = self._bias.t().view(1, 1, -1)
        else:
            assert False
        return x + bias


def get_leaf_nodes_with_factor(observation, batch_size, internal_node_holder, leaf_node_holder):
    unify_obs = observation.reshape((batch_size, -1, 9))
    leaf_nodes = unify_obs[:, internal_node_holder:internal_node_holder + leaf_node_holder, :]
    return unify_obs, leaf_nodes


def observation_decode_leaf_node(observation, internal_node_holder, internal_node_length, leaf_node_holder):
    """解析环境返回的观测张量，分出内部节点 / 叶子节点 / 当前物品 / 掩码。

    见 PCT/tools.py 注释：
    internal_nodes: 已放箱子 [x1,y1,z1,x2,y2,z2, density?]
    leaf_nodes:     候选放置 [x1,y1,z1,x2,y2,z2,?,?]
    current_box:    下一个待放物品 [density?,0,0,x,y,z]
    valid_flag:     候选是否可行
    full_mask:      该节点是否参与 GAT 编码
    """
    internal_nodes = observation[:, 0:internal_node_holder, 0:internal_node_length]
    leaf_nodes = observation[:, internal_node_holder:internal_node_holder + leaf_node_holder, 0:8]
    current_box = observation[:, internal_node_holder + leaf_node_holder:, 0:6]
    valid_flag = observation[:, internal_node_holder: internal_node_holder + leaf_node_holder, 8]
    full_mask = observation[:, :, -1]
    return internal_nodes, leaf_nodes, current_box, valid_flag, full_mask


def load_policy(load_path, policy):
    """加载 PCT 训练好的 state_dict 到 DRL_GAT。与 PCT/tools.py:load_policy 一致。"""
    assert os.path.exists(load_path), f'File does not exist: {load_path}'
    pretrained_state_dict = torch.load(load_path, map_location='cpu')
    if len(pretrained_state_dict) == 2:
        pretrained_state_dict, _ob_rms = pretrained_state_dict

    load_dict = {}
    for k, v in pretrained_state_dict.items():
        if 'actor.embedder.layers' in k:
            load_dict[k.replace('module.weight', 'weight')] = v
        else:
            load_dict[k.replace('module.', '')] = v

    load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
    load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}
    for k, v in load_dict.items():
        if len(v.size()) <= 3:
            load_dict[k] = v.squeeze(dim=-1)
    policy.load_state_dict(load_dict, strict=True)
    return policy
