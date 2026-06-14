# move 代码仓库技术理解文档

> 自动生成于 2026-06-13 | 分析方法：dl-code-analysis（适配机器人领域）

---

## 1. 背景与动机

### 任务目标

`move` 是一个**机器人码垛（palletizing）运动规划与仿真系统**。其核心任务是：控制一个双臂移动机器人（天翌 Tianyi）从桌面上抓取箱子，移动到托盘旁，并将箱子精确堆叠到托盘上指定的目标位置。

### 输入与输出

| 阶段 | 输入 | 输出 |
|------|------|------|
| 离线规划 | 几何常量（桌位、箱体尺寸、托盘尺寸）、URDF 机器人模型 | 关节角度轨迹、root 路线、IK 可行性报告 |
| Isaac Gym 仿真 | 离线规划结果、物理参数 | 可视化仿真、物理验证 |

### 核心困难

1. **多阶段 IK 链**：从"桌面抓取"到"移动抬升"再到"高位放置"，涉及不同的关节组（手臂关节 vs 腿/腰抬升关节），需要分阶段求解
2. **高度可达性**：堆垛高度达到 1.6m（4 层），超出纯手臂工作空间，需要腿部和腰部关节配合抬升
3. **狭窄空间操作**：三四层箱子放置时，手掌需要在已堆箱体之间收手、推入，避免碰撞
4. **抓取-放置几何耦合**：抓取姿态（掌心法向、手指方向）需要在移动和放置阶段保持，同时适应托盘侧的不同站位

### 为什么需要当前设计

- **离线规划 + 在线播放**架构：`planning.py` + `grab_test.py` 在无 Isaac Gym 环境下完成全部 IK 计算，生成关节轨迹；仿真脚本只负责播放和物理验证。这使得 IK 调试可以快速迭代，不依赖耗时的物理仿真。
- **模块化分层**：几何规划（EMS 空间分解）、IK 求解（手臂 + 抬升分离）、仿真播放三层分离，各自可独立测试。

---

## 2. 整体架构概览

### 核心模块及其职责

```
move/
├── planning.py          # [离线] 纯几何规划：EMS空间、站位、路线
├── utils.py             # [离线] URDF解析、前向运动学、双臂IK控制器
├── mobile_ik.py         # [离线] 移动状态抬升关节IK求解器
├── grab_test.py         # [离线] 完整抓取-移动-放置IK链路验证
├── tasks/
│   ├── move_test1.py    # [仿真] 基础root移动场景
│   ├── move_test2.py    # [仿真] root移动+抬升IK
│   ├── move_test3.py    # [仿真] 顶部堆叠高度IK测试
│   ├── move_test4.py    # [仿真] move_test3变体
│   ├── grab_test_task.py# [仿真] 完整抓取-移动-放置流程播放
│   ├── task1.py         # [仿真] 60箱标准垛型 (3×5×4)
│   ├── task1_2.py       # [仿真] 60箱调参最终版
│   ├── task2_1.py       # [仿真] 36箱新垛型基线 (9箱/层)
│   └── task2_2.py       # [仿真] 36箱新垛型最终版（含推入策略）
└── assets/
    └── integrated/
        └── tianyi_xhand_move.urdf  # 机器人URDF模型
```

### 文件规模

| 文件 | 行数 | 类别 |
|------|------|------|
| `planning.py` | 591 | 几何规划 |
| `utils.py` | 371 | IK/URDF 工具 |
| `mobile_ik.py` | 152 | 抬升 IK |
| `grab_test.py` | 783 | 离线 IK 链路 |
| `tasks/move_test1.py` | ~400 | 基础仿真 |
| `tasks/move_test2.py` | 236 | 移动+抬升仿真 |
| `tasks/move_test3.py` | ~450 | 高度 IK 仿真 |
| `tasks/move_test4.py` | ~45 | 场景变体 |
| `tasks/grab_test_task.py` | 804 | 完整流程仿真 |
| `tasks/task1.py` | 4231 | 60箱基线 |
| `tasks/task1_2.py` | 5007 | 60箱最终版 |
| `tasks/task2_1.py` | 1317 | 36箱基线 |
| `tasks/task2_2.py` | 1712 | 36箱最终版 |

### 数据从输入到输出的主要阶段

```
几何常量定义 → EMS空间分解 → 站位规划(四侧) → 路线规划(waypoints)
                                                          ↓
URDF模型加载 → 活动DOF解析 → 双掌抓取IK → 抬升关节IK → 放置IK
                                                          ↓
                                              Isaac Gym仿真播放 + 物理验证
```

### 离线规划 vs 在线仿真

| 维度 | 离线规划 (`planning.py`, `grab_test.py`) | 在线仿真 (`tasks/*.py`) |
|------|------------------------------------------|--------------------------|
| 依赖 | 仅 NumPy / PyTorch / 标准库 | Isaac Gym Preview 4 |
| 目的 | IK 可行性验证、轨迹生成 | 物理仿真、碰撞检测、可视化 |
| 速度 | 秒级 | 分钟级（取决于播放帧数） |
| 输出 | 关节角度 tensor、文本报告 | 3D 可视化、物理状态 |

---

## 3. 输入设定与符号定义

### 几何常量（定义于 [planning.py:19-43](planning.py#L19-L43)）

| 符号 | 值 | 语义 |
|------|-----|------|
| `TABLE_SIZE` | `(0.80, 0.60, 0.06)` m | 源桌尺寸 (长, 宽, 高) |
| `TABLE_POSE` | `(0.60, 0.0, 0.60)` m | 源桌世界坐标 (x, y, z) |
| `BOX_SIZE` | `(0.20, 0.20, 0.20)` m | 标准箱子尺寸 |
| `BOX_POSE` | 计算自 `TABLE_POSE` | 源箱世界坐标 |
| `PALLET_SIZE` | `(1.0, 1.0)` m | 托盘 XY 尺寸 |
| `PALLET_SURFACE_Z` | `0.10` m | 托盘表面高度 |
| `PALLET_DIAGONAL_DISTANCE` | `5.0` m | 托盘距桌对角距离 |
| `STACK_HEIGHT_LIMIT` | `1.0` m | 堆垛高度上限 |
| `MOVE_STAND_OFF` | `0.60` m | 站位距托盘面距离 |
| `DEFAULT_ROOT_Z` | `0.0` m | 默认 root 高度 |
| `DEFAULT_HEIGHT_CLEARANCE` | `0.18` m | 默认高度余量 |
| `SAFE_CORRIDOR_MARGIN` | `1.20` m | 安全通道边距 |
| `STACK_TMP_RETREAT` | `0.80` m | 临时站位退避距离 |
| `TABLE_RETREAT` | `0.90` m | 桌边退避距离 |

### 手部几何常量（定义于 [utils.py:43-51](utils.py#L43-L51)）

| 符号 | 值 | 语义 |
|------|-----|------|
| `PALM_SURFACE_X` | `0.025` m | 掌心沿法向偏移（link 原点 → 掌面） |
| `PALM_CENTER_Z` | `-0.020` m | 掌心沿手指方向偏移 |
| `PALM_BOX_CLEARANCE` | `-0.032` m | 掌面到箱面的压缩距离（负值 = 穿入箱体） |
| `BOX_SIDE_CONTACT_Z_RATIO` | `0.18` | 接触点高度比例（箱侧面的 18% 高度处） |
| `APPROACH_GAP_READY` | `0.16` m | 预备接近间隙 |
| `APPROACH_GAP_PRE` | `0.095` m | 预抓取间隙 |
| `APPROACH_GAP_GRASP` | `0.0` m | 抓取接触间隙 |

### 关节组定义

**左臂 7 关节**（[utils.py:20-28](utils.py#L20-L28)）：
`shoulder_pitch_l_joint`, `shoulder_roll_l_joint`, `shoulder_yaw_l_joint`, `elbow_pitch_l_joint`, `elbow_yaw_l_joint`, `wrist_pitch_l_joint`, `wrist_roll_l_joint`

**右臂 7 关节**（[utils.py:30-38](utils.py#L30-L38)）：镜像命名（后缀 `_r_`），注意 `wrist_roll_r_joinst` 存在拼写问题

**抬升 3 关节**（[mobile_ik.py:20-24](mobile_ik.py#L20-L24)）：
`first_leg_pitch_joint`, `second_leg_pitch_joint`, `waist_pitch_joint`

**末端 Link**（[utils.py:40-41](utils.py#L40-L41)）：
`xhand_left_left_hand_link`, `xhand_right_right_hand_link`

---

## 4. 核心方法流程

### 4.1 EMS 空间分解（Empty Space Management）

#### 4.1.1 代码位置
[planning.py:198-214](planning.py#L198-L214) — `compute_pct_like_ems()`

#### 4.1.2 输入与输出
- **输入**：`placed_boxes: Iterable[BoxPlacement]` — 已放置箱体列表，每个含 `min_corner(3,)` 和 `size(3,)`
- **输出**：`tuple[EmptySpace, ...]` — 当前所有可利用空位（EMS 叶子节点）

#### 4.1.3 具体计算过程

初始时整个托盘空间为一个 3D 空位：

$$EMS_0 = \{(0,0,0) \to (L_{pallet}, W_{pallet}, H_{limit})\}$$

对每个已放置箱体 $B_i$，对当前所有 EMS 空间逐一执行**减法切割**（`_subtract_box`, [行 150-173](planning.py#L150-L173)）：

给定空间 $S = (sx_1, sy_1, sz_1) \to (sx_2, sy_2, sz_2)$ 和箱体 $B = (bx_1, by_1, bz_1) \to (bx_2, by_2, bz_2)$，交集区域为：

$$ix_1 = \max(sx_1, bx_1), \quad iy_1 = \max(sy_1, by_1), \quad iz_1 = \max(sz_1, bz_1)$$
$$ix_2 = \min(sx_2, bx_2), \quad iy_2 = \min(sy_2, by_2), \quad iz_2 = \min(sz_2, bz_2)$$

生成 **5 个候选子空间**（在交集的 5 个面上分裂）：

1. **-X 侧**：$(sx_1, sy_1, sz_1) \to (ix_1, sy_2, sz_2)$
2. **+X 侧**：$(ix_2, sy_1, sz_1) \to (sx_2, sy_2, sz_2)$
3. **-Y 侧**：$(sx_1, sy_1, sz_1) \to (sx_2, iy_1, sz_2)$
4. **+Y 侧**：$(sx_1, iy_2, sz_1) \to (sx_2, sy_2, sz_2)$
5. **+Z 侧（顶部）**：$(sx_1, sy_1, iz_2) \to (sx_2, sy_2, sz_2)$

最后通过 `_eliminate_inscribed`（[行 176-195](planning.py#L176-L195)）消除被其他空间完全包含的冗余空间：对每个空间，如果存在另一个空间完全包含它（$\forall k: min_k \ge other.min_k \land max_k \le other.max_k$），则将其剔除。

#### 4.1.4 设计动机

此算法模仿了 PCT (Palletizing Configuration Tool) 的 EMS 管理策略。关键设计选择：
- **5-面分裂**而非 6-面：底面不分裂，因为箱子只能放在支撑面上方
- **最小尺寸过滤**（`min_size=0.05`m）：避免产生无用的微小空间碎片
- **包含消除**：保证 EMS 叶子集是完备且无冗余的

---

### 4.2 PCT 风格叶子候选生成

#### 4.2.1 代码位置
[planning.py:217-249](planning.py#L217-L249) — `leaf_candidates_for_box()`

#### 4.2.2 输入与输出
- **输入**：`spaces: Iterable[EmptySpace]`、`box_size: (sx, sy, sz)` — 目标箱子尺寸
- **输出**：`tuple[BoxPlacement, ...]` — 候选放置位置列表

#### 4.2.3 具体计算过程

对每个 EMS 空间 $(ex_1, ey_1, ez_1) \to (ex_2, ey_2, ez_2)$，检查尺寸是否足够容纳目标箱。如果足够，生成至多 **9 个候选点**：

- **4 个角点**：$(ex_1, ey_1)$, $(ex_2-s_x, ey_1)$, $(ex_1, ey_2-s_y)$, $(ex_2-s_x, ey_2-s_y)$ — 都在 $ez_1$ 高度
- **4 个边中点**：X 边中点 2 个、Y 边中点 2 个 — 也在 $ez_1$ 高度
- **1 个面中心点**：$(\frac{ex_1+ex_2-s_x}{2}, \frac{ey_1+ey_2-s_y}{2}, ez_1)$

所有坐标四舍五入到 6 位小数后去重。

#### 4.2.4 选择策略

`choose_next_box_leaf()`（[行 252-267](planning.py#L252-L267)）按优先级排序：

1. **最低 Z**（`b.min_corner[2]` 升序）— 先堆低层
2. **最接近托盘原点**（$|x|+|y|$ 升序）— 从内侧开始
3. **Y 最小** → **X 最小**

这保证了"从低到高、从内侧到外侧"的堆叠顺序。

---

### 4.3 四侧站位规划

#### 4.3.1 代码位置
[planning.py:295-344](planning.py#L295-L344) — `generate_stance_plans()`

#### 4.3.2 输入与输出
- **输入**：栈中心 `stack_center(3,)`、栈尺寸 `stack_size_xy(2,)`、站距 `stand_off`、目标支撑面高度 `target_support_z`、放置中心 `placement_center_xy`
- **输出**：`tuple[StancePlan, ...]` — 四侧（+X, -X, +Y, -Y）站位计划

#### 4.3.3 具体计算过程

对于每个侧面 $side \in \{+X, -X, +Y, -Y\}$，定义法向 $normal$ 和面板中心 $face\_xy$：

| Side | normal | face_xy |
|------|--------|---------|
| +X | (1, 0) | (cx + L/2, target_y) |
| -X | (-1, 0) | (cx - L/2, target_y) |
| +Y | (0, 1) | (target_x, cy + W/2) |
| -Y | (0, -1) | (target_x, cy - W/2) |

**站位计算**：

$$s_x = face\_x + normal_x \cdot stand\_off$$
$$s_y = face\_y + normal_y \cdot stand\_off$$
$$yaw = \text{atan2}(target\_y - s_y,\; target\_x - s_x)$$

**高度命令**：

$$height\_command = \max(min\_root\_z,\; start\_root\_z + (desired\_box\_center\_z - z\_box\_rel\_hold))$$

其中 $desired\_box\_center\_z = target\_support\_z + box\_z \cdot 0.5 + clearance$。

**临时退避位**：在站位基础上再沿法向外退 `STACK_TMP_RETREAT=0.80`m。

---

### 4.4 路线规划

#### 4.4.1 代码位置
[planning.py:455-485](planning.py#L455-L485) — `build_route_plans()`

#### 4.4.2 路线结构

每个侧面的路线包含以下 waypoint 序列：

```
table_start → table_retreat → safe_axis_1 → safe_corner → tmp_axis_1 → tmp → target → retreat
```

**关键安全节点**：
- **table_retreat**（[行 371-382](planning.py#L371-L382)）：从起始点沿"远离桌子"方向退 `TABLE_RETREAT=0.90`m
- **safe_corner**（[行 385-389](planning.py#L385-L389)）：托盘左下角外 `SAFE_CORRIDOR_MARGIN=1.20`m 的安全角
- **到 safe_corner 的路线**（[行 399-417](planning.py#L399-L417)，`_route_to_safe_corner`）：如果当前在托盘 X 范围外，先走 Y 轴对齐再走 X 轴对齐；否则先 X 后 Y
- **从 safe_corner 到临时位**（[行 420-437](planning.py#L420-L437)，`_route_from_safe_corner_to_tmp`）：类似的 L 形路径

Waypoint 去重（`_dedupe_waypoints`，[行 347-361](planning.py#L347-L361)）：相邻 waypoint 的 `(x, y, z, yaw)` 差值小于 $10^{-6}$ 时合并。

---

### 4.5 双臂 IK 控制器

#### 4.5.1 代码位置
[utils.py:189-371](utils.py#L189-L371) — `IKFlipBoxController`

#### 4.5.2 核心算法：阻尼最小二乘 (Damped Least Squares) IK

**9 维末端特征向量**（[行 339-346](utils.py#L339-L346)）：

$$f_{ee} = [palm\_center_{3\times1},\; palm\_normal \cdot s_{3\times1},\; finger\_dir \cdot s_{3\times1}]$$

其中 $s = 0.20$ 是方向误差的缩放因子（`_orientation_scale()`）。

**掌心世界坐标计算**：

$$palm\_center = T_{link}[:3,3] + R_{link}[:,0] \cdot PALM\_SURFACE\_X + R_{link}[:,2] \cdot PALM\_CENTER\_Z$$

这里 $R_{link}[:,0]$ 是 link 坐标系的 X 轴（掌心法向），$R_{link}[:,2]$ 是 Z 轴（手指方向）。

**雅可比矩阵**通过有限差分计算（[行 306-313](utils.py#L306-L313)）：

$$J_{:,col} = \frac{f_{ee}(q + \epsilon \cdot e_{col}) - f_{ee}(q)}{\epsilon}, \quad \epsilon = 10^{-4}$$

**阻尼最小二乘更新**（[行 315-316](utils.py#L315-L316)）：

$$\Delta q = J^T (JJ^T + \lambda^2 I)^{-1} \cdot error, \quad \lambda = 0.045$$

更新步长被 clip 到 $[-0.06, 0.06]$ rad，关节角度被 clip 到 $[lower, upper]$。

**收敛条件**（[行 301-304](utils.py#L301-L304)）：位置误差 < 4mm，掌心法向误差 < 0.08（缩放后），手指方向误差 < 0.08。

**腕关节正则化**（[行 325-333](utils.py#L325-L333)，`_regularize_approach_wrist`）：当 `regularize_wrist=True` 且 `APPROACH_WRIST_REGULARIZATION > 0` 时，腕关节向参考值拉回：
$$q_{wrist} = (1 - gain) \cdot q_{wrist} + gain \cdot q_{ref}$$

#### 4.5.3 初始化姿态

`_apply_seed_posture()`（[行 234-253](utils.py#L234-L253)）设置初始种子姿态：

| 关节 | 初始值 (rad) |
|------|-------------|
| shoulder_pitch_l/r | -0.40 |
| shoulder_roll_l/r | ±0.15 |
| shoulder_yaw_l/r | ∓0.40 |
| elbow_pitch_l/r | -1.00 |
| elbow_yaw_l/r | 0.00 |
| wrist_pitch_l/r | 0.00 |
| wrist_roll_l/r | ∓0.10 |

---

### 4.6 移动抬升 IK

#### 4.6.1 代码位置
[mobile_ik.py:28-124](mobile_ik.py#L28-L124) — `MoveLiftIK`

#### 4.6.2 核心思路

保持手臂/手关节不变（`hold_targets`），仅调整 3 个抬升关节（两腿 pitch + 腰 pitch），使掌心高度达到目标箱体中心高度。

**单目标模式**（`preserve_body_pitch=False, preserve_body_x=False`）：

误差标量：
$$z\_error = target\_box\_center\_z - current\_palm\_z$$

掌心 Z 计算（双手平均，[行 126-137](mobile_ik.py#L126-L137)）：
$$palm\_z = 0.5 \cdot (left\_palm\_center_z + right\_palm\_center_z)$$

其中单个掌心 Z：
$$palm\_center\_z = (T_{link}[:3,3] + R_{link}[:,0] \cdot PALM\_SURFACE\_X + R_{link}[:,2] \cdot PALM\_CENTER\_Z)[2]$$

雅可比矩阵（$1 \times 3$，仅对 3 个抬升关节）：
$$J_{0, col} = \frac{palm\_z(q + \epsilon \cdot e_{col}) - palm\_z(q)}{\epsilon}, \quad \epsilon = 10^{-4}$$

**多目标模式**（`preserve_body_pitch=True`）：

误差向量变为 $[z\_error,\; pitch\_weight \cdot pitch\_error]$，雅可比变为 $2 \times 3$。

身体俯仰角从 `body_yaw_link` 的旋转矩阵提取（[行 139-143](mobile_ik.py#L139-L143)）：
$$pitch = \arctan2(R[0,2],\; R[0,0])$$

Body X 从 `body_yaw_link` 的位置提取（[行 145-148](mobile_ik.py#L145-L148)）：
$$body\_x = T[0,3]$$

**阻尼最小二乘参数**：$\lambda = 0.03$，步长 clip = 0.045 rad。

**收敛条件**：$|z\_error| < 0.003$ m，$|pitch\_error| < 0.015$ rad，$|x\_error| < 0.010$ m。

#### 4.6.3 目标箱体中心高度

`target_box_center_z()`（[行 151-152](mobile_ik.py#L151-L152)）：

$$target\_z = target\_support\_z + box\_size\_z \cdot 0.5 + clearance$$

---

### 4.7 完整 IK 链路（[grab_test.py:608-733](grab_test.py#L608-L733)）

`run()` 函数编排完整的离线 IK 计算流程：

```
阶段 1: 双掌抓取 IK
  ┌──────────────────────────────────────────────────────────┐
  │ pick_ready_high → pick_descend → pick_pre_grasp          │
  │ → pick_clamp → pick_compress → pick_hold → pick_lift     │
  │                                                          │
  │ 掌心法向: 箱体 ±Y 方向                                     │
  │ 手指方向: +X                                              │
  │ 首帧 80 次迭代, 后续帧 16 次迭代                              │
  └──────────────────────────────────────────────────────────┘

阶段 2: 移动抬升 IK (MoveLiftIK)
  ┌──────────────────────────────────────────────────────────┐
  │ pick_lift_q → move_approach_q → move_lift_q              │
  │                                                          │
  │ 仅调整 first_leg_pitch, second_leg_pitch, waist_pitch     │
  │ 手臂关节保持 pick_lift_q 的值不变                           │
  └──────────────────────────────────────────────────────────┘

阶段 3: Root 路线生成
  ┌──────────────────────────────────────────────────────────┐
  │ table_start → table_retreat → safe_corner                 │
  │ → tmp → target (最终放置站位)                               │
  │                                                          │
  │ waypoint 间 smoothstep 采样 + 终点 30 帧保持                │
  └──────────────────────────────────────────────────────────┘

阶段 4: 双掌放置 IK
  ┌──────────────────────────────────────────────────────────┐
  │ 在 final_root 站位下, 将世界坐标放置路径转为 root 局部坐标     │
  │ upright_rotate → move_to_release → hold → release_open   │
  │                                                          │
  │ 关键帧 80 次迭代, 密集扩展 smoothstep 插值                   │
  └──────────────────────────────────────────────────────────┘
```

### 4.8 平滑轨迹采样

#### 4.8.1 代码位置
[planning.py:509-539](planning.py#L509-L539) — `sample_root_trajectory()`

#### 4.8.2 插值函数

使用 **smoothstep**（三次 Hermite 插值）：

$$smooth(\alpha) = \alpha^2 \cdot (3 - 2\alpha), \quad \alpha \in [0, 1]$$

该函数满足 $smooth(0)=0$, $smooth(1)=1$, $smooth'(0)=0$, $smooth'(1)=0$，确保起点和终点的速度和加速度为零。

#### 4.8.3 时间同步

| 维度 | 速度 | 所需时间 |
|------|------|---------|
| XY 平移 | 0.225 m/s | $d_{xy} / 0.225$ |
| 偏航旋转 | 0.30 rad/s | $d_{yaw} / 0.30$ |
| Z 抬升 | 0.075 m/s | $d_z / 0.075$ |

取三者所需时间的**最大值**作为总时长，保证所有维度同步到达终点。步数 = $\max(2, \lceil duration / dt \rceil + 1)$，$dt = 1/60$ s。

---

## 5. 数据/信号流总结

| 阶段 | 模块 | 输入数据 | 输出数据 | 流向 |
|------|------|---------|---------|------|
| 几何初始化 | `planning.py` | 常量定义 | `StackScene`, `MoveScenePlan` | → 仿真场景构建 |
| URDF 解析 | `utils.py:UrdfKinematics` | URDF XML 文件 | `JointInfo` 列表, link 变换链缓存 | → FK 计算 |
| 活动 DOF 提取 | `grab_test.py:_parse_active_dofs` | URDF | `(dof_names, lower, upper)` | → IK 控制器初始化 |
| 抓取 IK | `utils.py:IKFlipBoxController` | `CartesianKeyframe` 序列 (7帧) | `pick_targets: [7, num_dofs]` | → 移动阶段种子 |
| 抬升 IK | `mobile_ik.py:MoveLiftIK` | `q_pick_lift`, `target_box_center_z` | `q_move_approach`, `q_move_lift` | → 放置 IK 种子 |
| 路线规划 | `planning.py` | `start_pose`, `StackScene`, `StancePlan` | `RoutePlan` (waypoints) | → root 轨迹采样 |
| 放置 IK | `utils.py:IKFlipBoxController` | `pose_path_world` (世界坐标), `final_root` | 密集 `q_place` 轨迹 | → 仿真播放 |
| 仿真播放 | `tasks/*.py` | 全部 IK 轨迹, 物理参数 | Isaac Gym 可视化 | → 用户观察 |

---

## 6. 关键数据结构语义表

| 数据结构 | 代码位置 | 核心字段 | 语义 |
|---------|---------|---------|------|
| `Pose` | [planning.py:46-51](planning.py#L46-L51) | `x, y, z, yaw` | 机器人 root 在世界坐标系中的位姿（仅平移 + yaw 旋转） |
| `BoxPlacement` | [planning.py:54-67](planning.py#L54-L67) | `name, min_corner(3,), size(3,)` | 一个箱体的放置定义，最小角坐标 + 三轴尺寸 |
| `EmptySpace` | [planning.py:69-81](planning.py#L69-L81) | `min_corner(3,), max_corner(3,)` | EMS 空位空间区域 |
| `StackScene` | [planning.py:84-92](planning.py#L84-L92) | `pallet_center, pallet_size, pallet_surface_z, preset_boxes, next_box, empty_spaces, selected_leaf` | 完整堆垛场景描述 |
| `StancePlan` | [planning.py:95-103](planning.py#L95-L103) | `side, normal(2,), face_center(3,), pose, tmp_pose, height_command` | 一个侧面的站位计划 |
| `RoutePlan` | [planning.py:111-114](planning.py#L111-L114) | `side, waypoints: tuple[RouteWaypoint, ...]` | 一个侧面的完整路线 |
| `CartesianKeyframe` | [utils.py:67-80](utils.py#L67-L80) | `name, duration, left_pos(3,), right_pos(3,), left_palm_normal(3,), right_palm_normal(3,), left_finger_dir(3,), right_finger_dir(3,), hand_close, extra_joints` | 双手笛卡尔空间 IK 目标关键帧 |
| `JointInfo` | [utils.py:54-65](utils.py#L54-L65) | `name, joint_type, parent, child, xyz(3,), rpy(3,), axis(3,)` | 从 URDF 提取的关节最小运动学信息 |
| `GrabTestReport` | [grab_test.py:99-114](grab_test.py#L99-L114) | `pick_frames, place_frames, root_frames, target_world_center, final_root_pose, pick_feasible, place_feasible, ...` | 离线 IK 链路的可行性报告 |
| `OfflineTest3Scene` | [grab_test.py:117-125](grab_test.py#L117-L125) | `name, stack_scene, target_support_z, final_pose, tmp_pose, waypoints` | 离线场景描述（无 Isaac Gym 依赖） |

---

## 7. 离线规划流程与在线仿真流程

### 7.1 离线规划阶段（[grab_test.py:608-733](grab_test.py#L608-L733)）

**调用入口**：`python -m move.grab_test` 或编程调用 `run(save_path, place_mode, stand_off)`

**执行步骤**：

1. **URDF 解析**（`_parse_active_dofs`，[行 127-147](grab_test.py#L127-L147)）：遍历 XML 中所有 `type != "fixed"` 的 joint，读取 limit 的 lower/upper
2. **构造 7 帧抓取关键帧序列**（`_build_pick_keyframes`，[行 211-241](grab_test.py#L211-L241)）：ready_high → descend → pre_grasp → clamp → compress → hold → lift
3. **首帧大迭代 IK（80次）+ 后续帧小迭代 IK（16次）**，生成 `pick_targets: [7, num_dofs]`
4. **构建离线场景**（`build_offline_test3_scene`）：确定目标世界中心和支撑面高度
5. **MoveLiftIK 两阶段抬升**：approach 高度 → preplace 高度
6. **Root 路线帧序列**（waypoint 间 smoothstep 采样 + 30 帧终点保持）
7. **在最终站位下求解密集放置 IK**：世界坐标路径 → root 局部坐标 → 关键帧 → 密集轨迹扩展
8. **求解释放帧**（release_open，双手张开）
9. **组装 payload** 并可选保存为 `.pt` 文件

### 7.2 在线仿真阶段（`tasks/grab_test_task.py`, `tasks/task1.py` 等）

**调用入口**：`python -m move.tasks.grab_test_task` 等

**执行步骤**：

1. 调用 `grab_test.run()` 获取离线规划结果
2. 创建 Isaac Gym 仿真环境、地面、物理参数
3. 加载机器人 URDF 资产 + 场景物体（桌子、源箱、托盘、预置箱、目标 ghost）
4. 设置 DOF 控制模式（位置控制）、刚度和阻尼参数
5. **分阶段播放**：
   - **抓取阶段**：root 在桌边，播放 pick 关键帧轨迹，源箱是物理对象
   - **Attach 阶段**：抓取压缩完成后，将源箱运动学附着到双掌抓取框架（`gym.set_actor_root_state_tensor` 改为 kinematic）
   - **移动阶段**：播放 root 路线 + 抬升关节混合目标
   - **放置阶段**：root 固定，播放放置密集轨迹
   - **Detach 阶段**：在释放点分离箱子，恢复为物理对象，自然落到堆垛位置

### 7.3 一二层 vs 三四层策略差异

| 维度 | 一二层（直接放置） | 三四层（waist-arm 策略） |
|------|-------------------|------------------------|
| 高度范围 | 0 ~ 0.8 m | 0.8 ~ 1.6 m |
| 抬升关节 | 不需要（root z=0 可达） | 需要 `MoveLiftIK` 调整腿/腰 |
| 放置方式 | 手臂直接伸到目标位 | waist-arm 协同运动 |
| 收手策略 | 简单收手/推入 | 需要水平收手、推入 |
| 站距 | 标准站距 | 可能需要调整 |

---

## 8. 重要函数逐级逻辑解释

### 8.1 `UrdfKinematics.fk()` — 前向运动学

```
文件: utils.py:172-180
签名: fk(self, link: str, q_map: Dict[str, float]) -> np.ndarray
返回: 4×4 齐次变换矩阵 T_world_link
```

**逐步骤**：

1. `mat = I_{4×4}` — 初始化为单位阵
2. 调用 `_chain_to(link)` 获取从 root 到该 link 的运动链（关节列表，从根到叶，已缓存）
3. 对链上每个关节：
   - `mat = mat @ T_origin` — 乘关节原点变换（静态偏移 + 旋转，来自 `_origin_cache`）
   - 如果是转动关节：`mat = mat @ T_axis_angle(axis, angle)` — 乘关节轴角旋转矩阵
4. 返回累积的 4×4 变换矩阵

**设计要点**：`_chain_cache` 避免重复树遍历；`_origin_cache` 预计算所有关节的静态原点变换。

### 8.2 `IKFlipBoxController._solve_arm()` — 单臂阻尼最小二乘 IK

```
文件: utils.py:280-323
签名: _solve_arm(self, ee_link, joint_names, target_pos, target_palm_normal,
               target_finger_dir, iterations, regularize_wrist)
```

**逐步骤**：

1. **过滤活动关节**：从 `joint_names` 中选出存在于 `name_to_index` 的（确保是 DOF）
2. **阻尼最小二乘循环**（最多 `iterations` 次）：
   - 计算当前末端特征 $f_{current}$（`_ee_feature`）
   - 计算目标特征 $f_{target}$（`_target_feature`）
   - 计算误差 $error = f_{target} - f_{current}$（9维）
   - 检查收敛：$|pos\_error| < 0.004$ m 且 $|palm\_error| < 0.08$ 且 $|finger\_error| < 0.08$
   - 有限差分计算雅可比 $J_{9×n\_active}$（每列一次 FK + 差分）
   - 阻尼最小二乘：$\Delta q = J^T (JJ^T + 0.045^2 I)^{-1} error$
   - $\Delta q$ clip 到 $[-0.06, 0.06]$, $q$ clip 到 $[lower, upper]$
   - 可选：腕关节正则化拉回

### 8.3 `MoveLiftIK.solve_for_box_center_z()` — 抬升关节 IK

```
文件: mobile_ik.py:57-124
签名: solve_for_box_center_z(self, hold_targets, target_box_center_z,
                             iterations=80, preserve_body_pitch=False, ...)
返回: q_solved: torch.Tensor [num_dofs]
```

**逐步骤**：

1. 复制 `hold_targets` (numpy) 作为初始解
2. 阻尼最小二乘循环（最多 80 次）：
   - 计算当前掌心平均高度 $z = (left\_palm\_z + right\_palm\_z) / 2$
   - 可选：计算 body pitch 误差、body x 误差
   - 构建误差向量（1D/2D/3D）
   - 检查收敛：$|z\_error| < 3mm$, $|pitch\_error| < 0.015rad$, $|x\_error| < 10mm$
   - 有限差分计算雅可比（仅 3 个抬升关节）
   - 阻尼最小二乘更新（$\lambda=0.03$, step clip=0.045 rad）
   - Clamp 到关节限位
3. 返回 `torch.tensor(q)`

### 8.4 `_expand_targets_for_pose_path()` — 关键帧到密集轨迹

```
文件: grab_test.py:512-540
签名: _expand_targets_for_pose_path(key_indices, key_targets, total_frames)
返回: dense_targets: torch.Tensor [total_frames, num_dofs]
```

**逐步骤**：

1. 初始化 `frames` 列表（长度 `total_frames`），填充 None
2. 在 `key_indices` 位置填入对应的 `key_targets`
3. 对相邻关键帧之间的每一帧：
   - 计算 $\alpha = (frame - start\_idx) / span$
   - 应用 smoothstep: $\alpha' = \alpha^2(3-2\alpha)$
   - 线性插值: $q = q_{start} \cdot (1 - \alpha') + q_{goal} \cdot \alpha'$
4. 填充剩余 None 为最后已知值（前向传播填充）

### 8.5 `_pose_pair_for_box()` — 双手掌心目标计算

```
文件: grab_test.py:156-184
签名: _pose_pair_for_box(center, box_size, theta, side_gap, contact_z, contact_x)
返回: (left_pos, right_pos, left_normal, right_normal, left_finger, right_finger)
```

**计算过程**：

掌心在箱体侧面的局部坐标（以箱体中心为原点）：

$$left\_contact\_local = [contact\_x,\; half\_y + PALM\_BOX\_CLEARANCE + side\_gap,\; contact\_z]$$
$$right\_contact\_local = [contact\_x,\; -half\_y - PALM\_BOX\_CLEARANCE - side\_gap,\; contact\_z]$$

经过 Y 轴旋转 $\theta$（`_rot_y(theta)`）：

$$left\_pos = center + R_y(\theta) \cdot left\_contact\_local$$
$$right\_pos = center + R_y(\theta) \cdot right\_contact\_local$$

掌心法向：左手 = $R_y \cdot [-1, 0, 0]^T$（指向 -Y，即箱体侧面），右手 = $R_y \cdot [1, 0, 0]^T$

手指方向：$R_y \cdot [0, 0, 1]^T$（指向 +X，即前方）

注意 `PALM_BOX_CLEARANCE = -0.032`（负值），因此实际掌心在箱体表面内侧 3.2cm 处，需要压缩才能到达。

---

## 9. 模块之间的关系

```
planning.py ◄────────── grab_test.py ──────────► mobile_ik.py
    │                       │                         │
    │ 几何常量               │ IK链路编排               │ 抬升关节求解
    │ StackScene             │ 场景构建                 │
    │ StancePlan             │ 关键帧生成               │
    │ RoutePlan              │                         │
    │                       │                         │
    ▼                       ▼                         ▼
  move_test1.py        grab_test_task.py         move_test2.py
  move_test2.py        task1.py                  move_test3.py
  move_test3.py        task1_2.py
  move_test4.py        task2_1.py
                       task2_2.py
                             │
                             ▼
                       utils.py
                  (UrdfKinematics,
                   IKFlipBoxController,
                   手部几何常量)
```

**关键配合关系**：

1. **planning.py → 仿真任务**：提供 `StackScene`（场景几何）、`StancePlan`（四侧站位）、`RoutePlan`（waypoints 路线），仿真任务据此创建 Isaac Gym 场景和 root 轨迹
2. **grab_test.py → 仿真任务**：提供完整的离线 IK 计算结果（`pick_targets`, `move_lift_target`, `place_targets` 等），仿真任务直接播放这些 tensor
3. **utils.py → grab_test.py & mobile_ik.py**：`UrdfKinematics` 被两者共用做前向运动学；`IKFlipBoxController` 用于手臂 IK
4. **task2_1.py → task1.py**：以 `import move.tasks.task1 as base` 方式继承 task1 的全部场景构建和控制策略，只替换箱体尺寸和放置目标点

---

## 10. 与已有方法或常见结构的关系

| 方法/结构 | 在 move 中的体现 | 相似处 | 不同处 |
|----------|-----------------|--------|--------|
| PCT EMS 算法 | `compute_pct_like_ems()` | 5-面分裂、包含消除 | 本地 Python 实现，非原始 C++ PCT |
| 阻尼最小二乘 IK | `IKFlipBoxController._solve_arm()` | $J^T(JJ^T+\lambda^2I)^{-1}$ 标准形式 | 有限差分雅可比而非解析雅可比；9维特征向量融合位置+双方向 |
| Smoothstep 插值 | `sample_root_trajectory()` | $\alpha^2(3-2\alpha)$ 标准 Hermite 形式 | 同时用于位置、角度和关节空间三个不同域 |
| Isaac Gym 工作流 | `tasks/*.py` | 标准 gymapi 调用模式 (create_sim → create_env → create_actor → simulate) | 自定义 attach/detach 实现物理-运动学模式切换 |
| 分层任务策略 | task1 / task2 | 一二层直接放置、三四层 waist-arm | 策略代码内嵌而非插件式加载 |
| 姿态插值 | `_lerp_q()`, `smoothstep` | 标准 LERP + easing | 关节空间 smoothstep 而非笛卡尔空间 SLERP |

---

## 11. 复杂度与效率分析

| 模块 | 计算复杂度 | 瓶颈分析 |
|------|-----------|---------|
| EMS 空间分解 | $O(N \cdot M)$，$N$ = 已放置箱数，$M$ = 当前空间数 | $M$ 随箱数增长但被包含消除控制；对 60 箱场景，空间数通常在 20-50 |
| 单帧 IK 求解 | $O(I \cdot J \cdot K)$，$I$ = 迭代次数，$J$ = 活动关节数，$K$ = FK 复杂 | 有限差分雅可比每列需一次完整 FK；14 个手臂关节 × 9 维特征 |
| 密集轨迹扩展 | $O(T)$，$T$ = 总帧数 | 纯逐帧计算，smoothstep + LERP，非瓶颈 |
| Isaac Gym 仿真 | 取决于物理子步和渲染频率 | viewer 渲染是主要瓶颈；`--fast --viewer-render-every 8` 可大幅加速 |

**具体数字**（来自 `grab_test.py` 默认设置）：

| 阶段 | 帧数 | 说明 |
|------|------|------|
| 抓取阶段 | ~630 帧 | 7 帧关键帧 × ~90 帧/段 |
| 放置阶段 | ~785 帧 | rotate(120) + move(360+180) + hold(45) + release |
| Root 路线 | 每 pair 20-100 帧 | 取决于 waypoint 间距离 |
| 总计 | ~1500-2000 帧 | 约 25-33 秒（60fps） |

---

## 12. 容易误解或需要特别注意的地方

1. **`IKFlipBoxController` 名称具有误导性**（[utils.py:189-195](utils.py#L189-L195)）：它不再包含 flip 状态机，纯粹是 move-local 的双臂 IK 控制器。名称仅保留用于向后兼容。

2. **`wrist_roll_r_joinst` 是拼写错误**（[utils.py:37](utils.py#L37)）：应为 `wrist_roll_r_joint`，但需要核对该拼写是否与 URDF 文件中的实际 joint name 一致。正则化字典中也使用了这个拼写（[utils.py:222-228](utils.py#L222-L228)）。

3. **`PALM_BOX_CLEARANCE = -0.032` 是负值**：这个常量表示掌心穿入箱体侧面的压缩距离，而非间隙。在 `_pose_pair_for_box()` 中 `half_y + PALM_BOX_CLEARANCE + side_gap`（`side_gap` 最终为 0）使掌心目标位于箱体表面内侧 3.2cm，意味着机器人需要用力夹持才能到达。

4. **世界坐标 ↔ root 局部坐标转换**：`_world_to_root()`（[grab_test.py:384-389](grab_test.py#L384-L389)）和 `_transform_local_point()`（[grab_test.py:392-399](grab_test.py#L392-L399)）在放置 IK 中频繁使用。放置路径在世界坐标中规划，但 IK 求解器在 root 坐标系下工作。

5. **Attach/Detach 时机**：在 `grab_test_task.py` 中，箱子在抓取压缩后从物理 actor 切换为 kinematic actor（运动学附着到双掌框架）。在释放点 detach 后恢复为物理 actor。切换时机至关重要——过早 detach 导致掉落，过晚 detach 导致手-箱穿透。

6. **两个不同的接触高度比例**：`BOX_SIDE_CONTACT_Z_RATIO = 0.18`（放置用）vs `PICK_SIDE_CONTACT_Z_RATIO = 0.24`（抓取用）。抓取时使用更高的接触点以避免手指与桌面碰撞。

7. **多重站距值**：`MOVE_STAND_OFF = 0.60`（几何规划默认）、`TEST3_STAND_OFF = 0.72`（test3）、`SIM_PLACE_STAND_OFF = 0.35`（仿真可行模式）。不同场景使用不同站距，影响 IK 可行性和碰撞余量。

8. **smoothstep 应用范围**：smoothstep 同时用于 root 路线（`sample_root_trajectory`）和关节空间轨迹扩展（`_expand_targets_for_pose_path`），保证 $C^1$ 连续。

---

## 13. 总结

### 核心技术点

1. **EMS 空间分解**是几何规划的核心：通过 5-面分裂 + 包含消除将托盘空间分解为可放置空位，支持任意堆垛配置
2. **阻尼最小二乘 IK** 是运动学核心：双臂独立 IK + 抬升关节 IK 的分离策略，将高维问题分解为低维子问题
3. **离线规划 + 在线播放**架构是工程效率的关键：所有 IK 在无仿真环境下完成，仿真仅负责物理验证

### 主要数据流

```
URDF 模型 → DOF 解析 → 几何常量初始化 → 场景构建 (EMS+站位+路线)
    → 抓取IK关键帧 → 抬升关节IK → 放置IK密集轨迹
    → Isaac Gym 分阶段播放 (抓取→attach→移动→放置→detach)
```

### 设计原因

- **为什么离线/在线分离**：IK 调试需要快速迭代，不依赖耗时的物理仿真；仿真只用于最终验证
- **为什么 EMS 使用 5-面分裂**：底面不分裂因为箱子只能放在支撑面上方，简化计算且符合物理实际
- **为什么有限差分雅可比而非解析**：可适配任意 URDF 模型，无需为特定机器人推导解析雅可比
- **为什么 smoothstep 而非线性插值**：保证轨迹在起点和终点速度/加速度为零，避免冲击

### 最关键模块

- **[planning.py](planning.py)**：所有离线确定的几何信息源头，是仿真任务的共同基础
- **[grab_test.py:run()](grab_test.py#L608)**：完整的 IK 链路编排函数，所有仿真任务的 IK 数据来源
- **[utils.py:IKFlipBoxController](utils.py#L189)**：双臂 IK 核心实现，被所有夹持/放置流程调用

### 决定效率/效果的实现细节

- 有限差分雅可比使 IK 适配任意 URDF，但每迭代每关节需一次额外 FK
- smoothstep 确保 $C^1$ 连续轨迹，避免速度/加速度突变
- 关键帧扩展 + smoothstep 关节空间插值实现计算量与平滑度的平衡
- attach/detach 时机控制直接影响仿真中箱子行为的物理合理性
- 缓存机制（`_chain_cache`, `_origin_cache`）减少 URDF 树遍历开销
