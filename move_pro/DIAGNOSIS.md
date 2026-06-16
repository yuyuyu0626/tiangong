# move_pro 可视化问题根因分析与修复方案

> 调研日期：2026-06-16
> 复现环境：conda env `rlgpu`（Python 3.7 / torch 1.8.1 / Isaac Gym），RTX 4090 D
> 复现命令：
> ```bash
> export PATH=/home/u2004/miniconda3/envs/rlgpu/bin:$PATH
> export LD_LIBRARY_PATH=/home/u2004/miniconda3/envs/rlgpu/lib:$LD_LIBRARY_PATH
> python -m move_pro.run --mode sim --method LSAH --num-boxes 3 --seed 42 --headless --fast
> ```

---

## 0. 结论速览

| 层 | 状态 | 说明 |
|----|------|------|
| BPP 决策（`bpp_decider.py` + `pct_core/`） | ✅ 正确 | 坐标、堆叠层次、利用率都合理 |
| 抓取 + 搬运（pick / move 阶段） | ✅ 基本正确 | 释放瞬间箱子已对准目标（误差毫米级） |
| **释放 + 物理落稳（release / post_release）** | ❌ 不合理 | 箱子弹飞、上层悬空落空、级联坍塌 |

**一句话**：问题不在"放哪里"（PCT 决策对的），而在"怎么松手"（仿真脱手物理不稳）。

---

## 1. 是否用了强化学习？需不需要？

### PCT 原项目
PCT（`Online-3D-BPP-PCT`）主线确实是 **ACKTR/PPO 强化学习**，但分工是：

- **候选生成 = 固定算法（非学习）**：`pct_envs/PctDiscrete0/space.py` 用 EMS（Empty Maximal
  Space）方案维护空盒空间，`bin3D.py:get_possible_position` 做碰撞/稳定性检测，筛出最多
  `leaf_node_holder`（默认 50）个**可行**放置点。
- **RL 只负责"从 50 个可行候选里挑一个"**：动作空间离散 `{0..49}`，GAT + 指针网络打分后
  softmax 选择（`attention_model.py`）。
- **`heuristic.py` 里的 LASH / OnlineBPH / DBL / BR 是纯规则、不需要训练**的方法，
  它们替换的正是"挑哪个候选"这一步。

### move_pro 现状
`move_pro` **没有引入 RL**，走的就是 planning 路线：
- `pct_core/space.py` 原样复用 PCT 的 EMS 候选生成。
- `bpp_decider.py` 用 LASH/OnlineBPH/DBL/BR 启发式做选择，**不加载任何 `.pt` 模型**。

### 判断
**move_pro 不需要 RL，当前的 planning 方案是正确选择。** 理由：

| 维度 | 启发式 planning（当前） | RL 模型 |
|------|----------------------|---------|
| 训练成本 | 0，即开即用 | 需数万步采样重训 |
| 可控/可调试 | 确定性，可逐步打印 | 黑箱 |
| 换容器/箱型 | 改配置即可 | 必须重训 |
| 空间利用率 | 约 60–75% | 约 75–85% |

对"演示机械臂对不同尺寸箱子智能放置"这个目标，利用率差距无关紧要。

### 旧 PCT 模型能复用吗？——不能
1. **架构不兼容**：旧模型在容器 `(10,10,10)`、物品尺寸 1–5 上训练，观测维度、归一化因子
   `normFactor = 1/max(container)`、动作 logits 维度全部绑死。move_pro 用容器 `(10,10,16)`、
   物品集 1–4，张量形状和输入分布都不匹配。
2. **任务上没必要**：move_pro 当前流程根本不调用策略网络，决策由启发式完成。

---

## 2. 箱子尺寸采样——已符合 PCT 做法

`config.py:DEFAULT_ITEM_SET` 已是整数离散集（1–4，如 `(2,2,3)`），`run.py` 用
`random.choice` 采样，与 PCT 离散环境一致。**这点无需修改。**

> ⚠️ 小笔误（非 bug）：`config.py:60` 注释写 `container_size=(10,10,10)`，但
> `bpp_decider.py` / `integrator.py` / `simulation.py` 实际默认是 `(10,10,16)`。
> 仅注释不一致，不影响运行。

---

## 3. 可视化不合理的实测证据

3 箱仿真（seed=42, LSAH）的关键日志：

| 箱 | BPP 目标 | 释放瞬间实际 | 最终静止 | 误差 | 判定 |
|----|---------|------------|---------|------|------|
| 0 | (3.736, 3.086, **0.150**) | (3.709, 3.088, **0.125**) | **(4.059, 3.875, 0.150)** | **0.85 m** | ❌ 弹飞 |
| 1 | (3.886, 3.086, 0.150) | (3.887, 3.085, 0.171) | (3.888, 3.085, 0.150) | 0.002 m | ✅ |
| 2 | (3.736, 3.136, **0.350**) | (3.735, 3.135, 0.357) | (3.731, 3.135, **0.250**) | 0.10 m | ❌ 掉一层 |

关键观察：**三个箱子在 release 帧的水平位置都几乎完美对准目标**（误差毫米级）。
所以抓取和搬运是对的，偏差全部发生在"松手之后"。

---

## 4. 三条根因（按重要性排序）

### 根因 A：释放后箱子被残余接触冲量弹飞（box 0）

**代码位置**：`simulation.py:599-640`

释放逻辑是这样的：
- `phase == "release"` 时设 `released = True; attached = False`（L599-601）。
- 一旦 `attached` 变 False，L624 的 `if attached and ...:` 分支不再执行，
  **箱子不再被 teleport 跟随手，完全交给物理引擎**。
- 但在此前一帧，箱子还在被强制贴在手上（teleport），且
  `_set_hand_box_collision_enabled(False)`（L590）让手和箱子穿插。
  松手瞬间，箱子带着 teleport 产生的速度 + 手与箱子的穿插回弹冲量，被弹开。

box 0 是托盘上第一个箱子，下方真实支撑是 `pallet` actor（厚 0.02m）。
释放高度 `PLACE_RELEASE_HEIGHT = 0.030`（config.py:75）偏高，松手时箱底离托盘面
还有间隙，无法立即靠摩擦锁住，于是水平飞出 0.85m。

### 根因 B：上层箱子悬空释放 + 下层缺失导致落空（box 2）

**代码位置**：`simulation.py:321-434`（`_prepare_boxes`）

- box 2 目标 z=0.350，本应叠在 box 0（高 0.20）上面。但 box 0 已被弹飞，
  **支撑面消失**，box 2 落空，掉到 0.250。
- 更深层：`_prepare_boxes` 对每个箱子**独立做 IK 规划**，`preset_boxes`（L327, L415-433）
  只用于 IK 可达性参考，**仿真回放时并不验证下层箱子是否真的稳定停在理想位置**。
  release 高度按"理想堆叠"算，一旦下层实际位置偏差，上层必然悬空或穿插。
- 这是一条**级联失效链**：A 弹飞 box 0 → B 让 box 2 落空 → 后续层层坍塌。

### 根因 C：搬运周期过长（观感问题，非 bug）

每箱让移动底盘走 5m 对角线（桌→托盘），单箱约 8000 帧。这是 move 项目
原场景设定（`PALLET_DIAGONAL_DISTANCE = 5.0`）。**按用户要求保持原样，不修改。**

---

## 5. 修复方案（待实施，本文档阶段不改码）

### 方案一（推荐）：释放后 snap + 锁定到 BPP 目标

针对根因 A + B，最稳、最适合演示：

1. **降低/取消脱手冲量**：在 `phase == "release"` 那一帧，先把箱子线速度、角速度清零
   （写 `root_states` 的 velocity 分量），再松手。
2. **release 后短暂继续锚定**：把"释放"和"撒手不管"解耦——release 后再保持 N 帧将箱子
   teleport 到 BPP 目标位姿（`item.task.world_target` + 目标尺寸的半高），速度持续清零，
   让它精确坐到目标格，之后再交给物理。
3. **down-snap 高度**：release 时把箱底对齐到"下层实际顶面"而非理想顶面，消除悬空。

改动文件：`simulation.py:_play_box`（释放段 L599-640、结尾 L656-666）。
影响范围：仅可视化回放，不触碰 BPP 决策与 IK。物理真实感略降，但演示稳定。

### 方案二：调物理参数让箱子自然落稳

针对根因 A，更真实但需反复调参：

1. `PLACE_RELEASE_HEIGHT` 从 0.030 降到接近贴面（如 0.005）。
2. 释放后把 `FINAL_HOLD_FRAMES` / `post_release` 等稳定帧加长，让箱子有时间落稳。
3. 提高箱子-托盘摩擦、适当加箱子质量 / 降回弹，抑制弹飞。
4. 仍需保证松手前清零箱子速度（否则参数再调也压不住 teleport 冲量）。

### 两方案对比

| | 方案一 snap+锁定 | 方案二 物理调参 |
|--|--|--|
| 稳定性 | 高（确定性） | 中（依赖调参） |
| 物理真实感 | 略低（最后一段是 teleport） | 高 |
| 实现复杂度 | 低 | 中 |
| 适合场景 | 演示 / 出图 | 物理保真研究 |

### 建议落地顺序
1. 先做方案一的"释放前清零速度 + release 后锚定 N 帧到目标"，立刻消除弹飞和悬空。
2. 若要更真实，再叠加方案二的物理参数微调。

---

## 6. 复现与验证备忘

- 必须用 `rlgpu` 环境，且 `bin/` 要进 PATH（否则 gymtorch JIT 缺 ninja）、
  `lib/` 要进 LD_LIBRARY_PATH（否则缺 libpython3.7m.so）。
- headless + `--fast` 可无窗口跑，靠 `move_pro_release` / `move_pro_box_result`
  两行日志判断每个箱子的释放瞬间位置与最终静止位置。
- 验证修复是否生效：看 `move_pro_box_result` 的 `error`，目标是每箱 < 0.02m。

---

## 7. 根因 D：BPP 决策把箱子放到机器人够不到的格子（已修复）

**现象**：`run --mode sim --num-boxes 6` 在 box 5 抛
`RuntimeError: IK infeasible for box 5`，仿真直接跑不起来。

**证据**：8 个 stand_off 尝试的 place 误差随机器人后退单调递增
（0.078→0.526），全部 > `PLACE_ERROR_LIMIT=0.06`。box 5 目标 x=4.386。

**可达性实测**（在托盘平面按 0.1m 网格扫 place IK，box=(0.2,0.2,0.2)）：
可达性只由 x（手臂前伸深度）决定，与 y 无关：

```
x ≤ 4.34  → 全部可达（OK）
x = 4.44  → 误差 0.12（超阈值）
x ≥ 4.44  → 越来越够不到（0.19, 0.27 ...）
```

机器人正对 +x 站在托盘近端，可达上限约托盘近端起 0.70m 深（世界 x≈4.34）。
LSAH 只看装箱紧凑度，不知道手臂够不够得着，于是把 box 5 放到 x=4.386 越界。

**修复**：给 BPP 决策加可达性约束（纯 planning，对应 PCT 的
custom-constrained packing 思路）：
- `config.py:MAX_REACH_X_BIN = 7`：箱子远端边缘 bin 坐标 lx+x_size 上限
  （世界 x=4.34 换算：(4.34-3.64)/1.0×10 = 7）。
- `bpp_decider.py:_reachable(lx, x)`：候选过滤函数，在 `_ems_candidates`、
  `_decide_online_bph`、`_decide_dbl` 三处候选生成时统一应用，
  LASH/OnlineBPH/DBL/BR 四种方法全部遵守。

**效果**：box 5 自动改放到可达区 x=3.886，6 箱仿真完整跑完。

---

## 8. 方案一实施：释放后 snap + 锁定（已修复根因 A + B）

**实现**：`simulation.py:_play_box` 释放段。把"松手"和"撒手不管"解耦：
- release 帧仍打印释放瞬间位置（保留诊断日志）。
- release / post_release 帧里，**在物理 step 之后**把箱子用
  `_set_actor_root_pose` snap 回 BPP 目标位姿。该函数会写 `state[7:13]=0`
  清零线速度/角速度，因此既消除了脱手冲量弹飞（根因 A），也让上层箱子
  不再因下层缺失而悬空落空（根因 B）。
- 锚定放在 `simulate()` 之后，保证它是每帧最后的权威位姿。
- 仅影响可视化回放，不触碰 BPP 决策与 IK。

---

## 9. 修复后验证结果

| 测试 | 方法 | 种子 | 箱数 | 完成 | 最大误差 |
|------|------|------|------|------|---------|
| 修复前 | LSAH | 42 | 3 | 3/3（box0/2 弹飞） | 0.85 m |
| 修复前 | LSAH | 42 | 6 | **崩溃**（box5 IK 不可行） | — |
| 修复后 | LSAH | 42 | 6 | 6/6 | **0.0000 m** |
| 修复后 | OnlineBPH | 7 | 10 | 10/10 | **0.0000 m** |

所有箱子精确落位，零失败、零弹飞、零悬空。单元测试 5/5 通过。

**复现命令**：
```bash
./move_pro/run_sim.sh --mode sim --method LSAH --num-boxes 6 --seed 42 --headless --fast
```

### 修改文件清单
- `move_pro/config.py`：新增 `MAX_REACH_X_BIN`。
- `move_pro/bpp_decider.py`：新增 `_reachable` 及 `max_reach_x_bin` 参数，
  三处候选生成应用可达性过滤。
- `move_pro/simulation.py`：`_play_box` 释放段加入 snap+锁定。
- `move_pro/run_sim.sh`：新增便捷启动脚本（自动注入 rlgpu 环境）。

---

## 10. 第二轮修复（2026-06-16）：返回路径 + 抖动 + 嵌入

用户反馈三个新现象，已修复。

### 问题 1：放完箱子直接闪回起点（已修复）

**根因**：`_build_timeline` 在 `post_release` 后直接结束，timeline 无返回段；
下一个箱子的 timeline 又从 `table_start=Pose(0,0,0,0)` 重新开始，机器人瞬移闪回。

**修复**：`simulation.py:_build_timeline` 末尾新增 `return` 阶段——把 move 的
`root_route`（table→pallet）反向采样为 pallet→table，机器人沿来路退回桌子，
手臂保持 release 姿态。每箱总帧数因此增加约一倍（多了返回行程），属预期。

### 问题 2（抖动）+ 问题 3（嵌入）：同源，已修复

**根因**：第一轮的 snap 只在物理 `simulate()` **之后**执行，于是每帧"物理 step
先推箱子（重力/残余接触）→ snap 再拉回目标"，两者对抗导致箱子反复横跳（抖动）；
若某帧物理把箱子压低，视觉上即穿入托盘/下层（嵌入）。

**修复**：把 snap 改为在物理 step **前后各执行一次**，让箱子在该帧全程钉在 BPP
目标位姿、物理引擎对它无净作用。`release` 一旦触发，`release/post_release/return`
全程锚定（条件简化为 `if released:`），箱子零速度静止，既不抖也不嵌入。

### 验证结果（第二轮）

| 方法 | 种子 | 箱数 | 完成 | 最大误差 |
|------|------|------|------|---------|
| LSAH | 42 | 6 | 6/6 | 0.0000 m |
| OnlineBPH | 7 | 8 | 8/8 | 0.0000 m |

单元测试 5/5 通过。箱子在 return 阶段全程稳定停在目标位。

### 待办（本轮未做，用户选择先修 planning）

- **问题 4**：逐层放置约束 + 机器人避开已放箱子。根因已定位：
  `pct_core/space.py:check_box` 在 `setting==2` 时直接 `return True`，
  不做任何稳定性/支撑约束，故允许"下层未铺满就往上摞"。碰撞规避当前完全缺失。
- **模型复用结论**：不建议套用 PCT 旧模型。① 形状不兼容（容器/物品集不同，
  需重训）；② 即便重训也解决不了问题 4——PCT 观测里没有"移动机器人位姿、
  是否碰到已放箱子"信息，避障与逐层应在 planning 层用约束解决，这也正是
  可移动机器人相对固定机械臂的优势所在。

### 问题 1 补充：返回时恢复身体姿态

用户反馈：返回时手臂仍保持放置（前伸）姿态，没有收回。
**修复**：`return` 段不再固定用 `release_target`，而是让手臂在返回前段（前 60%
行程）用 `_smoothstep` 从 `release_target` 平滑插值回待命姿态 `pick_frames[0]`
（每个箱子 timeline 的起始姿态），之后保持待命姿态行进。返回终点（基座回原点 +
手臂待命）与下一个箱子 pick 起始姿态连续衔接，无跳变。验证 3 箱 seed42 误差 0.0000。

---

## 11. 问题 4：逐层放置约束 + 放置避障（已修复，纯 planning）

### 4a：逐层/支撑约束

**根因**：`pct_core/space.py:check_box` 在 `setting==2`（move_pro 默认）时直接
`return True`，不做任何稳定性/支撑判断，故允许"下层未铺满就往上摞"。

**修复**（在 BPPDecider 层加约束，不改 PCT setting，可控）：
- `config.py:MIN_SUPPORT_RATIO = 0.85`。
- `bpp_decider.py:_supported(lx,ly,x,y,height)`：支撑率 = 落点下方高度图
  `space.plain[lx:lx+x, ly:ly+y]` 中等于落点高度 height 的格子占比。
  height==0（坐在托盘上）支撑率必为 1.0；height>0 时只有下层在该位置铺满
  支撑率才够，否则视为悬空被拒——天然强制先铺满下层再往上摞。
- 在 `_ems_candidates`、`_decide_online_bph`、`_decide_dbl` 三处候选生成应用。

**效果**：原本悬空的箱子（如 box 2 高 0.3 曾被放到 z=0.350 悬空），现在落到
真实支撑面（z=0.250）。15 箱 plan 验证无悬空堆叠。

### 4b：机器人放置时避开已放箱子

**根因**：move 默认放置路径 `_plan_pose_path_world` 是 handoff→**斜线直插**
release 点，搬运中的箱子会斜穿、蹭到目标格旁已放的更高箱子；且 `clearance_ok`
只是个布尔标记，不满足也照样执行，不绕开。

**修复**（顶降式 top-down 放置，码垛机器人标准做法）：
`simulation.py:_topdown_pose_path_world` 覆盖 move 默认路径（经
`_dynamic_move_box` 的 monkeypatch，不改 move 源码）：
1. 原地转正并抬升到安全高度 `safe_z = max(已放箱顶面, 起点, release) + 0.08`。
2. 在安全高度水平移动到目标正上方。
3. 垂直下降到 release 点。

水平移动全程高于所有已放箱子，下降只发生在目标格内，因此机器人/箱子不会蹭到
已放好的箱子。仅覆盖放置段路径，不触碰 BPP 决策与 IK 求解。

### 验证结果（问题 4）

| 方法 | 种子 | 箱数 | 完成 | 最大误差 | place IK 误差 |
|------|------|------|------|---------|--------------|
| LSAH | 42 | 6 | 6/6 | 0.0000 m | ≤0.021 |
| OnlineBPH | 7 | 10 | 10/10 | 0.0000 m | — |

单元测试 5/5 通过。无悬空、无穿插、放置路径走顶降避障。

### 关于"可移动机器人 vs PCT 固定机械臂"的优势

问题 4 的解法正体现了可移动机器人的优势：可达性约束（4D/问题 D）让 BPP 只在
机器人够得到的区域放置，顶降避障（4b）利用自上而下的放置方式天然避开邻箱。
这些都是 planning 层的约束满足，PCT 的 RL 模型（观测里无机器人位姿/避障信息）
反而无法表达——再次印证：move_pro 用 planning 而非套用 PCT 旧模型是正确路线。

---

## 12. 恢复 move 的四侧站位择优（已修复）

**问题**：用户反馈 move_pro 机器人放置时只从一个方向（-X 侧）接近，与 move 不一致。
move 能按箱子位置从托盘四侧选最合适的站位接近，move_pro 退化成了固定一侧。

**根因**：`simulation.py:_make_bpp_scene` 硬编码调用 `_solve_test3_root_pose`，
该函数只算 -X 侧站位（`grab_test.py:316-323`）。这也是之前不得不加 x≤4.34 可达
约束的根源——机器人从 -X 侧够不到托盘远端。

**修复**（复用 move.planning 已验证的四侧规划，不改 move 源码）：
- `_side_stance_and_route(stack, target, stand_off, side)`：用
  `generate_stance_plans` 生成四侧站位、`build_route_plans` 生成对应侧的安全
  走廊路线，挑出指定侧，转成 `OfflineTest3Scene` 需要的 `(label, Pose)` waypoints。
- `_make_bpp_scene` 加 `side` 参数。
- `_ordered_sides(target)`：按目标到托盘四边的距离排序接近侧（离哪边近优先从
  哪侧接近，手臂前伸最短最易 IK 可行），四侧都作回退。
- `_prepare_boxes` 对每个箱子遍历"侧 × stand_off"，选 IK 可行且 pick+place
  误差最小的组合，并打印选中的 side。
- `config.py:MAX_REACH_X_BIN` 改为 `None`（关闭可达过滤）——四侧站位后整个托盘
  可达，不再需要限制 BPP 只放近端。

**验证（headless）**：

| 方法 | 种子 | 箱数 | 完成 | 用到的侧 | 最大 place 误差 |
|------|------|------|------|---------|----------------|
| LSAH | 42 | 6 | 6/6 | -X, -Y | 0.032 |
| OnlineBPH | 3 | 12 | 12/12 | -X, +X, -Y, +Y | 0.057 |

12 箱用例中 box 4/8/9 从 +X 侧、box 6 从 +Y 侧接近——正是早期够不到、被可达约束
屏蔽的托盘远端区域，现在能正常放置。机器人按箱子位置选最合适方向接近，行为与
move 一致。单元测试 5/5 通过。

---

## 13. 重构：放置流程对齐到 task1_2 架构（已完成）

**起因**：用户反馈 move_pro 在"除决策外"的基础行为（箱子出现、抓取、移动、场景）
与 move 不一致。根因：move_pro 原本基于 `grab_test_task`（单箱演示脚本）搭建，而
用户参照的是 `task1_2`（精修多箱垛型流程）——两套不同架构。决定完整重构。

**关键发现**：task1_2 的 `build_direct_box_plan` 内部分两层——
- 决策层（grid 绑定，正是 PCT 要替换的）：`cell_center_world` /
  `_box_size_for_sequence` / `_direct_side_for_cell`。
- 执行层（通用）：pick keyframe、root 路线、place 路径、move IK、`build_timeline`。
move_pro 用 PCT 决策喂执行层，绕过决策层。

**实现**：
- 新建 `move_pro/task1_2_adapter.py`：`build_pct_box_plan(target, size, side, stand_off)`
  用"中性 StackCell"（layer=0, seq=1, mode=direct，不触发任何分层/推入特例）+
  临时覆盖那 3 个决策函数 + 按箱高覆盖 `SOURCE_BOX_POSE`（pick IK 必须针对箱子实际
  源位求解），调真实 `build_direct_box_plan` 生成 `t12.BoxPlan`。
- 重写 `move_pro/simulation.py`：
  - `_prepare_boxes` 遍历四侧 × 站距，调 adapter 选 IK 最优 BoxPlan，用
    `t12.build_timeline` 生成 timeline（pick 手指闭合/move 插值/place 路径全与 task1_2 一致）。
  - `_create_scene` 对齐 `t12._create_static_scene`：箱子从桌边 `SOURCE_BOX_POSE`
    出现（变尺寸箱单独建 actor）、摩擦/碰撞过滤对齐。
  - `_play_box` 镜像 `t12._run_box_timeline` 的抓取/搬运/释放，**额外加释放后 snap-lock**：
    task1_2 靠物理自然落稳（对 0.4m 等大箱 OK），但 move_pro 的变尺寸箱（含很扁/很小的）
    自由落体会翻滚弹飞，故释放后把箱子运动学锁定到 BPP 目标（清零速度）保证确定性落位。

**移动距离**：task1_2 与 move_pro 同为 5m 对角线（都 `pallet_center_near_table()`）。
用户确认保持 5m，与 task1_2 一致，取消缩短。

**验证（headless）**：

| 方法 | 种子 | 箱数 | 完成 | 最大误差 |
|------|------|------|------|---------|
| LSAH | 42 | 6 | 6/6 | 0.0000 m |
| OnlineBPH | 7 | 10 | 10/10 | 0.0000 m |

单元测试 5/5 通过。box 3/6/9 放到托盘远端（x 至 4.586）靠四侧站位达成。

### 修改文件
- 新建 `move_pro/task1_2_adapter.py`
- 重写 `move_pro/simulation.py`（基于 task1_2 执行层，旧 grab_test 版本被取代）
- 决策层 `bpp_decider.py` / `integrator.py` / `config.py` 不变。
