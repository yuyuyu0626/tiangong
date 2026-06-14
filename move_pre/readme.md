#文件说明

`move_pre` 目录用于抓取后的移动、抬升、路径规划和堆垛放置实验。当前 Python 文件可以分成三类：离线规划/IK 工具、Isaac Gym 仿真任务、包初始化文件。

## 顶层文件

### `__init__.py`

把 `move_pre` 标记为 Python package，便于其他脚本使用 `from move_pre...` 形式导入模块。文件本身不包含运行逻辑。

### `planning.py`

纯几何规划模块，不依赖 Isaac Gym。

主要内容：

- 定义桌子、源箱、托盘、预置箱、目标箱等几何常量。
- 定义 `Pose`、`BoxPlacement`、`EmptySpace`、`StackScene`、`StancePlan`、`RoutePlan` 等数据结构。
- 计算托盘位置、EMS 空间、目标箱放置位置、机器人站位和移动路线。
- 提供命令行入口，可直接输出文本计划或 JSON 计划。

适合用于快速检查堆垛几何、站位和路线，不需要仿真环境。

### `utils.py`

`move` 本地的 URDF/IK 工具模块。

主要内容：

- Xhand 掌心几何常量，例如 `PALM_SURFACE_X`、`PALM_CENTER_Z`、`PALM_BOX_CLEARANCE`。
- 末端 link 名称、左右手臂关节名等常量。
- `JointInfo`：URDF joint 的最小运动学信息。
- `CartesianKeyframe`：双手笛卡尔 IK 目标关键帧。
- `UrdfKinematics`：轻量 URDF 前向运动学，只根据 URDF 和关节角计算 link 位姿。
- `IKFlipBoxController`：名字保留兼容旧调用，实际是 `move` 本地的双掌 IK 控制器，用于把左右掌心目标转换为机器人 DOF target。

### `mobile_ik.py`

移动状态下的抬升关节 IK 工具。

主要内容：

- `MoveLiftIK`：在手臂/手保持已有姿态的情况下，只求解移动状态相关的抬升关节：
  - `first_leg_pitch_joint`
  - `second_leg_pitch_joint`
  - `waist_pitch_joint`
- `palm_z()`：计算左右掌心平均高度。
- `solve_for_box_center_z()`：根据目标箱体中心高度，迭代调整抬升关节。
- `target_box_center_z()`：根据目标支撑面高度、箱体高度和 clearance 计算目标箱心高度。

该模块被 `grab_test.py`、`move_test2.py`、`move_test3.py` 等脚本使用。

### `grab_test.py`

离线抓取、移动、放置 IK 链路验证脚本，不依赖 Isaac Gym。

主要流程：

- 解析 `move/assets/integrated/tianyi_xhand_move.urdf` 的活动 DOF 和 limit。
- 构造双掌抓取源箱的关键帧。
- 使用 `move.utils.IKFlipBoxController` 求解抓取和抬起阶段的 DOF target。
- 使用 `MoveLiftIK` 求解移动状态下的抬升关节。
- 规划从桌边到托盘目标位的 root 路线。
- 在目标站位下继续求解放置阶段的双掌 IK。
- 返回 `GrabTestReport` 和包含各段轨迹 target 的 payload。

常用于在不启动 Isaac Gym 的情况下检查 IK 是否可行，也可以保存生成的计划给仿真任务使用。

## `tasks/` 仿真脚本

### `tasks/move_test1.py`

Isaac Gym 移动状态基础场景。

主要内容：

- 创建桌子、源箱、托盘、预置箱和目标箱 ghost。
- 加载 `move/assets/integrated/tianyi_xhand_move.urdf`。
- 使用 `planning.py` 生成的四侧站位路线，让机器人 root 沿路线移动。
- 手臂/手保持本地生成或本地记录的 carry 姿态。
- 提供通用仿真 helper，例如 `create_sim()`、`load_robot_asset()`、`_make_transform()`、`_set_color()`、`_set_shape_friction()`。

`move_test2.py`、`move_test3.py`、`grab_test_task.py` 会复用这里的一部分 helper。

### `tasks/move_test2.py`

在 `move_test1` 路线基础上加入实际抬升关节 IK 的场景。

主要内容：

- 复用 `move_test1` 的场景搭建和机器人资产加载逻辑。
- 使用 `MoveLiftIK` 计算目标箱心高度对应的抬升关节姿态。
- 根据路线阶段在 hold target 和 lift target 之间混合。
- 用于测试移动过程中靠腿/腰抬升而不是直接修改 root z 的效果。

### `tasks/move_test3.py`

顶部堆叠目标场景：把第三个 0.2m 箱放到第二个 0.2m 箱上方。

主要内容：

- 定义 `HeightTestScene`，描述一个需要高度 IK 的堆垛测试场景。
- 构造 “两个 0.2m 预置箱 + 第三个箱放到第二个箱上方” 的 EMS 场景。
- 规划机器人最终站位、临时站位和安全移动路径。
- 使用 `MoveLiftIK` 求解放置高度。
- 在 Isaac Gym 中显示源箱、托盘、预置箱、目标位置和机器人运动。

这是当前高度/抬升 IK 的主要测试场景之一。

### `tasks/move_test4.py`

基于 `move_test3.py` 的场景变体。

主要内容：

- 复用 `move_test3` 的仿真主流程和 helper。
- 覆盖 `build_height_test_scenes()`，构造新的 EMS 场景：
  - 一个 0.3m 基础箱。
  - 一个 0.2m 箱放在基础箱上方。
  - 第三个 0.2m 目标箱放在基础箱右侧的托盘地面空位。
- 适合测试不同目标叶子空间下的站位和抬升 IK 行为。

### `tasks/grab_test_task.py`

完整抓取、移动、放置流程的 Isaac Gym 播放脚本。

主要内容：

- 调用 `move.grab_test.run()` 生成离线 IK 计划。
- 在 Isaac Gym 中创建完整场景并播放：
  - 双掌侧向抓取源箱。
  - 抬起并切换到移动状态。
  - root 移动到 `move_test3` 修正后的放置站位。
  - 在目标位置执行放置轨迹。
- 抓取初期箱子是物理对象；夹住后会把箱子按双掌抓取框架做运动学 attach；到释放点附近再 detach，让箱子落到堆垛位置。
- 包含姿态插值、碰撞过滤、抓取中心估计、箱体 yaw/pitch 估计等仿真辅助逻辑。

这是目录里最接近完整任务流程的仿真入口。
