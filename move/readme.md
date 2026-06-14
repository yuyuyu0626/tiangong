#文件说明

`move` 目录用于抓取后的移动、抬升、路径规划和堆垛放置实验。当前 Python 文件可以分成三类：离线规划/IK 工具、Isaac Gym 仿真任务、包初始化文件。

## 最终脚本

本实验当前保留两个最终任务入口：

- 基础任务最终版本：`move/tasks/task1_2.py`
  - 标准 3 x 5 x 4 垛型，共 60 箱。
  - 包含一二层直接放置策略、三四层 waist-arm 策略，以及当前调好的释放等待、角块收手/推入、站距和抓取高度参数。
- 进阶任务最终版本：`move/tasks/task2_2.py`
  - 新 9 箱/层垛型，共 36 箱。
  - 在 `task2_1.py` 基础上加入第二层、第三层、第四层角块的收手和推入策略，是进阶任务当前调参后的最终版本。


基础任务加速运行完整脚本：

```bash
python -m move.tasks.task1_2 --fast --viewer-render-every 8 
```


进阶任务加速运行完整脚本：

```bash
python -m move.tasks.task2_2 --fast-viewer --viewer-render-every 8 
```




## 环境配置

当前已验证的主要依赖库版本：

- Python：`3.8.20`
- Isaac Gym：`1.0rc4`
- PyTorch：`2.4.1+cu121`
- PyTorch CUDA 构建版本：`12.1`
- NumPy：`1.24.4`
- SciPy：`1.10.1`
- torchvision：`0.19.1`
- ninja：`1.13.0`


## 顶层文件

### `__init__.py`

把 `move` 标记为 Python package，便于其他脚本使用 `from move...` 形式导入模块。文件本身不包含运行逻辑。

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

### `tasks/task1.py`

整合后的 60 箱完整堆叠仿真入口。

该脚本把原 `stack_60_task.py` 中一二层的环境和放置策略，以及原 `waist_arm_ik_task.py` 中三四层的 waist-arm 放置策略，重构到同一个脚本内。运行时只启动一个 Isaac Gym 仿真：先使用一二层策略完成第 1 到第 30 个箱子，机器人回到初始点后，继续使用三四层策略完成第 31 到第 60 个箱子。脚本不依赖从旧任务脚本导入策略代码，保留该文件即可运行完整 demo。

主要内容：

- 创建完整 3 x 5 x 4 堆垛场景，目标总数为 60 个箱子。
- 第 1 到第 30 个箱子使用 `stack_60_task.py` 的原有直接放置、间隙释放、手掌抬离、外侧推入等一二层策略。
- 第 31 到第 60 个箱子使用 `waist_arm_ik_task.py` 的三四层 waist-arm IK 策略。
- 在同一个仿真中切换三四层阶段所需的箱体/接触参数和机器人 DOF 控制参数。
- 只保留完整 60 箱运行模式，不再暴露单独跑某一层、代理层验证、后三十箱验证等旧调试入口。
- 保留仿真加速相关参数，便于快速播放或跳帧验证。

常用运行方式：

```bash
python -m move.tasks.task1
python -m move.tasks.task1 --headless
python -m move.tasks.task1 --fast
```

主要参数：

- `--headless`：不创建 Isaac Gym viewer。
- `--max-frames N`：最多运行 N 帧，`0` 表示跑到任务结束。
- `--fast`：快捷加速模式，会启用非实时 viewer 播放，并降低 viewer 渲染频率。
- `--fast-viewer`：viewer 不按真实时间同步播放。
- `--viewer-render-every N`：每 N 帧渲染一次 viewer。
- `--viewer-start-box N`：从第 N 个箱子开始渲染 viewer，前面的箱子仍正常仿真。
- `--waist-frame-stride N`：三四层 waist-arm 时间线按 N 帧步进，用于加速三四层播放。
- `--waist-stack-plan-ik-stride N`：三四层策略规划阶段的 IK 采样步长。

### `tasks/task1_2.py`

`task1_2.py` 是基础任务的最终版本。垛型仍是 `task1.py` 的 3 x 5 x 4，共 60 箱，但保留了分层代理调试入口，并包含近期针对第二层角块、第三四层角块、第四层中心块和第四层十字块的释放、收手、推入、站距、抓取高度等调参。

主要内容：

- 第 1、2 层使用 `stack_60_task.py` 路线，包含第二层释放等待和角块推入调试策略。
- 第 3、4 层使用嵌入式 waist-arm 路线，包含三四层角块水平收手和推入策略。
- 脚本内部嵌入三四层 waist-arm 相关代码，可以独立运行，不需要从旧任务脚本导入策略。
- 适合继续调试标准 60 箱垛型的局部问题。

常用运行方式：

```bash
python -m move.tasks.task1_2
python -m move.tasks.task1_2 --second-layer-proxy
python -m move.tasks.task1_2 --third-fourth-proxy
python -m move.tasks.task1_2 --fast --viewer-render-every 8
python -m move.tasks.task1_2 --fast --viewer-render-every 8 --waist-frame-stride 2
```

主要模式：

- 默认模式：跑完整 60 箱。
- `--second-layer-proxy`：只跑第 16 到第 30 个箱子，第 1 层用一个 `1.0 x 1.0 x 0.4 m` 代理块替代。
- `--third-fourth-proxy`：只跑第 31 到第 60 个箱子，第 1、2 层用一个 `1.0 x 1.0 x 0.8 m` 代理块替代。
- `--fast` / `--fast-viewer` / `--viewer-render-every N`：用于加速 viewer 播放或减少渲染频率。
- `--waist-frame-stride N`：三四层 waist-arm 时间线按 N 帧步进。

### `tasks/task2_1.py`

`task2_1.py` 是新垛型的基线脚本。整体尺寸仍为 `1.0 x 1.0 x 1.6 m`，共 4 层，每层 0.4 m，但每层改为 9 个箱子：中心块、四个十字块和四个角块。

主要内容：

- 中心块沿用 task1 中心块尺寸和放置策略。
- 第 2、3 个箱子使用 `0.4000 x 0.3333 x 0.4000 m` 尺寸，放在原 task1 第 4、5 号位置。
- 第 4、5 个箱子使用 `0.3333 x 0.3000 x 0.4000 m` 尺寸，放在原 task1 第 2、3 号位置。
- 第 6 到第 9 个角块使用 `0.3500 x 0.3333 x 0.4000 m` 尺寸。
- 第 1、2 层复用 task1 的直接放置环境和 IK 路径；第 3、4 层复用 task1 的 waist-stack 控制路径。

常用运行方式：

```bash
python -m move.tasks.task2_1
python -m move.tasks.task2_1 --until-box 9
python -m move.tasks.task2_1 --second-layer-proxy
python -m move.tasks.task2_1 --third-fourth-proxy
python -m move.tasks.task2_1 --fast-viewer --viewer-render-every 8
```

主要模式：

- 默认模式：跑 task2 新垛型完整 36 箱。
- `--until-box N`：只运行到第 N 个 task2 箱，便于逐箱验证。
- `--second-layer-proxy`：只调试第 2 层，第 1 层用代理块替代。
- `--third-fourth-proxy`：只调试第 3、4 层，第 1、2 层用代理块替代。
- `--stand-off X`：调整一二层直接放置基础站距。
- `--waist-frame-stride N` / `--waist-stack-plan-ik-stride N`：调整三四层 waist-arm 播放和规划步长。

### `tasks/task2_2.py`

`task2_2.py` 是进阶任务的最终版本，垛型和箱体尺寸与 `task2_1.py` 一致。它重点用于验证角块放置后的收手、碰撞恢复和推入距离。

主要内容：

- 第二层角块采用“先按中心块方式放置，再收内侧手，再外侧手推入”的策略。
- 第三、四层角块也加入水平收手和推入动作。
- 保留 `task2_1.py` 的第二层代理模式和三四层代理模式。
- 适合调试角块与前方块余量、手掌余量、推入距离、推入高度和站距。

常用运行方式：

```bash
python -m move.tasks.task2_2
python -m move.tasks.task2_2 --second-layer-proxy
python -m move.tasks.task2_2 --third-fourth-proxy
python -m move.tasks.task2_2 --fast-viewer --viewer-render-every 8
python -m move.tasks.task2_2 --fast-viewer --viewer-render-every 8 --waist-frame-stride 2
```

主要模式：

- 默认模式：跑 task2 新垛型完整 36 箱，使用带推入动作的角块策略。
- `--second-layer-proxy`：只调试第 2 层，第 1 层用代理块替代。
- `--third-fourth-proxy`：只调试第 3、4 层，第 1、2 层用代理块替代。
- `--until-box N`：限制运行到指定箱号。
- `--fast-viewer` / `--viewer-render-every N`：用于加快仿真观察。
