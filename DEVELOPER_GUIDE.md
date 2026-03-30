# 无人机路径规划算法教学试验系统 — 开发者指南

> 本指南面向希望**接入自定义导航算法**的开发者。  
> 整个系统围绕"注册表驱动"设计：新算法只需两步即可完成集成，无需修改 GUI 层代码。

---

## 目录

1. [系统架构总览](#1-系统架构总览)  
2. [关键接口约定](#2-关键接口约定)  
   - 2.1 [训练函数签名](#21-训练函数签名)  
   - 2.2 [路径构建函数签名](#22-路径构建函数签名)  
3. [接入自定义算法（完整步骤）](#3-接入自定义算法完整步骤)  
   - 步骤 1：创建算法文件  
   - 步骤 2：注册到中央注册表  
4. [完整示例：RRT 算法](#4-完整示例rrt-算法)  
5. [地图数据结构说明](#5-地图数据结构说明)  
6. [现有算法简介](#6-现有算法简介)  
7. [常见问题](#7-常见问题)

---

## 1. 系统架构总览

```
main.py
└── GUI.py  (StartWindow)
    └── map/map_qt_3d_view.py  (ThreeDWindow — 主窗口)
        └── map/map_qt_opengl_widget.py  (RotatingCubeWidget — OpenGL 渲染与算法调度)
            └── algorithm/algorithm_entrypoints.py  ← 算法注册表（唯一修改点）
                ├── algorithm/algorithm_a_star.py
                ├── algorithm/algorithm_q_learning.py
                ├── algorithm/algorithm_dqn.py
                ├── algorithm/algorithm_ppo.py
                └── algorithm/algorithm_her.py
```

### 层职责说明

| 层 | 文件 | 职责 |
|---|---|---|
| 入口 | `main.py` | 初始化环境、启动 GUI |
| 启动窗口 | `GUI.py` | 显示欢迎界面，加载地形 |
| 主窗口 | `map/map_qt_3d_view.py` | UI 布局、按钮事件绑定；算法列表**自动**从注册表生成 |
| 渲染/调度 | `map/map_qt_opengl_widget.py` | OpenGL 渲染、路径动画；算法训练与导航**均通过注册表统一调度** |
| **算法注册表** | `algorithm/algorithm_entrypoints.py` | **唯一需要修改的文件**，维护两个字典：`TRAIN_ENTRYPOINTS` 和 `PATH_ENTRYPOINTS` |
| 算法实现 | `algorithm/algorithm_*.py` | 各算法的独立实现，互不依赖 |

---

## 2. 关键接口约定

所有接入注册表的函数必须严格遵循以下签名，否则系统无法自动调度。

### 2.1 训练函数签名

```python
def train_xxx_full_map(
    occupancy,                                           # np.ndarray | None，形状 (nx, ny_h, nz)，1=障碍
    heights,                                             # np.ndarray | None，形状 (rows, cols)，高度场（备用）
    end_map_xyz: tuple[float, float, float] | None,      # 目标点在地图坐标中的位置，范围 [0, 100]
    map_to_index: Callable[[float, int, str], int],      # 坐标→格子索引转换函数（见下方说明）
    progress_callback: Callable[[int], None] | None = None,  # 训练进度回调 0-100
) -> tuple[bool, str, YourModel | None]:
    """
    返回值：
        bool   — 训练是否成功
        str    — 供 GUI 显示的状态消息
        model  — 训练结果（任意可序列化对象），失败时为 None
    """
```

> `map_to_index(coord: float, n: int, mode: str) -> int`  
> 将地图坐标（0.0–100.0）映射到格子索引（0–n-1）。  
> `mode` 通常传 `"round"`（四舍五入）。

### 2.2 路径构建函数签名

```python
def build_xxx_path(
    model,                                               # train_xxx_full_map 返回的模型对象
    start_xyz: tuple[float, float, float],               # 起点地图坐标
    end_xyz:   tuple[float, float, float],               # 终点地图坐标
    map_to_index: Callable[[float, int, str], int],
) -> list[tuple[float, float, float]]:
    """
    返回：路径点列表，每个元素为 (x, y, z)，坐标范围 [0, 100]。
    路径点数 < 2 表示规划失败，系统将不启动导航动画。
    """
```

---

## 3. 接入自定义算法（完整步骤）

### 步骤 1：创建算法文件

在 `algorithm/` 目录下新建 `algorithm_xxx.py`，实现两个函数。

```
algorithm/
    algorithm_xxx.py   ← 新建此文件
```

最小模板如下：

```python
# algorithm/algorithm_xxx.py
"""
XXX 导航算法实现。

函数约定：
  train_xxx_full_map  — 离线学习/预处理，返回 (bool, str, model)
  build_xxx_path      — 在线路径查询，返回路径点列表
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class XxxModel:
    """存储训练结果的数据类（可自由扩展字段）。"""
    free_mask: Any        # bool ndarray，表示哪些格子可以飞行
    grid_shape: tuple     # (nx, [ny_h,] nz)
    is_3d: bool
    # ... 其他算法相关字段


def train_xxx_full_map(
    occupancy,
    heights,
    end_map_xyz: tuple[float, float, float] | None,
    map_to_index: Callable[[float, int, str], int],
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[bool, str, XxxModel | None]:
    import numpy as np

    if end_map_xyz is None:
        return (False, "请先设置终点", None)

    # --- 从 occupancy 或 heights 构建 free_mask ---
    if occupancy is not None:
        occ = np.asarray(occupancy)
        free_mask = occ == 0          # 0 = 可通行
        is_3d = True
        grid_shape = occ.shape        # (nx, ny_h, nz)
    elif heights is not None:
        rows, cols = heights.shape
        free_mask = np.ones((cols, rows), dtype=bool)
        is_3d = False
        grid_shape = (cols, rows)
    else:
        return (False, "无地图数据", None)

    # TODO: 在此实现算法的离线训练/预处理逻辑
    # 可通过 progress_callback(0~100) 向 GUI 报告进度

    model = XxxModel(free_mask=free_mask, grid_shape=grid_shape, is_3d=is_3d)
    return (True, "XXX 训练完成", model)


def build_xxx_path(
    model: XxxModel,
    start_xyz: tuple[float, float, float],
    end_xyz:   tuple[float, float, float],
    map_to_index: Callable[[float, int, str], int],
) -> list[tuple[float, float, float]]:
    # TODO: 根据 model 从 start_xyz 规划到 end_xyz 的路径
    # 返回路径点列表，坐标范围 [0, 100]
    return []
```

### 步骤 2：注册到中央注册表

打开 `algorithm/algorithm_entrypoints.py`，添加两行：

```python
# ① 在文件顶部导入新算法
from algorithm.algorithm_xxx import build_xxx_path, train_xxx_full_map

# ② 将新算法名称和函数加入两个注册表

TRAIN_ENTRYPOINTS: dict[str, ...] = {
    "Q-learning": _train_q_learning_normalized,
    "DQN":        train_dqn_full_map,
    "PPO":        train_ppo_full_map,
    "HER":        train_her_full_map,
    "XXX":        train_xxx_full_map,   # ← 新增这一行
}

PATH_ENTRYPOINTS: dict[str, ...] = {
    "Q-learning": build_q_learning_path,
    "DQN":        build_dqn_path,
    "PPO":        build_ppo_path,
    "HER":        build_her_path,
    "XXX":        build_xxx_path,       # ← 新增这一行
}
```

**完成！** 重启程序后，算法下拉框会自动出现 "XXX" 选项，训练按钮和导航按钮均正常工作。

---

## 4. 完整示例：RRT 算法

下面是一个基于随机采样的 RRT（Rapidly-exploring Random Tree）简化实现，用于演示接入流程。

```python
# algorithm/algorithm_rrt.py
"""RRT 路径规划算法（简化演示版）。"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RRTModel:
    free_mask: Any
    grid_shape: tuple
    is_3d: bool
    start_xyz: tuple[float, float, float] | None = None
    end_xyz:   tuple[float, float, float] | None = None


def _free_at(free_mask, xi: int, yi: int, nx: int, ny: int) -> bool:
    return 0 <= xi < nx and 0 <= yi < ny and bool(free_mask[xi, yi])


def train_rrt_full_map(
    occupancy,
    heights,
    end_map_xyz: tuple[float, float, float] | None,
    map_to_index: Callable[[float, int, str], int],
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[bool, str, RRTModel | None]:
    """RRT 是在线规划算法，训练阶段仅构建 free_mask。"""
    import numpy as np

    if end_map_xyz is None:
        return (False, "请先设置终点", None)

    if occupancy is not None:
        occ = np.asarray(occupancy)
        free_mask = (occ == 0)
        is_3d = True
        grid_shape = occ.shape
    elif heights is not None:
        rows, cols = heights.shape
        free_mask = np.ones((cols, rows), dtype=bool)
        is_3d = False
        grid_shape = (cols, rows)
    else:
        return (False, "无地图数据", None)

    if progress_callback:
        progress_callback(100)

    model = RRTModel(free_mask=free_mask, grid_shape=grid_shape, is_3d=is_3d)
    return (True, "RRT 就绪（在线规划，无需离线训练）", model)


def build_rrt_path(
    model: RRTModel,
    start_xyz: tuple[float, float, float],
    end_xyz:   tuple[float, float, float],
    map_to_index: Callable[[float, int, str], int],
) -> list[tuple[float, float, float]]:
    """2D-only RRT 演示实现。"""
    if model.is_3d or model.free_mask is None:
        return []

    nx, ny = model.grid_shape
    sx = map_to_index(start_xyz[0], nx, "round")
    sy = map_to_index(start_xyz[1], ny, "round")
    ex = map_to_index(end_xyz[0], nx, "round")
    ey = map_to_index(end_xyz[1], ny, "round")

    tree: dict[tuple[int,int], tuple[int,int] | None] = {(sx, sy): None}
    step = 3
    goal_radius = 4
    MAX_ITER = 5000

    for _ in range(MAX_ITER):
        # 以 10% 概率直接朝目标采样
        if random.random() < 0.1:
            rx, ry = ex, ey
        else:
            rx = random.randint(0, nx - 1)
            ry = random.randint(0, ny - 1)

        # 找最近树节点
        nearest = min(tree, key=lambda n: math.hypot(n[0]-rx, n[1]-ry))
        dx = rx - nearest[0]; dy = ry - nearest[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        nx_i = nearest[0] + int(round(dx / dist * step))
        ny_i = nearest[1] + int(round(dy / dist * step))

        if not _free_at(model.free_mask, nx_i, ny_i, nx, ny):
            continue

        node = (nx_i, ny_i)
        tree[node] = nearest

        if math.hypot(nx_i - ex, ny_i - ey) <= goal_radius:
            tree[(ex, ey)] = node
            # 回溯路径
            path_idx = []
            cur: tuple[int,int] | None = (ex, ey)
            while cur is not None:
                path_idx.append(cur)
                cur = tree[cur]
            path_idx.reverse()
            return [
                (float(xi) / max(1, nx-1) * 100.0,
                 float(yi) / max(1, ny-1) * 100.0,
                 start_xyz[2])
                for xi, yi in path_idx
            ]

    return []  # 超过最大迭代次数，规划失败
```

然后在 `algorithm_entrypoints.py` 中添加：

```python
from algorithm.algorithm_rrt import build_rrt_path, train_rrt_full_map

TRAIN_ENTRYPOINTS = {
    ...
    "RRT": train_rrt_full_map,
}
PATH_ENTRYPOINTS = {
    ...
    "RRT": build_rrt_path,
}
```

---

## 5. 地图数据结构说明

系统使用 **HDF5 占用栅格地图**（Occupancy Grid）。

### occupancy（3D 占用格）

| 属性 | 说明 |
|---|---|
| 类型 | `numpy.ndarray`，`dtype=uint8` |
| 形状 | `(nx, ny_h, nz)`，各轴范围约 100 格 |
| 含义 | `0` = 可飞行；`1` = 障碍/地面 |
| 坐标 | x 轴（东西）、ny_h 轴（高度）、z 轴（南北） |

### heights（2D 高度场，备用）

当 `occupancy` 为 `None` 时提供，`shape = (rows, cols)`，每格值为地面高度。

### map_to_index 函数

```python
# 内部实现（位于 map_qt_opengl_widget.py）：
def _map_to_index(coord: float, n: int, mode: str) -> int:
    """将地图坐标 [0.0, 100.0] 映射为格子索引 [0, n-1]。"""
    val = coord / 100.0 * (n - 1)
    return max(0, min(n - 1, round(val)))
```

### 坐标系约定

- 所有路径点均以 **地图坐标** 表示，范围 `[0.0, 100.0]`（与地图 nx/ny/nz 无关）。  
- `(x, y, z)` 中 y 为平面方向（南北），z 为高度方向。  
- `map_to_index(coord, n, "round")` 是算法获取格子索引的统一入口。

---

## 6. 现有算法简介

| 算法 | 类型 | 文件 | 特点 |
|---|---|---|---|
| **A\*** | 搜索式 | `algorithm_a_star.py` | 无需训练，实时规划，路径最优 |
| **Q-learning** | 表格式 RL | `algorithm_q_learning.py` | 全局最优策略，内存占用随地图大小增长 |
| **DQN** | 深度 RL | `algorithm_dqn.py` | 神经网络策略，以 Q-learning 为教师蒸馏训练 |
| **PPO** | 深度 RL | `algorithm_ppo.py` | PPO 剪切策略优化 + 价值头，以 Q-learning 为教师 |
| **HER** | 深度 RL | `algorithm_her.py` | 后见经验回放，以 Q-learning 为教师 |

> DQN / PPO / HER 均以 Q-learning 作为"教师"预先提供全局最优 Q 值，再蒸馏到神经网络中，
> 因此训练速度快、质量高。自定义算法也可以调用 `TEACHER_TRAIN_ENTRY`（即 `train_q_learning_full_map`）获取教师数据。

---

## 7. 常见问题

**Q: 添加算法后下拉框没有出现新选项？**  
A: 检查 `TRAIN_ENTRYPOINTS` 中是否已添加新条目，并确认 Python 文件已保存、重启了程序。

**Q: 点击"训练"后进度条停在 0 不动？**  
A: 确认 `train_xxx_full_map` 函数内部在适当时刻调用了 `progress_callback(value)`（value 为 0–100 的整数）。

**Q: 导航开始后无人机不移动？**  
A: `build_xxx_path` 返回的路径点数必须 ≥ 2。若返回空列表，系统认为规划失败。

**Q: 如何在算法中访问教师策略（Q-learning 全局最优解）？**  
A: 导入并调用 `TEACHER_TRAIN_ENTRY`：
```python
from algorithm.algorithm_entrypoints import TEACHER_TRAIN_ENTRY

ok, teacher_model, *_ = TEACHER_TRAIN_ENTRY(
    occupancy=occupancy, heights=heights,
    end_map_xyz=end_map_xyz, map_to_index=map_to_index,
    progress_callback=progress_callback,
)
```

**Q: 算法需要持久化（保存/加载模型）怎么办？**  
A: `train_xxx_full_map` 返回的 `model` 对象在当前会话内一直保存在内存中。
如需跨会话持久化，可在算法文件内自行实现 `save_model` / `load_model`，并在
`build_xxx_path` 中先尝试加载缓存。

---

*Creator: Orange JYJ | License: Apache-2.0*
