# Reinforcement Teach System

基于强化学习与 A* 的 3D 地形导航演示系统，支持地图编辑、地形导入导出、算法训练与路径对比。

## 功能概览

- 3D 可视化地形与路径导航（Qt + OpenGL）
- 算法支持：A*、Q-learning、DQN、PPO、HER
- 地形编辑：立方体/圆柱体编辑、撤销、随机障碍
- 地形数据：HDF5 存储、CSV 导入导出
- 算法统一入口注册，便于切换与扩展

## 代码结构（按职责分层）

```text
.
├─ main.py                          # 程序启动入口（含 .venv 自动重启逻辑）
├─ GUI.py                           # 开始界面（Start / Exit）
├─ map/                             # 地图与UI相关模块
│  ├─ map_qt_3d_view.py             # Qt 主窗口与控制面板
│  ├─ map_qt_opengl_widget.py       # OpenGL 渲染与交互逻辑
│  ├─ map_3d_engine_vpython.py      # VPython 3D 驱动（可选）
│  ├─ map_terrain_edit.py           # 地形编辑 + CSV/HDF5 读写辅助
│  ├─ map_terrain_io.py             # 演示地形生成与HDF5地形存储
│  ├─ map_flat.h5                   # 内置平坦地形
│  ├─ map_mountain.h5               # 内置山地地形
│  └─ saved_terrains/               # 运行中保存的编辑地形
├─ algorithm/                       # 算法核心与统一入口
│  ├─ algorithm_a_star.py           # A* 路径规划
│  ├─ algorithm_q_learning.py       # Q-learning 训练/路径生成
│  ├─ algorithm_dqn.py              # DQN 训练/路径生成
│  ├─ algorithm_ppo.py              # PPO 训练/路径生成
│  ├─ algorithm_her.py              # HER 训练/路径生成
│  └─ algorithm_entrypoints.py      # 算法入口注册表
├─ requirements.txt                 # Python 依赖
├─ ALGORITHM_ENTRYPOINTS.md         # 算法入口说明
├─ MAP_MODULE_LAYOUT.md             # 地图模块结构说明
├─ TERRAIN_CSV_FORMAT.md            # CSV 地形格式规范
└─ README.md
```

## 关键模块说明

### 1. 启动与GUI

- `main.py`
	- 自动检测本地 `.venv` 并重启到对应解释器
	- 启动前确保演示地图存在（通过 `map/map_terrain_io.py`）
	- 进入 `GUI.py`
- `GUI.py`
	- 提供程序开始窗口
	- 点击 `Start` 进入 3D 主界面

### 2. 地图与显示（map）

- `map/map_qt_3d_view.py`
	- 管理算法选择、训练、导航、地形切换、编辑控制
	- 维护地形设置与编辑地形持久化逻辑
- `map/map_qt_opengl_widget.py`
	- OpenGL 绘制、鼠标交互、路径动画
	- 调用算法模块生成路径
- `map/map_terrain_edit.py`
	- 地形编辑数据处理
	- CSV 导入导出、HDF5 读取/保存辅助
- `map/map_terrain_io.py`
	- 生成演示用 `map_flat.h5` 与 `map_mountain.h5`

### 3. 算法核心（algorithm）

- `algorithm/algorithm_entrypoints.py` 为统一入口
	- 训练入口：Q-learning / DQN / PPO / HER
	- 路径入口：Q-learning / DQN / PPO / HER
	- 教师模型入口：Q-learning（供 DQN/PPO/HER 使用）

## 运行方式

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动程序

```bash
python main.py
```

说明：在 Windows 下，如果项目存在 `.venv`，`main.py` 会自动尝试切换到该环境运行。

## 常见说明

- Flat/Mountain 切换时，若存在 `map/saved_terrains/*_edited.h5`，会优先加载编辑后的地形。
- 若希望恢复内置原始地形，可在界面点击 `Clear Terrain Setting`，或删除 `map/saved_terrains/` 下对应文件。
