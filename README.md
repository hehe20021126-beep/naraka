# 永劫无间刀房实战建议（画面识别 Demo）

这是一个本地实时画面识别示例：
- 读取屏幕/视频帧。
- 检测敌我角色与武器状态（示例中使用占位检测器，可替换为 YOLOv8）。
- 根据距离、敌人朝向、你自身血量等状态输出实战建议。

> 说明：游戏识别模型需要你自行采集并标注数据后训练。本仓库提供可落地的代码骨架与建议策略引擎。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --source camera --show
```


## 在 PyCharm 中部署与运行

1. **打开项目**
   - 启动 PyCharm，选择 `Open`，打开仓库根目录 `naraka`。

2. **配置 Python 解释器（虚拟环境）**
   - `File -> Settings -> Project -> Python Interpreter`。
   - 点击齿轮 `Add Interpreter`，选择 `Virtualenv`。
   - 可选：
     - **新建环境**：位置设为项目内 `.venv`。
     - **已有环境**：选择已经创建好的 `.venv/bin/python`（Windows 是 `.venv\Scripts\python.exe`）。

3. **安装依赖**
   - 在 PyCharm 的 Terminal 执行：

```bash
pip install -r requirements.txt
```

4. **创建运行配置（Run Configuration）**
   - 右上角 `Add Configuration...` -> `Python`。
   - `Module name` 填：`src.main`（推荐用模块方式，避免相对导入问题）。
   - `Working directory` 设为项目根目录。
   - `Parameters` 示例：
     - 摄像头：`--source camera --show`
     - 屏幕：`--source screen --show`
     - 视频：`--source video --video_path your.mp4 --show`

5. **运行与调试**
   - 点击 `Run` 直接运行；点击 `Debug` 可在 `src/main.py`、`src/detector.py`、`src/advisor.py` 打断点调试。

6. **常见问题**
   - 若出现 `No module named src`：确认运行配置用的是 **Module name=src.main**，且工作目录是项目根目录。
   - 若摄像头打不开：检查系统权限，关闭占用摄像头的软件（如会议软件）。
   - 若屏幕捕获失败（`mss`）：尝试以管理员身份运行 PyCharm，或改用 `--source video` 先验证流程。

## 输入源

- `--source camera`：摄像头
- `--source screen`：屏幕捕获（默认主屏）
- `--source video --video_path your.mp4`：本地视频

## 主要模块

- `src/detector.py`：目标检测器接口和占位实现。
- `src/advisor.py`：刀房建议策略（近战压制 / 拉扯骗振 / 蓄力对拼等）。
- `src/main.py`：主流程，融合识别结果并渲染建议文本。

## 替换为真实模型（建议）

1. 采集刀房对局视频，按帧标注：
   - 角色框（self/enemy）
   - 武器状态（平击/蓄力/振刀动作）
   - 位移状态（闪避、钩锁）
2. 使用 YOLOv8 训练。
3. 在 `detector.py` 的 `DummyDetector` 位置替换为 `YOLODetector`。
4. 若有 OCR 血量区域，可在 `main.py` 增加血量读取逻辑并写入 `FrameState.self_hp`。

## 免责声明

本项目仅用于学习计算机视觉与策略分析，不保证在所有对局场景准确。
