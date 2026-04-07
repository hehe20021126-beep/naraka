# 运行说明（Python）

## 1) 创建并激活虚拟环境
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2) 安装依赖
```bash
pip install -U pip
pip install -r requirements.txt
```

## 3) 直接运行（会弹出概率图窗口）
```bash
python shear_wall_pzt_damage_imaging.py
```

> 默认会读取 `sensor_layout_example.csv`（可直接修改坐标）。

## 4) 无图形界面环境运行（推荐服务器）
```bash
python shear_wall_pzt_damage_imaging.py --no-show --save-path damage_map.png
```

## 5) 使用你自己的传感器坐标
传感器坐标文件为 CSV，两列 `x,y`（单位 m），例如：
```csv
x,y
0.2,0.2
0.7,0.2
...
```

运行：
```bash
python shear_wall_pzt_damage_imaging.py --sensor-csv your_layout.csv --no-show --save-path your_map.png
```

运行后会打印：
- 传感器数量与通道数量
- baseline/current 数据维度
- 真实损伤与估计损伤位置
- 第 1 通道前 10 个采样点（用于快速检查数据）
