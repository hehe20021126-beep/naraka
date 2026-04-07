"""
剪力墙压电传感器损伤概率成像示例

功能:
1) 构建二维传感器阵列（发射-接收对）
2) 基于导波信号的“基准-当前”差异计算损伤指数
3) 使用椭圆敏感区叠加形成概率成像图（RAPID 风格）
4) 提供合成数据演示，可验证损伤位置定位效果

说明:
- 若你已有真实信号，可直接调用 `rapid_damage_imaging(...)`
- 代码默认假设同质介质并使用常数波速，工程中可结合实测标定修正
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SensorArray2D:
    """二维传感器阵列定义。"""

    positions: np.ndarray  # shape=(N,2), 每个传感器坐标 [x,y]

    @property
    def n_sensors(self) -> int:
        return self.positions.shape[0]

    @staticmethod
    def rectangular_grid(
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        nx: int,
        ny: int,
    ) -> "SensorArray2D":
        """在边界内生成规则二维阵列。"""
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        return SensorArray2D(positions=pts)


def fixed_sensor_array() -> SensorArray2D:
    """固定二维阵列（6x6，共 36 个传感器）。

    说明:
    - 若现场已确定了固定阵列，直接替换 `positions` 即可。
    - 此处给出一组可复现实验的默认固定坐标（均匀但带边界留量）。
    """
    xs = np.array([0.20, 0.72, 1.24, 1.76, 2.28, 2.80], dtype=float)
    ys = np.array([0.20, 0.72, 1.24, 1.76, 2.28, 2.80], dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    return SensorArray2D(positions=positions)


def ricker_wavelet(t: np.ndarray, f0: float, t0: float) -> np.ndarray:
    """Ricker 子波，常用于超声/导波模拟。"""
    x = np.pi * f0 * (t - t0)
    return (1 - 2 * x**2) * np.exp(-x**2)


def pair_index(n: int, ordered: bool = False) -> List[Tuple[int, int]]:
    """生成传感器对索引。

    ordered=False: 仅 i<j 的无序对（互易性场景）
    ordered=True:  所有 i!=j 的有序对（发射接收区分）
    """
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if not ordered and i > j:
                continue
            pairs.append((i, j))
    return pairs


def simulate_signals(
    sensors: SensorArray2D,
    pairs: Iterable[Tuple[int, int]],
    t: np.ndarray,
    wave_speed: float,
    damage_xy: Tuple[float, float] | None = None,
    damage_strength: float = 0.25,
    f0: float = 70_000,
    noise_std: float = 0.01,
    attenuation_alpha: float = 1.2,
) -> Dict[Tuple[int, int], np.ndarray]:
    """生成每个通道的合成导波信号。

    baseline: 仅直达波
    current:  直达波 + 损伤散射波（若 damage_xy 不为空）
    """
    rng = np.random.default_rng(42)
    sigs: Dict[Tuple[int, int], np.ndarray] = {}

    for i, j in pairs:
        pi = sensors.positions[i]
        pj = sensors.positions[j]
        dij = np.linalg.norm(pi - pj)

        # 直达波到达时刻
        tau_dir = dij / wave_speed
        amp_dir = 1.0 / (1.0 + attenuation_alpha * dij)
        s = amp_dir * ricker_wavelet(t, f0=f0, t0=tau_dir)

        # 损伤散射波
        if damage_xy is not None:
            d = np.asarray(damage_xy, dtype=float)
            did = np.linalg.norm(pi - d)
            djd = np.linalg.norm(d - pj)
            tau_scat = (did + djd) / wave_speed
            amp_scat = damage_strength / (1.0 + attenuation_alpha * (did + djd))
            s = s + amp_scat * ricker_wavelet(t, f0=f0, t0=tau_scat)

        # 噪声
        s = s + rng.normal(0.0, noise_std, size=t.shape)
        sigs[(i, j)] = s

    return sigs


def normalized_signal_difference(
    baseline: np.ndarray,
    current: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """通道损伤指数 DI:

    DI = ||current - baseline||_2 / ||baseline||_2
    """
    return float(np.linalg.norm(current - baseline) / (np.linalg.norm(baseline) + eps))


def rapid_damage_imaging(
    sensors: SensorArray2D,
    pairs: Iterable[Tuple[int, int]],
    baseline_signals: Dict[Tuple[int, int], np.ndarray],
    current_signals: Dict[Tuple[int, int], np.ndarray],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    grid_nx: int = 120,
    grid_ny: int = 120,
    beta: float = 1.03,
    sigma: float = 0.18,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RAPID 类概率成像。

    参数:
    - beta: 椭圆阈值系数（>=1），越大敏感区越宽
    - sigma: 椭圆外扩到概率权重的指数衰减宽度
    """
    xs = np.linspace(*x_range, grid_nx)
    ys = np.linspace(*y_range, grid_ny)
    xx, yy = np.meshgrid(xs, ys)

    pmap = np.zeros_like(xx, dtype=float)

    for (i, j) in pairs:
        b = baseline_signals[(i, j)]
        c = current_signals[(i, j)]
        di = normalized_signal_difference(b, c)

        si = sensors.positions[i]
        sj = sensors.positions[j]

        d1 = np.sqrt((xx - si[0]) ** 2 + (yy - si[1]) ** 2)
        d2 = np.sqrt((xx - sj[0]) ** 2 + (yy - sj[1]) ** 2)
        dij = np.linalg.norm(si - sj)

        # 椭圆归一化距离比值 r<=beta 时认为位于该通道敏感区域
        r = (d1 + d2) / (dij + 1e-12)
        w = np.exp(-((r - 1.0) / sigma) ** 2)
        w[r > beta] = 0.0

        pmap += di * w

    # 归一化为 [0,1]
    pmap -= pmap.min()
    pmax = pmap.max()
    if pmax > 0:
        pmap /= pmax

    return xx, yy, pmap


def estimate_damage_location(
    xx: np.ndarray,
    yy: np.ndarray,
    pmap: np.ndarray,
) -> Tuple[float, float, float]:
    """返回概率峰值点 (x,y,p)。"""
    idx = np.unravel_index(np.argmax(pmap), pmap.shape)
    return float(xx[idx]), float(yy[idx]), float(pmap[idx])


def build_dataset_for_test() -> Dict[str, np.ndarray]:
    """生成一组可直接测试的数据（固定阵列 + 基准/当前信号矩阵）。"""
    wall_x = (0.0, 3.0)
    wall_y = (0.0, 3.0)
    sensors = fixed_sensor_array()
    pairs = pair_index(sensors.n_sensors, ordered=False)

    fs = 2_000_000
    t_end = 2.2e-3
    t = np.arange(0.0, t_end, 1 / fs)
    c_gw = 2300.0
    true_damage = (1.85, 1.10)

    baseline_dict = simulate_signals(
        sensors, pairs, t,
        wave_speed=c_gw,
        damage_xy=None,
        noise_std=0.01,
    )
    current_dict = simulate_signals(
        sensors, pairs, t,
        wave_speed=c_gw,
        damage_xy=true_damage,
        damage_strength=0.42,
        noise_std=0.01,
    )

    pair_arr = np.asarray(pairs, dtype=int)
    baseline = np.vstack([baseline_dict[tuple(p)] for p in pair_arr])
    current = np.vstack([current_dict[tuple(p)] for p in pair_arr])

    return {
        "wall_x": np.asarray(wall_x),
        "wall_y": np.asarray(wall_y),
        "sensor_positions": sensors.positions,
        "pairs": pair_arr,
        "time": t,
        "baseline": baseline,
        "current": current,
        "true_damage": np.asarray(true_damage),
    }


def demo() -> None:
    # 1) 固定二维阵列测试数据（36 传感器，630 通道）
    data = build_dataset_for_test()
    wall_x = tuple(data["wall_x"])
    wall_y = tuple(data["wall_y"])
    sensors = SensorArray2D(positions=data["sensor_positions"])
    pair_arr = data["pairs"]
    pairs = [tuple(p) for p in pair_arr]
    baseline = {tuple(p): s for p, s in zip(pair_arr, data["baseline"])}
    current = {tuple(p): s for p, s in zip(pair_arr, data["current"])}
    true_damage = tuple(data["true_damage"])

    # 2) 概率成像
    xx, yy, pmap = rapid_damage_imaging(
        sensors,
        pairs,
        baseline,
        current,
        x_range=wall_x,
        y_range=wall_y,
        grid_nx=180,
        grid_ny=180,
        beta=1.04,
        sigma=0.16,
    )

    est_x, est_y, peak = estimate_damage_location(xx, yy, pmap)
    print(f"传感器数量: {sensors.n_sensors}, 通道数量: {len(pairs)}")
    print(f"数据维度: baseline={data['baseline'].shape}, current={data['current'].shape}")
    print(f"真实损伤位置: ({true_damage[0]:.3f}, {true_damage[1]:.3f}) m")
    print(f"估计损伤位置: ({est_x:.3f}, {est_y:.3f}) m, 峰值概率={peak:.3f}")
    print("测试数据预览(第1通道前10个采样点):")
    print("baseline[0,:10] =", np.array2string(data["baseline"][0, :10], precision=5))
    print("current[0,:10]  =", np.array2string(data["current"][0, :10], precision=5))

    # 3) 可视化
    fig, ax = plt.subplots(figsize=(7, 6))
    hm = ax.contourf(xx, yy, pmap, levels=35, cmap="hot")
    plt.colorbar(hm, ax=ax, label="Damage Probability")

    spos = sensors.positions
    ax.scatter(spos[:, 0], spos[:, 1], c="cyan", s=45, edgecolors="k", label="PZT Sensors")
    ax.scatter(*true_damage, c="lime", s=110, marker="*", edgecolors="k", label="True Damage")
    ax.scatter(est_x, est_y, c="blue", s=80, marker="x", label="Estimated Peak")

    ax.set_xlim(*wall_x)
    ax.set_ylim(*wall_y)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Shear Wall Damage Probability Imaging (PZT-based)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo()
