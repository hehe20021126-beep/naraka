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

import argparse
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


@dataclass(frozen=True)
class MaterialConfig:
    """工程化参数（可按实测标定）。"""

    wave_speed: float = 2400.0      # 混凝土剪力墙导波群速度，常见量级 1800~3000 m/s
    attenuation_alpha: float = 0.85  # 距离衰减系数（经验参数）
    baseline_drift: float = 0.015     # 工况漂移（温度/湿度/载荷差异）


@dataclass(frozen=True)
class ExcitationConfig:
    fs: float = 2_000_000           # 采样率 2 MHz
    t_end: float = 2.2e-3           # 2.2 ms 记录长度
    center_freq: float = 50_000     # 50 kHz，工程常用 PZT 激励频段


@dataclass
class RealTimeMonitorConfig:
    """实时监测参数。"""

    alarm_threshold: float = 0.32       # 损伤告警阈值（0~1）
    smoothing_alpha: float = 0.25       # EWMA 平滑系数
    min_alarm_frames: int = 3           # 连续告警帧数阈值


def load_sensor_array_from_csv(csv_path: str) -> SensorArray2D:
    """从 CSV 读取传感器坐标，格式: x,y（单位 m）。"""
    positions = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("传感器 CSV 格式错误，应为两列: x,y")
    return SensorArray2D(positions=positions.astype(float))


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
    attenuation_alpha: float = 0.85,
    baseline_drift: float = 0.0,
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

        # 工况漂移：模拟环境变化导致的整体幅值偏移（工程中常见）
        if baseline_drift > 0:
            drift = baseline_drift * amp_dir * ricker_wavelet(t, f0=f0, t0=tau_dir + 8e-6)
            s = s + drift

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


class RealTimeDamageMonitor:
    """实时损伤监测器：逐帧输入 current_signals，输出健康分数与告警。"""

    def __init__(
        self,
        sensors: SensorArray2D,
        pairs: List[Tuple[int, int]],
        baseline_signals: Dict[Tuple[int, int], np.ndarray],
        wall_x: Tuple[float, float],
        wall_y: Tuple[float, float],
        config: RealTimeMonitorConfig | None = None,
    ) -> None:
        self.sensors = sensors
        self.pairs = pairs
        self.baseline_signals = baseline_signals
        self.wall_x = wall_x
        self.wall_y = wall_y
        self.cfg = config or RealTimeMonitorConfig()
        self.ewma_score = 0.0
        self.alarm_streak = 0

    def update(self, current_signals: Dict[Tuple[int, int], np.ndarray]) -> Dict[str, object]:
        xx, yy, pmap = rapid_damage_imaging(
            self.sensors,
            self.pairs,
            self.baseline_signals,
            current_signals,
            x_range=self.wall_x,
            y_range=self.wall_y,
            grid_nx=120,
            grid_ny=120,
            beta=1.04,
            sigma=0.16,
        )
        est_x, est_y, peak = estimate_damage_location(xx, yy, pmap)
        self.ewma_score = (1 - self.cfg.smoothing_alpha) * self.ewma_score + self.cfg.smoothing_alpha * peak

        if self.ewma_score >= self.cfg.alarm_threshold:
            self.alarm_streak += 1
        else:
            self.alarm_streak = 0

        alarm = self.alarm_streak >= self.cfg.min_alarm_frames
        return {
            "peak_score": float(peak),
            "ewma_score": float(self.ewma_score),
            "alarm": bool(alarm),
            "alarm_streak": int(self.alarm_streak),
            "estimated_xy": (float(est_x), float(est_y)),
            "probability_map": pmap,
        }


def estimate_damage_location(
    xx: np.ndarray,
    yy: np.ndarray,
    pmap: np.ndarray,
) -> Tuple[float, float, float]:
    """返回概率峰值点 (x,y,p)。"""
    idx = np.unravel_index(np.argmax(pmap), pmap.shape)
    return float(xx[idx]), float(yy[idx]), float(pmap[idx])


def build_dataset_for_test() -> Dict[str, np.ndarray]:
    """生成一组更贴近工程实际的测试数据（阵列可配置）。"""
    return build_dataset_for_test_with_positions("sensor_layout_example.csv")


def build_dataset_for_test_with_positions(sensor_csv: str) -> Dict[str, np.ndarray]:
    """使用外部可编辑传感器坐标生成数据。"""
    wall_x = (0.0, 3.0)
    wall_y = (0.0, 3.0)
    sensors = load_sensor_array_from_csv(sensor_csv)
    pairs = pair_index(sensors.n_sensors, ordered=False)

    material = MaterialConfig()
    ex = ExcitationConfig()
    t = np.arange(0.0, ex.t_end, 1 / ex.fs)
    true_damage = (1.85, 1.10)

    baseline_dict = simulate_signals(
        sensors, pairs, t,
        wave_speed=material.wave_speed,
        damage_xy=None,
        f0=ex.center_freq,
        noise_std=0.01,
        attenuation_alpha=material.attenuation_alpha,
    )
    current_dict = simulate_signals(
        sensors, pairs, t,
        wave_speed=material.wave_speed,
        damage_xy=true_damage,
        damage_strength=0.42,
        f0=ex.center_freq,
        noise_std=0.01,
        attenuation_alpha=material.attenuation_alpha,
        baseline_drift=material.baseline_drift,
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


def run_realtime_monitor_demo(sensor_csv: str) -> None:
    """实时监测演示：模拟损伤从无到有，连续帧触发告警。"""
    data = build_dataset_for_test_with_positions(sensor_csv)
    sensors = SensorArray2D(positions=data["sensor_positions"])
    pair_arr = data["pairs"]
    pairs = [tuple(p) for p in pair_arr]
    baseline = {tuple(p): s for p, s in zip(pair_arr, data["baseline"])}
    wall_x = tuple(data["wall_x"])
    wall_y = tuple(data["wall_y"])
    t = data["time"]
    true_damage = tuple(data["true_damage"])

    monitor = RealTimeDamageMonitor(
        sensors=sensors,
        pairs=pairs,
        baseline_signals=baseline,
        wall_x=wall_x,
        wall_y=wall_y,
        config=RealTimeMonitorConfig(alarm_threshold=0.30, smoothing_alpha=0.30, min_alarm_frames=3),
    )

    material = MaterialConfig()
    ex = ExcitationConfig()
    # 前 6 帧近似无损伤，后 9 帧损伤逐步增强
    strengths = [0.00, 0.00, 0.02, 0.01, 0.00, 0.03, 0.10, 0.15, 0.20, 0.26, 0.32, 0.38, 0.42, 0.46, 0.50]
    print("frame,damage_strength,peak_score,ewma_score,alarm,estimated_x,estimated_y")
    for k, ds in enumerate(strengths, start=1):
        current = simulate_signals(
            sensors, pairs, t,
            wave_speed=material.wave_speed,
            damage_xy=true_damage if ds > 0 else None,
            damage_strength=ds,
            f0=ex.center_freq,
            noise_std=0.01,
            attenuation_alpha=material.attenuation_alpha,
            baseline_drift=material.baseline_drift,
        )
        result = monitor.update(current)
        est_x, est_y = result["estimated_xy"]
        print(
            f"{k},{ds:.2f},{result['peak_score']:.3f},{result['ewma_score']:.3f},"
            f"{int(result['alarm'])},{est_x:.3f},{est_y:.3f}"
        )


def run_realtime_plot_demo(sensor_csv: str, interval_sec: float = 0.35) -> None:
    """实时图像演示：每帧更新概率热力图（不是单张静态图）。"""
    data = build_dataset_for_test_with_positions(sensor_csv)
    sensors = SensorArray2D(positions=data["sensor_positions"])
    pair_arr = data["pairs"]
    pairs = [tuple(p) for p in pair_arr]
    baseline = {tuple(p): s for p, s in zip(pair_arr, data["baseline"])}
    wall_x = tuple(data["wall_x"])
    wall_y = tuple(data["wall_y"])
    t = data["time"]
    true_damage = tuple(data["true_damage"])

    monitor = RealTimeDamageMonitor(
        sensors=sensors,
        pairs=pairs,
        baseline_signals=baseline,
        wall_x=wall_x,
        wall_y=wall_y,
        config=RealTimeMonitorConfig(alarm_threshold=0.30, smoothing_alpha=0.30, min_alarm_frames=3),
    )

    material = MaterialConfig()
    ex = ExcitationConfig()
    strengths = [0.00, 0.00, 0.02, 0.01, 0.00, 0.03, 0.10, 0.15, 0.20, 0.26, 0.32, 0.38, 0.42, 0.46, 0.50]

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 6))
    heat = ax.imshow(
        np.zeros((120, 120)),
        origin="lower",
        extent=[wall_x[0], wall_x[1], wall_y[0], wall_y[1]],
        cmap="hot",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    cbar = plt.colorbar(heat, ax=ax, label="Damage Probability")
    _ = cbar
    spos = sensors.positions
    ax.scatter(spos[:, 0], spos[:, 1], c="cyan", s=35, edgecolors="k", label="PZT Sensors")
    ax.scatter(*true_damage, c="lime", s=110, marker="*", edgecolors="k", label="True Damage")
    est_scatter = ax.scatter([], [], c="blue", s=80, marker="x", label="Estimated Peak")
    title = ax.set_title("Real-time Damage Imaging")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right")

    for k, ds in enumerate(strengths, start=1):
        current = simulate_signals(
            sensors, pairs, t,
            wave_speed=material.wave_speed,
            damage_xy=true_damage if ds > 0 else None,
            damage_strength=ds,
            f0=ex.center_freq,
            noise_std=0.01,
            attenuation_alpha=material.attenuation_alpha,
            baseline_drift=material.baseline_drift,
        )
        result = monitor.update(current)
        pmap = result["probability_map"]
        est_x, est_y = result["estimated_xy"]
        heat.set_data(pmap)
        est_scatter.set_offsets(np.array([[est_x, est_y]]))
        title.set_text(
            f"Real-time Damage Imaging | frame={k} | EWMA={result['ewma_score']:.3f} | alarm={int(result['alarm'])}"
        )
        fig.canvas.draw_idle()
        plt.pause(interval_sec)
    plt.ioff()
    plt.show()


def demo(
    show_plot: bool = True,
    save_path: str | None = None,
    sensor_csv: str = "sensor_layout_example.csv",
) -> None:
    # 1) 可编辑二维阵列测试数据（默认读取 sensor_layout_example.csv）
    data = build_dataset_for_test_with_positions(sensor_csv)
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

    if save_path:
        plt.savefig(save_path, dpi=180)
        print(f"已保存概率图到: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PZT 剪力墙损伤概率成像演示")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不弹出图窗（适合服务器/无桌面环境）",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="将概率图保存到指定路径，例如 result.png",
    )
    parser.add_argument(
        "--sensor-csv",
        type=str,
        default="sensor_layout_example.csv",
        help="传感器坐标 CSV（列名 x,y；单位 m）",
    )
    parser.add_argument(
        "--realtime-demo",
        action="store_true",
        help="运行实时监测演示（逐帧输出损伤分数与告警）",
    )
    parser.add_argument(
        "--realtime-plot",
        action="store_true",
        help="运行实时热力图演示（逐帧更新图像）",
    )
    args = parser.parse_args()
    if args.realtime_plot:
        run_realtime_plot_demo(sensor_csv=args.sensor_csv)
    elif args.realtime_demo:
        run_realtime_monitor_demo(sensor_csv=args.sensor_csv)
    else:
        demo(show_plot=not args.no_show, save_path=args.save_path, sensor_csv=args.sensor_csv)
