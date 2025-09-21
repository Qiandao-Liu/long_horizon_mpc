# /src/demo_processing/segment_rollout.py
import os, re, json, math, pickle, argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/demo_processing/segment_rollout.py
SRC_DIR      = PROJECT_ROOT / "src"
PHYSTWIN_DIR = PROJECT_ROOT / "third_party" / "PhysTwinFork"
DATA_DIR     = PROJECT_ROOT / "data"

"""
python -m src.demo_processing.segment_rollout
"""

# 可选：图像调试
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# -------------------- 基础工具 --------------------

def _ensure_task_dirs(task_dir: Path):
    (task_dir / "demo_rollout").mkdir(parents=True, exist_ok=True)
    (task_dir / "mpc_rollout").mkdir(parents=True, exist_ok=True)
    (task_dir / "segmented_status").mkdir(parents=True, exist_ok=True)

def _get_task_dir(tasks_root: Path, task_name: Optional[str]) -> Path:
    if task_name is None:
        ts = sorted([p for p in tasks_root.glob("task*") if p.is_dir()],
                    key=lambda p: int(re.findall(r"\d+", p.name)[0]))
        assert ts, f"No task* under {tasks_root}"
        return ts[-1]
    tdir = tasks_root / task_name
    assert tdir.exists(), f"Task dir not found: {tdir}"
    return tdir

def _load_latest_demo_pkl(task_dir: Path) -> Dict:
    demo_dir = task_dir / "demo_rollout"
    assert demo_dir.exists(), f"Not found: {demo_dir}"
    cand = sorted(demo_dir.glob("demo*.pkl"))
    assert cand, f"No demo*.pkl in {demo_dir}"
    latest = cand[-1]
    with open(latest, "rb") as f:
        data = pickle.load(f)
    print(f"[LOAD] {latest.name} | frames={len(data['ctrl_seq'])}")
    return data

def _unwrap_angle(a: np.ndarray) -> np.ndarray:
    return np.unwrap(a)

def _angle_between(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    nu = np.linalg.norm(u) + eps
    nv = np.linalg.norm(v) + eps
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.arccos(c))

def _percentile(vals: np.ndarray, p: float) -> float:
    return float(np.percentile(vals, p))

def _vector_xy_angle(vecs: np.ndarray) -> np.ndarray:
    """vecs: [T,3] -> angle in xy-plane, radians"""
    return np.arctan2(vecs[:, 1], vecs[:, 0])

def _ema(x: np.ndarray, beta: float = 0.7) -> np.ndarray:
    # 逐列 EMA
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, x.shape[0]):
        y[i] = beta * y[i-1] + (1 - beta) * x[i]
    return y.squeeze()

# -------------------- 分组与参考点 --------------------
# ------------- NEW: 左右手分组（DBSCAN + 兜底） -------------
def split_ctrl_pts_dbscan(ctrl_pts: np.ndarray,
                        eps: float = 0.05,
                        min_samples: int = 3,
                        fallback_mode: str = "first_half") -> Tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.cluster import DBSCAN
    except Exception:
        DBSCAN = None

    C = ctrl_pts.shape[0]
    if DBSCAN is None:
        return _split_left_right(C, mode=("first_half" if fallback_mode=="first_half" else "odd_even"))

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(ctrl_pts)
    labels = clustering.labels_

    # 收集非噪声簇（DBSCAN 中噪声为 -1）
    uniq = sorted([l for l in np.unique(labels) if l != -1])
    if len(uniq) != 2:
        return _split_left_right(C, mode=("first_half" if fallback_mode=="first_half" else "odd_even"))

    clusters = {i: np.where(labels == i)[0] for i in uniq}
    # 按 x 坐标均值决定左右
    c0, c1 = clusters[uniq[0]], clusters[uniq[1]]
    x0 = ctrl_pts[c0, 0].mean()
    x1 = ctrl_pts[c1, 0].mean()
    if x0 < x1:
        left_idx, right_idx = c0, c1
    else:
        left_idx, right_idx = c1, c0

    # 健壮性：空簇兜底
    if len(left_idx) == 0 or len(right_idx) == 0:
        return _split_left_right(C, mode=("first_half" if fallback_mode=="first_half" else "odd_even"))

    return left_idx.astype(int), right_idx.astype(int)

def _split_left_right(C: int, mode: str = "first_half") -> Tuple[np.ndarray, np.ndarray]:
    print("CAN NOT SPLIT HANDS")

def _pick_anchor_indices(ctrl_seq: np.ndarray,
                         left_idx: np.ndarray,
                         right_idx: np.ndarray,
                         anchor_idx_left: Optional[int],
                         anchor_idx_right: Optional[int]) -> Tuple[int, int]:
    """
    参考点（用于“手腕旋转”方向向量）：默认用 t=0 时该手里离质心最远的控制点；
    也可通过参数显式指定 index。
    """
    C = ctrl_seq.shape[1]
    if anchor_idx_left is not None:
        aL = int(anchor_idx_left)
    else:
        c0 = ctrl_seq[0, left_idx]       # [|L|,3]
        centroid = c0.mean(axis=0)
        d = np.linalg.norm(c0 - centroid, axis=1)
        aL = int(left_idx[np.argmax(d)])

    if anchor_idx_right is not None:
        aR = int(anchor_idx_right)
    else:
        c0 = ctrl_seq[0, right_idx]
        centroid = c0.mean(axis=0)
        d = np.linalg.norm(c0 - centroid, axis=1)
        aR = int(right_idx[np.argmax(d)])

    return aL, aR

# -------------------- 里程碑检测（通用） --------------------

def detect_milestones_generic(
    demo: Dict,
    left_idx: Optional[np.ndarray] = None,
    right_idx: Optional[np.ndarray] = None,
    left_right_split: str = "first_half",
    window: int = 20,                       # W: 最近 W 帧窗口
    turn_deg_thr: float = 40.0,             # 折线弯折角阈值（度）
    min_path_percentile: float = 60.0,      # 折线位移阈值（相对于全局速度分位数）
    min_gap_seconds: float = 0.3,           # 相邻 milestone 最小间隔（秒）
    ema_beta: float = 0.6,                  # EMA 平滑
    # 手腕/朝向旋转检测
    enable_wrist: bool = True,
    wrist_deg_thr: float = 90.0,            # 窗口累计朝向旋转阈值（度）
    wrist_quiet_ratio: float = 0.5,         # “旋转停止”定义：窗口后半段角速度 < 前半段的多少比例
    wrist_hold_frames: int = 8,             # 旋转检测窗口长度（帧）
    anchor_idx_left: Optional[int] = None,  # 可选：左手参考点索引
    anchor_idx_right: Optional[int] = None, # 可选：右手参考点索引
) -> List[int]:
    """
    通用策略：
    - 维护最近 W 帧的两只手的质心轨迹（两条折线）；若折线“明显折叠”（前半段方向 vs 后半段方向夹角 > 阈值，
      且该窗口内累计位移足够），触发 turning-point 里程碑。
    - 并行检测“质心→参考点”的朝向在窗口内累计旋转是否超过阈值；若超过，且**旋转速度显著下降**
      将该帧视为“剧烈旋转停止”的里程碑。
    """
    ctrl_seq = np.asarray(demo["ctrl_seq"])  # [T,C,3]
    T, C, _ = ctrl_seq.shape
    fps = float(demo.get("meta", {}).get("fps", 30.0))
    min_gap = max(1, int(min_gap_seconds * fps))

    # 分组
    if left_idx is None or right_idx is None:
        left_idx, right_idx = _split_left_right(C, mode=left_right_split)

    # 预计算每帧两手的质心
    cent_L = ctrl_seq[:, left_idx].mean(axis=1)   # [T,3]
    cent_R = ctrl_seq[:, right_idx].mean(axis=1)  # [T,3]

    # 各手速度（用于本手的“有足够移动量”阈值）
    vel_L = np.vstack([np.zeros((1,3)), cent_L[1:] - cent_L[:-1]])
    vel_R = np.vstack([np.zeros((1,3)), cent_R[1:] - cent_R[:-1]])
    speed_L = np.linalg.norm(vel_L, axis=1)
    speed_R = np.linalg.norm(vel_R, axis=1)

    # 每手自己的窗口累计位移阈值（分位数 * window）
    path_thr_L = max(1e-9, _percentile(speed_L, min_path_percentile)) * window
    path_thr_R = max(1e-9, _percentile(speed_R, min_path_percentile)) * window

    # 可选：再给一个绝对下限，防“静帧”扰动（按你数据尺度可调）
    ABS_PATH_EPS = 1e-6
    path_thr_L = max(path_thr_L, ABS_PATH_EPS)
    path_thr_R = max(path_thr_R, ABS_PATH_EPS)

    # 参考点（手腕朝向）
    aL, aR = _pick_anchor_indices(ctrl_seq, left_idx, right_idx, anchor_idx_left, anchor_idx_right)

    milestones: List[int] = [0]  # 起点

    def window_vectors(track: np.ndarray, t: int, W: int) -> Optional[np.ndarray]:
        if t < W: return None
        return track[t-W+1:t+1]  # [W,3]

    def bend_angle(line: np.ndarray) -> Tuple[float, float, float]:
        W = line.shape[0]
        mid = W // 2
        v1 = line[mid-1] - line[0]
        v2 = line[-1]    - line[mid-1]
        ang = _angle_between(v1, v2)
        path = np.sum(np.linalg.norm(line[1:] - line[:-1], axis=1))
        # 半程弦长：max(|v1|, |v2|)，衡量“半段的净位移”
        chord = max(float(np.linalg.norm(v1)), float(np.linalg.norm(v2)))
        return ang, path, chord

    def wrist_rotation_metrics(cent: np.ndarray, ref: np.ndarray, t: int, hold: int) -> Optional[Tuple[float, float]]:
        if t < hold or not enable_wrist:
            return None
        seg_c = cent[t-hold+1:t+1]
        seg_r = ref[t-hold+1:t+1]
        vec = seg_r - seg_c                               # [hold,3], 质心->参考点
        th  = _unwrap_angle(_vector_xy_angle(vec))        # 水平朝向变化
        dth = np.abs(np.diff(th, prepend=th[0]))          # 角速度(绝对)
        cum = float(np.sum(dth))                          # 累计旋转量（弧度）
        # “旋转停止”检测：后半段角速度均值显著小于前半段
        mid = hold // 2
        v1 = float(np.mean(dth[:mid])) + 1e-9
        v2 = float(np.mean(dth[mid:])) + 1e-9
        calm_ratio = v2 / v1
        return cum, calm_ratio

    # 参考点轨迹（取所选控制点）
    refL = ctrl_seq[:, aL]
    refR = ctrl_seq[:, aR]

    # 平滑（让阈值更稳健）
    cent_L_s = _ema(cent_L, beta=ema_beta)
    cent_R_s = _ema(cent_R, beta=ema_beta)
    refL_s   = _ema(refL,   beta=ema_beta)
    refR_s   = _ema(refR,   beta=ema_beta)

    # 角阈值（弧度）
    turn_thr = math.radians(turn_deg_thr)
    wrist_thr = math.radians(wrist_deg_thr)

    last_ms_L = -10**9
    last_ms_R = -10**9

    for t in range(T):
        # --- 1) 折线弯折检测（两手分别） ---
        fired = False
        for seg in ("L", "R"):
            track = cent_L_s if seg == "L" else cent_R_s
            line = window_vectors(track, t, window)
            if line is None:
                continue
            ang, path, chord = bend_angle(line)  # 注意 bend_angle 也要改，见下
            pth = path_thr_L if seg == "L" else path_thr_R

            # 强角度兜底：角度特别大时，放宽路径要求
            turn_deg_thr_strong = max(turn_deg_thr + 10.0, 45.0)
            strong_angle = ang >= math.radians(turn_deg_thr_strong)

            # 弦长兜底：前/后半段至少一段有“实打实”的位移（避免数值抖动）
            # 这里用“该手速度的 p50 * (window/2) * 0.8”作为本地最小弦长
            sp = speed_L if seg == "L" else speed_R
            local_chord_thr = max(1e-6, _percentile(sp, 50.0) * (window/2) * 0.8)

            trigger = ((ang >= turn_thr) and (path >= pth)) or \
                    (strong_angle and (chord >= local_chord_thr))

            last_seg = last_ms_L if seg == "L" else last_ms_R
            if trigger and (t - last_seg >= min_gap):
                milestones.append(t)
                if seg == "L": last_ms_L = t
                else:          last_ms_R = t
                fired = True
                break

        if fired:
            continue

        # --- 2) 手腕旋转停止检测（两手分别） ---
        if enable_wrist:
            for seg in ("L", "R"):
                cent = cent_L_s if seg == "L" else cent_R_s
                ref  = refL_s   if seg == "L" else refR_s
                met = wrist_rotation_metrics(cent, ref, t, wrist_hold_frames)
                if met is None: 
                    continue
                cum_rot, calm_ratio = met  # 弧度, 速度比
                if (cum_rot >= wrist_thr) and (calm_ratio <= wrist_quiet_ratio) and (t - last_ms >= min_gap):
                    milestones.append(t)
                    last_ms = t
                    break

    # 追加终点
    if milestones[-1] != (T-1):
        milestones.append(T-1)

    # 去重 & 排序 & 再次 enforce 最小间隔（保高质量帧）
    milestones = sorted(set(milestones))
    filtered = [milestones[0]]
    for t in milestones[1:]:
        if t - filtered[-1] >= min_gap:
            filtered.append(t)
        else:
            # 简单策略：保留较晚的（通常更接近真正拐点）
            filtered[-1] = max(filtered[-1], t)
    return filtered

# -------------------- 保存/可视化 --------------------

def save_milestones(task_dir: Path, milestones: List[int]):
    out = task_dir / "segmented_status" / "milestones.json"
    with open(out, "w") as f:
        json.dump({"milestones": milestones}, f, indent=2)
    print(f"[SAVE] {out}  -> {milestones}")

def plot_debug(task_dir: Path, demo: Dict, milestones: List[int],
               left_idx: np.ndarray, right_idx: np.ndarray):
    if not _HAS_MPL:
        print("[PLOT] matplotlib not found, skip.")
        return
    ctrl = np.asarray(demo["ctrl_seq"])
    T = ctrl.shape[0]
    cent_L = ctrl[:, left_idx].mean(axis=1)
    cent_R = ctrl[:, right_idx].mean(axis=1)
    vL = np.vstack([np.zeros((1,3)), cent_L[1:] - cent_L[:-1]])
    vR = np.vstack([np.zeros((1,3)), cent_R[1:] - cent_R[:-1]])
    spL = np.linalg.norm(vL, axis=1)
    spR = np.linalg.norm(vR, axis=1)
    thL = _unwrap_angle(_vector_xy_angle(vL + 1e-12))
    thR = _unwrap_angle(_vector_xy_angle(vR + 1e-12))
    xs = np.arange(T)

    out_png = task_dir / "segmented_status" / "milestones_debug.png"
    plt.figure(figsize=(12,7))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(xs, spL, label="speed L")
    ax1.plot(xs, spR, label="speed R", alpha=0.7)
    for t in milestones: ax1.axvline(t, color="k", alpha=0.25)
    ax1.set_ylabel("Speed")
    ax1.legend(loc="upper right")
    ax2 = plt.subplot(2,1,2)
    ax2.plot(xs, np.degrees(thL), label="theta_xy L")
    ax2.plot(xs, np.degrees(thR), label="theta_xy R", alpha=0.7)
    for t in milestones: ax2.axvline(t, color="k", alpha=0.25)
    ax2.set_ylabel("Heading (deg)")
    ax2.set_xlabel("Frame")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[PLOT] {out_png}")

# -------------------- MAIN --------------------

def main(
    tasks_root: str = "data/tasks",
    task_name: Optional[str] = None,
    cluster: str = "dbscan",
    left_right_split: str = "first_half",
    dbscan_eps: float = 0.05,
    dbscan_min_samples: int = 3,
    window: int = 20,
    turn_deg_thr: float = 30.0,
    min_path_percentile: float = 60.0,
    min_gap_seconds: float = 0.6,
    ema_beta: float = 0.6,
    enable_wrist: bool = True,
    wrist_deg_thr: float = 90.0,
    wrist_quiet_ratio: float = 0.5,
    wrist_hold_frames: int = 8,
    anchor_idx_left: Optional[int] = None,
    anchor_idx_right: Optional[int] = None,
    plot: bool = True,
):
    tasks_root = Path(tasks_root)
    task_dir = _get_task_dir(tasks_root, task_name)
    _ensure_task_dirs(task_dir)
    demo = _load_latest_demo_pkl(task_dir)

    ctrl0 = np.asarray(demo["ctrl_seq"][0])    # (C,3)
    C = ctrl0.shape[0]

    # 左右手分组
    if cluster == "dbscan":
        left_idx, right_idx = split_ctrl_pts_dbscan(
            ctrl0, eps=dbscan_eps, min_samples=dbscan_min_samples,
            fallback_mode=("first_half" if left_right_split=="first_half" else "odd_even")
        )
        print(f"[CLUSTER] DBSCAN -> left={len(left_idx)} right={len(right_idx)}")
        print(f"          left mean xyz = {ctrl0[left_idx].mean(axis=0)}")
        print(f"          right mean xyz= {ctrl0[right_idx].mean(axis=0)}")
    else:
        left_idx, right_idx = _split_left_right(C, mode=left_right_split)
        print(f"[CLUSTER] {left_right_split} -> left={len(left_idx)} right={len(right_idx)}")

    milestones = detect_milestones_generic(
        demo=demo,
        left_idx=left_idx,
        right_idx=right_idx,
        left_right_split=left_right_split,
        window=window,
        turn_deg_thr=turn_deg_thr,
        min_path_percentile=min_path_percentile,
        min_gap_seconds=min_gap_seconds,
        ema_beta=ema_beta,
        enable_wrist=enable_wrist,
        wrist_deg_thr=wrist_deg_thr,
        wrist_quiet_ratio=wrist_quiet_ratio,
        wrist_hold_frames=wrist_hold_frames,
        anchor_idx_left=anchor_idx_left,
        anchor_idx_right=anchor_idx_right,
    )

    save_milestones(task_dir, milestones)

    # === 新增：导出方向调试 CSV + 图 ===
    dump_direction_debug(task_dir, demo, left_idx, right_idx, ema_beta=ema_beta)

    if plot:
        plot_debug(task_dir, demo, milestones, left_idx, right_idx)


# ------------- NEW: 方向调试导出与可视化 -------------
import csv

def _compute_centroids(ctrl_seq: np.ndarray, left_idx: np.ndarray, right_idx: np.ndarray):
    cent_L = ctrl_seq[:, left_idx].mean(axis=1)   # [T,3]
    cent_R = ctrl_seq[:, right_idx].mean(axis=1)
    return cent_L, cent_R

def _vel_and_heading(cent: np.ndarray, ema_beta: float = 0.6):
    """
    给定 [T,3] 质心轨迹，返回：
    - v: 速度向量 [T,3] （首帧置0，其余为差分）
    - v_s: EMA 平滑后的速度 [T,3]
    - u: 单位方向 [T,3]（速度为0的地方置 0）
    - theta_xy_deg: XY 平面朝向角（度） [T]
    - speed: 速度模长 [T]
    """
    T = cent.shape[0]
    v = np.vstack([np.zeros((1,3)), cent[1:] - cent[:-1]])
    v_s = _ema(v, beta=ema_beta)
    speed = np.linalg.norm(v_s, axis=1)
    u = np.zeros_like(v_s)
    nz = speed > 1e-12
    u[nz] = v_s[nz] / speed[nz, None]
    theta_xy_deg = np.degrees(_unwrap_angle(_vector_xy_angle(v_s + 1e-12)))
    return v, v_s, u, theta_xy_deg, speed

def dump_direction_debug(task_dir: Path, demo: Dict,
                         left_idx: np.ndarray, right_idx: np.ndarray,
                         ema_beta: float = 0.6):
    out_dir = task_dir / "segmented_status"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "direction_debug.csv"

    ctrl_seq = np.asarray(demo["ctrl_seq"])
    T = ctrl_seq.shape[0]
    fps = float(demo.get("meta", {}).get("fps", 30.0))

    # 计算左右手质心、速度与朝向
    cent_L, cent_R = _compute_centroids(ctrl_seq, left_idx, right_idx)
    vL, vL_s, uL, thL_deg, spL = _vel_and_heading(cent_L, ema_beta=ema_beta)
    vR, vR_s, uR, thR_deg, spR = _vel_and_heading(cent_R, ema_beta=ema_beta)

    # 若已有 milestones.json，加载以便画图
    ms_path = out_dir / "milestones.json"
    milestones = []
    if ms_path.exists():
        try:
            milestones = json.loads(ms_path.read_text())["milestones"]
        except Exception:
            pass

    # 写 CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame","time_s",
            # left
            "centL_x","centL_y","centL_z",
            "vLx","vLy","vLz","vL_sx","vL_sy","vL_sz",
            "uLx","uLy","uLz","thetaL_xy_deg","speedL",
            # right
            "centR_x","centR_y","centR_z",
            "vRx","vRy","vRz","vR_sx","vR_sy","vR_sz",
            "uRx","uRy","uRz","thetaR_xy_deg","speedR",
            "is_milestone"
        ])
        for t in range(T):
            is_ms = int(t in milestones)
            w.writerow([
                t, t/ max(1e-9, fps),
                *cent_L[t].tolist(),
                *vL[t].tolist(), *vL_s[t].tolist(),
                *uL[t].tolist(), thL_deg[t], spL[t],
                *cent_R[t].tolist(),
                *vR[t].tolist(), *vR_s[t].tolist(),
                *uR[t].tolist(), thR_deg[t], spR[t],
                is_ms
            ])

    print(f"[DUMP] {out_csv}  (frames={T})")

    # 可视化：角度与速度
    if _HAS_MPL:
        import matplotlib.pyplot as plt
        xs = np.arange(T)

        # 角度图
        fig1 = plt.figure(figsize=(12,4))
        ax1 = fig1.gca()
        ax1.plot(xs, thL_deg, label="theta_xy_L (deg)")
        ax1.plot(xs, thR_deg, label="theta_xy_R (deg)", alpha=0.75)
        for t in milestones:
            ax1.axvline(t, color="k", alpha=0.25, linewidth=1)
        ax1.set_xlabel("frame"); ax1.set_ylabel("deg")
        ax1.legend(loc="upper right")
        fig1.tight_layout()
        fig1.savefig(out_dir / "direction_theta_xy.png", dpi=150)
        plt.close(fig1)

        # 速度图
        fig2 = plt.figure(figsize=(12,4))
        ax2 = fig2.gca()
        ax2.plot(xs, spL, label="speed_L")
        ax2.plot(xs, spR, label="speed_R", alpha=0.75)
        for t in milestones:
            ax2.axvline(t, color="k", alpha=0.25, linewidth=1)
        ax2.set_xlabel("frame"); ax2.set_ylabel("|v|")
        ax2.legend(loc="upper right")
        fig2.tight_layout()
        fig2.savefig(out_dir / "direction_speed.png", dpi=150)
        plt.close(fig2)

        print(f"[PLOT] {out_dir/'direction_theta_xy.png'}")
        print(f"[PLOT] {out_dir/'direction_speed.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generic milestone detection from human demo rollout")
    parser.add_argument("--tasks_root", type=str, default="data/tasks")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--left_right_split", type=str, choices=["first_half","odd_even"], default="first_half")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--turn_deg_thr", type=float, default=30.0)
    parser.add_argument("--min_path_percentile", type=float, default=60.0)
    parser.add_argument("--min_gap_seconds", type=float, default=0.6)
    parser.add_argument("--ema_beta", type=float, default=0.6)
    parser.add_argument("--enable_wrist", action="store_true")
    parser.add_argument("--wrist_deg_thr", type=float, default=90.0)
    parser.add_argument("--wrist_quiet_ratio", type=float, default=0.5)
    parser.add_argument("--wrist_hold_frames", type=int, default=8)
    parser.add_argument("--anchor_idx_left", type=int, default=None)
    parser.add_argument("--anchor_idx_right", type=int, default=None)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--cluster", type=str, choices=["dbscan","first_half","odd_even"], default="dbscan")
    parser.add_argument("--dbscan_eps", type=float, default=0.05)
    parser.add_argument("--dbscan_min_samples", type=int, default=3)

    args = parser.parse_args()

    main(
        tasks_root=args.tasks_root,
        task_name=args.task_name,
        cluster=args.cluster,
        left_right_split=args.left_right_split,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,        
        window=args.window,
        turn_deg_thr=args.turn_deg_thr,
        min_path_percentile=args.min_path_percentile,
        min_gap_seconds=args.min_gap_seconds,
        ema_beta=args.ema_beta,
        enable_wrist=args.enable_wrist,
        wrist_deg_thr=args.wrist_deg_thr,
        wrist_quiet_ratio=args.wrist_quiet_ratio,
        wrist_hold_frames=args.wrist_hold_frames,
        anchor_idx_left=args.anchor_idx_left,
        anchor_idx_right=args.anchor_idx_right,
        plot=not args.no_plot,
    )
