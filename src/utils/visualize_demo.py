#!/usr/bin/env python3
import os, sys, pickle, argparse, math, time
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
python src/utils/visualize_demo.py \
  --pkl /home/qiandaoliu/long_horizon_mpc/data/tasks/task7/demo_rollout/demo.pkl \
  --out /home/qiandaoliu/long_horizon_mpc/data/tasks/task7/demo_rollout \
  --fps 30 \
  --size 800
"""

"""
Structure of a .pkl Rollout:
{
    "ctrl_seq": [
        np.ndarray shape (C,3), # frame 0
        np.ndarray shape (C,3), # frame 1
        ...
    ],
    "wp_x_seq": [
        np.ndarray shape (N,3), # farme 0
        ...
    ],
    "timestamps": [float, ...], # timestemp for every frames
    "pressed_keys": [
        list of str,
        ...
    ],
    "meta": {
        "fps": float,
        "n_ctrl_parts": int,
        "eps_ctrl": float,
        "eps_wp": float,
    },
    "init_state": {
        "wp_x0": np.ndarray (N,3),
        "wp_v0": np.ndarray (N,3),
        "ctrl0": np.ndarray (C,3),
        "spring_indices": np.ndarray (M,2),
        "spring_rest_len": np.ndarray (M,),
    }
}
"""
def load_demo(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # data["ctrl_seq"] : list[T] of (C,3)
    # data["wp_x_seq"] : list[T] of (N,3)
    # data["timestamps"]: list[T] (sec)
    # data["pressed_keys"]: list[T] of list[str]
    # data["init_state"]: dict(...)
    # data["meta"]: dict(...)
    ctrl_seq = np.asarray(data.get("ctrl_seq", []), dtype=object)
    wp_x_seq = np.asarray(data.get("wp_x_seq", []), dtype=object)
    ts_seq   = np.asarray(data.get("timestamps", []), dtype=float)
    keys_seq = data.get("pressed_keys", [])
    meta     = data.get("meta", {})
    return ctrl_seq, wp_x_seq, ts_seq, keys_seq, meta

def to_ndarray_list(obj_array):
    # obj_array: np.array(dtype=object) of arrays
    return [np.asarray(x, dtype=np.float32) for x in obj_array.tolist()]

def compute_disp_stats(wp_list):
    if len(wp_list) < 2:
        return np.zeros(0), np.zeros(0)
    means, maxs = [], []
    for t in range(1, len(wp_list)):
        d = wp_list[t] - wp_list[t-1]
        n = np.linalg.norm(d, axis=1)
        means.append(float(n.mean()))
        maxs.append(float(n.max()))
    return np.asarray(means), np.asarray(maxs)

def normalize_points_for_canvas(points_list, margin=0.08):
    all_xy = []
    for pts in points_list:
        if pts.size == 0: 
            continue
        all_xy.append(pts[:, :2])  # XY
    if not all_xy:
        raise ValueError("empty point list")
    all_xy = np.concatenate(all_xy, axis=0)
    mins = all_xy.min(axis=0)
    maxs = all_xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    s = float(max(span[0], span[1]))

    def norm_xy(xy):
        center = (mins + maxs) * 0.5
        xy0 = xy - center
        xy1 = xy0 / s * (1.0 - 2*margin)
        xy1 = xy1 + 0.5
        return xy1

    return norm_xy, (mins, maxs, s)

def draw_frame(canvas_size, cloth_xy, ctrl_xy, text, thickness=1):
    H = W = canvas_size
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    def to_px(xy):
        xy_px = np.clip(xy, 0.0, 1.0) * (W-1)
        return xy_px.astype(np.int32)

    if cloth_xy.size > 0:
        pts = to_px(cloth_xy)
        for p in pts:
            cv2.circle(canvas, (p[0], p[1]), 1, (0,0,0), -1, lineType=cv2.LINE_AA)

    if ctrl_xy.size > 0:
        cps = to_px(ctrl_xy)
        for p in cps:
            cv2.circle(canvas, (p[0], p[1]), 3, (0,0,255), -1, lineType=cv2.LINE_AA)

    cv2.putText(canvas, text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), thickness, cv2.LINE_AA)
    return canvas

def save_video(frames, out_path, fps=30):
    if len(frames) == 0:
        print("[WARN] no frames to save")
        return
    H, W, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    for img in frames:
        vw.write(img)
    vw.release()

def plot_disp_stats(disp_mean, disp_max, out_png):
    plt.figure()
    plt.plot(np.arange(1, len(disp_mean)+1), disp_mean, label="mean")
    plt.plot(np.arange(1, len(disp_max)+1), disp_max, label="max")
    plt.xlabel("frame index (delta from previous)")
    plt.ylabel("point displacement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_ctrl_xy_traj(ctrl_list, left_mask=None, right_mask=None, out_png="demo_ctrl_xy_traj.png"):
    if len(ctrl_list) == 0:
        return
    ctrl = ctrl_list  # list[T] of (C,3)
    mean_xy = []
    for c in ctrl:
        if c.size == 0:
            mean_xy.append([0,0]); continue
        mean_xy.append(c[:, :2].mean(axis=0))
    mean_xy = np.asarray(mean_xy)

    plt.figure()
    plt.plot(mean_xy[:,0], mean_xy[:,1])
    plt.xlabel("ctrl mean X")
    plt.ylabel("ctrl mean Y")
    plt.title("Controller mean XY trajectory")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="path to demo_*.pkl")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, default=800, help="canvas size (pixels)")
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    ctrl_seq, wp_x_seq, ts_seq, keys_seq, meta = load_demo(str(pkl_path))
    ctrl_list = to_ndarray_list(ctrl_seq)
    wp_list   = to_ndarray_list(wp_x_seq)

    T = len(wp_list)
    if T == 0:
        print("[ERR] Empty demo.")
        sys.exit(1)

    disp_mean, disp_max = compute_disp_stats(wp_list)
    plot_disp_stats(disp_mean, disp_max, out_dir / "demo_displacement_stats.png")

    plot_ctrl_xy_traj(ctrl_list, out_png=str(out_dir / "demo_ctrl_xy_traj.png"))

    norm_fn, bounds = normalize_points_for_canvas(wp_list)

    frames = []
    for t in range(T):
        cloth_xy = norm_fn(wp_list[t][:, :2]) if wp_list[t].size else np.zeros((0,2), dtype=np.float32)
        ctrl_xy  = norm_fn(ctrl_list[t][:, :2]) if ctrl_list[t].size else np.zeros((0,2), dtype=np.float32)

        ts_text = f"t={t:03d}/{T-1}"
        if t < len(ts_seq):
            ts_text += f"  time={ts_seq[t]:.2f}s"
        if t < len(keys_seq) and len(keys_seq[t])>0:
            ts_text += f"  keys={','.join(keys_seq[t])}"

        img = draw_frame(args.size, cloth_xy, ctrl_xy, ts_text)
        frames.append(img)

    out_mp4 = out_dir / (pkl_path.stem + ".mp4")
    save_video(frames, out_mp4, fps=args.fps)
    print(f"Saved video: {out_mp4}")
    print(f"Saved plots: {out_dir/'demo_displacement_stats.png'}, {out_dir/'demo_ctrl_xy_traj.png'}")

if __name__ == "__main__":
    main()
