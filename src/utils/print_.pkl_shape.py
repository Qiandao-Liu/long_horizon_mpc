import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # utils → src → long_horizon_mpc
DATA_DIR = PROJECT_ROOT / "data"

"""
📂 File: ./data/mpc_init/init_000.pkl
         ctrl_pts: ndarray    shape=(30, 3), dtype=float32, min=-0.1842, max=0.2629, mean=0.0244
           gs_pts: ndarray    shape=(44445, 3), dtype=float32, min=-0.1355, max=0.1964, mean=0.0327
             wp_x: ndarray    shape=(4742, 3), dtype=float32, min=-0.1335, max=0.2029, mean=0.0340
   spring_indices: ndarray    shape=(103975, 2), dtype=int32, min=0.0000, max=4768.0000, mean=2375.3387
  spring_rest_len: ndarray    shape=(103975,), dtype=float32, min=0.0004, max=0.0650, mean=0.0127

📂 File: ./data/mpc_target_U/target_000.pkl
         ctrl_pts: ndarray    shape=(30, 3), dtype=float32, min=-0.1442, max=0.2229, mean=0.0427
           gs_pts: ndarray    shape=(44445, 3), dtype=float32, min=-0.1145, max=0.1955, mean=0.0437
    object_points: ndarray    shape=(4742, 3), dtype=float32, min=-0.1129, max=0.1985, mean=0.0420

Left hand (14 pts):
  idx= 1, coord=[ 0.18358071  0.19438893 -0.08767821]
  idx= 4, coord=[ 0.16195802  0.18084352 -0.1384158 ]
  idx= 5, coord=[ 0.16926992  0.24382359 -0.11852317]
  idx= 6, coord=[ 0.17891896  0.19079114 -0.11152633]
  idx= 8, coord=[ 0.17983519  0.18438709 -0.06355261]
  idx= 9, coord=[ 0.17758381  0.21943977 -0.09458151]
  idx=10, coord=[ 0.15138318  0.2500299  -0.1314926 ]
  idx=13, coord=[ 0.17661527  0.21618927 -0.12317456]
  idx=15, coord=[ 0.16627313  0.23980671 -0.13833432]
  idx=17, coord=[ 0.15812823  0.2606224  -0.10993294]
  idx=18, coord=[ 0.13374755  0.22183838 -0.15289594]
  idx=22, coord=[ 0.16270438  0.210459   -0.14650834]
  idx=24, coord=[ 0.09765279  0.26290873 -0.12373855]
  idx=25, coord=[ 0.12703013  0.25543097 -0.14012796]

Right hand (16 pts):
  idx= 0, coord=[ 0.17777632 -0.16024068 -0.13623904]
  idx= 2, coord=[ 0.19218616 -0.14260937 -0.08622006]
  idx= 3, coord=[ 0.15466814 -0.16392526 -0.1402209 ]
  idx= 7, coord=[ 0.17874154 -0.1842325  -0.12442087]
  idx=11, coord=[ 0.18020776 -0.1355114  -0.12763448]
  idx=12, coord=[ 0.18030404 -0.1221493  -0.106456  ]
  idx=14, coord=[ 0.18686341 -0.15472113 -0.11026772]
  idx=16, coord=[ 0.1465779  -0.12829809 -0.08649073]
  idx=19, coord=[ 0.13523878 -0.15589614 -0.14452526]
  idx=20, coord=[ 0.15852474 -0.1282072  -0.10442565]
  idx=21, coord=[ 0.15329488 -0.12997073 -0.1268976 ]
  idx=23, coord=[ 0.17586331 -0.12603536 -0.08433791]
  idx=26, coord=[ 0.17207932 -0.12491016 -0.05966926]
  idx=27, coord=[ 0.11109328 -0.1367846  -0.10972201]
  idx=28, coord=[ 0.12790558 -0.13884035 -0.12793939]
  idx=29, coord=[ 0.13093665 -0.12717521 -0.10255807]
"""

def inspect_init_state(filename):
    path = DATA_DIR / "mpc_init" / filename
    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"\n📂 File: {path}")
    for k, v in data.items():
        info = f"  {k:>15}: {type(v).__name__:<10} shape={v.shape}"
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            arr = v.flatten()
            info += f", dtype={v.dtype}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}"
        elif isinstance(v, np.ndarray):
            info += f", dtype={v.dtype}"
        print(info)

inspect_init_state("init_000.pkl")

def inspect_target_state(filename):
    path = DATA_DIR / "mpc_target_U" / filename
    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"\n📂 File: {path}")
    for k, v in data.items():
        print(f"  {k}: {v.shape}")

inspect_target_state("target_000.pkl")

from sklearn.cluster import DBSCAN

def split_ctrl_pts_dbscan(ctrl_pts: np.ndarray):
    clustering = DBSCAN(eps=0.05, min_samples=3).fit(ctrl_pts)
    labels = clustering.labels_

    clusters = {}
    for i in range(max(labels) + 1):
        clusters[i] = np.where(labels == i)[0]
    if len(clusters) != 2:
        print(f"⚠️ Warning: DBSCAN did not find exactly 2 clusters! Found {len(clusters)}")
        return np.arange(15), np.arange(15, 30)

    c0_mean_x = ctrl_pts[clusters[0], 0].mean()
    c1_mean_x = ctrl_pts[clusters[1], 0].mean()

    if c0_mean_x < c1_mean_x:
        left_idx, right_idx = clusters[0], clusters[1]
    else:
        left_idx, right_idx = clusters[1], clusters[0]

    return left_idx, right_idx

with open("./mpc_init/init_000.pkl", "rb") as f:
    data = pickle.load(f)

ctrl_pts = data["ctrl_pts"]
left_idx, right_idx = split_ctrl_pts_dbscan(ctrl_pts)
left, right = ctrl_pts[left_idx], ctrl_pts[right_idx]

print(f"\nLeft hand ({len(left_idx)} pts):")
for i in left_idx:
    print(f"  idx={i:2d}, coord={ctrl_pts[i]}")

print(f"\nRight hand ({len(right_idx)} pts):")
for i in right_idx:
    print(f"  idx={i:2d}, coord={ctrl_pts[i]}")

import matplotlib.pyplot as plt

def plot_ctrl_pts(ctrl_pts, left, right):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ctrl_pts[:, 0], ctrl_pts[:, 1], ctrl_pts[:, 2], c='gray', label='All')
    ax.scatter(left[:, 0], left[:, 1], left[:, 2], c='blue', label='Left hand')
    ax.scatter(right[:, 0], right[:, 1], right[:, 2], c='red', label='Right hand')
    ax.legend()
    plt.show()

plot_ctrl_pts(ctrl_pts, left, right)

