# ./long_horizon_mpc/src/env/phystwin_starter.py
import torch
import os, pickle, glob, sys
import open3d as o3d
import numpy as np
import warp as wp
from sklearn.cluster import KMeans
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR      = PROJECT_ROOT / "src"
PHYSTWIN_DIR = PROJECT_ROOT / "third_party" / "PhysTwinFork"
DATA_DIR     = PROJECT_ROOT / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PHYSTWIN_DIR) not in sys.path:
    sys.path.insert(0, str(PHYSTWIN_DIR))

from qqtt.engine.trainer_warp import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gs_render import remove_gaussians_with_low_opacity

import time
class Timer:
    def __init__(self):
        self.reset()

    def start(self):
        self.t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def stop(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t1 = time.time()
        return self.t1 - self.t0

    def reset(self):
        self.t0 = 0
        self.t1 = 0

class PhysTwinStarter():
    def __init__(self, 
                 case_name,
                 pure_inference_mode=True,
                 ):
        self.case_name = case_name
        self.n_ctrl_parts = 2 

        # ===== 1) Load Model =====
        exp_root      = PHYSTWIN_DIR / "experiments" / case_name / "train"
        best_model_glob = list(exp_root.glob("best_*.pth"))
        if not best_model_glob:
            raise FileNotFoundError(f"No best_*.pth under {exp_root}")
        self.best_model_path = str(best_model_glob[0])

        data_path   = PHYSTWIN_DIR / "data" / "different_types" / case_name / "final_data.pkl"
        base_dir    = PHYSTWIN_DIR / "temp_experiments" / case_name
        optimal_path = PHYSTWIN_DIR / "experiments_optimization" / case_name / "optimal_params.pkl"
        calibrate   = PHYSTWIN_DIR / "data" / "different_types" / case_name / "calibrate.pkl"
        metadata    = PHYSTWIN_DIR / "data" / "different_types" / case_name / "metadata.json"

        exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
        gs_root  = PHYSTWIN_DIR / "gaussian_output" / case_name
        gs_dir   = gs_root / exp_name
        ply_path = gs_dir / "point_cloud" / "iteration_10000" / "point_cloud.ply"
        self.gaussians_path = str(ply_path)

        # ===== 2) Load Config =====
        cfg_path = PHYSTWIN_DIR / "configs" / ("cloth.yaml" if ("cloth" in case_name or "package" in case_name) else "real.yaml")
        cfg.load_from_yaml(str(cfg_path))

        logger.info(f"Load optimal parameters from: {optimal_path}")
        if not optimal_path.exists():
            raise FileNotFoundError(f"{case_name}: Optimal parameters not found: {optimal_path}")
        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)
        cfg.set_optimal_params(optimal_params)

        # ===== 3. Init TrainerWarp & SimulatorWarp =====
        cfg.device = torch.device("cuda:0")
        trainer = InvPhyTrainerWarp(
            data_path=str(data_path),
            base_dir=str(base_dir),
            pure_inference_mode=pure_inference_mode
        )

        self.trainer = trainer
        self.simulator = trainer.simulator

    def step(self, left_delta=None, right_delta=None):
        """
        推动一帧仿真。接受外部控制器输入（如 Isaac 双臂末端位姿变化），
        更新 simulator 的控制点并推进一步。
        """

    def get_state(self):
        """
        返回当前仿真状态，包括:
        - wp_x (N,3): 物理节点位置
        - gs_xyz (M,3): Gaussian 位置
        - gs_sigma (M,): Gaussian 大小
        - gs_color (M,3): 颜色
        - ctrl_pts (K,3): 控制点位置
        """

    def set_ctrl_from_robot(self, left_pose, right_pose):
        """
        把 Isaac Sim 中的 gripper pose 映射到 controller_points
        可使用 split_ctrl_pts_kmeans() 的左右索引。
        """

    def run(self, bridge=False, socket_port=None):
        """
        如果 bridge=False 则本地纯物理运行；
        如果 bridge=True 则开启 socket 通信线程：
            Isaac Sim ↔ PhysTwin
        """


    def split_ctrl_pts_kmeans(self, ctrl_pts: torch.Tensor, n_ctrl_parts=2):
        ctrl_np = ctrl_pts.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
        labels = kmeans.fit_predict(ctrl_np)

        clusters = [np.where(labels == i)[0] for i in range(n_ctrl_parts)]
        if len(clusters) != 2:
            raise RuntimeError("KMeans failed: expected 2 clusters but got {len(clusters)}")

        c0_mean_x = ctrl_np[clusters[0], 0].mean()
        c1_mean_x = ctrl_np[clusters[1], 0].mean()
        left_idx, right_idx = (clusters[0], clusters[1]) if c0_mean_x < c1_mean_x else (clusters[1], clusters[0])

        # Return torch.LongTensor
        device = ctrl_pts.device
        return (torch.as_tensor(left_idx, dtype=torch.long, device=device),
                torch.as_tensor(right_idx, dtype=torch.long, device=device))
    
    
    
        