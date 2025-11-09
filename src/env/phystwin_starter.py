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

        # ===== 4. Info from Simulator
        # 初始化控制点分组与局部坐标
        ctrl_init = self.simulator.controller_points[0].detach().cpu().numpy()
        self.left_idx, self.right_idx = self.split_ctrl_pts_kmeans(torch.from_numpy(ctrl_init))
        self.prev_target = torch.from_numpy(ctrl_init).to(cfg.device)

        # 记录局部坐标（初始化时 gripper pose 由 Isaac 第一次传入后再更新）
        self.local_ctrl_left  = None
        self.local_ctrl_right = None
        self.current_target = self.prev_target.clone()

        # ===== 5. 加载 Gaussian Splatting 模型 =====
        logger.info(f"Loading Gaussian model from {self.gaussians_path}")
        self.gaussians = GaussianModel(sh_degree=3)
        self.gaussians.load_ply(self.gaussians_path)
        self.gaussians = remove_gaussians_with_low_opacity(self.gaussians, 0.1)
        self.gaussians.isotropic = True


    def step(self, left_delta=None, right_delta=None):
        """
        推动一帧仿真。接受外部控制器输入（如 Isaac 双臂末端位姿变化），
        更新 simulator 的控制点并推进一步。
        """
        logger.info("[PhysTwinStarter] stepping...")

        # 将控制点目标传入 spring-mass 系统
        self.simulator.set_controller_interactive(self.prev_target, self.current_target)

        # 执行 Warp CUDA 计算图
        wp.capture_launch(self.simulator.forward_graph)

        # 准备下一帧初始状态
        self.simulator.set_init_state(
            self.simulator.wp_states[-1].wp_x,
            self.simulator.wp_states[-1].wp_v
        )

        # 记录 prev_target
        self.prev_target = self.current_target.clone()
        logger.info("[PhysTwinStarter] step finished.")


    def get_state(self):
        wp_x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
        ctrl_pts = self.simulator.controller_points[0].detach().clone()

        # Gaussian 信息
        gs = self.gaussians
        gs_xyz   = gs.get_xyz.detach().clone()
        gs_sigma = gs.get_scaling.detach().clone()

        # === 获取颜色 ===
        if hasattr(gs, "get_color"):
            gs_color = gs.get_color.detach().clone()
        elif hasattr(gs, "get_features_dc"):
            # feature_dc 是 (N, SH_coeffs, 3)，只取 DC (base color)
            features = gs.get_features_dc.detach().clone()
            if features.shape[-1] == 3:
                gs_color = torch.sigmoid(features)
            else:
                # fallback to gray
                gs_color = torch.ones_like(gs_xyz) * 0.5
        else:
            gs_color = torch.ones_like(gs_xyz) * 0.5

        logger.info(f"[PhysTwinStarter] wp_x={wp_x.shape}, gs={gs_xyz.shape}, ctrl={ctrl_pts.shape}")
        return wp_x, gs_xyz, gs_sigma, gs_color, ctrl_pts

    def set_ctrl_from_robot(self, left_pose, right_pose):
        """
        将 robot 每一步移动的末端位姿 (R, t) 应用到左右控制点簇。
        left_pose / right_pose: dict with keys {"R": (3,3), "t": (3,)}
        """
        ctrl_pts = self.current_target.clone()

        # 解析输入
        R_left,  t_left  = np.array(left_pose["R"], dtype=np.float32),  np.array(left_pose["t"], dtype=np.float32)
        R_right, t_right = np.array(right_pose["R"], dtype=np.float32), np.array(right_pose["t"], dtype=np.float32)

        # 如果是第一次调用，保存局部坐标
        if self.local_ctrl_left is None:
            ctrl_np = ctrl_pts.detach().cpu().numpy()
            self.local_ctrl_left  = (R_left.T  @ (ctrl_np[self.left_idx]  - t_left).T).T
            self.local_ctrl_right = (R_right.T @ (ctrl_np[self.right_idx] - t_right).T).T

        # 应用刚体变换
        left_world  = (R_left  @ self.local_ctrl_left.T).T  + t_left
        right_world = (R_right @ self.local_ctrl_right.T).T + t_right

        # 更新控制点
        ctrl_pts[self.left_idx]  = torch.from_numpy(left_world).to(ctrl_pts.device, dtype=ctrl_pts.dtype)
        ctrl_pts[self.right_idx] = torch.from_numpy(right_world).to(ctrl_pts.device, dtype=ctrl_pts.dtype)

        self.current_target = ctrl_pts

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


