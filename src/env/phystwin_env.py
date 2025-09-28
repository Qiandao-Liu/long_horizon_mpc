# /src/env/phystwin_env.py
import torch
import os, pickle, glob, sys
import open3d as o3d
import numpy as np
import warp as wp
from sklearn.cluster import KMeans
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/env/phystwin_env.py
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

class PhysTwinEnv():
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

        # ===== 4. Init Scenario =====
        timer = Timer()
        self.timer = timer

        self.prev_target = self.simulator.controller_points[0].clone()
        self.current_target = self.simulator.controller_points[0].clone()
        self.prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()
        self.masks_ctrl_pts = []

        self.init_scenario(self.best_model_path)

    def init_scenario(self, best_model_path):
        self.timer.start()
        
        logger.info(f"Load model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )
        print(f"[init_scenario] Done in {self.timer.stop():.3f}s.")

    # 在 PhysTwinEnv 里补：快照/恢复（在同一 sim 上高效复现一个状态）
    def snapshot(self):
        return {
            "wp_x": wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False).detach().cpu().numpy(),
            "wp_v": wp.to_torch(self.simulator.wp_states[-1].wp_v, requires_grad=False).detach().cpu().numpy(),
            "ctrl": self.simulator.controller_points[0].detach().cpu().numpy(),
        }

    def restore(self, snap: dict):
        x = wp.array(snap["wp_x"], dtype=wp.vec3f, device="cuda")
        v = wp.array(snap["wp_v"], dtype=wp.vec3f, device="cuda")
        self.simulator.set_init_state(x, v)
        ctrl = torch.tensor(snap["ctrl"], dtype=torch.float32, device="cuda")
        self.simulator.controller_points[0].copy_(ctrl)

    def set_ctrl_targets(self, left_target: torch.Tensor, right_target: torch.Tensor, left_idx, right_idx):
        # 写入当前帧控制点目标（把两团控制点分别平移到期望位姿，可做 nearest/rigid 分配）
        tgt = self.simulator.controller_points[0].clone()
        tgt[left_idx]  = left_target
        tgt[right_idx] = right_target
        self.prev_target = self.current_target
        self.current_target = tgt

    def step_phys(self, is_first=False):
        # 调用你在 playground 中的 forward（one_step_from_action / forward_graph）
        self.simulator.set_controller_interactive(self.prev_target, self.current_target)
        if self.simulator.object_collision_flag:
            self.simulator.update_collision_graph()
        wp.capture_launch(self.simulator.forward_graph)
        # 将新状态作为下一步 init
        self.simulator.set_init_state(
            self.simulator.wp_states[-1].wp_x,
            self.simulator.wp_states[-1].wp_v,
        )

    def split_ctrl_pts_kmeans(self, ctrl_pts: torch.Tensor, n_ctrl_parts=2):
        ctrl_np = ctrl_pts.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
        labels = kmeans.fit_predict(ctrl_np)

        clusters = [np.where(labels == i)[0] for i in range(n_ctrl_parts)]
        if len(clusters) != 2 or len(clusters[0]) == 0 or len(clusters[1]) == 0:
            raise RuntimeError(f"KMeans failed: expected 2 clusters but got {len(clusters)}")

        c0_mean_x = ctrl_np[clusters[0], 0].mean()
        c1_mean_x = ctrl_np[clusters[1], 0].mean()
        if c0_mean_x < c1_mean_x:
            left_idx, right_idx = clusters[0], clusters[1]
        else:
            left_idx, right_idx = clusters[1], clusters[0]

        return left_idx, right_idx
    