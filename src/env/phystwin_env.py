# /workspace/src/env/phystwin_env.py
import numpy as np
import warp as wp
wp.config.mode = "debug"
wp.config.verify_cuda = False
import torch
import glob
import os
import pickle
import wandb
import imageio
import math
import open3d as o3d
from tqdm import trange
from PhysTwin.qqtt.engine.trainer_warp import InvPhyTrainerWarp
import PhysTwin.qqtt.model.diff_simulator.spring_mass_warp as smw
from PhysTwin.qqtt.utils import logger, cfg
from sklearn.cluster import DBSCAN

from PhysTwin.gaussian_splatting.scene.gaussian_model import GaussianModel
from PhysTwin.gaussian_splatting.scene.cameras import Camera
from PhysTwin.gaussian_splatting.gaussian_renderer import render as render_gaussian
from PhysTwin.gaussian_splatting.dynamic_utils import (
    interpolate_motions_speedup,
    knn_weights,
    knn_weights_sparse,
    get_topk_indices,
    calc_weights_vals_from_indices,
)
from PhysTwin.gaussian_splatting.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from PhysTwin.gs_render import (
    remove_gaussians_with_low_opacity,
    remove_gaussians_with_point_mesh_distance,
)
from PhysTwin.gaussian_splatting.rotation_utils import quaternion_multiply, matrix_to_quaternion
from sklearn.cluster import KMeans


class PhysTwinEnv():
    """
    Gym-style version of the PhysTwin Env
    """ 
    def __init__(self, 
                 case_name,
                 pure_inference_mode=True,
                 ):
        self.case_name = case_name
        self.n_ctrl_parts = 2 

        # ===== 1. Set Path =====
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        PHYSTWIN_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../PhysTwin"))
        BEST_MODEL_GLOB = os.path.join(CURRENT_DIR, "../../PhysTwin/experiments", case_name, "train", "best_*.pth")
        exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
        best_model_files = glob.glob(BEST_MODEL_GLOB)
        if not best_model_files:
            raise FileNotFoundError(f"No best_*.pth found at {BEST_MODEL_GLOB}")

        data_path = os.path.join(PHYSTWIN_DIR, "data", "different_types", case_name, "final_data.pkl")
        base_dir = os.path.join(PHYSTWIN_DIR, "temp_experiments", case_name)
        optimal_path = os.path.join(PHYSTWIN_DIR, "experiments_optimization", case_name, "optimal_params.pkl")
        calibrate = os.path.join(PHYSTWIN_DIR, "data", "different_types", case_name, "calibrate.pkl")
        metadata = os.path.join(PHYSTWIN_DIR, "data", "different_types", case_name, "metadata.json")

        self.best_model_path = best_model_files[0]

        self.gaussians_path = os.path.join(
            PHYSTWIN_DIR, "gaussian_output", case_name,
            "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0",
            "point_cloud", "iteration_10000", "point_cloud.ply"
        )

        # ===== 2. Load Config =====
        if "cloth" in self.case_name or "package" in self.case_name:
            cfg.load_from_yaml(os.path.join(PHYSTWIN_DIR, "configs", "cloth.yaml"))
        else:
            cfg.load_from_yaml(os.path.join(PHYSTWIN_DIR, "configs", "real.yaml"))  

        logger.info(f"Load optimal parameters from: {optimal_path}")
        assert os.path.exists(
            optimal_path
        ), f"{case_name}: Optimal parameters not found: {optimal_path}"
        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)
        cfg.set_optimal_params(optimal_params)

        # ===== 3. Init a Trainer Warp =====
        cfg.device = torch.device("cuda:0")
        trainer = InvPhyTrainerWarp(
            data_path=data_path,
            base_dir=base_dir,
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

    """
    Init the Gym-Style Env
    """
    def init_scenario(self, best_model_path):
        self.timer.start()
        
        # Load the model
        logger.info(f"Load model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        # print(f"[DEBUG] Simulator spring count: {self.simulator.n_springs}")
        # print(f"[DEBUG] Checkpoint spring_Y count: {spring_Y.shape[0]}")

        # assert (
        #     len(spring_Y) == self.simulator.n_springs
        # ), "Check if the loaded checkpoint match the config file to connect the springs"

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        print(f"[init_scenario] Done in {self.timer.stop():.3f}s.")

    """
    Reset gs_pts and ctrl_pts to the original state
    Reset and clean spring_mass system
    """
    def reset_to_origin(self, n_ctrl_parts=2):
        # print(f"[reset] Reset at time {self.timer.stop():.3f}s.")
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        self.prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()
        
        self.current_target = self.simulator.controller_points[0]
        self.prev_target = self.current_target

        vis_controller_points = self.current_target.cpu().numpy()

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(self.gaussians_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation

        if n_ctrl_parts > 1:
            kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(vis_controller_points)
            for i in range(n_ctrl_parts):
                mask = cluster_labels == i
                self.masks_ctrl_pts.append(torch.from_numpy(mask))

            # cluster center x 坐标判断左右
            center0 = np.mean(vis_controller_points[self.masks_ctrl_pts[0]], axis=0)
            center1 = np.mean(vis_controller_points[self.masks_ctrl_pts[1]], axis=0)

            if center0[0] > center1[0]:  # x 坐标大的是右边
                # print("Switching the control parts")
                self.masks_ctrl_pts = [self.masks_ctrl_pts[1], self.masks_ctrl_pts[0]]
        else:
            self.masks_ctrl_pts = None
        self.n_ctrl_parts = n_ctrl_parts
        self.mask_ctrl_pts = self.masks_ctrl_pts
        self.scale_factors = 1.0
        assert n_ctrl_parts <= 2, "Only support 1 or 2 control parts"
        if n_ctrl_parts > 1:
            hand_positions = []
            for i in range(2):
                target_points = torch.from_numpy(
                    vis_controller_points[self.mask_ctrl_pts[i]]
                ).to("cuda")
                # print(f"[DEBUG] Hand {i} cluster points shape: {target_points.shape}")
                # print(f"[DEBUG] Hand {i} cluster points: {target_points}")
                hand_positions.append(self.trainer._find_closest_point(target_points))
            self.hand_left_pos, self.hand_right_pos = hand_positions
        else:
            target_points = torch.from_numpy(vis_controller_points).to("cuda")
            self.hand_left_pos = self.trainer._find_closest_point(target_points)

    def step(self, n_ctrl_parts, action):
        # self.simulator.set_controller_interactive(self.prev_target, self.current_target)
        self.simulator.set_controller_interactive(self.prev_target.detach(), self.current_target.detach())
        if self.simulator.object_collision_flag:
            self.simulator.update_collision_graph()
        wp.capture_launch(self.simulator.forward_graph)
        x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=True)

        # Set initial state for next step
        self.simulator.set_init_state(
            self.simulator.wp_states[-1].wp_x,
            self.simulator.wp_states[-1].wp_v,
        )

        torch.cuda.synchronize()

        self.prev_x = x

        self.prev_target = self.current_target
        """        
        ctrl_pts shape: [n_ctrl_parts, 3]
        """
        target_change = action
        if self.masks_ctrl_pts is not None:
            for i in range(n_ctrl_parts):
                if self.masks_ctrl_pts[i].sum() > 0:
                    self.current_target[self.masks_ctrl_pts[i]] += torch.tensor(
                        target_change[i], dtype=torch.float32, device=cfg.device
                    )
                    
                    if i == 0:
                        self.hand_left_pos += torch.tensor(
                            target_change[i], dtype=torch.float32, device=cfg.device
                        )
                    if i == 1:
                        self.hand_right_pos += torch.tensor(
                            target_change[i], dtype=torch.float32, device=cfg.device
                        )
        else:
            self.current_target += torch.tensor(
                target_change, dtype=torch.float32, device=cfg.device
            )
            self.hand_left_pos += torch.tensor(
                target_change, dtype=torch.float32, device=cfg.device
            )

    def get_obs(self):
        ctrl_pts = self.current_target.clone().detach().cpu().numpy()
        state_pts = self.prev_x.detach().cpu().numpy()
        return {"ctrl_pts": ctrl_pts, "state": state_pts}

    def reset_clusters(self, vis_controller_points, n_ctrl_parts=None):
        """
        Re-run clustering logic to assign masks_ctrl_pts.
        """
        if n_ctrl_parts is None:
            n_ctrl_parts = self.n_ctrl_parts

        self.masks_ctrl_pts = []
        if n_ctrl_parts > 1:
            kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(vis_controller_points)
            for i in range(n_ctrl_parts):
                mask = cluster_labels == i
                self.masks_ctrl_pts.append(torch.from_numpy(mask))

            # Sort left/right by x
            center0 = np.mean(vis_controller_points[self.masks_ctrl_pts[0]], axis=0)
            center1 = np.mean(vis_controller_points[self.masks_ctrl_pts[1]], axis=0)
            if center0[0] > center1[0]:
                # print("Switching the control parts (left/right)")
                self.masks_ctrl_pts = [self.masks_ctrl_pts[1], self.masks_ctrl_pts[0]]

        else:
            self.masks_ctrl_pts = None
        self.mask_ctrl_pts = self.masks_ctrl_pts
        self.n_ctrl_parts = n_ctrl_parts

    def get_ctrl_pts(self):
        return self.simulator.get_controller_state()
    
    def get_gs_pts(self):
        return self.get_obs()["state"]
    
    # 辅助函数: 分左右手idx
    def split_ctrl_pts_dbscan(self, ctrl_pts: torch.Tensor):
        clustering = DBSCAN(eps=0.05, min_samples=3).fit(ctrl_pts.cpu().numpy())
        labels = clustering.labels_

        clusters = {}
        for i in range(max(labels) + 1):
            clusters[i] = np.where(labels == i)[0]  # 保存索引而不是值

        if len(clusters) != 2:
            print(f"⚠️ Warning: DBSCAN did not find exactly 2 clusters! Found {len(clusters)}")
            # fallback: 左右各取一半
            return np.arange(15), np.arange(15, 30)

        ctrl_np = ctrl_pts.cpu().numpy()
        c0_mean_x = ctrl_np[clusters[0], 0].mean()
        c1_mean_x = ctrl_np[clusters[1], 0].mean()

        if c0_mean_x < c1_mean_x:
            left_idx, right_idx = clusters[0], clusters[1]
        else:
            left_idx, right_idx = clusters[1], clusters[0]

        return left_idx, right_idx
    
    # 可视化：初始状态和目标状态
    def visualize_initial_vs_target(self, init_data, target_data):
        """
        可视化 init 和 target
        - cloth 点云：蓝 vs 红
        - 左手/右手中心：绿 vs 黄
        - 用线把每个 init cloth 点和对应 target cloth 点连接
        """
        # Cloth 点云
        init_pts  = np.asarray(init_data["wp_x"])             # (M,3)
        target_pts = np.asarray(target_data["object_points"]) # (M,3)
        
        pcd_init  = o3d.geometry.PointCloud()
        pcd_init.points = o3d.utility.Vector3dVector(init_pts)
        pcd_init.paint_uniform_color([0.1, 0.1, 0.7])  # 深蓝
        
        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(target_pts)
        pcd_target.paint_uniform_color([0.7, 0.1, 0.1])  # 深红

        # 计算左右手中心
        ctrl_init  = torch.tensor(init_data["ctrl_pts"], dtype=torch.float32)
        ctrl_tgt   = torch.tensor(target_data["ctrl_pts"], dtype=torch.float32)
        left_i, right_i = self.split_ctrl_pts_dbscan(ctrl_init)    # numpy idx arrays
        left_t, right_t = self.split_ctrl_pts_dbscan(ctrl_tgt)
        
        left_mean_i  = ctrl_init[left_i].mean(dim=0).cpu().numpy()
        right_mean_i = ctrl_init[right_i].mean(dim=0).cpu().numpy()
        left_mean_t  = ctrl_tgt[left_t].mean(dim=0).cpu().numpy()
        right_mean_t = ctrl_tgt[right_t].mean(dim=0).cpu().numpy()

        # 为中心点创建小球辅助几何
        def make_sphere(center, color, radius=0.005):
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius)
            mesh.translate(center)
            mesh.paint_uniform_color(color)
            return mesh

        sphere_li = make_sphere(left_mean_i,  [0.0, 0.8, 0.0])  # 绿
        sphere_ri = make_sphere(right_mean_i, [0.0, 0.5, 0.0])  # 深绿
        sphere_lt = make_sphere(left_mean_t,  [0.9, 0.9, 0.1])  # 黄
        sphere_rt = make_sphere(right_mean_t, [0.8, 0.8, 0.0])  # 深黄

        # 构造连线 (LineSet) —— cloth 上的对应点
        num_pts = init_pts.shape[0]
        # 合并点阵: 前半段是 init，后半段是 target
        all_pts = np.vstack([init_pts, target_pts])
        # 每条线 (i, i+num_pts)
        lines = [[i, i + num_pts] for i in range(num_pts)]
        colors = [[0.5, 0.5, 0.5] for _ in lines]  # 灰色
        
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(all_pts),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # 最后把所有几何体一起显示
        o3d.visualization.draw_geometries([
            pcd_init, pcd_target,
            sphere_li, sphere_ri, sphere_lt, sphere_rt,
            line_set
        ],
        window_name="Init vs Target",
        width=800, height=600,
        left=50, top=50,
        point_show_normal=False)

    # 可视化：simulator
    def visualize_simulator_state(self):
        """
        从 simulator 里读取当前状态并可视化：
         - 布料点云：灰色
         - 左右手控制点：大球，绿/黄
        """
        n_ctrl_pts = self.simulator.wp_states[0].wp_control_x.shape[0]
        print("len(wp_states):", len(self.simulator.wp_states))
        print("wp_states[0].wp_control_x shape:", self.simulator.wp_states[0].wp_control_x.shape)
        print("wp_states[0].wp_x shape:", self.simulator.wp_states[0].wp_x.shape)

        print("simulator.controller_points type:", type(self.simulator.controller_points))
        print("simulator.controller_points shape:", self.simulator.controller_points.shape)
        print("simulator.controller_points ndim: ", self.simulator.controller_points.ndim)
        print("simulator.controller_points dtype:", self.simulator.controller_points.dtype)
        print("simulator.controller_points device:", self.simulator.controller_points.device)
        print("simulator.controller_points requires_grad:", self.simulator.controller_points.requires_grad)

        gs_pts = wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False).detach().cpu().numpy()  # (P,3)

        # 2. 读出当前 controller points
        ctrl_pts = self.simulator.get_controller_state().detach().cpu().numpy()  # (N,3)

        # 3. 构建点云几何
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gs_pts)
        pcd.paint_uniform_color([0.6, 0.6, 0.6])  # 灰

        # 4. 识别左右手索引 & 计算中心
        ctrl_tensor = torch.from_numpy(ctrl_pts)
        left_idx, right_idx = self.split_ctrl_pts_dbscan(ctrl_tensor)
        left_mean  = ctrl_tensor[left_idx].mean(dim=0).numpy()
        right_mean = ctrl_tensor[right_idx].mean(dim=0).numpy()

        # 5. 用小球标出左右手
        def sphere(center, color, radius=0.01):
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius)
            mesh.translate(center)
            mesh.paint_uniform_color(color)
            return mesh

        sphere_l = sphere(left_mean,  [0.0, 0.8, 0.0])  # 绿
        sphere_r = sphere(right_mean, [0.9, 0.9, 0.1])  # 黄

        # 6. 显示
        o3d.visualization.draw_geometries(
            [pcd, sphere_l, sphere_r],
            window_name="Simulator State",
            width=800, height=600
        )

    # 可视化：cloth springs
    def visualize_springs(self):
        # 1) 确保 Warp 状态是最新的
        wp.capture_launch(self.simulator.forward_graph)

        # 2) 读顶点坐标，并 detach 掉 grad
        verts = (
            wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)
            .detach()
            .cpu()
            .numpy()
        )  # (V,3)

        # 3) 读弹簧索引
        springs = (
            wp.to_torch(self.simulator.wp_springs, requires_grad=False)
            .detach()
            .cpu()
            .numpy()
        )  # (S,2)

        # 4) 构造点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        pcd.paint_uniform_color([0.6, 0.6, 0.6])

        # 5) 构造 LineSet
        lines = springs.tolist()
        colors = [[0.8, 0.2, 0.2] for _ in lines]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(verts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        ls.colors = o3d.utility.Vector3dVector(colors)

        # 6) 显示
        o3d.visualization.draw_geometries(
            [pcd, ls],
            window_name="Spring Network",
            width=800, height=600
        )


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
