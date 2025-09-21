import numpy as np
import torch, mujoco
from mujoco import mjx  # 若用加速可选
from mujoco.glfw import glfw
from mujoco import viewer as mj_viewer  # 或自建 offscreen renderer
from pathlib import Path

class CoSimRunner:
    def __init__(self, mj_xml:str, case_name:str, dt:float=0.01, substeps:int=2,
                 cam_name:str="camera0"):
        # 1) MuJoCo
        self.mj_model = mujoco.MjModel.from_xml_path(mj_xml)
        self.mj_data  = mujoco.MjData(self.mj_model)
        self.dt = dt
        self.substeps = substeps
        # 如果 model.dt 与 dt 不同，可以设置 mj_model.opt.timestep = dt
        self.cam_name = cam_name
        self.cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

        # 2) PhysTwin
        from src.env.phystwin_env import PhysTwinEnv
        self.env = PhysTwinEnv(case_name)
        # 拆左右索引（复用你的 DBSCAN）
        ctrl = self.env.simulator.controller_points[0].detach()
        self.left_idx, self.right_idx = self.env.split_ctrl_pts_dbscan(ctrl)

        # 3) 相机标定：对齐 MJ 摄像机到 GS camera（或反之）
        # 建议存到 configs：K(intrinsic), T_wc(extrinsic), 并在两端复用
        # 这里略，假设你有一个函数 get_shared_camera()

        # 4) 渲染器（可选离屏）
        # 你已有 GS 渲染；MJ 可用 mujoco.Renderer(width,height)

    def _get_gripper_targets(self):
        # 从 MJ 读取两个末端/夹爪 site 的位置（世界系）
        sid_L = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripperL_site")
        sid_R = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripperR_site")
        pL = self.mj_data.site_xpos[sid_L].copy()
        pR = self.mj_data.site_xpos[sid_R].copy()
        return torch.tensor(pL, dtype=torch.float32, device="cuda"), \
               torch.tensor(pR, dtype=torch.float32, device="cuda")

    def step(self, ctrl_cmd=None):
        """执行一个主时间步：MJ->PhysTwin 子步->渲染合成"""
        # 1) MJ 推进一步（机器人受控）
        if ctrl_cmd is not None:
            ctrl_cmd(self.mj_model, self.mj_data)  # 例如设关节力/位姿
        mujoco.mj_step(self.mj_model, self.mj_data)

        # 2) 读取 gripper 目标，驱动布料控制点
        tL, tR = self._get_gripper_targets()
        self.env.set_ctrl_targets(tL, tR, self.left_idx, self.right_idx)

        # 3) PhysTwin 子步（保障稳定）
        for k in range(self.substeps):
            self.env.step_phys(is_first=(k==0))

        # 4) 渲染（可选）：MJ 渲染 + GS 渲染 → alpha 合成
        # mj_rgb = render_mujoco(self.mj_model, self.mj_data, self.cam_id)
        # gs_rgba = render_gaussians(self.env, shared_camera)
        # frame = alpha_composite(mj_rgb, gs_rgba)

        # 5) 返回观测（位姿、点云、图像等）
        obs = {
            "robot_qpos": self.mj_data.qpos.copy(),
            "gripper_LR": (tL.detach().cpu().numpy(), tR.detach().cpu().numpy()),
            "cloth_wp": wp.to_torch(self.env.simulator.wp_states[-1].wp_x, requires_grad=False).detach().cpu().numpy(),
            # "image": frame
        }
        return obs
