# src/planning/gradient_core.py
import os, time, pickle, sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import warp as wp

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR      = PROJECT_ROOT / "src"
PHYSTWIN_DIR = PROJECT_ROOT / "third_party" / "PhysTwinFork"
DATA_DIR     = PROJECT_ROOT / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PHYSTWIN_DIR) not in sys.path:
    sys.path.insert(0, str(PHYSTWIN_DIR))

from src.env.phystwin_starter import PhysTwinEnv
from src.planning.action_param import ActionParam, rowwise_normalized_step
from src.planning.losses import mpc_loss_shape_relative

@dataclass
class MPCOpts:
    horizon: int = 10
    max_iters: int = 30
    step_row: float = 3e-2
    pre_tanh_clip: float = 3.0
    grad_clip: float = 1.0
    max_delta: float = 0.25
    w_action: float = 0.0
    early_tol: float = 1e-3

class GradientCore:
    def __init__(self, case_name: str, data_dir: Path, device="cuda"):
        self.case = case_name
        self.data_dir = Path(data_dir)
        self.device = device

        self.env = PhysTwinEnv(case_name)
        self.sim = self.env.simulator

        # 左右 mask
        self.left_idx = None
        self.right_idx = None
        self.left_wp_mask = None
        self.right_wp_mask = None

        # clean ctrl-pts
        self._ctrl0_wp = None

        # 预备 springs（来自 init.pkl）
        self.springs_wp = None

    def set_init_from_pkl(self, init_pkl: dict):
        # 写入初始状态
        wp_x0 = wp.array(init_pkl["wp_x"], dtype=wp.vec3f, device="cuda", requires_grad=True)
        wp_v0 = wp.array(init_pkl["wp_v"], dtype=wp.vec3f, device="cuda", requires_grad=True)
        self.sim.set_init_state(wp_x0, wp_v0)

        ctrl = torch.tensor(init_pkl["ctrl_pts"], dtype=torch.float32, device="cuda")
        self.sim.wp_original_control_point = wp.from_torch(ctrl, dtype=wp.vec3, requires_grad=False)
        self.sim.wp_target_control_point = wp.clone(self.sim.wp_original_control_point, requires_grad=True)

        # 保存“初始控制点”的只读快照，供每个 it 复位
        self._ctrl0_wp = wp.clone(self.sim.wp_original_control_point, requires_grad=False)

        # 弹簧
        springs_np = init_pkl.get("spring_indices", None)
        if springs_np is None:
            raise ValueError("init_pkl missing 'spring_indices'")
        self.springs_wp = wp.array(springs_np.reshape(-1).astype(np.int32), dtype=wp.int32, device="cuda")

        # —— 左右分组（使用真实 ctrl）——
        L, R = self.env.split_ctrl_pts_kmeans(ctrl)
        if len(L) == 0 or len(R) == 0:
            raise RuntimeError("KMeans returned empty cluster(s)")
        self.left_idx, self.right_idx = L.detach().cpu().numpy(), R.detach().cpu().numpy()

        # 构建/更新 warp mask
        left_mask  = torch.zeros(ctrl.shape[0], dtype=torch.int32, device="cuda")
        right_mask = torch.zeros_like(left_mask)
        left_mask[L] = 1; right_mask[R] = 1
        self.left_wp_mask  = wp.array(left_mask.detach().cpu().numpy(), dtype=wp.int32, device="cuda")
        self.right_wp_mask = wp.array(right_mask.detach().cpu().numpy(), dtype=wp.int32, device="cuda")

    def visualize_split_once(self, init_pkl: dict, target_pkl: dict):
        """在 set_init_from_pkl / set_target_from_pkl 之后调用，检查左右手"""
        init_mass = init_pkl["wp_x"].astype(np.float32)
        tgt_mass  = target_pkl["object_points"].astype(np.float32)
        init_ctrl = init_pkl["ctrl_pts"].astype(np.float32)
        tgt_ctrl  = target_pkl["ctrl_pts"].astype(np.float32)

        # 目标帧也做一次 KMeans（可能手在另一侧）
        L_tgt, R_tgt = self.env.split_ctrl_pts_kmeans(torch.from_numpy(tgt_ctrl).to("cuda"))
        L_tgt = L_tgt.detach().cpu().numpy()
        R_tgt = R_tgt.detach().cpu().numpy()

        from src.vis.vis_ctrl_split import visualize_ctrl_and_mass_split
        visualize_ctrl_and_mass_split(
            init_mass=init_mass,
            tgt_mass=tgt_mass,
            init_ctrl=init_ctrl,
            tgt_ctrl=tgt_ctrl,
            left_idx_init=self.left_idx,
            right_idx_init=self.right_idx,
            left_idx_tgt=L_tgt,
            right_idx_tgt=R_tgt,
            z_lift_target=0.01,
            bg_color="black",
            mass_point_size=1.5, 
            ctrl_radius=0.015, 
            show_window=True,
            save_path=None,
        )

    def set_target_from_pkl(self, target_pkl: dict):
        target_gs = torch.tensor(target_pkl["object_points"], dtype=torch.float32, device="cuda")
        self.sim.wp_current_object_points = wp.from_torch(target_gs, dtype=wp.vec3, requires_grad=False)

    # —— 单步展开（左右2×3 → full）——
    def _expand_lr_to_full(self, a2x3: torch.Tensor, scale: float):
        a_wp = wp.from_torch(a2x3, dtype=wp.vec3, requires_grad=False)
        dfull = wp.zeros(self.sim.num_control_points, dtype=wp.vec3, device="cuda", requires_grad=False)
        wp.launch(self.sim.expand_row2_squash_scale, dim=self.sim.num_control_points,
                  inputs=[a_wp, self.left_wp_mask, self.right_wp_mask, float(scale)], outputs=[dfull])
        return dfull

    def _rollout_no_tape(self, action_seq: torch.Tensor, scale: float, record=False):
        """纯前向 rollout H 步"""
        H = action_seq.shape[0] // 2
        frames = []
        if record:
            frames.append(wp.to_torch(self.sim.wp_states[-1].wp_x).detach().cpu().numpy())
        for j in range(H):
            a2 = action_seq[2*j:2*j+2]
            df = self._expand_lr_to_full(a2, scale)
            self.sim.rollout(df)
            if record:
                frames.append(wp.to_torch(self.sim.wp_states[-1].wp_x).detach().cpu().numpy())
        return frames if record else None

    def optimize_online(self, opts: MPCOpts, action: ActionParam):
        """
        在当前 sim 初始状态上，对 H 步动作做在线优化（方案二）；
        返回 best_loss
        """
        # ---- 抓初态快照 ----
        x0_np = wp.to_torch(self.sim.wp_states[0].wp_x).detach().cpu().numpy()
        v0_np = wp.to_torch(self.sim.wp_states[0].wp_v).detach().cpu().numpy()

        best = float("inf")
        for it in range(opts.max_iters):
            # 每次迭代先恢复到初态
            wp_x0 = wp.array(x0_np, dtype=wp.vec3f, device="cuda", requires_grad=True)
            wp_v0 = wp.array(v0_np, dtype=wp.vec3f, device="cuda", requires_grad=True)
            self.sim.set_init_state(wp_x0, wp_v0)

            # 还原控制点
            # 1) 在 set_init_from_pkl(...) 里缓存一份 ctrl0
            #    self._ctrl0_wp = wp.clone(self.sim.wp_original_control_point, requires_grad=False)
            # 2) 每个 it 复位
            self.sim.wp_original_control_point = wp.clone(self._ctrl0_wp, requires_grad=False)
            self.sim.wp_target_control_point = wp.clone(self._ctrl0_wp,   requires_grad=True)

            if getattr(self.sim, "object_collision_flag", False):
                self.sim.update_collision_graph()

            # 每轮迭代是否清理的检查
            def _checksum(wp_arr):
                t = wp.to_torch(wp_arr).detach()
                return float(t.float().sum().cpu())
            print("[CHK] it", it,
                "x0=", _checksum(self.sim.wp_states[0].wp_x),
                "v0=", _checksum(self.sim.wp_states[0].wp_v),
                "ctrl_orig=", _checksum(self.sim.wp_original_control_point),
                "ctrl_tgt=",  _checksum(self.sim.wp_target_control_point))
            

            with wp.Tape() as tape:
                a_seq_wp = wp.from_torch(action.param, dtype=wp.vec3, requires_grad=True)

                # 组装 H 步 full action
                df_list = []
                for j in range(opts.horizon):
                    dlr = wp.zeros(2, dtype=wp.vec3, device="cuda", requires_grad=True)
                    wp.launch(self.sim.copy_row_vec3, dim=2, inputs=[a_seq_wp, j, 2], outputs=[dlr])
                    df = wp.zeros(self.sim.num_control_points, dtype=wp.vec3, device="cuda", requires_grad=True)
                    wp.launch(self.sim.expand_row2_squash_scale, dim=self.sim.num_control_points,
                              inputs=[dlr, self.left_wp_mask, self.right_wp_mask, float(opts.max_delta)],
                              outputs=[df])
                    df_list.append(df)

                # rollout
                self.sim.rollout(df_list)

                # loss
                loss_t = mpc_loss_shape_relative(self.sim, self.springs_wp,
                                                 w_action=opts.w_action,
                                                 action_seq_wp=a_seq_wp)

            # backward
            try:
                tape.backward(self.sim.loss)
            except Exception as e:
                print(f"[MPC][it={it}] tape.backward error: {e}")
                break

            # 2) warp.grad -> torch 并净化
            g = wp.to_torch(a_seq_wp.grad)          # g: [H*2, 3]
            if g is None:
                g_none, g_finite, g_norm = True, False, 0.0
            else:
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                g_none  = False
                g_finite= bool(torch.isfinite(g).all())
                g_norm  = float(g.norm().detach().cpu())

            # 3) 用 g 更新参数（你当前的“优化器”）
            if not g_none:
                action.param.grad = g
                rowwise_normalized_step(action.param, action.param.grad, opts.step_row)
                action.clamp_pre_tanh_(opts.pre_tanh_clip)
                action.param.grad = None  # 现在才清

            # 4) 打印（只用上面缓存到的 g_* 数）
            p_norm = float(action.param.detach().norm().cpu())
            x_last = wp.to_torch(self.sim.wp_states[-1].wp_x).detach()
            nan_x  = torch.isnan(x_last).any().item()
            inf_x  = torch.isinf(x_last).any().item()
            curr   = float(loss_t.detach().cpu())
            print(f"[MPC][it={it}] loss={curr:.6e} | grad_none={g_none} finite={g_finite} "
                f"| gnorm={g_norm:.3e} | |param|={p_norm:.3e} | x[nan]={nan_x} x[inf]={inf_x}")
            
            g_wp_norm = 0.0 if a_seq_wp.grad is None else float(wp.to_torch(a_seq_wp.grad).norm().detach().cpu())
            print(f"[DBG] a_seq_wp.grad_norm={g_wp_norm:.3e}")


            if curr < opts.early_tol:
                print(f"[MPC] early stop at it={it}, loss={curr:.6f}")
                break

            # 评估一次 loss（纯前向）
            with torch.no_grad():
                # 重置到优化前初态：当前 sim.wp_states[0] 已是 rollout 前的初态
                # 简化处理：直接再 roll 一次（Tape 外）
                # （更严谨可在进入 optimize_online 前抓一次 snapshot 并恢复）
                pass

        return best

    def run_segment(self, init_pkl: dict, target_pkl: dict,
                    opts: MPCOpts, exec_last_horizon: bool = True,
                    save_path: Path = None):
        # 写入 init / target
        self.set_init_from_pkl(init_pkl)
        self.set_target_from_pkl(target_pkl)

        # 优化 H 步
        action = ActionParam(H=opts.horizon, device="cuda")
        best = self.optimize_online(opts, action)

        # 执行 H 步 & 记录（最后一轮）
        frames = self._rollout_no_tape(action.param.detach(), opts.max_delta, record=True)

        # 保存 rollout.pkl
        if save_path is None:
            out_dir = Path("data/mpc_logs")
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / "rollout.pkl"

        rollout = {
            "frames": np.asarray(frames, dtype=np.float32),       # [H+1, N, 3]
            "action_seq": action.param.detach().cpu().numpy(),    # [H*2, 3]
            "max_delta": float(opts.max_delta),
            "left_idx": np.asarray(self.left_idx, dtype=np.int32),
            "right_idx": np.asarray(self.right_idx, dtype=np.int32),
            "best_loss": float(best),
        }
        with open(save_path, "wb") as f:
            pickle.dump(rollout, f)
        print(f"[MPC] saved rollout to {save_path}")
        return save_path
