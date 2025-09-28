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

from src.env.phystwin_env import PhysTwinEnv
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

        # 预备 springs（来自 init.pkl）
        self.springs_wp = None

    def set_init_from_pkl(self, init_pkl: dict):
        # 写入初始状态
        wp_x0 = wp.array(init_pkl["wp_x"], dtype=wp.vec3f, device="cuda", requires_grad=True)
        wp_v0 = wp.array(init_pkl["wp_v"], dtype=wp.vec3f, device="cuda", requires_grad=True)
        self.sim.set_init_state(wp_x0, wp_v0)

        ctrl = torch.tensor(init_pkl["ctrl_pts"], dtype=torch.float32, device="cuda")
        self.sim.wp_original_control_point = wp.from_torch(ctrl, dtype=wp.vec3, requires_grad=False)
        self.sim.wp_target_control_point   = wp.clone(self.sim.wp_original_control_point, requires_grad=True)

        # 弹簧
        springs_np = init_pkl.get("spring_indices", None)
        if springs_np is None:
            raise ValueError("init_pkl missing 'spring_indices'")
        self.springs_wp = wp.array(springs_np.reshape(-1).astype(np.int32), dtype=wp.int32, device="cuda")

        # —— 左右分组（使用真实 ctrl）——
        L, R = self.env.split_ctrl_pts_kmeans(ctrl)
        if not isinstance(L, (list, tuple)) or len(L) == 0 or len(R) == 0:
            raise RuntimeError("KMeans returned empty cluster(s)")

        self.left_idx, self.right_idx = L.detach().cpu().numpy(), R.detach().cpu().numpy()

        # 构建/更新 warp mask
        left_mask  = torch.zeros(ctrl.shape[0], dtype=torch.int32, device="cuda")
        right_mask = torch.zeros_like(left_mask)
        left_mask[L] = 1; right_mask[R] = 1
        self.left_wp_mask  = wp.array(left_mask.detach().cpu().numpy(), dtype=wp.int32, device="cuda")
        self.right_wp_mask = wp.array(right_mask.detach().cpu().numpy(), dtype=wp.int32, device="cuda")


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
        """纯前向 rollout H 步；可选记录状态"""
        H = action_seq.shape[0] // 2
        frames = []
        if record:
            frames.append(wp.to_torch(self.sim.wp_states[-1].wp_x).detach().cpu().numpy())
        for j in range(H):
            a2 = action_seq[2*j:2*j+2]
            df = self._expand_lr_to_full(a2, scale)
            self.sim.one_step_from_action(df, is_first_step=(j==0))
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

            # Tape 区：H 步可导 rollout
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

                # loss（相对几何）
                loss_t = mpc_loss_shape_relative(self.sim, self.springs_wp,
                                                 w_action=opts.w_action,
                                                 action_seq_wp=a_seq_wp)

            # backward（到 warp）
            try:
                tape.backward(self.sim.loss)
            except Exception as e:
                print(f"[MPC][it={it}] tape.backward error: {e}")
                break

            # warp grad → torch grad
            action.param.grad = wp.to_torch(a_seq_wp.grad)

            # 数值保护
            if action.param.grad is None or not torch.isfinite(action.param.grad).all():
                print(f"[MPC] grad invalid at it={it}, skip update")
            else:
                # 行归一化步长
                rowwise_normalized_step(action.param, action.param.grad, opts.step_row)
                # pre-tanh 限幅
                action.clamp_pre_tanh_(opts.pre_tanh_clip)
                # 清梯度
                action.param.grad = None

            # 评估一次 loss（纯前向）
            with torch.no_grad():
                # 重置到优化前初态：当前 sim.wp_states[0] 已是 rollout 前的初态
                # 简化处理：直接再 roll 一次（Tape 外）
                # （更严谨可在进入 optimize_online 前抓一次 snapshot 并恢复）
                pass

            # 读取标量 loss
            curr = float(loss_t.detach().cpu())
            best = min(best, curr)

            if curr < opts.early_tol:
                print(f"[MPC] early stop at it={it}, loss={curr:.6f}")
                break


            # 读取 loss
            curr = float(loss_t.detach().cpu())

            # 打印 loss 与数值健康状况
            if (it % 1) == 0:
                x_last = wp.to_torch(self.sim.wp_states[-1].wp_x).detach()
                nan_x = torch.isnan(x_last).any().item()
                inf_x = torch.isinf(x_last).any().item()
                g = action.param.grad
                g_none = (g is None)
                g_finite = (False if g_none else torch.isfinite(g).all().item())
                g_norm = (0.0 if (g_none or not g_finite) else float(g.norm().detach().cpu()))
                p_norm = float(action.param.detach().norm().cpu())
                print(f"[MPC][it={it}] loss={curr:.6e} | grad_none={g_none} finite={g_finite} "
                    f"| gnorm={g_norm:.3e} | |param|={p_norm:.3e} | x[nan]={nan_x} x[inf]={inf_x}")

            # 如果 grad 是 None，进一步检查 a_seq_wp 是否 requires_grad、Tape 是否记录到了内核
            if action.param.grad is None:
                print(f"[MPC][it={it}] a_seq_wp.requires_grad={a_seq_wp.requires_grad}")
                # 简单扰动参数做一次前向，验证 loss 是否变化（数值梯度探针）
                with torch.no_grad():
                    eps = 1e-3
                    idx = 0
                    action.param[idx, 0] += eps
                    # 恢复初态
                    self.sim.set_init_state(wp.array(x0_np, dtype=wp.vec3f, device='cuda', requires_grad=True),
                                            wp.array(v0_np, dtype=wp.vec3f, device='cuda', requires_grad=True))
                    df_list = []
                    a_seq_wp2 = wp.from_torch(action.param, dtype=wp.vec3, requires_grad=False)
                    for j in range(opts.horizon):
                        dlr2 = wp.zeros(2, dtype=wp.vec3, device="cuda", requires_grad=False)
                        wp.launch(self.sim.copy_row_vec3, dim=2, inputs=[a_seq_wp2, j, 2], outputs=[dlr2])
                        df2 = wp.zeros(self.sim.num_control_points, dtype=wp.vec3, device="cuda", requires_grad=False)
                        wp.launch(self.sim.expand_row2_squash_scale, dim=self.sim.num_control_points,
                                inputs=[dlr2, self.left_wp_mask, self.right_wp_mask, float(opts.max_delta)], outputs=[df2])
                        df_list.append(df2)
                    self.sim.rollout(df_list)
                    loss_probe = mpc_loss_shape_relative(self.sim, self.springs_wp, w_action=0.0, action_seq_wp=None)
                    print(f"[MPC][it={it}] loss probe (param+eps) = {float(loss_probe.detach().cpu()):.6e}")
                    # 撤销扰动
                    action.param[idx, 0] -= eps


        return best

    def run_segment(self, init_pkl: dict, target_pkl: dict,
                    opts: MPCOpts, exec_last_horizon: bool = True,
                    save_path: Path = None):
        """
        最小可用：在一个段上在线优化；非最后一轮由外层调用时只执行 1 步；
        这里提供“执行整段 H 步并保存 rollout.pkl”的便捷接口（用于最后一轮）。
        """
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
