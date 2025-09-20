# workspace/src/planning/gradient_mpc.py
import os, sys, pickle, math, time
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PhysTwin")))

from src.env.phystwin_env import PhysTwinEnv
import warp as wp

@torch.no_grad()
def _export_state_for_outer(sim):
    """从“真·world”sim 导出下一轮 inner-iter 的干净起点(CPU numpy方便跨对象传递)"""
    x0 = wp.to_torch(sim.wp_states[0].wp_x).detach().cpu().numpy()
    v0 = wp.to_torch(sim.wp_states[0].wp_v).detach().cpu().numpy()
    orig_cp = wp.to_torch(sim.wp_original_control_point).detach().cpu().numpy()
    return {"wp_x": x0, "wp_v": v0, "ctrl_pts": orig_cp}

def _build_eval_env(case_name, base_state, target_data, left_idx, right_idx):
    """重建一个“评估用”的全新 env + sim 并把起点/目标/mask 塞好（不做任何共享）"""
    env = PhysTwinEnv(case_name)
    sim = env.simulator

    # init & target
    wp_x = wp.array(base_state["wp_x"], dtype=wp.vec3f, device="cuda", requires_grad=True)
    wp_v = wp.array(base_state["wp_v"], dtype=wp.vec3f, device="cuda", requires_grad=True)
    sim.set_init_state(wp_x, wp_v)

    init_ctrl = torch.tensor(base_state["ctrl_pts"], dtype=torch.float32, device="cuda")
    sim.wp_original_control_point = wp.from_torch(init_ctrl, dtype=wp.vec3, requires_grad=False)
    sim.wp_target_control_point   = wp.clone(sim.wp_original_control_point, requires_grad=True)

    target_gs = torch.tensor(target_data["object_points"], dtype=torch.float32, device="cuda")
    sim.wp_current_object_points  = wp.from_torch(target_gs, dtype=wp.vec3, requires_grad=False)

    # masks（固定左右分组，不随 outer 改）
    left_mask  = torch.zeros(init_ctrl.shape[0], dtype=torch.int32, device="cuda")
    right_mask = torch.zeros_like(left_mask)
    left_mask[left_idx]  = 1
    right_mask[right_idx] = 1
    left_wp_mask  = wp.array(left_mask.detach().cpu().numpy(), dtype=wp.int32, device="cuda")
    right_wp_mask = wp.array(right_mask.detach().cpu().numpy(), dtype=wp.int32, device="cuda")

    # 可见性/motion-valid 兜底
    if not hasattr(sim, "wp_current_object_visibilities") or sim.wp_current_object_visibilities is None:
        num_surface = target_gs.shape[0]
        sim.wp_current_object_visibilities = wp.array(
            torch.ones(num_surface, dtype=torch.int32, device="cuda").cpu().numpy(), dtype=wp.int32, device="cuda"
        )
        sim.num_valid_visibilities = int(num_surface)

    if not hasattr(sim, "wp_current_object_motions_valid") or sim.wp_current_object_motions_valid is None:
        num_orig = getattr(sim, "num_original_points", target_gs.shape[0])
        sim.wp_current_object_motions_valid = wp.array(
            torch.ones(num_orig, dtype=torch.int32, device="cuda").cpu().numpy(), dtype=wp.int32, device="cuda"
        )
        sim.num_valid_motions = int(num_orig)

    return env, sim, left_wp_mask, right_wp_mask

# 可视化 & 保存可视化
def _wp_vec3_to_torch(wp_arr):
    # wp.array(vec3) -> torch.float32 [N,3] (cuda) -> cpu
    return wp.to_torch(wp_arr).detach().cpu()

# 可视化 & 保存可视化
@torch.no_grad()
def debug_visualize_rollout(case_name, base_state, target_data, left_idx, right_idx,
                            action_seq, max_delta=None, tag="debug"):
    """
    纯前向 roll 出 horizon 步，保存：
      - frames:        每步后的 cloth 点云 [T, N, 3]
      - ctrl_frames:   每步控制点的轨迹    [T, C, 3]  （按「命令的增量」累积得到）
      - a_full_all:    每步展开后的 full 控制场 [H, C, 3]
      - a0, a0_full, scale, left_idx, right_idx 供可视化
    """
    import numpy as np
    os.makedirs("PhysTwin/mpc_logs", exist_ok=True)

    # 1) 重建 eval_env
    env, sim, lm, rm = _build_eval_env(case_name, base_state, target_data, left_idx, right_idx)
    scale = getattr(sim, "MAX_DELTA", 1e-5) if max_delta is None else float(max_delta)

    # 2) 组动作（纯前向）
    a_seq_wp = wp.from_torch(action_seq, dtype=wp.vec3, requires_grad=False)
    horizon = action_seq.shape[0] // 2

    action_slice = []
    for j in range(horizon):
        dlr = wp.zeros(2, dtype=wp.vec3, device="cuda", requires_grad=False)
        wp.launch(sim.copy_row_vec3, dim=2, inputs=[a_seq_wp, j, 2], outputs=[dlr])

        df = wp.zeros(sim.num_control_points, dtype=wp.vec3, device="cuda", requires_grad=False)
        wp.launch(sim.expand_row2_squash_scale, dim=sim.num_control_points,
                  inputs=[dlr, lm, rm, scale], outputs=[df])
        action_slice.append(df)

    # 3) 布料点云轨迹
    frames = []
    # 初始帧：start-of-rollout 的起点（保持读取 [0]）
    frames.append(_wp_vec3_to_torch(sim.wp_states[0].wp_x).numpy())

    for j in range(horizon):
        sim.one_step_from_action(action_slice[j], is_first_step=(j == 0))
        # ⬇️ 关键改动：记录“本步的结果”，即 [-1]
        frames.append(_wp_vec3_to_torch(sim.wp_states[-1].wp_x).numpy())

    # 4) 控制点轨迹（按命令的 full 控制场累积）
    #    初始控制点 = base_state["ctrl_pts"]，每一步加 action_slice[j]
    ctrl0 = torch.tensor(base_state["ctrl_pts"], dtype=torch.float32).cpu().numpy()  # [C,3]
    a_full_all = [ _wp_vec3_to_torch(df).numpy() for df in action_slice ]            # H × [C,3]
    ctrl_frames = [ctrl0.copy()]
    acc = ctrl0.copy()
    for j in range(horizon):
        acc = acc + a_full_all[j]
        ctrl_frames.append(acc.copy())
    ctrl_frames = np.asarray(ctrl_frames, dtype=np.float32)   # [H+1, C, 3]
    a_full_all = np.asarray(a_full_all, dtype=np.float32)     # [H, C, 3]

    # 5) 位移统计（cloth）
    disp_mean, disp_max = [], []
    for t in range(1, len(frames)):
        d = frames[t] - frames[t-1]
        n = np.linalg.norm(d, axis=1)
        disp_mean.append(float(n.mean()))
        disp_max.append(float(n.max()))

    # 6) 也存下第一步信息
    a0 = action_seq[0:2].detach().cpu().numpy()
    a0_full = _wp_vec3_to_torch(action_slice[0]).numpy()

    out_path = os.path.join("PhysTwin", "mpc_logs", f"debug_{tag}.npz")
    np.savez_compressed(
        out_path,
        frames=np.asarray(frames, dtype=np.float32),        # [H+1, N, 3]
        ctrl_frames=ctrl_frames,                            # [H+1, C, 3]
        a_full_all=a_full_all,                              # [H,   C, 3]
        disp_mean=np.asarray(disp_mean, dtype=np.float32),
        disp_max=np.asarray(disp_max, dtype=np.float32),
        action_seq=action_seq.detach().cpu().numpy(),       # [H*2, 3]
        a0=a0, a0_full=a0_full,
        scale=float(scale),
        left_idx=np.asarray(left_idx, dtype=np.int32),
        right_idx=np.asarray(right_idx, dtype=np.int32),
    )
    print(f"[DEBUG-VIS] saved {out_path} | "
          f"cloth disp_mean[0..4]={disp_mean[:5]} | disp_max[0..4]={disp_max[:5]}")


def main(
    case_name="double_lift_cloth_1",
    init_idx=1,
    target_idx=2,
    max_num_actions=10,
    iteration=32,
    horizon=1,
    lr=1e-4,
    lam_action=1e-3,
    grad_clip=1.0,
    max_delta=None,   # 如果 None 就用 sim.MAX_DELTA
):
    # 1) 载入数据
    init_path   = f"PhysTwin/mpc_init/init_{init_idx:03d}.pkl"
    target_path = f"PhysTwin/mpc_target_U/target_{target_idx:03d}.pkl"
    with open(init_path, "rb") as f:
        init_data = pickle.load(f)
    with open(target_path, "rb") as f:
        target_data = pickle.load(f)

    # 2) 真·world env：仅用于“外层真实推进一步”
    truth_env = PhysTwinEnv(case_name)
    sim_true  = truth_env.simulator

    # 用初始 pkl 打好起点
    wp_x0 = wp.array(init_data["wp_x"], dtype=wp.vec3f, device="cuda")
    wp_v0 = wp.zeros_like(wp_x0, requires_grad=True)  # 初速为 0（评估 env 要可导，这里无所谓）
    sim_true.set_init_state(wp_x0, wp_v0)

    init_ctrl = torch.tensor(init_data["ctrl_pts"], dtype=torch.float32, device="cuda")
    truth_env.left_idx, truth_env.right_idx = truth_env.split_ctrl_pts_dbscan(init_ctrl)
    sim_true.wp_original_control_point = wp.from_torch(init_ctrl, dtype=wp.vec3, requires_grad=False)
    sim_true.wp_target_control_point = wp.clone(sim_true.wp_original_control_point, requires_grad=False)

    target_gs = torch.tensor(target_data["object_points"], dtype=torch.float32, device="cuda")
    sim_true.wp_current_object_points  = wp.from_torch(target_gs, dtype=wp.vec3, requires_grad=False)

    # masks index（只算一次）
    left_idx, right_idx = truth_env.left_idx, truth_env.right_idx

    # 可视化
    truth_env.visualize_initial_vs_target(init_data, target_data)

    early_tol = 1e-3           # ✅ 早停阈值
    early_stop = False         # ✅ 标记位

    # 3) 待优化的动作序列（torch Param）
    action_seq = torch.zeros(horizon * 2, 3, device="cuda", dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([action_seq], lr=lr)

    # 4) 外层循环
    print(f"[STAT-MPC] max_num_actions={max_num_actions}, iteration={iteration}, horizon={horizon}, lr={lr}, lam={lam_action}, max_delta={max_delta}")
    base_state = _export_state_for_outer(sim_true)

    for outer in range(max_num_actions):
        # inner: 每次都重建“评估用”env，确保没有残留
        best_loss = float("inf")
        t_outer0 = time.time()

        for it in range(iteration):
            # --- 重建评估 env ---
            eval_env, sim_eval, left_wp_mask, right_wp_mask = _build_eval_env(
                case_name, base_state, target_data, left_idx, right_idx
            )

            # 选 MAX_DELTA
            scale = getattr(sim_eval, "MAX_DELTA", 1e-5) if max_delta is None else float(max_delta)
            print(f"[STAT-MPC] outer={outer:03d} it={it:03d} scale={scale}")

            # --- 前向 + 反传（Warp） ---
            with wp.Tape() as tape:
                # torch → warp 视图（这一步每 iter 都新建，Tape 只看 warp）
                a_seq_wp = wp.from_torch(action_seq, dtype=wp.vec3, requires_grad=True)

                # 组装 rollout 动作：每步把 (2,3) 展成 (num_ctrl,3) 并 tanh 限幅再 * scale
                action_slice = []
                for j in range(horizon):
                    delta_lr = wp.zeros(2, dtype=wp.vec3, device="cuda", requires_grad=True)
                    wp.launch(sim_eval.copy_row_vec3, dim=2, inputs=[a_seq_wp, j, 2], outputs=[delta_lr])

                    delta_full = wp.zeros(sim_eval.num_control_points, dtype=wp.vec3, device="cuda", requires_grad=True)
                    wp.launch(sim_eval.expand_row2_squash_scale, dim=sim_eval.num_control_points,
                              inputs=[delta_lr, left_wp_mask, right_wp_mask, scale], outputs=[delta_full])
                    action_slice.append(delta_full)
                    
                    if j == (horizon-1):
                        print(f"[DEBUG] step {j} wp_status[] =", wp.to_torch(delta_full).detach().cpu().numpy())

                sim_eval.rollout(action_slice)  # rollout 整个 horizon

                print(f"[STAT-MPC] outer={outer:03d} it={it:03d} rollout done")
                # loss_t = sim_eval.calculate_mpc_loss_simple(action_seq_wp=a_seq_wp, lam=lam_action)
                loss_t = sim_eval.calculate_mpc_loss(w_chamfer=1.0, w_track=0.2, lambda_drift=1e-2)
                print(f"[STAT-MPC] outer={outer:03d} it={it:03d} loss computed = {float(loss_t.detach().cpu()):.6f}")

            # backward 到 warp
            tape.backward(sim_eval.loss)
            print(f"[STAT-MPC] outer={outer:03d} it={it:03d} backward done")

            # warp grad → torch grad
            action_seq.grad = wp.to_torch(a_seq_wp.grad)
            if action_seq.grad is None:
                print(f"[STAT-MPC] outer={outer:03d} it={it:03d} grad=None")
            else:
                gnorm = float(action_seq.grad.norm().detach().cpu())
                print(f"[STAT-MPC] outer={outer:03d} it={it:03d} grad_norm={gnorm:.3e}")

            # warp grad → torch grad
            action_seq.grad = wp.to_torch(a_seq_wp.grad)

            if action_seq.grad is not None and torch.isfinite(action_seq.grad).all():
                with torch.no_grad():
                    g = action_seq.grad

                    # === 逐行归一化（每个时间步都有“固定步长”）===
                    step_row = 3e-2
                    for r in range(action_seq.shape[0]):  # (H*2, 3)
                        gr = g[r]
                        nr = gr.norm() + 1e-12
                        action_seq[r].add_(-step_row * gr / nr)

                    # 限制 pre-tanh 不进饱和（避免 tanh'(x)≈0）
                    action_seq.clamp_(-3.0, 3.0)

                # 清梯度（我们没用 optimizer 了）
                action_seq.grad = None
            else:
                print(f"[STAT-MPC] outer={outer} it={it} grad skipped (non-finite)")

            # **用 Adam 更新**
            # optimizer.step()
            # optimizer.zero_grad(set_to_none=True)

            # 记录
            loss_val = float(loss_t.detach().cpu()) if torch.isfinite(loss_t) else float("inf")
            best_loss = min(best_loss, loss_val)

            # 早停
            if loss_val < early_tol:
                print(f"[STAT-MPC] 🎉 Early stop at outer={outer:03d} it={it:03d}, loss={loss_val:.6f} < {early_tol}")
                # 最后可视化一次
                with torch.no_grad():
                    debug_visualize_rollout(
                        case_name, base_state, target_data, left_idx, right_idx,
                        action_seq=action_seq, max_delta=max_delta,
                        tag=f"earlystop_outer{outer:03d}_it{it:03d}"
                    )
                early_stop = True
                break

            # 额外诊断：当前参数下，动作向量的范数
            with torch.no_grad():
                act_norm = float(action_seq.norm().detach().cpu())

            # 额外诊断：复评估一次 step 后的 loss（用全新 eval_env，保证纯前向）
            re_eval_loss = None
            with torch.no_grad():
                _env2, _sim2, _lm2, _rm2 = _build_eval_env(case_name, base_state, target_data, left_idx, right_idx)
                a_seq_wp2 = wp.from_torch(action_seq, dtype=wp.vec3, requires_grad=False)
                action_slice2 = []
                scale2 = getattr(_sim2, "MAX_DELTA", 1e-5) if max_delta is None else float(max_delta)
                for j in range(horizon):
                    dlr2 = wp.zeros(2, dtype=wp.vec3, device="cuda", requires_grad=False)
                    wp.launch(_sim2.copy_row_vec3, dim=2, inputs=[a_seq_wp2, j, 2], outputs=[dlr2])
                    df2 = wp.zeros(_sim2.num_control_points, dtype=wp.vec3, device="cuda", requires_grad=False)
                    wp.launch(_sim2.expand_row2_squash_scale, dim=_sim2.num_control_points,
                            inputs=[dlr2, _lm2, _rm2, scale2], outputs=[df2])
                    action_slice2.append(df2)
                _sim2.rollout(action_slice2)
                # re_eval_loss_t = _sim2.calculate_mpc_loss_simple(action_seq_wp=a_seq_wp2, lam=lam_action)
                re_eval_loss_t = _sim2.calculate_mpc_loss(w_chamfer=1.0, w_track=0.2, lambda_drift=1e-2)
                re_eval_loss = float(re_eval_loss_t.detach().cpu())
                del _env2, _sim2, _lm2, _rm2, a_seq_wp2
                torch.cuda.empty_cache()

            # 打印更高精度 + 新增指标
            if it % max(1, iteration // 8) == 0 or it == iteration - 1:
                gnorm = (float(action_seq.grad.norm().detach().cpu()) 
                        if action_seq.grad is not None else 0.0)
                print(
                    f"[STAT-MPC] outer={outer:03d} it={it:03d} "
                    f"loss={loss_val:.12f} re_eval={re_eval_loss:.12f} "
                    f"|Δact|={act_norm:.3e} grad_norm={gnorm:.3e}"
                )

            # 可视化
            if (it % 8) == 0:
                debug_visualize_rollout(
                    case_name, base_state, target_data, left_idx, right_idx,
                    action_seq=action_seq, max_delta=max_delta,
                    tag=f"outer{outer:03d}_it{it:03d}"
                )

            # 释放评估 env，防止显存堆积
            del eval_env, sim_eval, left_wp_mask, right_wp_mask, a_seq_wp
            torch.cuda.empty_cache()

        if early_stop:
            break

        # --- 外层执行第一步动作（真·world env） ---
        # 把动作第 0 步拿出来，展开成 (num_ctrl,3) 然后走一步
        eval_env, sim_eval, left_wp_mask, right_wp_mask = _build_eval_env(
            case_name, base_state, target_data, left_idx, right_idx
        )
        scale = getattr(sim_eval, "MAX_DELTA", 1e-5) if max_delta is None else float(max_delta)
        with torch.no_grad():
            a0 = action_seq[0:2].contiguous()  # (2,3)
            a0_wp = wp.from_torch(a0, dtype=wp.vec3, requires_grad=False)
            delta_full0 = wp.zeros(sim_true.num_control_points, dtype=wp.vec3, device="cuda", requires_grad=False)
            wp.launch(sim_eval.expand_row2_squash_scale, dim=sim_true.num_control_points,
                      inputs=[a0_wp, left_wp_mask, right_wp_mask, scale], outputs=[delta_full0])

        # 用真·world sim 执行
        sim_true.one_step_from_action(delta_full0, is_first_step=True)

        # 更新下轮起点
        base_state = _export_state_for_outer(sim_true)

        # 动作 warm start：左移一格，末尾置零
        with torch.no_grad():
            if horizon > 1:
                prev = action_seq.data.clone()
                action_seq.data.copy_(
                    torch.cat([prev[2:], torch.zeros_like(prev[:2])], dim=0)
                )
            else:
                action_seq.data.zero_()

        del eval_env, sim_eval, left_wp_mask, right_wp_mask, a0_wp, delta_full0
        torch.cuda.empty_cache()

        dt = time.time() - t_outer0
        print(f"[STAT-MPC] >>> outer={outer:03d} done  best_inner_loss={best_loss:.6f}  took={dt:.2f}s")

if __name__ == "__main__":
    main(
        case_name="double_lift_cloth_1",
        init_idx=0,
        target_idx=1,
        max_num_actions=30,
        iteration=64,
        horizon=10,
        lr=1e-4,
        lam_action=0,
        grad_clip=1.0,
        max_delta=0.25,
    )
