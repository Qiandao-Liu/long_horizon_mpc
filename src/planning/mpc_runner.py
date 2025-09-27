# src/planning/mpc_runner.py
"""
python -m src.planning.mpc_runner --task_name task11 --frames_per_step 5 --horizon 10
"""
import json, pickle, math, sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
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

from src.planning.gradient_core import MPCRunner, MPCOpts
from src.planning.action_param import ActionParam, rowwise_normalized_step

# ------------------------------ I/O helpers ------------------------------

def _load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)

def _save_pickle(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(obj, f)

def _task_paths(task_name: str) -> Dict[str, Path]:
    base = DATA_DIR / "tasks" / task_name
    return dict(
        base=base,
        demo=base / "demo_rollout" / "demo.pkl",
        milestones=base / "segmented_status" / "milestones.json",
        out_dir=base / "mpc_rollout",
    )


# ------------------------------ Segment plan ------------------------------

@dataclass
class SegmentPlan:
    seg_id: int
    t_start: int
    t_goal: int
    steps_required: int  # S
    rounds: int          # R
    last_exec_steps: int # 最后一轮实际执行的步数：min(H, S-(R-1))

def build_segment_plans(milestones: List[int], frames_per_step: int, horizon: int) -> List[SegmentPlan]:
    plans: List[SegmentPlan] = []
    for k in range(len(milestones) - 1):
        t0, t1 = milestones[k], milestones[k+1]
        delta = max(0, int(t1 - t0))
        S = int(math.ceil(delta / float(frames_per_step)))  # 向上取整，保证到达
        if S <= 0:
            S = 1
        if S <= horizon:
            R = 1
            last_exec = S
        else:
            # 前 R-1 轮各执行 1 步，最后一轮执行 horizon 步
            R = S - (horizon - 1)
            last_exec = min(horizon, S - (R - 1))
        plans.append(SegmentPlan(
            seg_id=k, t_start=t0, t_goal=t1,
            steps_required=S, rounds=R, last_exec_steps=last_exec
        ))
    return plans


# ------------------------------ Demo frame → pkls ------------------------------

def make_init_from_demo(demo: dict, t: int) -> dict:
    """从 demo.pkl 的第 t 帧构造 init_pkl 形状"""
    wp_x = np.asarray(demo["wp_x_seq"][t], dtype=np.float32)
    # 如果没有速度序列，就置零
    if "wp_v_seq" in demo:
        wp_v = np.asarray(demo["wp_v_seq"][t], dtype=np.float32)
    else:
        wp_v = np.zeros_like(wp_x, dtype=np.float32)
    ctrl = np.asarray(demo["ctrl_seq"][t], dtype=np.float32)
    springs = np.asarray(demo["spring_indices"], dtype=np.int32)
    return {
        "wp_x": wp_x,
        "wp_v": wp_v,
        "ctrl_pts": ctrl,
        "spring_indices": springs,
    }

def make_target_from_demo(demo: dict, t: int) -> dict:
    """把第 t 帧位置当作目标（相对形状 loss 用边长，不关心绝对位姿）"""
    return {"object_points": np.asarray(demo["wp_x_seq"][t], dtype=np.float32)}


# ------------------------------ Execution helpers ------------------------------

def _expand_lr_to_full(runner: MPCRunner, a2x3: torch.Tensor, scale: float):
    """便捷封装：左右 2×3 → full 控制场（warp 数组）"""
    a_wp = wp.from_torch(a2x3, dtype=wp.vec3, requires_grad=False)
    dfull = wp.zeros(runner.sim.num_control_points, dtype=wp.vec3, device="cuda", requires_grad=False)
    wp.launch(runner.sim.expand_row2_squash_scale, dim=runner.sim.num_control_points,
              inputs=[a_wp, runner.left_wp_mask, runner.right_wp_mask, float(scale)], outputs=[dfull])
    return dfull

def _execute_k_steps_and_record(runner: MPCRunner, action_param: torch.Tensor,
                                k_steps: int, scale: float, record: bool):
    """执行前 k_steps；若 record=True 则返回 frames 序列（含起点）"""
    frames = []
    if record:
        frames.append(wp.to_torch(runner.sim.wp_states[-1].wp_x).detach().cpu().numpy())
    for s in range(k_steps):
        a2 = action_param[2*s:2*s+2]
        df = _expand_lr_to_full(runner, a2, scale)
        runner.sim.one_step_from_action(df, is_first_step=(s == 0))
        if record:
            frames.append(wp.to_torch(runner.sim.wp_states[-1].wp_x).detach().cpu().numpy())
    return frames if record else None


# ------------------------------ Main multi-segment MPC ------------------------------

def run_task_multiseg(task_name: str,
                      case_name: str = "double_lift_cloth_1",
                      frames_per_step: int = 5,
                      horizon: int = 10,
                      max_iters: int = 30,
                      max_delta: float = 0.25,
                      step_row: float = 3e-2,
                      pre_tanh_clip: float = 3.0,
                      w_action: float = 0.0,
                      early_tol: float = 1e-3):
    paths = _task_paths(task_name)
    demo = _load_pickle(paths["demo"])
    with open(paths["milestones"], "r") as f:
        milestones = json.load(f)["milestones"]

    plans = build_segment_plans(milestones, frames_per_step, horizon)
    print(f"[PLAN] milestones={milestones} | frames_per_step={frames_per_step} | horizon={horizon}")
    for p in plans:
        print(f"  - seg{p.seg_id}: {p.t_start}->{p.t_goal} | steps S={p.steps_required} | rounds R={p.rounds} | last_exec={p.last_exec_steps}")

    # Runner + 统一 opts
    runner = MPCRunner(case_name=case_name, data_dir=DATA_DIR)
    opts = MPCOpts(
        horizon=horizon, max_iters=max_iters,
        max_delta=max_delta, step_row=step_row,
        pre_tanh_clip=pre_tanh_clip, w_action=w_action,
        early_tol=early_tol,
    )

    out_dir = paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 汇总容器
    final_frames: List[np.ndarray] = []
    final_actions_2x3: List[np.ndarray] = []
    seg_summaries: List[Dict] = []

    # —— 逐段执行 —— #
    for plan in plans:
        print(f"\n[SEG{plan.seg_id}] start={plan.t_start} goal={plan.t_goal}  S={plan.steps_required}  R={plan.rounds}")

        # 段起点/目标
        init_pkl = make_init_from_demo(demo, plan.t_start)
        tgt_pkl  = make_target_from_demo(demo, plan.t_goal)

        # 放入 sim
        runner.set_init_from_pkl(init_pkl)
        runner.set_target_from_pkl(tgt_pkl)

        # 每轮在线优化
        remaining = plan.steps_required
        seg_frames_all: List[np.ndarray] = []
        seg_actions_2x3: List[np.ndarray] = []

        for r in range(1, plan.rounds + 1):
            # 新建动作参数（H×），在线优化
            action = ActionParam(H=opts.horizon, device="cuda")
            _ = runner.optimize_online(opts, action)

            if r < plan.rounds:
                # —— 非最后一轮：只执行 1 步 —— #
                k = 1
                frames = _execute_k_steps_and_record(runner, action.param.detach(), k_steps=k,
                                                     scale=opts.max_delta, record=False)
                remaining -= k

                # 记录动作（2×3）
                seg_actions_2x3.append(action.param[:2].detach().cpu().numpy())

                # Warm start 思想体现在下轮“重新优化”的初值；这里简单重来
                # 更新 init：把当前 sim 的 -1 帧设置为下一轮起点
                # （注意：set_init_from_pkl 会重置 wp_states，准备下一轮优化）
                with torch.no_grad():
                    wp_x = wp.to_torch(runner.sim.wp_states[-1].wp_x).detach().cpu().numpy()
                    wp_v = wp.to_torch(runner.sim.wp_states[-1].wp_v).detach().cpu().numpy()
                init_pkl = {
                    "wp_x": wp_x,
                    "wp_v": wp_v,
                    "ctrl_pts": init_pkl["ctrl_pts"],  # 控制点原样即可（相对形状 loss 不依赖绝对位置）
                    "spring_indices": init_pkl["spring_indices"],
                }
                runner.set_init_from_pkl(init_pkl)  # 为下一轮优化准备
                # 目标不变
                runner.set_target_from_pkl(tgt_pkl)

            else:
                # —— 最后一轮：执行 last_exec_steps（通常=H）并记录 —— #
                k = int(plan.last_exec_steps)
                # 将动作裁剪为前 k 步
                act_k = action.param[:2*k].detach()
                frames = _execute_k_steps_and_record(runner, act_k, k_steps=k,
                                                     scale=opts.max_delta, record=True)
                remaining -= k

                # 记录动作与帧
                seg_actions_2x3.append(act_k.detach().cpu().numpy())
                seg_frames_all.extend(frames)  # frames 是 [k+1] 张，直接串起来

                # 段 rollout 存盘
                seg_out = {
                    "seg_id": plan.seg_id,
                    "t_start": plan.t_start,
                    "t_goal": plan.t_goal,
                    "frames": np.asarray(seg_frames_all, dtype=np.float32),     # [K_total+1, N, 3]
                    "actions_2x3": np.asarray(seg_actions_2x3, dtype=np.float32),  # (num_exec_rounds, 2, 3) 累积
                    "frames_per_step": frames_per_step,
                    "horizon": horizon,
                    "max_iters": max_iters,
                    "max_delta": float(max_delta),
                }
                seg_path = out_dir / f"seg{plan.seg_id}_rollout.pkl"
                _save_pickle(seg_out, seg_path)
                print(f"[SEG{plan.seg_id}] saved {seg_path}")

                # 汇入总结果
                # 注意：拼接时避免重复起点，取本段 frames 全部，跨段由下一段接着追加
                if len(final_frames) == 0:
                    final_frames.extend(seg_out["frames"])
                else:
                    # 去掉与上一段末帧重复的首帧
                    final_frames.extend(seg_out["frames"][1:])
                final_actions_2x3.append(np.asarray(seg_actions_2x3, dtype=np.float32))

                seg_summaries.append({
                    "seg_id": plan.seg_id,
                    "steps_required": plan.steps_required,
                    "rounds": plan.rounds,
                    "executed_last_steps": k,
                })

        assert remaining == 0, f"Segment {plan.seg_id} remaining steps={remaining} (logic bug)"

    # —— 汇总 final —— #
    final = {
        "milestones": milestones,
        "frames": np.asarray(final_frames, dtype=np.float32),
        "actions_2x3": np.concatenate(final_actions_2x3, axis=0) if len(final_actions_2x3)>0 else np.zeros((0,2,3), dtype=np.float32),
        "frames_per_step": frames_per_step,
        "horizon": horizon,
        "max_iters": max_iters,
        "max_delta": float(max_delta),
        "segments": seg_summaries,
    }
    final_path = paths["out_dir"] / "final_rollout.pkl"
    _save_pickle(final, final_path)
    print(f"\n[FINAL] saved {final_path}")
    print(f"[FINAL] total segments={len(plans)} | total frames={final['frames'].shape[0]} | total executed rounds={sum(s['rounds'] for s in seg_summaries)}")


# ------------------------------ CLI ------------------------------

def main(task_name: str = "task11",
         case_name: str = "double_lift_cloth_1",
         frames_per_step: int = 5,
         horizon: int = 10,
         max_iters: int = 30,
         max_delta: float = 0.25,
         step_row: float = 3e-2,
         pre_tanh_clip: float = 3.0,
         w_action: float = 0.0,
         early_tol: float = 1e-3):
    run_task_multiseg(
        task_name=task_name,
        case_name=case_name,
        frames_per_step=frames_per_step,
        horizon=horizon,
        max_iters=max_iters,
        max_delta=max_delta,
        step_row=step_row,
        pre_tanh_clip=pre_tanh_clip,
        w_action=w_action,
        early_tol=early_tol,
    )

if __name__ == "__main__":
    # 你也可以把这些参数做成 argparse；这里给一个默认可跑配置
    main(
        task_name="task11",
        case_name="double_lift_cloth_1",
        frames_per_step=5,
        horizon=10,
        max_iters=30,
        max_delta=0.25,
        step_row=3e-2,
        pre_tanh_clip=3.0,
        w_action=1e-4,
        early_tol=1e-3,
    )
