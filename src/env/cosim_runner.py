# /src/env/cosim_runner.py
"""
python -m src.env.cosim_runner --case double_lift_cloth_1
"""
import time
import os
from pathlib import Path
import numpy as np
import torch
import warp as wp
import mujoco
import mujoco.viewer as mj_viewer

from .mujoco_loader import load_scene, resolve_ur5e_xml
from .phystwin_env import PhysTwinEnv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]  # long_horizon_mpc/


def get_site_pose(model, data, site_name):
    """返回世界系下的 site 位姿 (pos[3], R[3x3])."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    pos = data.site_xpos[sid].copy()                    # (3,)
    R = data.site_xmat[sid].reshape(3, 3).copy()        # (3,3), local->world
    return pos, R


def two_fingers_from_attachment(model, data, site_name="attachment_site", half_gap=0.03):
    """
    以 ur5e.xml 中的 attachment_site 为中心，沿其局部 y 轴 ±half_gap 生成左右“手指”世界坐标。
    返回 (p_left, p_right) 两个 (3,) numpy 数组。
    """
    pos, R = get_site_pose(model, data, site_name)
    y_axis_world = R[:, 1]   # 列 1 是局部 y 轴映射到世界
    pL = pos + half_gap * y_axis_world
    pR = pos - half_gap * y_axis_world
    return pL, pR


def build_ctrl_target_from_two_points(env: PhysTwinEnv, pL_world: np.ndarray, pR_world: np.ndarray):
    """
    给定左右抓取点的世界坐标，构建一帧完整的控制点目标 (C,3)：
    - 左簇索引覆盖 pL
    - 右簇索引覆盖 pR
    其余保持为上一帧的值（平滑）
    """
    # 当前控制点（上一帧）
    cur = env.simulator.controller_points[0].detach().clone()   # [C,3] cuda
    C = cur.shape[0]
    assert C >= 2, "Expect >=2 control points"

    # 确保已分簇（左右）
    if not hasattr(env, "left_idx") or not hasattr(env, "right_idx"):
        ctrl = env.simulator.controller_points[0].detach()
        env.left_idx, env.right_idx = env.split_ctrl_pts_dbscan(ctrl)

    device = cur.device
    pL = torch.tensor(pL_world, dtype=torch.float32, device=device)
    pR = torch.tensor(pR_world, dtype=torch.float32, device=device)

    # 用左/右抓取点覆盖对应簇（简化：整簇同步到抓取点）
    cur[env.left_idx]  = pL
    cur[env.right_idx] = pR
    return cur


def step_phys(env: PhysTwinEnv, target_ctrl_pts: torch.Tensor):
    """
    推进 PhysTwin 一小步：把上一帧/本帧控制点作为 set_controller_interactive 的 prev/current，
    执行 forward_graph，并把新状态设置为下一步 init。
    """
    # prev/current切换
    env.prev_target = env.current_target
    env.current_target = target_ctrl_pts

    env.simulator.set_controller_interactive(env.prev_target, env.current_target)
    if env.simulator.object_collision_flag:
        env.simulator.update_collision_graph()

    # 前向一步
    wp.capture_launch(env.simulator.forward_graph)

    # 下一步起点
    env.simulator.set_init_state(
        env.simulator.wp_states[-1].wp_x,
        env.simulator.wp_states[-1].wp_v,
    )


def main(
    scene_xml: str = None,
    case_name: str = "double_lift_cloth_1",
    substeps: int = 2,
    finger_half_gap: float = 0.03,   # 左右“手指”间距一半（m）
    hz: float = 60.0,                # GUI 刷新目标帧率
):
    """
    - 打开 MuJoCo GUI
    - 每个可视帧：推 MJ 一步、从 attachment_site 生成 L/R 抓取点 → 推 PhysTwin 子步
    - 没有给机器人 IK 控制；MJ 默认停在 keyframe 'home'（若存在），方便你先验证联动
    """
    repo = _repo_root()
    if scene_xml is None:
        scene_xml = repo / "assets" / "mujoco" / "scene" / "ur5e_room.xml"

    # 1) 载入 MJ 模型
    model, data = load_scene(str(scene_xml))

    # 2) reset 到 keyframe 'home'（若有）
    try:
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    except Exception:
        pass
    mujoco.mj_forward(model, data)

    # 3) 启动 PhysTwin
    env = PhysTwinEnv(case_name=case_name, pure_inference_mode=True)
    # 初次拆左右簇
    ctrl = env.simulator.controller_points[0].detach()
    env.left_idx, env.right_idx = env.split_ctrl_pts_dbscan(ctrl)
    env.prev_target = ctrl.clone()
    env.current_target = ctrl.clone()

    # 4) 打开 MJ GUI
    dt = model.opt.timestep
    frame_dt = 1.0 / hz
    print(f"[cosim_viewer] MJ dt={dt:.4f}s, GUI target hz={hz}, substeps={substeps}")

    with mj_viewer.launch_passive(model, data) as viewer:
        t_last = time.time()
        while viewer.is_running():
            # === MJ 推进一步（此处没有关节控制，保持 home 姿态）===
            mujoco.mj_step(model, data)  # 用当前 ctrl（默认 0）前进一步

            # === 从末端 attachment 生成左右抓取点 → 写到 PhysTwin 控制点目标 ===
            try:
                pL, pR = two_fingers_from_attachment(model, data, "attachment_site", half_gap=finger_half_gap)
            except Exception as e:
                print(f"[cosim_viewer] Cannot find 'attachment_site': {e}")
                pL = data.xpos[0].copy()  # fallback
                pR = pL.copy()

            target_ctrl = build_ctrl_target_from_two_points(env, pL, pR)

            # === PhysTwin 子步 ===
            for _ in range(max(1, substeps)):
                step_phys(env, target_ctrl)

            # === 刷新 GUI ===
            viewer.sync()

            # === 节流到目标帧率 ===
            now = time.time()
            sleep_t = frame_dt - (now - t_last)
            if sleep_t > 0:
                time.sleep(sleep_t)
            t_last = time.time()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=None,
                    help="Path to scene xml. Default: assets/mujoco/scene/ur5e_room.xml")
    ap.add_argument("--case", type=str, default="double_lift_cloth_1")
    ap.add_argument("--substeps", type=int, default=2)
    ap.add_argument("--gap", type=float, default=0.03, help="half gap between two fingers (meters)")
    ap.add_argument("--hz", type=float, default=60.0)
    args = ap.parse_args()

    main(scene_xml=args.scene, case_name=args.case, substeps=args.substeps, finger_half_gap=args.gap, hz=args.hz)
