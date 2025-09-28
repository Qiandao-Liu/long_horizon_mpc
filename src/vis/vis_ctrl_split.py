# src/vis/vis_ctrl_split.py
from pathlib import Path
import numpy as np
import open3d as o3d

def _to_pcd(points_xyz: np.ndarray, rgb: tuple, translate=None):
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3
    pcd = o3d.geometry.PointCloud()
    pts = points_xyz.astype(np.float32).copy()
    if translate is not None:
        pts += np.asarray(translate, dtype=np.float32)
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    color = np.array(rgb, dtype=np.float64)[None, :].repeat(len(pts), axis=0)
    pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def _spheres_at(points_xyz: np.ndarray, radius: float, rgb: tuple, translate=None):
    """用若干小球来渲染控制点，避免 point_size 全局共享的限制。"""
    pts = points_xyz.astype(np.float32).copy()
    if translate is not None:
        pts += np.asarray(translate, dtype=np.float32)

    color = np.array(rgb, dtype=np.float64)
    spheres = []
    # 预建一个单位球再拷贝会更快，但数量通常不大，直接逐个建即可
    for p in pts:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
        s.paint_uniform_color(color)
        s.translate(p.astype(np.float64))
        s.compute_vertex_normals()
        spheres.append(s)
    return spheres

def visualize_ctrl_and_mass_split(
    init_mass: np.ndarray,
    tgt_mass:  np.ndarray,
    init_ctrl: np.ndarray,
    tgt_ctrl:  np.ndarray,
    left_idx_init: np.ndarray,
    right_idx_init: np.ndarray,
    left_idx_tgt:  np.ndarray,
    right_idx_tgt: np.ndarray,
    z_lift_target: float = 0.01,     # 让 target 轻微抬高，避免完全重合
    bg_color: str = "black",         # "black" / "darkgray"
    mass_point_size: float = 1.5,    # 质量点 point size（全局生效）
    ctrl_radius: float = 0.01,       # 控制点小球半径（只影响控制点）
    show_window: bool = True,
    save_path: Path | None = None,
):
    """
    同一窗口显示：
      - init/tgt 的布质量点（灰度区分，tgt 整体上移 z_lift_target）
      - 四组控制点：init-left / init-right / tgt-left / tgt-right，用小球放大显示
    颜色：
      mass init  : 深灰 (0.5, 0.5, 0.5)
      mass target: 浅灰 (0.8, 0.8, 0.8)
      init-left  : 红   (0.91, 0.30, 0.24)
      init-right : 蓝   (0.20, 0.60, 0.86)
      tgt-left   : 粉   (0.95, 0.58, 0.67)
      tgt-right  : 青   (0.52, 0.76, 0.91)
    """
    # 1) mass clouds
    mass_init_pcd = _to_pcd(init_mass, (0.50, 0.50, 0.50), translate=None)
    mass_tgt_pcd  = _to_pcd(tgt_mass,  (0.80, 0.80, 0.80), translate=(0, 0, z_lift_target))

    # 2) controller spheres（四组，tgt 同样抬高一点）
    init_left_spheres  = _spheres_at(init_ctrl[left_idx_init],  ctrl_radius, (0.91, 0.30, 0.24), translate=None)
    init_right_spheres = _spheres_at(init_ctrl[right_idx_init], ctrl_radius, (0.20, 0.60, 0.86), translate=None)
    tgt_left_spheres   = _spheres_at(tgt_ctrl[left_idx_tgt],    ctrl_radius, (0.95, 0.58, 0.67), translate=(0, 0, z_lift_target))
    tgt_right_spheres  = _spheres_at(tgt_ctrl[right_idx_tgt],   ctrl_radius, (0.52, 0.76, 0.91), translate=(0, 0, z_lift_target))

    geoms = [mass_init_pcd, mass_tgt_pcd] + init_left_spheres + init_right_spheres + tgt_left_spheres + tgt_right_spheres

    # 3) 渲染
    if show_window:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Init/Target + Left/Right Controllers", width=1280, height=800, visible=True)
        for g in geoms:
            vis.add_geometry(g)

        opt = vis.get_render_option()
        opt.background_color = np.array((0, 0, 0) if bg_color == "black" else (0.12, 0.12, 0.12), dtype=np.float64)
        opt.point_size = float(mass_point_size)
        opt.show_coordinate_frame = False

        vis.poll_events()
        vis.update_renderer()
        # 可选：截图
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            vis.capture_screen_image(str(save_path.with_suffix(".png")), do_render=True)
        vis.run()
        vis.destroy_window()
    else:
        # 仅保存点云（可选）
        if save_path is not None:
            o3d.io.write_point_cloud(str(Path(save_path).with_suffix(".ply")), mass_init_pcd + mass_tgt_pcd)
