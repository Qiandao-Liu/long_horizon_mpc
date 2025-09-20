# /home/qiandaoliu/workspace/src/planning/play_rollout.py
"""
python /src/planning/play_rollout.py \
  --file /home/qiandaoliu/workspace/PhysTwin/mpc_logs/debug_outer000_it000.npz \
  --fps 1 --point-size 5 --bg-white
"""
import os, time, argparse
import numpy as np
import open3d as o3d

def load_npz(path):
    data = np.load(path)
    frames = data["frames"]  # (T, N, 3)
    extra = {k: data[k] for k in data.files if k != "frames"}
    return frames, extra

def auto_setup_camera(vis, pts, zoom=0.7):
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(pts)
    )
    ctr = vis.get_view_control()
    ctr.set_lookat(bbox.get_center())
    ctr.set_front([0.5, -0.5, 0.7])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(zoom if zoom else 0.7)
    return bbox

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str,
                    default="workspace/PhysTwin/mpc_logs/debug_outer000_it000.npz",
                    help="npz rollout file")
    ap.add_argument("--fps", type=float, default=10, help="playback fps")
    ap.add_argument("--point-size", type=float, default=1.0, help="cloth point size")
    ap.add_argument("--ctrl-size", type=float, default=4.0, help="control point size (if supported)")
    ap.add_argument("--loop", action="store_true", help="loop playback")
    ap.add_argument("--downsample", type=int, default=0,
                    help="randomly sample N cloth points per frame (0 = no ds)")
    ap.add_argument("--export", type=str, default="",
                    help="export to MP4 at the given path (e.g. out.mp4)")
    ap.add_argument("--bg-white", action="store_true", help="white background")
    args = ap.parse_args()

    assert os.path.exists(args.file), f"File not found: {args.file}"
    frames, extra = load_npz(args.file)
    T, N, C = frames.shape
    assert C == 3, "frames must be (T, N, 3)"

    # 下采样 cloth 点
    if args.downsample and args.downsample < N:
        idx = np.random.choice(N, args.downsample, replace=False)
        frames = frames[:, idx, :]
        N = args.downsample
        print(f"[info] downsampled cloth to {N} points")

    # 读取控制点轨迹（若存在）
    ctrl_frames = extra.get("ctrl_frames", None)  # (T, C, 3)
    left_idx = extra.get("left_idx", None)
    right_idx = extra.get("right_idx", None)
    has_ctrl = (ctrl_frames is not None) and (left_idx is not None) and (right_idx is not None)
    if has_ctrl:
        ctrl_frames = np.asarray(ctrl_frames)
        left_idx = np.asarray(left_idx, dtype=np.int32)
        right_idx = np.asarray(right_idx, dtype=np.int32)
        left_idx = np.unique(left_idx)
        right_idx = np.unique(right_idx)
        assert ctrl_frames.shape[0] == T, "ctrl_frames T must match frames T"
        print(f"[info] ctrl loaded: T={ctrl_frames.shape[0]}, C={ctrl_frames.shape[1]}, "
              f"L={len(left_idx)}, R={len(right_idx)}")

    # 打印 extra 信息
    print(f"[info] loaded: T={T}, N={N}")
    if extra:
        for k, v in extra.items():
            try:
                print(f" - {k}: shape={tuple(v.shape)}")
            except Exception:
                print(f" - {k}: type={type(v)}")

    # 可视化器
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=os.path.basename(args.file), width=1280, height=960)
    ropt = vis.get_render_option()
    ropt.point_size = args.point_size
    ropt.background_color = (1,1,1) if args.bg_white else (0,0,0)

    # 坐标系
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(axis)

    # 布料点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frames[0])
    vis.add_geometry(pcd)

    # 控制点点云（左右手两层）
    if has_ctrl:
        ctrl0 = ctrl_frames[0]
        ctrl_L0 = ctrl0[left_idx] if left_idx.size > 0 else np.zeros((0,3))
        ctrl_R0 = ctrl0[right_idx] if right_idx.size > 0 else np.zeros((0,3))

        pcd_L = o3d.geometry.PointCloud()
        pcd_L.points = o3d.utility.Vector3dVector(ctrl_L0)
        pcd_L.paint_uniform_color([1.0, 0.2, 0.2])  # 红

        pcd_R = o3d.geometry.PointCloud()
        pcd_R.points = o3d.utility.Vector3dVector(ctrl_R0)
        pcd_R.paint_uniform_color([0.2, 0.4, 1.0])  # 蓝

        vis.add_geometry(pcd_L)
        vis.add_geometry(pcd_R)

        # Open3D 目前无法为不同几何设置不同 point_size；这里尽量把整体点径调大一些以便看清控制点
        ropt.point_size = max(args.point_size, args.ctrl_size)

    auto_setup_camera(vis, frames[0])

    # 播放控制
    state = {"t": 0, "paused": False, "quit": False}
    frame_time = 1.0 / max(1e-6, args.fps)

    def goto_frame(t):
        t = max(0, min(T-1, t))
        state["t"] = t
        pcd.points = o3d.utility.Vector3dVector(frames[t])
        vis.update_geometry(pcd)

        if has_ctrl:
            ctrl_t = ctrl_frames[t]
            if left_idx.size > 0:
                pcd_L.points = o3d.utility.Vector3dVector(ctrl_t[left_idx])
                vis.update_geometry(pcd_L)
            if right_idx.size > 0:
                pcd_R.points = o3d.utility.Vector3dVector(ctrl_t[right_idx])
                vis.update_geometry(pcd_R)

    def on_space(_):
        state["paused"] = not state["paused"];  return False
    def on_left(_):
        goto_frame(state["t"] - 1);             return False
    def on_right(_):
        goto_frame(state["t"] + 1);             return False
    def on_quit(_):
        state["quit"] = True;                   return False

    vis.register_key_callback(ord(" "), on_space)
    vis.register_key_callback(262, on_right)  # →
    vis.register_key_callback(263, on_left)   # ←
    vis.register_key_callback(ord("Q"), on_quit)
    vis.register_key_callback(27, on_quit)    # ESC

    # 可选：导出视频
    writer = None
    if args.export:
        try:
            import imageio
            writer = imageio.get_writer(args.export, fps=int(args.fps))
            print(f"[info] recording to {args.export}")
        except Exception as e:
            print(f"[warn] failed to open video writer: {e}")

    last = time.time()
    while not state["quit"]:
        now = time.time()
        if now - last >= frame_time and not state["paused"]:
            goto_frame(state["t"] + 1)
            if state["t"] == T - 1:
                if args.loop:
                    goto_frame(0)
                else:
                    state["paused"] = True
            last = now

        vis.poll_events()
        vis.update_renderer()

        if writer is not None:
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=False))
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            writer.append_data(frame)

    if writer is not None:
        writer.close()
        print(f"[info] saved video: {args.export}")

    vis.destroy_window()

if __name__ == "__main__":
    main()
