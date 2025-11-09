#!/usr/bin/env python3
# ./long_horizon_mpc/src/env/phystwin_server.py
"""
cd ~/long_horizon_mpc
python -m src.env.phystwin_server --case double_lift_cloth_1 --bind tcp://127.0.0.1:5556
"""
import argparse, json, io, sys, time
import numpy as np
import zmq
from pathlib import Path
import torch

# ====== import PhysTwin Starter ======
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from src.env.phystwin_starter import PhysTwinStarter


def np_from_maybe_torch(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def send_array_frames(sock, arr: np.ndarray, name: str, more_after: bool):
    """按两帧发送一个 ndarray: [json header][raw npy]，是否还有后续帧由 more_after 控制。"""
    meta = {"name": name, "dtype": str(arr.dtype), "shape": arr.shape}
    header = json.dumps(meta).encode("utf-8")
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    raw = buf.getvalue()

    # 发送 header 帧
    sock.send(header, flags=zmq.SNDMORE)
    # 发送 raw 帧
    sock.send(raw, flags=(zmq.SNDMORE if more_after else 0))


def recv_poses(msg: bytes):
    """Parse JSON pose message"""
    data = json.loads(msg.decode("utf-8"))
    L = data["left"]
    R_ = data["right"]
    left_pose = {"R": np.array(L["R"], dtype=np.float32), "t": np.array(L["t"], dtype=np.float32)}
    right_pose = {"R": np.array(R_["R"], dtype=np.float32), "t": np.array(R_["t"], dtype=np.float32)}
    return left_pose, right_pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, default="cloth_double_hand")
    ap.add_argument("--bind", type=str, default="tcp://127.0.0.1:5556")
    args = ap.parse_args()

    starter = PhysTwinStarter(case_name=args.case, pure_inference_mode=True)
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)
    print(f"[PhysTwinServer] Listening at {args.bind}")

    while True:
        try:
            # === 1. Wait for Isaac request ===
            msg = sock.recv()
            left_pose, right_pose = recv_poses(msg)

            # === 2. Step simulation ===
            starter.set_ctrl_from_robot(left_pose, right_pose)
            starter.step()

            # === 3. Get state ===
            wp_x, gs_xyz, gs_sigma, gs_color, ctrl_pts = starter.get_state()

            # === 4. Send one multipart reply ===
            meta = {"arrays": ["wp_x", "gs_xyz", "gs_sigma", "gs_color", "ctrl_pts"]}
            sock.send_json(meta, flags=zmq.SNDMORE)

            send_array_frames(sock, np_from_maybe_torch(wp_x), "wp_x", more_after=True)
            sock.send(b"", flags=zmq.SNDMORE)

            send_array_frames(sock, np_from_maybe_torch(gs_xyz), "gs_xyz", more_after=True)
            sock.send(b"", flags=zmq.SNDMORE)

            send_array_frames(sock, np_from_maybe_torch(gs_sigma), "gs_sigma", more_after=True)
            sock.send(b"", flags=zmq.SNDMORE)

            send_array_frames(sock, np_from_maybe_torch(gs_color), "gs_color", more_after=True)
            sock.send(b"", flags=zmq.SNDMORE)

            send_array_frames(sock, np_from_maybe_torch(ctrl_pts), "ctrl_pts", more_after=False)

        except KeyboardInterrupt:
            print("[PhysTwinServer] Shutting down...")
            break
        except Exception as e:
            print(f"[PhysTwinServer][ERROR] {e}")
            try:
                sock.send_json({"error": str(e)})
            except Exception:
                pass


if __name__ == "__main__":
    main()
