# /src/bridge/isaac_phystwin_bridge.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim
from pxr import Gf, UsdGeom

from src.env.phystwin_starter import PhysTwinStarter


# === 1. 启动 PhysTwin 模型 ===
phys = PhysTwinStarter("cloth_double_hand")
world = World(stage_units_in_meters=1.0)
stage = world.stage


# === 2. 创建 USD 里显示 PhysTwin cloth 的 prim ===
cloth_path = "/World/PhysTwinCloth"
cloth_prim = create_prim(cloth_path, "Points")


# === 3. 每帧回调：同步 Isaac → PhysTwin → Isaac ===
def on_step(dt):
    # (1) 从双臂 articulation 读取 gripper pose
    left_pose  = get_gripper_pose(world, "/World/so101_left/gripper")
    right_pose = get_gripper_pose(world, "/World/so101_right/gripper")

    # (2) 更新控制点到 PhysTwin
    phys.set_ctrl_from_robot(left_pose, right_pose)

    # (3) 推动一帧布料仿真
    phys.step()

    # (4) 取回状态并更新 Isaac 可视化
    wp_x, gs_xyz, gs_sigma, gs_color = phys.get_state()
    update_point_cloud(cloth_prim, gs_xyz)

def get_gripper_pose(world, path):
    pass

def update_point_cloud(points_prim, points):
    pass


world.add_physics_callback("phystwin_bridge", on_step)
world.reset()

while simulation_app.is_running():
    world.step(render=True)
