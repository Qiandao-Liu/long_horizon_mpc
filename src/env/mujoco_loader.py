# /src/env/mujoco_loader.py
"""
加载自定义 MuJoCo 场景 XML
- 支持在 scene.xml 中写 <include file="..."/>（MuJoCo 本身不支持，所以这里做预展开）
- 支持占位符 ${UR5E_XML}，会解析为 third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml
- 将被包含文件的顶层节点（compiler/option/default/asset/worldbody/actuator/keyframe等）按语义合并
"""

from pathlib import Path
import os
import copy
import xml.etree.ElementTree as ET
import mujoco
from mujoco import Renderer
import imageio


# ============== 路径解析 ==============

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]  # long_horizon_mpc/


def resolve_ur5e_xml() -> Path:
    # 1) 环境变量
    env_path = os.environ.get("UR5E_XML")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # 2) 默认 third_party 位置
    default = _repo_root() / "third_party" / "mujoco_menagerie" / "universal_robots_ur5e" / "ur5e.xml"
    if default.exists():
        return default

    raise FileNotFoundError(
        "UR5e XML not found. Run `python scripts/fetch_menagerie.py` "
        "or set env var: export UR5E_XML=/path/to/universal_robots_ur5e/ur5e.xml"
    )


# ============== XML 合并工具 ==============

_MJ_TOP_TAGS_ORDER = [
    "compiler", "option", "size", "visual", "statistic", "default",
    "asset", "worldbody", "tendon", "actuator", "sensor", "keyframe", "custom", "contact",
    "equality",
]

def _get_or_create(parent: ET.Element, tag: str) -> ET.Element:
    node = parent.find(tag)
    if node is None:
        node = ET.Element(tag)
        parent.append(node)
    return node

def _merge_children(dst_parent: ET.Element, src_parent: ET.Element):
    for child in list(src_parent):
        dst_parent.append(copy.deepcopy(child))

def _merge_root(dst_root: ET.Element, src_root: ET.Element):
    """
    将 src_root（<mujoco>）的顶层子节点按类型合并到 dst_root（<mujoco>）。
    - compiler/option/size/visual/statistic/default/asset：合并（若目标不存在则直接复制，存在则把子节点追加）
    - worldbody：把 src 的 worldbody 中的所有 child 直接追加到 dst 的 worldbody 下
    - actuator/keyframe/...：同样合并
    """
    for tag in _MJ_TOP_TAGS_ORDER:
        src = src_root.find(tag)
        if src is None:
            continue

        if tag == "worldbody":
            dst_wb = _get_or_create(dst_root, "worldbody")
            _merge_children(dst_wb, src)  # 直接把所有 child 挂到 dst 的 worldbody 下
        else:
            dst = dst_root.find(tag)
            if dst is None:
                # 直接整体复制一份
                dst_root.append(copy.deepcopy(src))
            else:
                # 追加其子节点（避免两个同名顶层标签并列）
                _merge_children(dst, src)

def _remove_element(root: ET.Element, target: ET.Element):
    # ElementTree 没有 parent 指针，只能遍历找
    for parent in root.iter():
        for child in list(parent):
            if child is target:
                parent.remove(child)
                return

def _expand_includes(scene_root: ET.Element, base_dir: Path, ur5e_xml: Path):
    """
    递归展开 <include file="..."/>：
    - 如果 file 里有 ${UR5E_XML}，替换为实际路径
    - 解析被包含 XML（必须也是 <mujoco> 根），将其**顶层结构**合并到 scene_root
    - 移除原 <include> 节点
    """
    # 注意：遍历过程中修改树结构，先收集所有 include 再处理
    includes = []
    for parent in scene_root.iter():
        for child in list(parent):
            if child.tag == "include" and "file" in child.attrib:
                includes.append((parent, child))

    for parent, inc in includes:
        file_attr = inc.attrib["file"]
        file_path = file_attr.replace("${UR5E_XML}", str(ur5e_xml))
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = (base_dir / file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Included XML not found: {file_path}")

        # 解析被包含文件
        src_root = ET.parse(str(file_path)).getroot()
        if src_root.tag != "mujoco":
            raise ValueError(f"Included file root must be <mujoco>, got <{src_root.tag}> from {file_path}")

        # 合并顶层元素到场景根
        _merge_root(scene_root, src_root)

        # 删除 include 节点
        parent.remove(inc)

    # 若展开一次后仍有 include（出现链式 include），可递归再展开
    # 这里简单检查一次即可；如有需要可循环直到不再有 include：
    more = any(el.tag == "include" for el in scene_root.iter())
    if more:
        _expand_includes(scene_root, base_dir, ur5e_xml)


def load_scene(scene_xml_path: str):
    scene_xml_path = Path(scene_xml_path)
    ur5e_xml = resolve_ur5e_xml()

    scene_root = ET.parse(str(scene_xml_path)).getroot()
    if scene_root.tag != "mujoco":
        raise ValueError(f"Scene root must be <mujoco>, got <{scene_root.tag}>")

    # 统一成弧度
    compiler = scene_root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        scene_root.insert(0, compiler)
    compiler.set("angle", "radian")

    # 展开 include（把 ur5e.xml 合并进来）
    _expand_includes(scene_root, scene_xml_path.parent, ur5e_xml)

    # 关键修复：设置 meshdir 为 ur5e 的 assets 绝对路径
    ur5e_assets_dir = ur5e_xml.parent / "assets"
    compiler = scene_root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        scene_root.insert(0, compiler)
    compiler.set("meshdir", str(ur5e_assets_dir.resolve()))

    final_xml = ET.tostring(scene_root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(final_xml)
    data = mujoco.MjData(model)
    return model, data


if __name__ == "__main__":
    repo_root = _repo_root()
    scene_path = repo_root / "assets" / "mujoco" / "scene" / "ur5e_room.xml"
    print(f"[mujoco_loader] Loading {scene_path}")
    model, data = load_scene(str(scene_path))
    print(f"[mujoco_loader] Loaded model with nq={model.nq}, nv={model.nv}, nu={model.nu}")

    # 1) 先把数据前向一次（很关键，否则很多可见物体矩阵未更新）
    try:
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    except Exception:
        pass
    mujoco.mj_forward(model, data)  # ← 必须：让几何/光照/相机一次性前向

    # 2) 如果场景里没光，这里临时加一个“强光”到 worldbody（更稳）
    #    （如果你已经在 ur5e_room.xml 里放了 light，可以注释掉这段）
    try:
        # 只有在模型里没有任何光时才添加
        if model.nlight == 0:
            # 直接修改 model 里光源很麻烦；简单做法是在 XML 里加个 light。
            # 这里就不动态加了，提醒你确认 ur5e_room.xml 里 worldbody 下有 <light .../>
            print("[mujoco_loader] WARNING: no lights found in model; please add <light> in XML")
    except Exception:
        pass

    # 3) 选择相机：优先用 "camera0"，否则用第一个相机；都没有则用 free camera
    cam_name = "camera0"
    cam_id = -1
    try:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    except Exception:
        if model.ncam > 0:
            cam_id = 0
            cam_name = None  # 用 id 渲染
        else:
            cam_id = -1  # 用 free camera

    # 4) 渲染
    renderer = Renderer(model, 480, 640)
    try:
        if cam_id >= 0:
            # 用指定相机
            if cam_name is None:
                renderer.update_scene(data, camera=cam_id)
            else:
                renderer.update_scene(data, camera=cam_name)
        else:
            # fallback：free camera，MuJoCo 会用默认自由视角（可能离得很近/很远）
            renderer.update_scene(data)

        img = renderer.render()
        out = _repo_root() / "assets" / "mujoco" / "scene" / "test_render.png"
        imageio.imwrite(out, img)
        print(f"[mujoco_loader] Wrote {out}")
    finally:
        renderer.close()

    # 5) 额外自检：打印一下相机和两个site（如果你加了 overlay weld 的话）
    try:
        sidL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperL_site")
        sidR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperR_site")
        print("gripperL_site:", data.site_xpos[sidL], "gripperR_site:", data.site_xpos[sidR])
    except Exception:
        print("Sites not found (overlay没生效？)")
