"""基于 MuJoCo 的自定义可视化工具模块。

提供坐标系框绘制函数和交互式可视化处理函数，
供 MuJoCo 仿真运行时调试与状态可视化使用。
"""
import mujoco
import numpy as np


def draw_frame(scn, pos, rot_matrix, axis_len=0.02, axis_radius=0.0008):
    """向 mjvScene 中绘制一个 XYZ 坐标系（三个箭头）。

    X = 红色，Y = 绿色，Z = 蓝色。
    pos       : (3,) 坐标系原点的世界坐标。
    rot_matrix: (3,3) 旋转矩阵，列向量分别为 X/Y/Z 轴方向。
    """
    colors = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ], dtype=np.float32)
    for i in range(3):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        tip = pos + rot_matrix[:, i] * axis_len
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            colors[i],
        )
        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_ARROW,
            axis_radius,
            pos,
            tip,
        )
        scn.ngeom += 1


def draw_body_frames(model, data, viewer, body_names=None, axis_len=0.02, axis_radius=0.0008):
    """向 viewer.user_scn 中绘制机器人各体的 XYZ 坐标系箭头。

    应在 mj_fwdPosition 之后的查看器循环中调用，以确保 xpos/xmat 已更新。

    Parameters
    ----------
    body_names : list[str] | None
        需要可视化的体名称列表。None 表示模型中的所有体。
    axis_len   : float   箭头长度（米）。
    axis_radius: float   箭头轴半径（米）。
    """
    viewer.user_scn.ngeom = 0
    if body_names is None:
        body_ids = range(model.nbody)
    else:
        body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in body_names
        ]
    for bid in body_ids:
        pos = data.xpos[bid].copy()
        rot = data.xmat[bid].reshape(3, 3).copy()
        draw_frame(viewer.user_scn, pos, rot, axis_len, axis_radius)


def _resolve_joints(model):
    """解析模型中所有可控单自由度关节（铰链/滑动），供 GUI 使用。"""
    names = []
    ids = []
    for jid in range(model.njnt):
        jtype = model.jnt_type[jid]
        if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname is None:
                jname = f"joint_{jid}"
            names.append(jname)
            ids.append(jid)
    return names, ids
