# src/planning/losses.py
import warp as wp
import torch

@wp.kernel(enable_backward=True)
def edge_length2_mse(
    x_cur: wp.array(dtype=wp.vec3),
    x_tgt: wp.array(dtype=wp.vec3),
    springs: wp.array(dtype=wp.int32),
    invM: float,
    out: wp.array(dtype=wp.float32),
):
    k = wp.tid()
    i = springs[2*k + 0]
    j = springs[2*k + 1]
    u  = x_cur[i] - x_cur[j]
    ut = x_tgt[i] - x_tgt[j]
    lij2  = wp.dot(u,  u)      # ||u||^2
    ltij2 = wp.dot(ut, ut)     # ||ut||^2
    d = lij2 - ltij2
    wp.atomic_add(out, 0, invM * d * d)

@wp.kernel(enable_backward=True)
def mean_action_l2(actions: wp.array(dtype=wp.vec3),
                   invM: float,
                   out: wp.array(dtype=wp.float32)):
    k = wp.tid()
    v = actions[k]
    wp.atomic_add(out, 0, invM * wp.dot(v, v))

def mpc_loss_shape_relative(sim, springs_wp, w_action=0.0, action_seq_wp=None):
    """
    刚体不变的 形状 loss: 对齐拓扑的边长 MSE
    - sim.wp_states[-1].wp_x: 当前 cloth N*vec3
    - sim.wp_current_object_points: 目标 cloth N*vec3 同索引
    - springs_wp: M*2 的 warp int32 数组 同拓扑
    """
    # 边长项
    M = springs_wp.shape[0] // 2
    invM = 1.0 / float(M)
    out = wp.zeros(1, dtype=wp.float32, device="cuda", requires_grad=True)
    wp.launch(edge_length2_mse, dim=M,
              inputs=[sim.wp_states[-1].wp_x,
                      sim.wp_current_object_points,
                      springs_wp, invM],
              outputs=[out])

    # 动作 L2 正则（可选）
    if action_seq_wp is not None and w_action > 0.0:
        A = action_seq_wp.shape[0]
        invA = 1.0 / float(A)
        reg = wp.zeros(1, dtype=wp.float32, device="cuda", requires_grad=True)
        wp.launch(mean_action_l2, dim=A, inputs=[action_seq_wp, invA], outputs=[reg])

        @wp.kernel(enable_backward=True)
        def axpy(alpha: float, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32)):
            y[0] = y[0] + alpha * x[0]
        wp.launch(axpy, dim=1, inputs=[float(w_action), reg], outputs=[out])

    # sim 侧保存标量，供 tape.backward
    sim.loss = out
    # 返回 torch view
    return wp.to_torch(out, requires_grad=True)
