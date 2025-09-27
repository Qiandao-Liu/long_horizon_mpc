# src/planning/action_param.py
import torch

class ActionParam:
    """动作参数：(H*2,3)，每步左右两行；提供 clamp/shift 等原地操作"""
    def __init__(self, H: int, device="cuda"):
        self.H = int(H)
        self.param = torch.zeros(H*2, 3, device=device, dtype=torch.float32, requires_grad=True)

    def zero_(self):
        with torch.no_grad():
            self.param.zero_()

    def clamp_pre_tanh_(self, clip: float = 3.0):
        with torch.no_grad():
            self.param.clamp_(-float(clip), float(clip))

    def shift_left_and_pad_(self, steps: int = 1):
        """执行了 steps 步后，把后续动作左移，尾部补零"""
        s = steps * 2
        with torch.no_grad():
            if s >= self.param.shape[0]:
                self.param.zero_()
            else:
                tail = self.param[s:].clone()
                self.param.zero_()
                self.param[:tail.shape[0]].copy_(tail)

def rowwise_normalized_step(param: torch.Tensor, grad: torch.Tensor, step_row: float):
    """每行固定步长的归一化梯度下降"""
    with torch.no_grad():
        for r in range(param.shape[0]):
            g = grad[r]
            n = g.norm() + 1e-12
            param[r].add_(-float(step_row) * g / n)
