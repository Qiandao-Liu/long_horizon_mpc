# /src/planning/config.py
from pathlib import Path

@dataclass
class MPCConfig:
    case_name: str = "double_lift_cloth_1"
    horizon: int = 10
    max_iters: int = 30
    frames_per_step: int = 5
    max_delta: float = 0.25
    step_row: float = 3e-2
    pre_tanh_clip: float = 3.0
    grad_clip: float = 1.0
    line_search: bool = True
    early_tol: float = 1e-3
    # loss
    w_chamfer: float = 1.0
    w_track: float = 0.2
    lambda_drift: float = 1e-2
    w_smooth_v: float = 0.0
    w_smooth_a: float = 0.0
    # io
    tasks_root: Path = Path("data/tasks")
    log_root: Path = Path("data/mpc_logs")
