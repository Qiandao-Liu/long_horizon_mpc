class SimManager:
    def __init__(self, cfg: MPCConfig):
        self.env, self.sim = self._build_env(cfg.case_name)
        self.left_idx, self.right_idx = self._split_ctrl_groups()
        self.left_mask_wp, self.right_mask_wp = self._build_masks()
        # target buffers / visibility / motion-valid 初始化……

    def set_init_from_pkl(self, init_pkl: dict) -> None:
        """把 init 的 wp_x/wp_v/ctrl_pts 写入 sim；准备初始快照S0"""
    def set_target_from_pkl(self, target_pkl: dict, pad_before: int = 1) -> None:
        """把目标点云（或里程碑帧）写入 sim 的 target buffers"""

    # —— 核心：快照/恢复（CPU numpy）——
    def snapshot(self) -> dict:
        """返回 {wp_x, wp_v, ctrl_pts, …} 的 numpy 拷贝（轻量）"""
    def restore(self, snap: dict) -> None:
        """把 snapshot 写回 sim（避免重建对象）"""

    # —— 单步/多步前向（无 Tape）——
    def step_from_lr_delta(self, delta_lr_2x3: torch.Tensor, scale: float) -> None:
        """(2,3)->expand_row2_squash_scale->one_step_from_action"""

    def rollout_from_action_seq(self, action_seq: torch.Tensor, scale: float) -> None:
        """按 (H*2,3) 做 H 步前向（无 Tape）"""

    # —— 构造 warp 端需要的切片、kernel 运行辅助（给优化器用）——
    def build_wp_view(self, torch_slice: torch.Tensor, requires_grad=True):
        """torch→warp 视图；返回 a_seq_wp"""
