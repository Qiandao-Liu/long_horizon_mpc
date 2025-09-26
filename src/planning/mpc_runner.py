class MPCRunner:
    def __init__(self, cfg: MPCConfig, sim_mgr: SimManager):
        self.cfg = cfg
        self.sim_mgr = sim_mgr

    def build_segment_plans(self, milestones: List[int]) -> List[SegmentPlan]:
        """根据 milestones 与 frames_per_step 生成每段计划 S 与 R"""

    def run_segment(self, plan: SegmentPlan, init_snap: dict,
                    target_pkl: dict) -> Tuple[dict, Dict]:
        """
        执行一个段：
        - 切目标
        - R 轮 inner；前 R-1 轮各执行 1 步，最后一轮执行 min(H,剩余步)
        - 返回：段末真实世界快照、诊断信息（loss 曲线、执行步数、iters）
        """

    def run_task(self, milestones: List[int], demo_pkl: dict) -> Dict:
        """
        1) set_init_from_pkl(demo_pkl @ t0)
        2) for each segment -> run_segment
        3) 汇总统计与保存
        """
