"""Translate existing progress signals without controlling the solver."""


class JobProgress:
    def __init__(self, status):
        self.status=status;self.solver=None;self.phase='主优化'

    def __call__(self, stage, current=0, total=0, *info):
        status=self.status;solver=self.solver;optimization_phase=self.phase
        if stage.startswith('solve'):
            group = '%s 第%d/%d组' % (optimization_phase,
                solver.optimization_stage_index, solver.optimization_stage_count)
            if stage in ('solve', 'solve_iteration', 'solve_trial'):
                suffix = ('线搜索 %d/16' % int(info[0]) if stage == 'solve_trial'
                          else {'solve': '已接受', 'solve_iteration': '计算修正方向'}[stage])
                status('%s %d/%d步 %s' % (group, int(current)+1, int(total), suffix))
            elif stage == 'solve_done':
                status('%s 本组结束：已接受 %d/%d步（允许提前结束）' % (group, int(current), int(total)))
            else:
                label = {'solve_stage': '更新碰撞约束', 'solve_prepare': '准备整段样条',
                         'solve_initial': '计算初始碰撞与梯度',
                         'solve_temporal': '预计算时间平滑矩阵'}[stage]
                status(group + '：' + label)
            return
        label = {'hands': '采样完整手掌手指', 'hand_fit': '拟合左右手掌椭球',
                 'colliders': '建立身体连续碰撞缓存', 'collider_seed': '拆分身体碰撞区域',
                 'collider_pack': '整理身体碰撞缓存', 'planes': '筛选有效身体碰撞平面',
                 'refit_hands': '重包修正后混合腕缝'}.get(stage, '处理进度')
        status('%s %d/%d' % (label, int(current), int(total)))
