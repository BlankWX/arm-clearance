"""Area 8: lifecycle policy, result ownership, cancellation and reports.

Opaque host handles are owned only at this boundary; numerical features never
receive them. Scene operations are supplied by the job I/O port.
"""
import time

def _retain_review_action(arm, source, candidate, report, validation_status):
    """Keep a complete bake saveable without calling it a verified result."""
    source.use_fake_user = True
    candidate.use_fake_user = True
    candidate['arm_clearance_validation_status'] = validation_status
    surface = report.get('surface_verification') or {}
    candidate['arm_clearance_problem_frames'] = list(surface.get('problem_frames', []))
    arm['ac_source_action'] = source.name
    arm['ac_review_action'] = candidate.name
    report.update(result_preserved=True, candidate_discarded=False,
                  review_action=candidate.name, result_action=candidate.name,
                  validation_passed=False)


class JobGuard:
    def __init__(self, io):
        self.io=io
        self.started=time.perf_counter()
        self.complete=False;self.candidate=None;self.text=None
        start=io.start;end=io.end;source=io.source;previous=io.previous
        targets=io.targets;wrists=io.wrists
        self.report = {'version': 2, 'status': 'running', 'range': [start, end],
                  'source_action': source.name, 'previous_action': previous.name,
                  'targets': targets, 'wrist_location_targets': wrists,
                  'proxy': {}, 'surface_verification': None, 'attempts': [],
                  'scope': ('整段六骨旋转与双腕位置修正，手指动画保持原样；'
                            '代理残差与真实变形网格验证分别报告；'
                            '烘焙数据独立保留，未通过复核的动作明确标记为待修正；'
                            '离散时刻验证不构成全部连续时刻无穿模的数学证明')}

    def begin(self, status):
        self.io.activate_source()
        self.text=self.io.new_report()
        self.io.publish_report(self.text)
        status('读取原动作，提取身体、双臂与完整手掌手指')
        self.write_report()

    def write_report(self):
        self.io.write_report(self.text,self.report)

    def stage_candidate(self, corrected, attempt):
        candidate=self.candidate;report=self.report
        self.io.restore_links()
        if self.text is not None:
            candidate['arm_clearance_report']=self.text.name
        elif 'arm_clearance_report' in candidate:
            del candidate['arm_clearance_report']
        candidate['arm_clearance_validation_status']='unverified'
        report['baked_candidate_name']=candidate.name
        report['surface_verification']=None
        attempt_report={'attempt':attempt+1,'proxy':corrected,
                        'candidate_name':candidate.name,'surface_verification':None}
        report['attempts'].append(attempt_report)
        return attempt_report

    def finish(self):
        arm=self.io.arm;source=self.io.source;previous=self.io.previous
        ad=self.io.ad;source_slot=self.io.source_slot;previous_links=self.io.previous_links
        candidate=self.candidate;report=self.report;settings=self.io.settings
        start=self.io.start;end=self.io.end
        _slot_for=self.io.slot_for
        source.use_fake_user = True
        previous.use_fake_user = True
        candidate.use_fake_user = True
        # Bind the copied slot explicitly, including multi-slot source actions.
        ad.action = candidate
        ad.action_slot = _slot_for(candidate, source_slot.identifier)
        prior_result = self.io.find_action(previous_links.get('ac_review_action') or '')
        prior_result = prior_result or self.io.find_action(previous_links.get('ac_result_action') or '')
        if prior_result is None and previous != source:
            prior_result = previous
        if prior_result is not None and prior_result != candidate:
            prior_result.use_fake_user = True
            arm['ac_previous_result_action'] = prior_result.name
        arm['ac_source_action'] = source.name
        validation_passed = 'validation_failure' not in report
        report.update(result_action=candidate.name, result_preserved=True,
                      candidate_discarded=False, validation_passed=validation_passed)
        if validation_passed:
            arm['ac_result_action'] = candidate.name
            if 'ac_review_action' in arm:
                del arm['ac_review_action']
            candidate['arm_clearance_validation_status'] = 'passed'
            report['status'] = 'complete'
            settings.status = '完成 %d–%d 帧，复核通过，烘焙动作已保留；保存工程即可保存数据' % (start, end)
        else:
            _retain_review_action(arm, source, candidate, report, 'needs_review')
            report['status'] = 'needs_review'
            reason = ('身体代理仍有深残差' if report['validation_failure'] == 'deep_body_proxy_residual'
                      else '仍有穿模，请播放检查并二次修正')
            settings.status = '已保留 %d–%d 帧烘焙，复核未通过：%s；可保存和切换查看' % (start, end, reason)
        self.complete = True
        return candidate

    def verification_terminal(self, surface, corrected, scale, attempt_report):
        if surface is None or not surface['complete']:
            raise RuntimeError('真实网格复核未完成')
        body_depth_world_mm = float(corrected['body_max_mm']) * scale
        attempt_report['body_proxy_max_world_mm'] = body_depth_world_mm
        if body_depth_world_mm > 20.:
            self.report['validation_failure'] = 'deep_body_proxy_residual'
            return True
        return not surface['collision_samples']

    def retries_exhausted(self, attempt):
        if attempt == 3:
            self.report['validation_failure'] = 'actual_surface_intersection'
            return True
        return False

    def cancel(self):
        self.report['status']='cancelled'
        self.io.settings.status='已取消，恢复先前动作'

    def error(self, error):
        self.report['status']='cancelled' if isinstance(error,KeyboardInterrupt) else 'error'
        self.report['error']=str(error)
        self.io.settings.status='处理失败，已恢复先前动作：' + str(error)

    def cleanup(self):
        report=self.report;candidate=self.candidate
        if not self.complete:
            self.io.restore_original()
            if candidate is not None:
                _retain_review_action(self.io.arm,self.io.source,candidate,report,'unverified')
                self.io.settings.status += '；最近完整烘焙已保留，可切换查看'
        self.io.restore_frame()
        report['seconds']=time.perf_counter()-self.started
        if self.text is not None:
            self.write_report()
