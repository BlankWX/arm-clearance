"""Area 8 console lifecycle around the ordinary generator."""
from .progress import ConsoleProgress

def logged_run(operation, scene, settings, status, *, console=None):
    """Log lifecycle outcomes after the original generator restores scene state."""
    log = console if console is not None else ConsoleProgress()
    start, end = int(scene.frame_start), int(scene.frame_end)
    log.start('场景范围 %d–%d，共 %d 帧' % (start, end, max(0, end-start+1)))
    def report(message):
        status(message)
        log.update(message)
    try:
        result = yield from operation(scene, settings, report)
    except (GeneratorExit, KeyboardInterrupt):
        log.finish('cancelled', '用户取消，本次运行已结束')
        raise
    except BaseException as error:
        log.finish('error', str(error))
        raise
    else:
        validation = result.get('arm_clearance_validation_status') if result is not None else None
        outcome = {'passed': 'complete', 'needs_review': 'needs_review'}.get(validation, 'error')
        log.finish(outcome, settings.status if result is not None else '本次未生成完整烘焙')
        return result
