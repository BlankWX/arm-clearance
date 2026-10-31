"""Compatibility entry; active orchestration is application.py."""
from .application import run_job, _run_job
from .adapters.validation import validate
from .adapters.job_io import _slot_for, _source_for, _restore_links, _write_report, _LINKS
from .features.safety.job import _retain_review_action
from .features.repair.policy import failed_hand_categories as _failed_hand_categories
from .features.repair.policy import repair_free_frames as _repair_free_frames


def _replace_hand_envelopes(model, hand):
    model.replace_hand_envelopes(hand)
