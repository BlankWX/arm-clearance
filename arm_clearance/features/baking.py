"""Area 7: candidate writing/checkpoint replacement through action I/O ports."""


class CandidateBaker:
    def __init__(self, writer, discard, arm, source, targets):
        self.writer = writer
        self.discard = discard
        self.arm = arm
        self.source = source
        self.targets = targets

    def write(self, clip, pose):
        return self.writer(self.arm, self.source, self.targets, clip.frames,
                           pose.quaternions, 'AC_v2_' + self.source.name)

    def discard_replaced(self, previous_candidate):
        if previous_candidate is not None:
            self.discard(previous_candidate)
