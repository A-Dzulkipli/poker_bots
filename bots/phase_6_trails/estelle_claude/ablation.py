"""Scratch: range-off ablations (same strategy deciding against a uniformly random hand)."""
from . import v1, v3, v4, v5


class V1NoRange(v1.Bot):
    def __init__(self):
        super().__init__(use_range=False)


class V4NoRange(v4.Bot):
    def __init__(self):
        super().__init__(use_range=False)


class V1LooseCall(v1.Bot):
    def __init__(self):
        super().__init__(call_margin=-0.06)


class V1TightRaise(v1.Bot):
    def __init__(self):
        super().__init__(value_raise_equity=0.8, value_bet_equity=0.66)


class V1TightOpen(v1.Bot):
    def __init__(self):
        super().__init__(open_equity=0.55)


class V1Aggro(v1.Bot):
    def __init__(self):
        super().__init__(value_raise_equity=0.62, value_bet_equity=0.52, bluff_frequency=0.3)


class V1TightCall(v1.Bot):
    def __init__(self):
        super().__init__(call_margin=0.06)


class V1Aggro2(v1.Bot):
    def __init__(self):
        super().__init__(value_raise_equity=0.56, value_bet_equity=0.48, bluff_frequency=0.45, open_equity=0.38)


class V1AggroBig(v1.Bot):
    def __init__(self):
        super().__init__(value_raise_equity=0.62, value_bet_equity=0.52, bluff_frequency=0.3, bet_pot_fraction=1.0)


class V5NoRange(v5.Bot):
    def __init__(self):
        super().__init__(use_range=False)
