"""Scratch: board-texture/draw features switched off by parameters (for the README ablation).
Disables: scary-board call discount, wet-board bet sizing (wet = dry 0.5 pot), draw semi-bluffs.
Not disabled: the stab gate (stabs only on boards without draws / pairs)."""
from .v4 import Bot as Bot4
from .v5 import Bot as Bot5


class NoTexture4(Bot4):
    def __init__(self):
        super().__init__(scary_board_penalty=0.0, wet_bet_fraction=0.5, semi_bluff_outs=99)


class NoTexture5(Bot5):
    def __init__(self):
        super().__init__(scary_board_penalty=0.0, wet_bet_fraction=0.5, semi_bluff_outs=99)


from .v6 import Bot as Bot6


class NoTexture6(Bot6):
    def __init__(self):
        super().__init__(scary_board_penalty=0.0, wet_bet_fraction=0.5, semi_bluff_outs=99)
