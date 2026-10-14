"""Ablation bot (scratch, not a version): the deployed v3 with texture and draw effects off.

Same decision code as v3, but texture_effect_scale=0 (wetness no longer grows bet sizing or
call margins, air bluffs no longer prefer dry boards) and draw_effect_scale=0 (outs no longer
trigger semi-bluffs, draw raises or implied-odds credit). Used only for the README evidence.
"""
from .v3 import Bot as ViciousV3


class Bot(ViciousV3):
    def __init__(self):
        super().__init__(texture_effect_scale=0.0, draw_effect_scale=0.0)

    def metadata(self) -> dict:
        info = super().metadata()
        info["bot_name"] = "vicious_glm_flat_ablation"
        info["params"] = {**info["params"], "texture_effect_scale": 0.0, "draw_effect_scale": 0.0}
        return info
