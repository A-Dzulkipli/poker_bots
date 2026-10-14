"""Rubick, strict table: switches away from windrunner_glm only where another bot is better by more than 2 SE.

See rubick_final.py for how the tables were built.
"""
from bots.rubick.rubick_final import Bot as _Rubick


class Bot(_Rubick):
    def __init__(self):
        super().__init__("strict")
