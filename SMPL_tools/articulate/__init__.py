"""Minimal articulate package: only the forward-kinematics model used by
paper/B/B3_TCN_FK.py. The original package also includes armature/evaluator/
utils modules (physics simulation, RBDL/Bullet bindings) that are not needed
for this paper and were omitted to keep the public repo minimal.
"""
from .model import *
from . import math
