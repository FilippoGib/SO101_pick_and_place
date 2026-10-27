"""Reset-phase sampling weights over the physically generated task horizon."""

from __future__ import annotations

# Phase order in the reset dataset:
#   0 home, 1 approach, 2 grasp, 3 lift, 4 transport, 5 place, 6 release.
# Every phase is a physics-validated state reached by executing the task from the home pose.
CANONICAL_START = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
"""Evaluation and interactive play start every episode from the real robot's home pose."""
ALL_PHASES = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
