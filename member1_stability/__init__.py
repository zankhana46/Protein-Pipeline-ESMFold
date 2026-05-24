"""
member1_stability — Structure & Stability Analysis Module

Public API
----------
StabilityAnalyzer  : high-level interface (use this from other modules)
StabilityReport    : result dataclass
StabilityClass     : enum (STABLE / MODERATE / FLEXIBLE)
ResidueStability   : per-residue result dataclass
"""
from .analyzer import StabilityAnalyzer, StabilityReport, StabilityClass, ResidueStability

__all__ = [
    "StabilityAnalyzer",
    "StabilityReport",
    "StabilityClass",
    "ResidueStability",
]
