"""Statistical significance testing for experiment results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy import stats


@dataclass
class StatisticalTester:
    """Run pairwise statistical tests on model metric arrays."""

    alpha: float = 0.05

    def paired_t_test(self, scores_a: np.ndarray, scores_b: np.ndarray) -> Dict[str, float]:
        statistic, p_value = stats.ttest_rel(scores_a, scores_b, nan_policy="omit")
        return {
            "t_statistic": float(statistic),
            "p_value": float(p_value),
            "significant": bool(p_value < self.alpha),
        }


__all__ = ["StatisticalTester"]
