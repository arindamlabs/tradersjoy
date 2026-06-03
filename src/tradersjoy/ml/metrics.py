"""The honest scorecard for a yes/no predictor, with the right baselines.

Accuracy alone is a trap for this problem. If 54% of days are up, a model that
blindly screams "up!" every day is 54% accurate while knowing nothing. So every
number here is reported next to the baseline it must beat:

- **base rate**: the share of up-days. The score a know-nothing model gets for
  free, and the bar accuracy has to clear to mean anything.
- **accuracy**: share of correct up/down calls (threshold 0.5).
- **AUC** (area under the ROC curve): the probability the model ranks a random
  up-day above a random down-day. ``0.5`` is pure coin-flip, ``1.0`` is perfect.
  This is the single most honest summary, because it measures *ranking* skill and
  ignores how the base rate happens to fall. For us ranking is what matters: the
  strategy buys the top-ranked names, it does not need a calibrated probability.
- **top-decile lift**: among the 10% of days the model was most confident about,
  what fraction actually rose, and what was their average forward return, versus
  the whole sample. This is the bridge from "is it accurate" to "would acting on
  it have made money".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Summary of a set of probabilistic up/down predictions.

    Attributes:
        n: Number of predictions scored.
        base_rate: Fraction of rows whose true label was up (the baseline).
        accuracy: Fraction of correct calls at a 0.5 threshold.
        auc: Area under the ROC curve; 0.5 is chance, higher is better.
        top_decile_hit_rate: Up-rate among the most-confident 10% of predictions.
        top_decile_avg_return: Mean forward return of that same top decile.
        avg_return_all: Mean forward return across all rows (the comparison).
    """

    n: int
    base_rate: float
    accuracy: float
    auc: float
    top_decile_hit_rate: float
    top_decile_avg_return: float
    avg_return_all: float

    def summary(self) -> str:
        """Render a short, plain-language scorecard."""
        edge = self.accuracy - self.base_rate
        lift = self.top_decile_avg_return - self.avg_return_all
        return (
            f"  rows scored:        {self.n:,}\n"
            f"  base rate (up):     {self.base_rate:6.2%}   <- the bar to beat\n"
            f"  accuracy:           {self.accuracy:6.2%}   "
            f"({edge:+.2%} vs base rate)\n"
            f"  AUC (ranking):      {self.auc:6.3f}    "
            f"(0.50 = coin flip)\n"
            f"  top-decile up-rate: {self.top_decile_hit_rate:6.2%}\n"
            f"  top-decile return:  {self.top_decile_avg_return:+6.2%}   "
            f"(all rows {self.avg_return_all:+.2%}, lift {lift:+.2%})"
        )


def _auc(labels: list[int], scores: list[float]) -> float:
    """Area under the ROC curve via scikit-learn, with a safe degenerate case."""
    if len(set(labels)) < 2:
        return 0.5  # only one class present; ranking is undefined, call it chance
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def evaluate(
    labels: list[int],
    scores: list[float],
    fwd_returns: list[float],
    decile: float = 0.10,
) -> ClassificationMetrics:
    """Score predictions against the truth and report the honest baselines.

    Args:
        labels: True ``0/1`` outcomes.
        scores: Predicted probability of the up-label, same length/order.
        fwd_returns: Actual forward returns for each row (for the lift metric).
        decile: Top fraction (by score) to measure lift over. Defaults to 0.10.

    Returns:
        A :class:`ClassificationMetrics`.

    Raises:
        ValueError: If the inputs are empty or of mismatched length.
    """
    n = len(labels)
    if n == 0:
        raise ValueError("no predictions to evaluate")
    if not (len(scores) == len(fwd_returns) == n):
        raise ValueError("labels, scores, and fwd_returns must be the same length")

    base_rate = sum(labels) / n
    accuracy = sum((s >= 0.5) == bool(y) for s, y in zip(scores, labels, strict=True)) / n
    auc = _auc(labels, scores)

    k = max(1, int(n * decile))
    top = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    top_hit = sum(labels[i] for i in top) / k
    top_ret = sum(fwd_returns[i] for i in top) / k
    avg_ret = sum(fwd_returns) / n

    return ClassificationMetrics(
        n=n,
        base_rate=base_rate,
        accuracy=accuracy,
        auc=auc,
        top_decile_hit_rate=top_hit,
        top_decile_avg_return=top_ret,
        avg_return_all=avg_ret,
    )
