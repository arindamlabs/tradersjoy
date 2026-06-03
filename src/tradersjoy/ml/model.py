"""A thin wrapper around the gradient-boosted-tree model we actually fit.

A *gradient-boosted tree* is, in plain English, a large pile of small yes/no
decision rules ("is the 20-day return above 3%? is volume unusually high?") that
the training process discovers automatically and stacks so each new rule mostly
fixes the previous pile's mistakes. It is the workhorse for tabular problems like
ours: a few thousand rows, a handful of numeric features, no images or text.

This wrapper exists so the rest of the system never imports scikit-learn
directly. It hides three things behind a small surface:

- the model choice (``HistGradientBoostingClassifier``) and its settings,
- the conversion to/from plain Python lists (numpy stays inside this file), and
- the saved-feature-order check, so a model trained on one feature layout can
  never be silently fed a different one at predict time.

Heavy imports (numpy, scikit-learn, joblib) are done lazily inside the methods so
that merely importing a strategy, or running ``tradersjoy --help``, stays fast
and does not require the ML stack to be present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tradersjoy.ml.features import FEATURE_NAMES

#: Default hyper-parameters. Kept deliberately modest: shallow-ish trees, a small
#: learning rate, and built-in L2 regularisation all lean against overfitting,
#: which is the dominant failure mode here. These are sane starting points, not
#: a tuned configuration; tuning is itself done under walk-forward, never by
#: peeking at the test years.
DEFAULT_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "max_depth": 3,
    "max_iter": 300,
    "l2_regularization": 1.0,
    "min_samples_leaf": 50,
    "early_stopping": False,
    "random_state": 7,
}


class GBMModel:
    """Fit/predict wrapper over a gradient-boosted-tree classifier.

    The model predicts the probability that a row's label is ``1`` (the stock
    rises over the forward horizon). It records the feature order it was trained
    on and refuses to predict on a mismatched layout.

    Attributes:
        feature_names: The feature columns, in order, the model expects.
        params: The hyper-parameters the underlying estimator was built with.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
    ) -> None:
        """Configure (but do not yet fit) the model.

        Args:
            params: Hyper-parameters to override :data:`DEFAULT_PARAMS`.
            feature_names: Expected feature order; defaults to the canonical
                :data:`~tradersjoy.ml.features.FEATURE_NAMES`.
        """
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.feature_names = tuple(feature_names)
        self._estimator: Any = None

    def fit(self, X: list[list[float]], y: list[int]) -> GBMModel:
        """Fit the model on a feature matrix and its labels.

        Args:
            X: Feature rows, each in :attr:`feature_names` order.
            y: Matching ``0/1`` labels.

        Returns:
            ``self``, fitted, for chaining.

        Raises:
            ValueError: If ``X`` is empty or ``X`` and ``y`` lengths differ.
        """
        if not X:
            raise ValueError("cannot fit on an empty feature matrix")
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)} labels")

        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier

        self._estimator = HistGradientBoostingClassifier(**self.params)
        self._estimator.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        """Probability of the up-label (class ``1``) for each feature row.

        Args:
            X: Feature rows in :attr:`feature_names` order.

        Returns:
            One probability in ``[0, 1]`` per row. An empty input yields an empty
            list.

        Raises:
            RuntimeError: If called before :meth:`fit` or :meth:`load`.
        """
        if self._estimator is None:
            raise RuntimeError("model is not fitted; call fit() or load() first")
        if not X:
            return []

        import numpy as np

        proba = self._estimator.predict_proba(np.asarray(X, dtype=float))
        # Column 1 is P(label == 1); guard the degenerate single-class case.
        classes = list(self._estimator.classes_)
        if 1 not in classes:
            return [0.0] * len(X)
        col = classes.index(1)
        return [float(p[col]) for p in proba]

    def save(self, path: str | Path) -> Path:
        """Persist the fitted model and its feature order to ``path``.

        Args:
            path: Destination file (parent directories are created).

        Returns:
            The path written.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        if self._estimator is None:
            raise RuntimeError("refusing to save an unfitted model")
        import joblib

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self._estimator,
                "feature_names": self.feature_names,
                "params": self.params,
            },
            dest,
        )
        return dest

    @classmethod
    def load(cls, path: str | Path) -> GBMModel:
        """Load a model previously written by :meth:`save`.

        Args:
            path: File to read.

        Returns:
            A fitted :class:`GBMModel` ready to :meth:`predict_proba`.

        Raises:
            ValueError: If the saved feature order does not match the current
                :data:`~tradersjoy.ml.features.FEATURE_NAMES` (the code and the
                model have drifted apart and predictions would be meaningless).
        """
        import joblib

        blob = joblib.load(Path(path))
        saved = tuple(blob["feature_names"])
        if saved != FEATURE_NAMES:
            raise ValueError(
                "saved model feature order does not match the current code: "
                f"{saved} != {FEATURE_NAMES}. Retrain before using this model."
            )
        model = cls(params=blob.get("params"), feature_names=saved)
        model._estimator = blob["estimator"]
        return model
