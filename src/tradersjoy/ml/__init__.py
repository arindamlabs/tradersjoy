"""Model-building pipeline: features, labels, datasets, the GBM, walk-forward.

Everything in this package is about *building and evaluating* a model offline.
The trained model is consumed at trade time by :mod:`tradersjoy.strategy.ml`,
which turns its scores into orders.
"""
