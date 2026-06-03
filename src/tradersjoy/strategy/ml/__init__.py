"""The runtime ML strategy: load a trained model and turn its scores into orders.

This package consumes a model built offline by :mod:`tradersjoy.ml`. It computes
the *same* features (via :mod:`tradersjoy.ml.features`) live, so there is no
train/serve skew, and ranks the universe by the model's predicted probability of
rising, buying the strongest names.
"""
