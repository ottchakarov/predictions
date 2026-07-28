"""L4 prediction models.

Each model exposes the same minimal interface so it is a drop-in for the others
in the pipeline:

* ``fit(matches)``                       -> self
* ``predict(home, away, neutral=...)``   -> {"home", "draw", "away"}
* ``match_probabilities(...)``           -> alias of ``predict`` (pipeline parity)
"""

from .dixon_coles import DixonColesConfig, DixonColesModel

__all__ = ["DixonColesModel", "DixonColesConfig"]
