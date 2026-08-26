"""Modules that call a cloud API at run time.

Kept apart from ``data/``, ``features/``, ``train/`` and ``evaluate/``, which are pure transforms
over local data. Nothing in the training pipeline imports from here, and nothing produced here is
a model input.
"""
