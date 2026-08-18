"""Modules that call a cloud API at run time.

Kept apart from ``data/``, ``features/``, ``train/`` and ``evaluate/`` — everything there is a
pure transform over local data, and that boundary is worth being able to see. Nothing in the
training pipeline imports from here, and nothing produced here is a model input.
"""
