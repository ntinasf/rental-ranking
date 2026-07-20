"""Aspect sentiment via Azure AI Language opinion mining, computed once and cached.

Aspect scores (cleanliness, location, communication, value) per listing. The API
bills per text record: sample reviews per listing (e.g. most recent 20), cache
results to Blob, one full run total. Never call the API inside a training loop.
"""

# TODO: sample the most recent ~20 reviews per listing.
# TODO: call Azure AI Language opinion mining in batches; credentials from .env.
# TODO: cache raw API responses to Blob before any aggregation.
# TODO: aggregate to per-listing aspect scores; neutral imputation + has_reviews flag
#       for listings without reviews (cold-start signal).
