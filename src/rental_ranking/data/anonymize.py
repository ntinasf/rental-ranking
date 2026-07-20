"""Strip/hash host and reviewer PII before anything is committed or published.

Ported conceptually from the Thessaloniki project. Must run on every dataset
before it leaves the local environment (commits, Blob uploads, notebook outputs).
"""

# TODO: drop or hash host names, host IDs (hash, keep joinable), reviewer names and IDs.
# TODO: remove free-text fields that can contain PII (host_about, etc.) or scrub them.
# TODO: make the transformation deterministic so re-runs produce identical output.
