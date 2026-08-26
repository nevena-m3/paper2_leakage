# Local data

Participant-level and recording-level data belong on the local machine and are not committed.

Expected working areas:

- `raw/` â€” frozen starting inputs, unchanged
- `interim/` â€” audit/merge intermediates
- `processed/` â€” canonical analysis matrices
- `manifests/` â€” de-identified machine-readable analysis manifests that may be versioned after review

Never place raw audio/video or identifiable participant data in Git.
