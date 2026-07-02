from pathlib import Path

# Robot descriptions (unitree_description, pnd_description, ...) live in the
# repo-level ``assets/`` directory (a separate git repo), not inside the python
# package. ``assets.py`` is at ``<repo>/source/whole_body_tracking/whole_body_tracking/``
# so the repo root is ``parents[3]``.
ASSET_DIR = Path(__file__).resolve().parents[3] / "assets"
