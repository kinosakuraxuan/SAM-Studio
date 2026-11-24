"""DAVIS evaluation metrics from the bundled official implementation."""

import sys

from training.data import ROOT

sys.path.append(str(ROOT / "vendor/davis2017-evaluation"))
from davis2017.metrics import db_eval_boundary as db_eval_boundary  # noqa: E402
from davis2017.metrics import db_eval_iou as db_eval_iou  # noqa: E402
