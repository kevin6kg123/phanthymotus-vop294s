"""
plugins/vop294.py — board #294 (VOP object recognition) model bundle + decoding.

What this adds on top of the stock vop plugin:

* A size/SHA256-pinned engine bundle served from an internal HTTP mirror
  (`vop294_bundle.json` beside this file lists the files and sources). The
  judge cannot reliably reach GitHub, and the stock COS bundle is the 144-class
  zero-shot engine; ours is fine-tuned on the 220 board classes.
* `vocab.json` carries, besides the class list, per-class confidence
  thresholds (tuned with cross-validation for Macro-F1 @ IoU 0.7), the decode
  mode and the NMS IoU.
* Two decoders: `e2e` (the NMS-free one2one head, (1,300,6), same as the stock
  runtime) and `nms` (the one2many head (1,4+nc,N) + class-aware NMS, fully
  vectorised in numpy; ~3 Macro-F1 points better offline).

Everything is opt-in: without the bundle json the stock behaviour is unchanged.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_JSON = os.path.join(_HERE, "vop294_bundle.json")


def bundle_spec() -> Optional[dict]:
    if os.environ.get("VOP294_DISABLE") == "1" or not os.path.isfile(BUNDLE_JSON):
        return None
    with open(BUNDLE_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def ensure_bundle(progress_cb=None) -> tuple[str, str]:
    """Download/verify the pinned bundle; return (engine_path, vocab_path)."""
    from utils.model_downloader import ensure_verified_bundle

    spec = bundle_spec()
    family = _family()
    entry = spec["families"][family]
    model_dir = os.path.join(os.environ.get("VOP_MODEL_DIR", "/models/vop"), f"vop294-{spec['version']}-{family}")
    sources = [s.rstrip("/") + f"/{family}" for s in spec["sources"]]
    log.info(f"[vop294] bundle {spec['version']} family={family} sources={sources}")
    paths = ensure_verified_bundle(f"vop294/{family}", model_dir, sources, entry["files"],
                                   progress_cb=progress_cb)
    engine = next(p for n, p in paths.items() if n.endswith(".engine"))
    return engine, paths["vocab.json"]


def _family() -> str:
    """jp61 (TensorRT 10.x) or jp511 (TensorRT 8.x) — an engine only loads on the TRT that built it."""
    fam = os.environ.get("VOP294_FAMILY")
    if fam:
        return fam
    try:
        import tensorrt as trt
        return "jp61" if int(trt.__version__.split(".")[0]) >= 10 else "jp511"
    except Exception:  # noqa: BLE001
        return "jp61"


class BoardMeta:
    """Per-class thresholds and decode settings read from vocab.json."""

    def __init__(self, classes: list[str], thresholds: dict, default_conf: float,
                 decode: str = "e2e", nms_iou: float = 0.7, max_det: int = 300):
        self.classes = classes
        self.decode = decode
        self.nms_iou = float(nms_iou)
        self.max_det = int(max_det)
        self.thr = np.array([float(thresholds.get(c, default_conf)) for c in classes], np.float32)
        self.min_thr = float(self.thr.min()) if len(self.thr) else float(default_conf)

    @classmethod
    def from_vocab(cls, path: Optional[str]) -> Optional["BoardMeta"]:
        if not path or not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if not isinstance(d, dict) or "thresholds" not in d:
            return None
        return cls(list(d["classes"]), d["thresholds"], float(d.get("default_conf", 0.15)),
                   d.get("decode", "e2e"), d.get("nms_iou", 0.7), d.get("max_det", 300))


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy NMS, vectorised per kept box. boxes are pre-offset per class."""
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        ov = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[ov <= iou]
    return np.asarray(keep, dtype=np.int64)


def decode(outputs, meta_lb, board: BoardMeta):
    """Return (boxes_xyxy_orig_pixels, scores, class_ids) after per-class thresholds."""
    from plugins.vision_runtime import decode_detections, undo_letterbox

    if board.decode == "nms":
        arr = None
        for a in (outputs if isinstance(outputs, (list, tuple)) else [outputs]):
            a = np.asarray(a)
            if a.ndim == 3 and a.shape[1] == 4 + len(board.classes):
                arr = a[0]
                break
        if arr is not None:
            cls_scores = arr[4:]                                 # (nc, N)
            cid = cls_scores.argmax(0)
            sc = cls_scores[cid, np.arange(cls_scores.shape[1])]
            m = sc >= board.thr[cid]
            if not m.any():
                e = np.empty((0, 4), np.float32)
                return e, np.empty(0, np.float32), np.empty(0, np.int32)
            xywh, sc, cid = arr[:4, m].T, sc[m], cid[m]
            xyxy = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
            off = cid[:, None].astype(np.float32) * 8192.0           # class-aware NMS
            keep = _nms_xyxy(xyxy + off, sc, board.nms_iou)[: board.max_det]
            return (undo_letterbox(xyxy[keep].astype(np.float32), meta_lb),
                    sc[keep].astype(np.float32), cid[keep].astype(np.int32))
        log.warning("[vop294] decode=nms but no one2many output found; falling back to e2e")
    boxes, scores, classes = decode_detections(outputs, meta_lb, board.min_thr)
    if len(scores):
        keep = scores >= board.thr[np.clip(classes, 0, len(board.thr) - 1)]
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    return boxes, scores, classes
