# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""Reference and exact, tie-preserving postprocessing for fixed YOLO26n."""
import numpy as np


def decode(boxes, scores):
    """Original stable-sort implementation; keep as the reference."""
    boxes, scores = boxes[0].T, scores[0].T
    anchors = np.argsort(-scores.max(axis=1), kind='stable')[:300]
    flat = scores[anchors].reshape(-1)
    indices = np.argsort(-flat, kind='stable')[:300]
    return np.column_stack((boxes[anchors[indices // 80]], flat[indices], indices % 80)).astype(np.float32)


def stable_topk(values, k=300):
    values = np.asarray(values)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('top-k requires a finite vector')
    if not 0 < k <= values.size:
        raise ValueError('invalid k')
    threshold = np.partition(values, values.size - k)[values.size - k]
    greater = np.flatnonzero(values > threshold)
    equal = np.flatnonzero(values == threshold)[:k - greater.size]
    selected = np.concatenate((greater, equal))
    # Widen before negation, including the INT16 minimum.
    keys = values[selected].astype(np.int64) if values.dtype.kind in 'iu' else values[selected]
    return selected[np.lexsort((selected, -keys))]


def decode_partial(boxes, scores):
    boxes, scores = boxes[0].T, scores[0].T
    anchors = stable_topk(scores.max(axis=1))
    flat = scores[anchors].reshape(-1)
    indices = stable_topk(flat)
    return np.column_stack((boxes[anchors[indices // 80]], flat[indices], indices % 80)).astype(np.float32)


def dequantize(values, meta):
    scale = np.asarray(meta['scale'], dtype=np.float32)
    zero = np.asarray(meta['zero_point'], dtype=np.float32)
    if scale.size != 1:
        shape = [1] * values.ndim
        shape[meta['axis']] = scale.size
        scale = scale.reshape(shape)
        if zero.size != 1:
            zero = zero.reshape(shape)
    return (values.astype(np.float32) - zero) * scale


def scalar_order_preserved(meta):
    """Prove integer ordering including ties for all INT16 values."""
    if np.size(meta['scale']) != 1 or np.size(meta['zero_point']) != 1:
        return False
    if not np.isfinite(meta['scale']).all() or float(np.asarray(meta['scale']).item()) <= 0:
        return False
    codes = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16)
    decoded = dequantize(codes, meta)
    return bool(np.isfinite(decoded).all() and np.all(np.diff(decoded) > 0))


def decode_integer(boxes, scores, output_meta, partial=False):
    bm, sm = output_meta
    if not scalar_order_preserved(sm) or np.size(bm['scale']) != 1:
        return decode(dequantize(boxes, bm), dequantize(scores, sm))
    return integer_decoder(output_meta, partial)(boxes, scores)


def integer_decoder(output_meta, partial=False):
    """Validate scale ordering once, outside the timed loop."""
    bm, sm = output_meta
    if not scalar_order_preserved(sm) or np.size(bm['scale']) != 1:
        return lambda boxes, scores: decode(dequantize(boxes, bm), dequantize(scores, sm))
    def select(v):
        return stable_topk(v) if partial else np.argsort(-v.astype(np.int32), kind='stable')[:300]
    def run(boxes, scores):
        boxes, scores = boxes[0].T, scores[0].T
        anchors = select(scores.max(axis=1))
        flat = scores[anchors].reshape(-1)
        indices = select(flat)
        return np.column_stack((dequantize(boxes[anchors[indices // 80]], bm),
                                dequantize(flat[indices], sm), indices % 80)).astype(np.float32)
    return run
