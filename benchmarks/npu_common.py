# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""Shared input preparation, provenance, and validation (no NPU dependency)."""
import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def no_overwrite(*paths):
    for path in paths:
        if Path(path).exists():
            raise FileExistsError(f'Refusing to overwrite {path}')


def letterbox(im):
    h, w = im.shape[:2]
    ratio = min(640 / h, 640 / w)
    nh, nw = round(h * ratio), round(w * ratio)
    canvas = np.full((640, 640, 3), 114, np.uint8)
    top, left = (640 - nh) // 2, (640 - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(im, (nw, nh))
    return canvas


def preprocess(im):
    return np.ascontiguousarray(letterbox(im)[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255


def load_cases(manifest=None, synthetic=False):
    if manifest:
        manifest = Path(os.path.expandvars(str(manifest))).expanduser().resolve()
        records = json.loads(manifest.read_text())['images']
        base = manifest.parent
    else:
        raise ValueError('Supply --images-manifest with your own images and SHA256 hashes')
    cases = []
    for record in records:
        record = dict(record)
        path = (base / Path(os.path.expandvars(record['path'])).expanduser()).resolve()
        if sha256(path) != record['sha256']:
            raise ValueError(f'Input hash mismatch: {path}')
        record['path'] = str(path)
        if record.get('kind', 'image') == 'npz':
            with np.load(path) as archive:
                x = archive['input'].copy()
        else:
            im = cv2.imread(str(path))
            if im is None:
                raise ValueError(f'Cannot read {path}')
            x = preprocess(im)
        if x.shape != (1, 3, 640, 640) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f'Invalid input: {path}')
        cases.append((record, np.ascontiguousarray(x)))
    if not cases:
        raise ValueError('Empty input manifest')
    if synthetic:
        for name, value in [('black', 0), ('white', 1)]:
            cases.append((dict(kind='synthetic', name=name, role='equivalence'), np.full((1, 3, 640, 640), value, np.float32)))
        cases.append((dict(kind='synthetic', name='seed101', role='equivalence'),
                      np.random.default_rng(101).random((1, 3, 640, 640), dtype=np.float32)))
    return cases


def quantize_input(x, scale, zero, scratch, output, skip_zero=False):
    np.divide(x, np.float32(scale), out=scratch)
    np.rint(scratch, out=scratch)
    if not (skip_zero and int(zero) == 0):
        np.add(scratch, np.float32(zero), out=scratch)
    np.clip(scratch, -32768, 32767, out=scratch)
    np.copyto(output, scratch, casting='unsafe')
    return output


def make_pixel_lut(scale, zero):
    pixels = np.arange(256, dtype=np.float32) / np.float32(255)
    return quantize_input(pixels, scale, zero, np.empty_like(pixels), np.empty(256, np.int16))


def preprocess_lut(im, lut):
    canvas = letterbox(im)
    return np.ascontiguousarray(lut[canvas[:, :, ::-1]].transpose(2, 0, 1)[None])


def compare_detections(reference, candidate, tolerance=.05):
    rd, gd = reference[reference[:, 4] >= .25], candidate[candidate[:, 4] >= .25]
    valid = bool(np.isfinite(candidate).all() and np.all(gd[:, 2:4] >= gd[:, :2])
                 and np.all(gd[:, 5] == np.round(gd[:, 5])))
    matches, used = [], set()
    for row in rd:
        choices = []
        for j, other in enumerate(gd):
            if j in used or row[5] != other[5]:
                continue
            wh = np.maximum(0, np.minimum(row[2:4], other[2:4]) - np.maximum(row[:2], other[:2]))
            inter = float(np.prod(wh))
            union = float(np.prod(row[2:4] - row[:2]) + np.prod(other[2:4] - other[:2]) - inter)
            choices.append((inter / max(union, 1e-9), j))
        if not choices:
            matches.append(dict(iou=0, score_diff=None))
            continue
        iou, j = max(choices)
        used.add(j)
        matches.append(dict(iou=iou, score_diff=float(abs(row[4] - gd[j, 4]))))
    passed = valid and len(rd) == len(gd) and all(m['iou'] >= .9 and m['score_diff'] is not None and m['score_diff'] <= tolerance for m in matches)
    return dict(passed=bool(passed), geometry_valid=valid, cpu_detections=rd.tolist(),
                npu_detections=gd.tolist(), matches=matches)


def exact_outputs(left, right):
    return len(left) == len(right) and all(np.isfinite(a).all() and np.isfinite(b).all()
                                          and np.array_equal(a, b) for a, b in zip(left, right))


def cpu_session(model):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    return ort.InferenceSession(model if isinstance(model, bytes) else str(model), so, providers=['CPUExecutionProvider'])


class Executor:
    """CPU memory I/O binding; never claims EP-internal zero-copy."""
    def __init__(self, session, mode='run'):
        self.session, self.mode = session, mode
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, 640, 640]:
            raise ValueError('Expected fixed batch=1 640 input')
        self.name = inputs[0].name
        types = {'tensor(float)': np.float32, 'tensor(int16)': np.int16}
        self.input = np.empty(inputs[0].shape, types[inputs[0].type])
        self.outputs = [np.empty(o.shape, types[o.type]) for o in session.get_outputs()]
        if [list(x.shape) for x in self.outputs] != [[1, 4, 8400], [1, 80, 8400]]:
            raise ValueError('Expected ordered boxes/scores outputs')
        if mode == 'iobinding':
            self.binding = session.io_binding()
            self.binding.bind_cpu_input(self.name, self.input)
            for node, buffer in zip(session.get_outputs(), self.outputs):
                self.binding.bind_output(node.name, 'cpu', 0, buffer.dtype.type, buffer.shape, buffer.ctypes.data)
        elif mode != 'run':
            raise ValueError(f'Unknown execution mode {mode}')

    def run(self):
        if self.mode == 'run':
            return self.session.run(None, {self.name: self.input})
        self.binding.synchronize_inputs()
        self.session.run_with_iobinding(self.binding)
        self.binding.synchronize_outputs()
        return self.outputs
