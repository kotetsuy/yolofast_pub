#!/usr/bin/env python3
"""Validate a fixed-shape YOLO26 ONNX with ROCm 10.1 MIGraphX on gfx1150."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

MODEL = Path.home() / 'yolofast/models/migraphx/yolo26n-448-fp32.onnx'
EP = 'MIGraphXExecutionProvider'


def compare(reference, actual, confidence, min_iou, max_score_diff):
    import numpy as np
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('Non-finite output')
    ref = reference[0][reference[0, :, 4] >= confidence]
    got = actual[0][actual[0, :, 4] >= confidence]
    if len(ref) != len(got):
        raise ValueError(f'Detection count mismatch: CPU={len(ref)}, MIGraphX={len(got)}')
    if not len(ref):
        raise ValueError('No detections above confidence threshold; use a suitable image or lower --confidence')
    unmatched = set(range(len(got)))
    ious, scores = [], []
    for a in ref:
        candidates = []
        for j in unmatched:
            b = got[j]
            if a[5] != b[5]:
                continue
            wh = np.maximum(0, np.minimum(a[2:4], b[2:4]) - np.maximum(a[:2], b[:2]))
            intersection = float(wh[0] * wh[1])
            area_a = float(np.prod(np.maximum(0, a[2:4] - a[:2])))
            area_b = float(np.prod(np.maximum(0, b[2:4] - b[:2])))
            union = area_a + area_b - intersection
            iou = intersection / union if union > 0 else 0.0
            score_diff = abs(float(a[4]) - float(b[4]))
            if iou >= min_iou and score_diff <= max_score_diff:
                candidates.append((iou, -score_diff, j))
        if not candidates:
            raise ValueError(f'No matching detection for class {int(a[5])}, score {a[4]:.6f}')
        iou, neg_diff, j = max(candidates)
        unmatched.remove(j)
        ious.append(iou)
        scores.append(-neg_diff)
    return {'count': len(ref), 'min_iou': min(ious), 'max_score_diff': max(scores)}


def make_input(path):
    import cv2
    import numpy as np
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    h, w = image.shape[:2]
    scale = min(448 / h, 448 / w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, top = (448 - nw) // 2, (448 - nh) // 2
    canvas = np.full((448, 448, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def run(args):
    import numpy as np
    import onnxruntime as ort
    rocm = Path(args.rocm_path).resolve()
    version = (rocm / '.info/version').read_text().strip()
    if not re.match(r'^10\.1(?:\.|$)', version):
        raise RuntimeError(f'Expected ROCm 10.1; {rocm} reports {version}')
    if os.environ.get('HSA_OVERRIDE_GFX_VERSION'):
        raise RuntimeError('Unset HSA_OVERRIDE_GFX_VERSION to verify the actual GPU architecture')
    env = os.environ.copy()
    env['ROCM_PATH'] = str(rocm)
    probe = subprocess.run([str(rocm / 'bin/rocminfo')], capture_output=True, text=True, env=env)
    if probe.returncode:
        raise RuntimeError(f'rocminfo failed (exit {probe.returncode}): {(probe.stderr or probe.stdout).strip()}')
    info = probe.stdout
    architectures = re.findall(r'\bName:\s*(gfx[0-9a-z]+)', info)
    if architectures != ['gfx1150']:
        raise RuntimeError(f'Expected exactly one visible gfx1150 GPU, found: {architectures}')
    if os.environ.get('ORT_MIGRAPHX_FP16_ENABLE', '0') not in ('', '0'):
        raise RuntimeError('Unset ORT_MIGRAPHX_FP16_ENABLE for FP32 validation')
    model = args.model.expanduser().resolve(strict=True)
    x = make_input(args.image)
    try:
        import onnxruntime_ep_migraphx as plugin
    except ImportError as exc:
        raise RuntimeError('Install the ROCm 10.1 onnxruntime-ep-migraphx plugin in this Python environment') from exc
    names, paths = plugin.get_ep_names(), plugin.get_library_paths()
    if not names or len(names) != len(paths):
        raise RuntimeError('MIGraphX plugin did not provide valid library paths')
    for name, path in zip(names, paths):
        ort.register_execution_provider_library(name, path)
    devices = [d for d in ort.get_ep_devices() if d.ep_name == EP]
    if len(devices) != 1:
        raise RuntimeError(f'Expected one MIGraphX EP device; found {len(devices)}')
    options = ort.SessionOptions()
    options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    options.add_provider_for_devices(devices, {})
    gpu = ort.InferenceSession(str(model), sess_options=options, providers=[])
    gpu.disable_fallback()
    if EP not in gpu.get_providers():
        raise RuntimeError(f'MIGraphX provider not active: {gpu.get_providers()}')
    inputs, outputs = gpu.get_inputs(), gpu.get_outputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 3, 448, 448] or inputs[0].type != 'tensor(float)' or len(outputs) != 1 or outputs[0].shape != [1, 300, 6] or outputs[0].type != 'tensor(float)':
        raise RuntimeError('Model must have FP32 input [1,3,448,448] and output [1,300,6]')
    cpu = ort.InferenceSession(str(model), providers=['CPUExecutionProvider'])
    feed = {inputs[0].name: x}
    expected = cpu.run(None, feed)[0]
    start = time.perf_counter()
    actual = gpu.run(None, feed)[0]
    elapsed = time.perf_counter() - start
    comparison = compare(expected, actual, args.confidence, args.min_iou, args.max_score_diff)
    packages = {}
    for package in ('onnxruntime', 'onnxruntime-ep-migraphx', 'migraphx-libs'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {'status': 'passed', 'rocm_path': str(rocm), 'rocm_version': version,
            'architecture': architectures[0], 'providers': gpu.get_providers(), 'cpu_fallback_disabled': True,
            'packages': packages, 'model': str(model), 'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
            'image': str(args.image.resolve()), 'confidence': args.confidence, 'min_iou_required': args.min_iou,
            'max_score_diff_allowed': args.max_score_diff, 'comparison': comparison,
            'first_inference_seconds': elapsed, 'note': 'CPU and MIGraphX run the same ONNX model; timing includes host transfers.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--image', type=Path, required=True, help='Image containing at least one detectable object')
    p.add_argument('--rocm-path', type=Path, default=Path('/opt/rocm/core'))
    p.add_argument('--confidence', type=float, default=0.25)
    p.add_argument('--min-iou', type=float, default=0.95)
    p.add_argument('--max-score-diff', type=float, default=0.03)
    p.add_argument('--output', type=Path, help='Optional JSON report path')
    args = p.parse_args()
    if not (0 <= args.confidence <= 1 and 0 <= args.min_iou <= 1 and 0 <= args.max_score_diff <= 1):
        p.error('Thresholds must be between 0 and 1')
    try:
        report = run(args)
    except Exception as exc:
        report = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
    result = json.dumps(report, indent=2, ensure_ascii=False)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + '\n', encoding='utf-8')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
