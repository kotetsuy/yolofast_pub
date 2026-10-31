# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""Validate saved final detections against same-model CPU and FP32 outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from npu_common import ROOT, cpu_session, load_cases, no_overwrite, sha256, write_json


def stats(y):
    d = y[0][y[0, :, 4] >= .25]
    return dict(finite=bool(np.isfinite(y).all()), detections=d.tolist(),
                inverted_boxes=int(np.sum(np.any(d[:, 2:4] < d[:, :2], axis=1))),
                noninteger_classes=int(np.sum(d[:, 5] != np.round(d[:, 5]))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--reference-model', type=Path, required=True)
    p.add_argument('--images-manifest', type=Path, required=True)
    a = p.parse_args()
    out = a.results / 'correctness'
    no_overwrite(out)
    out.mkdir()
    run = json.loads((a.results / 'result.json').read_text())
    model = Path(run['model'])
    if sha256(model) != run['sha256']:
        raise ValueError('Model has changed since benchmark')
    ref_path = a.reference_model
    q = cpu_session(model)
    # Match the benchmark reference session options, including thread defaults.
    fp = ort.InferenceSession(str(ref_path), providers=['CPUExecutionProvider'])
    saved_npu = np.load(a.results / 'quantized-output.npz')['output0']
    saved_fp = np.load(a.results / 'fp32-output.npz')['output0']
    report = dict(model=str(model), sha256=sha256(model),
                  reference_sha256=sha256(ref_path),
                  scope='Smoke validation; no independent labeled accuracy evaluation', cases=[])
    for i, (record, x) in enumerate(load_cases(a.images_manifest)):
        y = q.run(None, {q.get_inputs()[0].name: x})[0]
        f = fp.run(None, {fp.get_inputs()[0].name: x})[0]
        np.savez(out / f'case-{i}.npz', input=x, quantized_cpu=y, fp32_cpu=f)
        row = dict(input=record, quantized_cpu=stats(y), fp32_cpu=stats(f))
        if i == 0:
            # Reproducing the stored FP32 output verifies the preprocessing and input.
            row['saved_fp32_exact'] = bool(np.array_equal(f, saved_fp))
            if not row['saved_fp32_exact']:
                raise ValueError('Saved benchmark input/reference could not be reproduced')
            row['npu'] = stats(saved_npu)
            row['cpu_npu_exact'] = bool(np.array_equal(y, saved_npu))
            row['cpu_npu_max_abs_by_column'] = np.max(np.abs(y - saved_npu), axis=(0, 1)).tolist()
        report['cases'].append(row)
    m = onnx.load(model)
    producers = {v: n for n in m.graph.node for v in n.output}
    initializers = {v.name: numpy_helper.to_array(v) for v in m.graph.initializer}
    node = producers[m.graph.output[0].name]
    report['output_producer'] = dict(name=node.name, op_type=node.op_type, inputs=list(node.input))
    if node.op_type == 'DequantizeLinear':
        scale = initializers[node.input[1]]
        zero = initializers[node.input[2]]
        report['output_quantization'] = dict(scale=scale.tolist(), zero_point=zero.tolist(),
            class_5_after_qdq=( (np.round(np.float32(5) / scale) + zero - zero) * scale).tolist())
    report['correctness_passed'] = all(
        row[k]['finite'] and row[k]['inverted_boxes'] == 0 and row[k]['noninteger_classes'] == 0
        for row in report['cases'] for k in ['quantized_cpu', 'fp32_cpu', *(['npu'] if 'npu' in row else [])])
    write_json(out / 'validation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
