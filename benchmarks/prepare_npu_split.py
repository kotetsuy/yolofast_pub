# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""Extract FP32 boxes/scores BEFORE quantization and verify CPU equivalence."""
import argparse
import copy
from pathlib import Path
import numpy as np
import onnx
from npu_common import ROOT, cpu_session, load_cases, no_overwrite, sha256, write_json, exact_outputs, compare_detections
from npu_postprocess import decode

NAMES = ['/model/model.23/Mul_2_output_0', '/model/model.23/Sigmoid_output_0']
SHAPES = [[1, 4, 8400], [1, 80, 8400]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT/'models/npu/yolo26n-640.onnx')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--images-manifest', type=Path)
    a = p.parse_args()
    no_overwrite(a.output, a.output.with_suffix('.json'))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    m = onnx.load(a.input)
    producers = {v: n for n in m.graph.node for v in n.output}
    for name, op, shape in zip(NAMES, ['Mul', 'Sigmoid'], SHAPES):
        if name not in producers or producers[name].op_type != op:
            raise ValueError(f'Unexpected producer: {name}')
        known = [v for v in [*m.graph.input, *m.graph.output, *m.graph.value_info] if v.name == name]
        for v in known:
            dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
            if dims and all(dims) and dims != shape:
                raise ValueError(f'Unexpected shape: {name}: {dims}')
        if not known:
            m.graph.value_info.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
    diagnostic = copy.deepcopy(m)
    diagnostic.graph.output.extend([onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, s) for n, s in zip(NAMES, SHAPES)])
    extracted = onnx.utils.Extractor(m).extract_model(['images'], NAMES)
    onnx.checker.check_model(extracted)
    original = cpu_session(diagnostic.SerializeToString())
    candidate = cpu_session(extracted.SerializeToString())
    report = dict(source=str(a.input.resolve()), source_sha256=sha256(a.input),
                  precision='FP32', checks=[], accuracy_status='unqualified')
    for record, x in load_cases(a.images_manifest, synthetic=True):
        old = original.run(None, {'images': x})
        new = candidate.run(None, {'images': x})
        exact = exact_outputs(old[-2:], new)
        if [list(y.shape) for y in new] != SHAPES:
            raise ValueError('Output shape mismatch')
        check = dict(input=record, raw_exact=exact, detections_order_exact=bool(np.array_equal(old[0][0], decode(*new))),
                     detections=compare_detections(old[0][0], decode(*new), tolerance=0))
        report['checks'].append(check)
        if not exact or not check['detections']['passed']:
            write_json(a.output.with_suffix('.json'), dict(report, status='failed'))
            raise RuntimeError('FP32 split verification failed')
    onnx.save(extracted, a.output)
    report.update(status='complete', model_sha256=sha256(a.output))
    write_json(a.output.with_suffix('.json'), report)
    print(a.output)


if __name__ == '__main__':
    main()
