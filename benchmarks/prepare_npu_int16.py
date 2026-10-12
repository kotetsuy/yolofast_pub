# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""Remove only input Q; validate CPU equality before publishing a new model."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
from npu_common import ROOT, cpu_session, load_cases, no_overwrite, sha256, write_json, quantize_input, exact_outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT/'models/npu/yolo26n-640-a16w8-split.onnx')
    p.add_argument('--output', type=Path)
    p.add_argument('--images-manifest', type=Path)
    a = p.parse_args()
    source = a.input.resolve()
    target = a.output or source.with_name(source.stem + '-int16.onnx')
    no_overwrite(target, target.with_suffix('.json'))
    target.parent.mkdir(parents=True, exist_ok=True)
    m = onnx.load(source)
    if len(m.graph.input) != 1 or m.graph.input[0].name != 'images':
        raise ValueError('Expected one images input')
    consumers = [n for n in m.graph.node if 'images' in n.input]
    if len(consumers) != 1 or consumers[0].op_type != 'QuantizeLinear':
        raise ValueError('Expected a single input QuantizeLinear')
    q = consumers[0]
    init = {x.name: numpy_helper.to_array(x) for x in m.graph.initializer}
    scale, zero = init[q.input[1]], init[q.input[2]]
    if scale.size != 1 or zero.size != 1 or zero.dtype != np.int16 or not np.isfinite(scale).all() or scale.item() <= 0:
        raise ValueError('Expected positive scalar INT16 quantization')
    output = q.output[0]
    m.graph.node.remove(q)
    for n in m.graph.node:
        for i, name in enumerate(n.input):
            if name == output:
                n.input[i] = 'images'
    m.graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.INT16
    for v in list(m.graph.value_info):
        if v.name == output:
            m.graph.value_info.remove(v)
    onnx.checker.check_model(m)
    meta = dict(scale=float(scale.item()), zero_point=int(zero.item()), source=str(source),
                source_sha256=sha256(source), checks=[], input_type='INT16')
    old, new = cpu_session(source), cpu_session(m.SerializeToString())
    scratch = np.empty((1, 3, 640, 640), np.float32)
    quantized = np.empty(scratch.shape, np.int16)
    for record, x in load_cases(a.images_manifest, synthetic=True):
        quantize_input(x, scale.item(), zero.item(), scratch, quantized)
        left, right = old.run(None, {'images': x}), new.run(None, {'images': quantized})
        exact = exact_outputs(left, right)
        meta['checks'].append(dict(input=record, exact=bool(exact), max_abs=[float(np.max(np.abs(u-v))) for u,v in zip(left,right)]))
        if not exact:
            write_json(target.with_suffix('.json'), dict(meta, status='failed'))
            raise RuntimeError('INT16 input CPU equivalence failed')
    onnx.save(m, target)
    meta.update(status='complete', model_sha256=sha256(target))
    write_json(target.with_suffix('.json'), meta)
    print(target)


if __name__ == '__main__':
    main()
