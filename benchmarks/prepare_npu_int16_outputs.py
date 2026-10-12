# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""Expose terminal INT16 tensors; retain all internal QDQ and verify on CPU."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
from npu_common import cpu_session, load_cases, no_overwrite, sha256, write_json, quantize_input, exact_outputs
from npu_postprocess import decode, dequantize, integer_decoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--images-manifest', type=Path)
    a = p.parse_args()
    no_overwrite(a.output, a.output.with_suffix('.json'))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(a.input.with_suffix('.json').read_text())
    if meta['model_sha256'] != sha256(a.input): raise ValueError('Input metadata mismatch')
    m = onnx.load(a.input)
    init = {v.name:numpy_helper.to_array(v) for v in m.graph.initializer}
    producers = {v:n for n in m.graph.node for v in n.output}
    outputs, records = [], []
    if len(m.graph.output) != 2: raise ValueError('Two outputs required')
    for output in m.graph.output:
        node = producers[output.name]
        if node.op_type != 'DequantizeLinear': raise ValueError('Expected terminal DQ')
        scale, zero = init[node.input[1]], init[node.input[2]]
        if zero.dtype != np.int16 or scale.dtype != np.float32 or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Expected INT16/FP32 output quantization')
        shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
        records.append(dict(source_name=output.name, name=node.input[0], scale=scale.tolist(), zero_point=zero.tolist(),
                            axis=next((onnx.helper.get_attribute_value(v) for v in node.attribute if v.name=='axis'),1), shape=shape, dtype='INT16'))
        outputs.append(onnx.helper.make_tensor_value_info(node.input[0],onnx.TensorProto.INT16,shape))
    for value in outputs:
        if not any(v.name == value.name for v in m.graph.value_info): m.graph.value_info.append(value)
    m = onnx.utils.Extractor(m).extract_model([v.name for v in m.graph.input], [v.name for v in outputs])
    onnx.checker.check_model(m)
    reference = Path(meta.get('source', a.input.with_name('yolo26n-640-a16w8-split.onnx'))).resolve()
    report = dict(meta, source=str(a.input.resolve()), source_sha256=sha256(a.input),
                  reference_model=str(reference), reference_sha256=sha256(reference),
                  output_quantization=records, checks=[], status='started')
    old, new = cpu_session(a.input), cpu_session(m.SerializeToString())
    decode_int = integer_decoder(records)
    scratch, q = np.empty((1,3,640,640),np.float32), np.empty((1,3,640,640),np.int16)
    for record, x in load_cases(a.images_manifest, synthetic=True):
        quantize_input(x,meta['scale'],meta['zero_point'],scratch,q)
        left, raw = old.run(None,{'images':q}), new.run(None,{'images':q})
        right = [dequantize(v,r) for v,r in zip(raw,records)]
        exact = exact_outputs(left,right)
        detections_exact = np.array_equal(decode(*left),decode_int(*raw))
        report['checks'].append(dict(input=record,raw_exact=bool(exact),detections_exact=bool(detections_exact)))
        if not exact or not detections_exact:
            write_json(a.output.with_suffix('.json'),dict(report,status='failed'))
            raise RuntimeError('Output conversion is not equivalent')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(m,a.output)
    report.update(model_sha256=sha256(a.output),status='complete')
    write_json(a.output.with_suffix('.json'),report)
    print(a.output)


if __name__ == '__main__': main()
