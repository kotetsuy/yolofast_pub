# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""Separate quantization error from NPU error; CPU-only, no speed claims."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
from npu_common import ROOT,cpu_session,load_cases,write_json,sha256,compare_detections,no_overwrite
from npu_postprocess import decode


def output_scales(path):
    m=onnx.load(path)
    producers={v:n for n in m.graph.node for v in n.output}
    init={v.name:numpy_helper.to_array(v) for v in m.graph.initializer}
    result=[]
    for value in m.graph.output:
        node=producers[value.name]
        if node.op_type!='DequantizeLinear': raise ValueError('Expected output DQ')
        result.append(dict(name=value.name,scale=init[node.input[1]].tolist(),zero_point=init[node.input[2]].tolist()))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fp32',type=Path,required=True)
    p.add_argument('--v1',type=Path,required=True)
    p.add_argument('--v2',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--images-manifest',type=Path,required=True)
    a=p.parse_args();no_overwrite(a.output)
    a.output.mkdir(parents=True)
    sessions={k:cpu_session(getattr(a,k)) for k in ['fp32','v1','v2']}
    report=dict(accuracy_status='unqualified: smoke inputs include calibration images, no independent labels',
                models={k:dict(path=str(getattr(a,k)),sha256=sha256(getattr(a,k))) for k in sessions},
                output_quantization={k:output_scales(getattr(a,k)) for k in ['v1','v2']},images=[])
    for i,(record,x) in enumerate(load_cases(a.images_manifest,synthetic=True)):
        raw={k:s.run(None,{'images':x}) for k,s in sessions.items()}
        np.savez(a.output/f'outputs-{i}.npz',input=x,**{f'{k}_{t}':v for k,values in raw.items() for t,v in zip(['boxes','scores'],values)})
        row=dict(input=record,comparisons={})
        for k in ['v1','v2']:
            ref,got=raw['fp32'],raw[k]
            row['comparisons'][k]=dict(raw_max_abs=[float(np.max(np.abs(r-g))) for r,g in zip(ref,got)],
                score_unique_values=int(np.unique(got[1]).size), detections=compare_detections(decode(*ref),decode(*got)))
        report['images'].append(row)
    write_json(a.output/'validation.json',report)
    print(report['output_quantization'])


if __name__=='__main__':main()
