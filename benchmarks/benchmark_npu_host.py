# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""CPU helper microbenchmarks only. No inference and no NPU FPS estimate."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from npu_common import ROOT,sha256,write_json,no_overwrite,quantize_input,preprocess,preprocess_lut,make_pixel_lut
from npu_postprocess import decode,decode_partial,integer_decoder,dequantize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--saved-outputs',type=Path,required=True)
    p.add_argument('--quant-meta',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=1)
    p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    if a.seconds<=0 or a.repeats<1:p.error('positive durations/repeats required')
    no_overwrite(a.output);a.output.mkdir(parents=True)
    source=a.saved_outputs
    with np.load(source) as data:x,boxes,scores=[data[k].copy() for k in ['input','npu_boxes','npu_scores']]
    meta=json.loads(a.quant_meta.read_text())
    scratch,q=np.empty_like(x),np.empty(x.shape,np.int16)
    im=cv2.imread(str(a.image))
    if im is None: raise ValueError(f'Cannot read image: {a.image}')
    lut=make_pixel_lut(meta['scale'],meta['zero_point'])
    def quant(skip=False):return quantize_input(x,meta['scale'],meta['zero_point'],scratch,q,skip)
    def full_preprocess():
        return quantize_input(preprocess(im),meta['scale'],meta['zero_point'],scratch,q)
    groups={
        'topk_only': {'stable':lambda:decode(boxes,scores),'partial':lambda:decode_partial(boxes,scores)},
        'normalized_fp32_to_int16':{'reference':quant,'skip_zero':lambda:quant(True)},
        'decoded_image_to_int16':{'reference':full_preprocess,'pixel_lut':lambda:preprocess_lut(im,lut)}
    }
    report=dict(scope='CPU helper microbenchmarks; no NPU inference, no NPU FPS projection',
                source_sha256=sha256(source),seconds=a.seconds,repeats=a.repeats,groups={})
    for group,methods in groups.items():
        expected=next(iter(methods.values()))().copy()
        for name,fn in methods.items():
            if not np.array_equal(fn(),expected):raise RuntimeError(f'{group}/{name} mismatch')
        rows=[]
        for trial in range(a.repeats):
            names=list(methods)
            if trial%2:names.reverse()
            for name in names:
                fn=methods[name]
                for _ in range(100):fn()
                lat=[];start=time.perf_counter()
                while time.perf_counter()-start<a.seconds:
                    t=time.perf_counter();fn();lat.append((time.perf_counter()-t)*1000)
                elapsed=time.perf_counter()-start
                write_json(a.output/f'{group}-{name}-{trial}.json',lat)
                rows.append(dict(name=name,trial=trial,mean_ms=float(np.mean(lat)),p99_ms=float(np.percentile(lat,99)),calls=len(lat),elapsed_s=elapsed))
        report['groups'][group]=rows
        write_json(a.output/'result.json',report)
    print(a.output)


if __name__=='__main__':main()
