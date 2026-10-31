# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""Regression tests for tie order, quantization boundaries and buffer reuse."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
from npu_postprocess import decode,decode_partial,stable_topk,dequantize,integer_decoder
from npu_common import quantize_input,make_pixel_lut,preprocess,preprocess_lut,load_cases,Executor,cpu_session
from npu_split import profile_summary
from compare_npu_variants import accepted_result


class Correctness(unittest.TestCase):
    def test_failed_or_cpu_runs_cannot_win(self):
        good=dict(status='complete',execution_exact=[True]*12,validation_passed=True,
                  baseline_npu_exact=[True]*3,pooled_fps=100,runs=[{'frames':100}])
        self.assertTrue(accepted_result(good,0,True))
        for change in [dict(cpu_check=True),dict(args={'target':'X1'},target_verified=False),dict(status='failed'),dict(baseline_npu_exact=[]),
                       dict(baseline_npu_exact=[True,False,True]),dict(execution_exact=[]),
                       dict(validation_passed=False),dict(pooled_fps=0)]:
            self.assertFalse(accepted_result(dict(good,**change),0,True))
        self.assertFalse(accepted_result(good,-1,True))
        self.assertFalse(accepted_result(good,2,True))

    def test_topk_ties_and_signed_minimum(self):
        rng=np.random.default_rng(73)
        for values in [np.zeros(8400,np.float32),rng.integers(-32768,32768,8400,dtype=np.int16),
                       np.full(8400,-32768,np.int16),rng.integers(-3,4,24000).astype(np.float32)]:
            wide=values.astype(np.int64) if values.dtype.kind in 'iu' else values
            np.testing.assert_array_equal(stable_topk(values),np.argsort(-wide,kind='stable')[:300])
        values=np.zeros(8400,np.float32);values[::28]=1
        np.testing.assert_array_equal(stable_topk(values),np.argsort(-values,kind='stable')[:300])
        for bad in [np.array([np.nan]),np.array([np.inf])]:
            with self.assertRaises(ValueError):stable_topk(bad,1)

    def test_decode_saved_and_random(self):
        rng=np.random.default_rng(83)
        boxes=rng.random((1,4,8400),dtype=np.float32)
        for scores in [np.zeros((1,80,8400),np.float32),rng.integers(0,4,(1,80,8400)).astype(np.float32),rng.random((1,80,8400),dtype=np.float32)]:
            np.testing.assert_array_equal(decode(boxes,scores),decode_partial(boxes,scores))

    @unittest.skipUnless(os.environ.get('YOLOFAST_SAVED_OUTPUTS'), 'Set YOLOFAST_SAVED_OUTPUTS to three external outputs-*.npz fixtures')
    def test_saved_outputs(self):
        paths=sorted(Path(os.environ['YOLOFAST_SAVED_OUTPUTS']).glob('outputs-*.npz'))
        self.assertEqual(len(paths),3)
        for p in paths:
            with np.load(p) as z:
                for device in ['cpu','npu']:
                    b,s=z[device+'_boxes'],z[device+'_scores']
                    np.testing.assert_array_equal(decode(b,s),decode_partial(b,s))

    def test_integer_decode_order_and_fallback(self):
        rng=np.random.default_rng(2)
        boxes=rng.integers(-32768,32768,(1,4,8400),dtype=np.int16)
        scores=rng.integers(-32768,32768,(1,80,8400),dtype=np.int16)
        for meta in [[dict(scale=.025,zero_point=0,axis=1)]*2,
                     [dict(scale=.025,zero_point=0,axis=1),dict(scale=[.01]*80,zero_point=[0]*80,axis=1)]]:
            expected=decode(dequantize(boxes,meta[0]),dequantize(scores,meta[1]))
            for partial in [False,True]:
                np.testing.assert_array_equal(integer_decoder(meta,partial)(boxes,scores),expected)

    def test_quantization_half_steps_and_saturation(self):
        scale=np.float32(3.0518509447574615e-05)
        half=(np.arange(-32770,32771,dtype=np.float32)+np.float32(.5))*scale
        x=np.concatenate([half,np.nextafter(half,np.float32(-np.inf)),np.nextafter(half,np.float32(np.inf)),np.array([-2,-0.,0.,2],np.float32)])
        expected=np.clip(np.rint(x/scale)+np.float32(0),-32768,32767).astype(np.int16)
        for skip in [False,True]:
            got=quantize_input(x,scale,0,np.empty_like(x),np.empty(x.shape,np.int16),skip)
            np.testing.assert_array_equal(got,expected)
        # Nonzero zero-point must not be omitted.
        expected=np.clip(np.rint(x/scale)+np.float32(7),-32768,32767).astype(np.int16)
        np.testing.assert_array_equal(quantize_input(x,scale,7,np.empty_like(x),np.empty(x.shape,np.int16),True),expected)

    def test_pixel_lut_and_letterbox(self):
        import cv2
        scale=np.float32(3.0518509447574615e-05)
        lut=make_pixel_lut(scale,0)
        values=np.arange(256,dtype=np.float32)/np.float32(255)
        np.testing.assert_array_equal(lut,np.clip(np.rint(values/scale),-32768,32767).astype(np.int16))
        rng=np.random.default_rng(31)
        for shape in [(480,640,3),(720,320,3),(640,640,3)]:
            im=rng.integers(0,256,shape,dtype=np.uint8);x=preprocess(im)
            expected=quantize_input(x,scale,0,np.empty_like(x),np.empty(x.shape,np.int16))
            np.testing.assert_array_equal(preprocess_lut(im,lut),expected)

    def test_profile_excludes_warmup(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'profile.json'
            events=[]
            for i in range(150):
                events.extend([dict(name='model_run',ph='X',ts=i*100,dur=80),
                               dict(name='kernel',cat='Node',ts=i*100+10,dur=10 if i<100 else 20,args={'provider':'test','op_name':'test'})])
            p.write_text(json.dumps(events))
            r=profile_summary(p,100,50)
            self.assertEqual(r['node_events'][0]['calls'],50)
            self.assertAlmostEqual(r['node_events'][0]['mean_ms'],.02)


if __name__=='__main__':unittest.main()
