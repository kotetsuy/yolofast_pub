# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""Validate fixed YOLO26n split outputs before timing the NPU."""
import argparse
import json
import platform
import os
import sys
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from npu_common import (ROOT, Executor, compare_detections, cpu_session, exact_outputs,
                        load_cases, quantize_input, sha256, write_json)
from npu_postprocess import decode, decode_partial, dequantize, integer_decoder

V1 = ROOT / 'models/npu/yolo26n-640-a16w8-split.onnx'
V1_HASH = '02361c4aab579cb3adb72e301fd2db2196ff391e0dc8cd7a31e2f5b2cface383'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--int16-input', action='store_true')
    p.add_argument('--model', type=Path)
    p.add_argument('--reference-model', type=Path, help='FP32-input version of the SAME quantized model')
    p.add_argument('--quant-meta', type=Path)
    p.add_argument('--execution', choices=['run', 'iobinding'], default='run')
    p.add_argument('--postprocess', choices=['stable', 'partial'], default='stable')
    p.add_argument('--quantization', choices=['reference', 'skip-zero'], default='reference')
    p.add_argument('--images-manifest', type=Path)
    p.add_argument('--intra-op-threads', type=int)
    p.add_argument('--profile-runs', type=int, default=50)
    p.add_argument('--opt-level', type=int, choices=[0, 1, 2, 3, 65536], default=None)
    p.add_argument('--target', choices=['X2', 'X1'])
    p.add_argument('--phase-file', type=Path, help='Diagnostic lifecycle journal; not for formal FPS')
    p.add_argument('--phase-ack-dir', type=Path, help='Collector handshake directory (requires phase-file)')
    p.add_argument('--skip-profile', action='store_true')
    p.add_argument('--disk-cache', action='store_true')
    p.add_argument('--compare-npu', type=Path)
    p.add_argument('--diagnostic-score-tolerance', type=float)
    p.add_argument('--cpu-check', action='store_true', help='CPU correctness only; NEVER reports NPU FPS')
    return p


class Pipeline:
    def __init__(self, session, args, meta):
        self.executor = Executor(session, args.execution)
        self.meta, self.args = meta, args
        self.scratch = np.empty((1, 3, 640, 640), np.float32)
        self.integer_input = self.executor.input.dtype == np.int16
        self.output_meta = meta.get('output_quantization')
        if self.integer_input and not {'scale', 'zero_point'} <= meta.keys():
            raise ValueError('INT16 input requires quantization metadata')
        if self.output_meta:
            if len(self.output_meta) != 2 or any(v.dtype != np.int16 for v in self.executor.outputs):
                raise ValueError('INT16 output metadata mismatch')
            self.decode = integer_decoder(self.output_meta, args.postprocess == 'partial')
        else:
            if any(v.dtype != np.float32 for v in self.executor.outputs):
                raise ValueError('Integer outputs require metadata')
            self.decode = decode_partial if args.postprocess == 'partial' else decode

    def prepare(self, x):
        if self.integer_input:
            quantize_input(x, self.meta['scale'], self.meta['zero_point'], self.scratch,
                           self.executor.input, self.args.quantization == 'skip-zero')
        elif self.args.execution == 'run':
            self.executor.input = x  # original FP32 run path has no input copy
        else:
            np.copyto(self.executor.input, x)

    def raw_float(self, raw):
        return [dequantize(v, m) for v, m in zip(raw, self.output_meta)] if self.output_meta else raw

    def run(self, x):
        self.prepare(x)
        return self.decode(*self.executor.run())


def profile_summary(path, warmup, count):
    events = json.loads(Path(path).read_text())
    runs = sorted([e for e in events if e.get('name') == 'model_run' and e.get('ph') == 'X'], key=lambda e:e['ts'])
    selected = runs[warmup:warmup + count]
    if len(selected) != count:
        raise RuntimeError(f'Cannot identify measured profile runs: {len(runs)} total')
    nodes = {}
    for event in events:
        if event.get('cat') != 'Node' or not any(r['ts'] <= event.get('ts', -1) < r['ts'] + r.get('dur', 0) for r in selected):
            continue
        key = (event.get('name'), event.get('args', {}).get('provider'), event.get('args', {}).get('op_name'))
        entry = nodes.setdefault(key, [])
        entry.append(event.get('dur', 0) / 1000)
    return dict(warmup_runs=warmup, measured_runs=count,
                clock='ORT host timestamps, microseconds converted to milliseconds',
                model_run_mean_ms=float(np.mean([r['dur']/1000 for r in selected])),
                device_compute_ms=None, dma_ms=None, pdi_load_ms=None,
                node_events=[dict(name=k[0], provider=k[1], op=k[2], calls=len(v), mean_ms=float(np.mean(v))) for k,v in nodes.items()],
                limitation='Host EP events, not pure NPU device time; overlapping events are not summed.')


def main():
    p = parser()
    a = p.parse_args()
    if a.phase_ack_dir and not a.phase_file: p.error('phase-ack-dir requires phase-file')
    if a.seconds <= 0 or a.repeats < 1 or a.profile_runs < 50 or (a.intra_op_threads is not None and a.intra_op_threads < 1):
        p.error('positive durations/repeats/threads and profile-runs>=50 required')
    if a.diagnostic_score_tolerance not in (None, .05, .051):
        p.error('only the legacy diagnostic tolerance 0.051 is allowed')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').exists():
        raise FileExistsError(f'Refusing to overwrite {out}/result.json')
    model = (a.model or (V1.with_name(V1.stem+'-int16.onnx') if a.int16_input else V1)).resolve()
    report = dict(status='started', ort=ort.__version__, python=platform.python_version(),
                  model=str(model), model_sha256=sha256(model), validation=[], runs=[],
                  validation_scope='Smoke and equivalence only; independent mAP untested.',
                  accuracy_status='unqualified', cpu_check=a.cpu_check,
                  boundary='CPU normalized FP32 input -> external quantization -> ORT -> CPU top-k; excludes preprocessing/JSON',
                  npu_execution='not measured' if a.cpu_check else 'pending profile review',
                  args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                  telemetry={k:dict(value=None, reason='No telemetry collector configured') for k in ['temperature','clock','power','other_load']})
    def phase(name):
        if not a.phase_file: return
        a.phase_file.parent.mkdir(parents=True, exist_ok=True)
        with a.phase_file.open('a') as journal:
            journal.write(json.dumps(dict(phase=name, monotonic_ns=time.monotonic_ns(), pid=os.getpid()))+'\n')
        if a.phase_ack_dir:
            deadline=time.monotonic()+120
            while not (a.phase_ack_dir/name).exists():
                if time.monotonic()>deadline: raise TimeoutError('Collector did not acknowledge '+name)
                time.sleep(.05)
    def save(): write_json(out/'result.json', report)
    save()
    try:
        meta = {}
        meta_path = a.quant_meta or model.with_suffix('.json')
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get('model_sha256') != report['model_sha256']:
                raise ValueError('Quantization metadata/model hash mismatch')
        elif a.quant_meta or a.int16_input:
            raise ValueError('Quantization metadata missing')
        if 'scale' in meta:
            reference = (a.reference_model or Path(meta.get('reference_model', meta.get('source', str(V1))))).resolve()
            expected = meta.get('reference_sha256', meta.get('source_sha256'))
        else:
            reference = (a.reference_model or model).resolve()
            expected = None
        if expected and sha256(reference) != expected:
            raise ValueError('Reference model hash mismatch')
        if a.diagnostic_score_tolerance == .051 and sha256(reference) != V1_HASH:
            raise ValueError('0.051 is restricted to original v1 and equivalent transformations')
        report.update(reference_model=str(reference), reference_sha256=sha256(reference))
        if not a.cpu_check and 'VitisAIExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('VitisAI EP unavailable')
        # Fail before a hidden CPU fallback or an expensive compile without hardware.
        if not a.cpu_check and not list(Path('/dev/accel').glob('accel*')):
            raise RuntimeError('NPU device nodes unavailable; use host script or --cpu-check for correctness only')
        cpu = cpu_session(reference)
        opts = {'cache_dir': str(out/'cache')}
        if a.opt_level is not None: opts['opt_level'] = str(a.opt_level)
        if a.target is not None: opts['target'] = a.target
        if a.disk_cache: opts['enable_cache_file_io_in_mem'] = '0'
        report['provider_options'] = opts
        def session(profile=False):
            so = ort.SessionOptions()
            if a.intra_op_threads is not None: so.intra_op_num_threads = a.intra_op_threads
            elif a.cpu_check: so.intra_op_num_threads = 4
            if profile:
                so.enable_profiling = True
                so.profile_file_prefix = str(out/'profile')
            if a.cpu_check:
                return ort.InferenceSession(str(model), so, providers=['CPUExecutionProvider'])
            result = ort.InferenceSession(str(model), so, providers=['VitisAIExecutionProvider','CPUExecutionProvider'], provider_options=[opts,{}])
            result.disable_fallback()
            if 'VitisAIExecutionProvider' not in result.get_providers(): raise RuntimeError('Session fell back to CPU')
            return result
        t = time.perf_counter()
        npu = session()
        pipeline = Pipeline(npu, a, meta)
        if a.int16_input and not pipeline.integer_input:
            raise ValueError('--int16-input requires an INT16 model')
        report.update(init_s=time.perf_counter()-t, int16_input=pipeline.integer_input)
        if a.target and not a.cpu_check:
            contexts = list((out/'cache').rglob('context.json'))
            observed = sorted(set(json.loads(f.read_text()).get('config', {}).get('target', '') for f in contexts))
            report['compiled_targets'] = observed
            report['target_verified'] = bool(observed and observed == [a.target])
            if not report['target_verified']:
                raise RuntimeError(f'Requested target {a.target} not confirmed by cache contexts: {observed}')
        report['session_io'] = {kind:[dict(name=v.name, shape=v.shape, dtype=v.type) for v in values]
                                for kind,values in [('inputs',npu.get_inputs()),('outputs',npu.get_outputs())]}
        phase('loaded')
        cases = load_cases(a.images_manifest)
        if len(cases) < 3: raise ValueError('At least three smoke images required')
        write_json(out/'manifest.json', dict(command=sys.argv, images=[c[0] for c in cases], model_sha256=sha256(model),
                   reference_sha256=sha256(reference), code={str(f.relative_to(ROOT)):sha256(f) for f in (ROOT/'benchmarks').glob('npu*.py')},
                   provider_options=opts, input_metadata=meta, boundary=report['boundary'], python=sys.version, ort=ort.__version__))
        all_pass, diagnostic_pass, reference_count = True, True, 0
        for i, (record, x) in enumerate(cases):
            ref = cpu.run(None, {'images':x})
            pipeline.prepare(x)
            raw = pipeline.executor.run()
            got = pipeline.raw_float(raw)
            gd = pipeline.decode(*raw)
            reference_decode = decode(*got)
            post_exact = bool(np.array_equal(gd, reference_decode))
            np.savez(out/f'outputs-{i}.npz', cpu_boxes=ref[0],cpu_scores=ref[1],npu_boxes=got[0],npu_scores=got[1],input=x)
            if not post_exact: raise RuntimeError('Postprocess differs from stable reference')
            if a.compare_npu:
                with np.load(a.compare_npu/f'outputs-{i}.npz') as prior:
                    exact = bool(np.array_equal(x,prior['input']) and exact_outputs(got,[prior['npu_boxes'],prior['npu_scores']])
                                 and np.array_equal(gd,decode(prior['npu_boxes'],prior['npu_scores'])))
                report.setdefault('baseline_npu_exact',[]).append(exact)
                if not exact: raise RuntimeError('Candidate differs from baseline NPU outputs')
            result = compare_detections(decode(*ref), gd)
            result.update(image=record['path'], raw_max_abs=[float(np.max(np.abs(r-g))) for r,g in zip(ref,got)],
                          raw_finite=bool(all(np.isfinite(z).all() for z in got)), postprocess_exact=post_exact)
            result['passed'] &= result['raw_finite']
            report['validation'].append(result)
            all_pass &= result['passed']
            diagnostic_pass &= result['raw_finite'] and compare_detections(decode(*ref), gd, a.diagnostic_score_tolerance or .05)['passed']
            reference_count += len(result['cpu_detections'])
            if a.cpu_check and not exact_outputs(ref,got): raise RuntimeError('CPU transformation is not exactly equivalent')
            save()
        # Fresh/alternating inputs: protect binding, quantization, and output lifetime.
        extra = load_cases(a.images_manifest, synthetic=True)
        report['execution_exact'] = []
        for record,x in extra + list(reversed(extra)):
            pipeline.prepare(x)
            direct = npu.run(None, {pipeline.executor.name:pipeline.executor.input})
            expected = [v.copy() for v in direct]
            actual = pipeline.executor.run()
            ok = exact_outputs(expected, actual) and np.array_equal(pipeline.decode(*actual),decode(*pipeline.raw_float(expected)))
            report['execution_exact'].append(bool(ok))
            if not ok:
                np.savez(out/'execution-mismatch.npz', input=x, expected_boxes=expected[0],expected_scores=expected[1],actual_boxes=actual[0],actual_scores=actual[1])
                raise RuntimeError('Execution path mismatch')
        all_pass = bool(all_pass and reference_count > 0)
        diagnostic_pass = bool(diagnostic_pass and reference_count > 0)
        report.update(validation_passed=all_pass, diagnostic_gate_passed=diagnostic_pass,
                      diagnostic_score_tolerance=a.diagnostic_score_tolerance,
                      performance_qualification='smoke-test passed; mAP untested' if all_pass else 'diagnostic only; strict output validation FAILED')
        save()
        if not all_pass and not (a.diagnostic_score_tolerance is not None and diagnostic_pass):
            raise RuntimeError('Strict output validation failed; timing skipped')
        if a.cpu_check:
            report.update(status='complete', performance_qualification='CPU correctness only; NPU not measured')
            save()
            return
        x = cases[0][1]
        phase('warmup')
        for _ in range(100): pipeline.run(x)
        phase('steady')
        for trial in range(1,a.repeats+1):
            lat, start = [], time.perf_counter()
            while time.perf_counter()-start < a.seconds:
                t = time.perf_counter(); pipeline.run(x); lat.append((time.perf_counter()-t)*1000)
            elapsed = time.perf_counter()-start
            row = dict(trial=trial,frames=len(lat),elapsed_s=elapsed,fps=len(lat)/elapsed,mean_ms=float(np.mean(lat)),p50_p95_p99_ms=np.percentile(lat,[50,95,99]).tolist())
            report['runs'].append(row)
            write_json(out/f'times-{trial}.json',lat)
            save(); print(row,flush=True)
        report['pooled_fps'] = sum(r['frames'] for r in report['runs'])/sum(r['elapsed_s'] for r in report['runs'])
        phase('measured_end')
        stages = []
        for _ in range(100):
            t0=time.perf_counter(); pipeline.prepare(x); t1=time.perf_counter()
            raw=pipeline.executor.run(); t2=time.perf_counter(); pipeline.decode(*raw); t3=time.perf_counter()
            stages.append([(t1-t0)*1000,(t2-t1)*1000,(t3-t2)*1000])
        report['separate_stage_mean_ms'] = dict(zip(['external_quantization','session_run','cpu_topk'],np.mean(stages,axis=0).tolist()))
        write_json(out/'stage-times.json',stages)
        del pipeline,npu
        phase('released')
        if not a.skip_profile:
            prof = session(profile=True)
            prof_pipeline = Pipeline(prof,a,meta)
            for _ in range(100): prof_pipeline.run(x)
            for _ in range(a.profile_runs): prof_pipeline.run(x)
            report['profile'] = prof.end_profiling()
            report['profile_summary'] = profile_summary(report['profile'],100,a.profile_runs)
        report['status']='complete'; save()
    except Exception as exc:
        report.update(status='failed',error=repr(exc)); save(); raise
    finally:
        write_json(out/'validation.json', {k:report.get(k) for k in ['validation','validation_passed','diagnostic_gate_passed','baseline_npu_exact','execution_exact','accuracy_status','cpu_check','error']})
    print(f'Results: {out}',flush=True)


if __name__ == '__main__':
    main()
