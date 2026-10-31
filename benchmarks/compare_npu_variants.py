# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""Isolated NPU variant screening, reverse-order recheck and sustained tests."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from npu_common import ROOT,sha256,write_json


def accepted_result(result, returncode, exact_required):
    if returncode != 0 or result.get('status') != 'complete' or result.get('cpu_check'):
        return False
    if result.get('args',{}).get('target') and not result.get('target_verified'): return False
    checks=result.get('execution_exact',[])
    if len(checks)<6 or not all(checks):return False
    if not result.get('validation_passed') and not (result.get('diagnostic_score_tolerance')==.051 and result.get('diagnostic_gate_passed')):
        return False
    checks=result.get('baseline_npu_exact',[])
    if exact_required and (len(checks)<3 or not all(checks)):return False
    return bool(result.get('pooled_fps',0)>0 and result.get('runs'))


def improvement(base,candidate):
    return dict(fps_ratio=candidate['fps']/base['fps'],p99_ratio=candidate['p99_ms']/base['p99_ms'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--images-manifest',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--sustain-seconds',type=float,default=300)
    p.add_argument('--timeout',type=float,default=1800)
    a=p.parse_args()
    if a.seconds<30 or a.repeats<3 or a.sustain_seconds<300 or a.timeout<=a.sustain_seconds:
        p.error('Formal comparison requires >=30s x3, sustain>=300s, timeout>sustain')
    os.chdir(ROOT)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'comparison.json').exists():raise FileExistsError('Comparison already exists')
    config=json.loads(a.config.read_text())
    labels=[v['label'] for v in [config['baseline'],*config['candidates']]]
    if len(set(labels))!=len(labels) or any(not re.fullmatch(r'[a-zA-Z0-9_-]+',s) for s in labels):raise ValueError('Unique safe labels required')
    report=dict(status='started',config=config,config_sha256=sha256(a.config),runs=[],
                accuracy_status='unqualified: no independent mAP acceptance',boundary='normalized FP32 CPU input through CPU top-k')
    def save():write_json(out/'comparison.json',report)
    save()
    def run(variant,label,reference=None,seconds=None,repeats=None):
        dest=out/label;dest.mkdir()
        cmd=[sys.executable,'-B',str(ROOT/'benchmarks/npu_split.py'),'--output',str(dest),
             '--images-manifest',str(a.images_manifest.resolve()),
             '--model',str(ROOT/variant['model']),'--int16-input','--opt-level',str(variant.get('opt-level',2)),'--disk-cache','--skip-profile',
             '--seconds',str(seconds or a.seconds),'--repeats',str(repeats or a.repeats)]
        for key in ['execution','postprocess','quantization','intra-op-threads','target']:
            if key in variant:cmd += ['--'+key,str(variant[key])]
        if variant.get('legacy_diagnostic'):cmd += ['--diagnostic-score-tolerance','.051']
        if reference:cmd += ['--compare-npu',str(reference)]
        row=dict(label=label,command=cmd,directory=str(dest))
        env=os.environ.copy();env['XLNX_ONNX_EP_REPORT_FILE']='vitisai_ep_report.json'
        with (dest/'run.log').open('w') as log:
            try:
                proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=a.timeout)
                row['returncode']=proc.returncode
            except subprocess.TimeoutExpired:
                row.update(returncode=-1,error='timeout')
        if (dest/'result.json').exists():
            result=json.loads((dest/'result.json').read_text())
            row['valid']=accepted_result(result,row['returncode'],reference is not None)
            row['error']=result.get('error',row.get('error'))
            if row['valid']:
                times=[]
                for trial in result['runs']:times.extend(json.loads((dest/f"times-{trial['trial']}.json").read_text()))
                row.update(fps=result['pooled_fps'],p99_ms=float(np.percentile(times,99)),
                           p50_p95_p99_ms=np.percentile(times,[50,95,99]).tolist(),
                           stages_ms=result['separate_stage_mean_ms'],strict_smoke_passed=result['validation_passed'])
        report['runs'].append(row);save();print(json.dumps(row),flush=True)
        return row
    try:
        if not list(Path('/dev/accel').glob('accel*')):raise RuntimeError('NPU device unavailable; no timings performed')
        baseline=config['baseline']
        control=run(baseline,'baseline-screen',ROOT/config['baseline_reference'])
        if not control.get('valid'):raise RuntimeError('Baseline failed')
        valid=[]
        for variant in config['candidates']:
            candidate=run(variant,variant['label']+'-screen',out/'baseline-screen' if variant['equivalent_to_baseline'] else None)
            if candidate.get('valid'):valid.append((variant,candidate))
        if not valid and config.get('allow_rejected_candidates'):
            report.update(status='complete', decision='All optional targets rejected; baseline retained', selected=None)
            save(); return 0
        if not valid:raise RuntimeError('All candidates failed')
        best,screen=max(valid,key=lambda pair:pair[1]['fps'])
        report['screening_best']=best['label'];save()
        # Same-candidate exactness on recheck, even for a newly quantized model.
        winner=run(best,'winner-recheck',Path(screen['directory']))
        reverse=run(baseline,'baseline-recheck',out/'baseline-screen')
        if not winner.get('valid') or not reverse.get('valid'):raise RuntimeError('Reverse recheck failed')
        report['screening_improvement']=improvement(control,screen)
        report['reverse_improvement']=improvement(reverse,winner)
        if any(v['fps_ratio']<1.05 or v['p99_ratio']>1.05 for v in [report['screening_improvement'],report['reverse_improvement']]):
            report.update(status='complete',decision='no performance candidate meets provisional 5% gate',selected=None)
            save();return 0
        base_long=run(baseline,'baseline-sustained',out/'baseline-screen',a.sustain_seconds,1)
        best_long=run(best,'candidate-sustained',Path(screen['directory']),a.sustain_seconds,1)
        if not base_long.get('valid') or not best_long.get('valid'):raise RuntimeError('Sustained validation failed')
        report['sustained_improvement']=improvement(base_long,best_long)
        sustained=report['sustained_improvement']
        accepted=sustained['fps_ratio']>=1.05 and sustained['p99_ratio']<=1.05 and best_long['fps']>=winner['fps']*.95
        report.update(status='complete',selected=best['label'] if accepted else None,
                      decision='provisional performance candidate; accuracy unqualified' if accepted else 'sustained performance gate failed')
        save();return 0
    except Exception as exc:
        report.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':sys.exit(main())
