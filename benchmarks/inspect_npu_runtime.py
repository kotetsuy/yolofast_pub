# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""Capability-discovered read-only telemetry; monitored FPS is diagnostic only."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from audit_npu_compile import ROOT, digest

REPORTS = ('aie-partitions', 'platform', 'aie', 'aiemem', 'aieshim', 'telemetry')


def discover_bdfs(text):
    return sorted(set(re.findall(r'(?<![\da-fA-F])(?:[\da-fA-F]{4}:)?[\da-fA-F]{2}:[\da-fA-F]{2}\.[0-7](?![\da-fA-F])', text)))


def supported_reports(help_text):
    # Match standalone report names, not substrings such as aie within aieshim.
    return [name for name in REPORTS if re.search(r'(?<![\w-])'+re.escape(name)+r'(?![\w-])', help_text)]


def parse_partitions(text):
    """Accept XRT's explicit JSON schema or documented flat table; never guess."""
    rows = []
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        data = None
    def walk(value):
        if isinstance(value, dict):
            if {'start_col', 'num_cols', 'hw_contexts'} <= value.keys():
                start, count = int(value['start_col']), int(value['num_cols'])
                if start < 0 or not 0 < count <= 32: return
                for ctx in value['hw_contexts']:
                    if not {'pid','context_id'} <= ctx.keys(): continue
                    row = dict(pid=int(ctx['pid']), context_id=str(ctx['context_id']),
                               columns=list(range(start,start+count)), status=ctx.get('status'))
                    for key in ['command_submissions','command_completions','migrations','errors']:
                        row[key] = int(ctx[key]) if key in ctx else None
                    rows.append(row)
            else:
                for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    if data is not None:
        walk(data)
        return rows
    columns, headers = None, None
    for line in text.splitlines():
        match = re.search(r'Columns:\s*\[([\d, ]+)\]',line)
        if match:
            columns = [int(v.strip()) for v in match[1].split(',')]; headers = None
        if '|' not in line or columns is None: continue
        cells = [v.strip() for v in line.strip().strip('|').split('|')]
        if 'PID' in cells and 'Ctx ID' in cells and 'Status' in cells:
            headers = cells; continue
        if not headers or len(cells)!=len(headers): continue
        record = dict(zip(headers,cells))
        if not record['PID'].isdigit() or not record['Ctx ID'].isdigit(): continue
        row = dict(pid=int(record['PID']),context_id=record['Ctx ID'],columns=columns,status=record['Status'])
        for key, names in [('command_submissions',['Sub','Submissions']),('command_completions',['Compl','Completions']),('migrations',['Migr','Migrations']),('errors',['Err'])]:
            cell = next((record[n] for n in names if n in record),None)
            row[key] = int(cell) if cell and cell.isdigit() else None
        rows.append(row)
    return rows


def summarize_allocation(observations, pid, width):
    steady = [o for o in observations if o['phase']=='steady']
    worker = [(o,[r for r in o['contexts'] if r['pid']==pid]) for o in steady]
    worker = [(o,rows) for o,rows in worker if rows]
    if not worker: return dict(classification='unknown', reason='No recognized steady-state partition rows for worker PID')
    snapshots = [sorted(set(c for row in rows for c in row['columns'])) for _,rows in worker]
    if width is not None and any(c>=width or c<0 for cols in snapshots for c in cols):
        return dict(classification='unknown', reason='Report column indices inconsistent with enumerated topology')
    all_rows = [dict(monotonic_ns=o['monotonic_ns'],**row) for o in steady for row in o['contexts']]
    deltas, prior = [], {}
    for o, rows in worker:
        for row in rows:
            key=row['context_id']; old=prior.get(key)
            if old:
                counters={}
                for name in ['command_submissions','command_completions','migrations','errors']:
                    now,before=row[name],old[name]
                    counters[name]=now-before if now is not None and before is not None and now>=before else None
                deltas.append(dict(context_id=key,monotonic_ns=o['monotonic_ns'],deltas=counters,
                                   limitation='Reset/recreated context or missing counter yields null; commands are not frames.'))
            prior[key]=row
    full = width==8 and any(cols==list(range(8)) for cols in snapshots)
    return dict(classification='all_8_columns_allocated' if full else 'allocation_observed',
                allocated_columns=dict(value=snapshots,reason='Per-snapshot worker PID allocation; not a union across time'),
                hw_contexts=dict(value=all_rows,reason='Includes other PIDs and context statuses; Active is not MAC utilization'),
                submissions_completions_delta=dict(value=deltas,reason='Only consecutive observations of matching context ID'),
                migrations_errors=dict(value=[{k:r[k] for k in ['pid','context_id','migrations','errors']} for r in all_rows], reason='Reported counters'),
                observed_max_columns=max(map(len,snapshots)),
                reason='8-column allocation confirmed; simultaneous compute utilization unmeasured' if full else 'Observed allocation saved; <=6 columns does not by itself establish a configured limit')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', help='Select BDF if multiple devices are enumerated')
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--interval', type=float, default=1)
    p.add_argument('--timeout', type=float, default=1800)
    a = p.parse_args()
    if a.seconds < 60 or a.interval <= 0 or a.timeout <= a.seconds: p.error('seconds>=60, interval>0, timeout>seconds required')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'topology.json').exists(): raise FileExistsError(out/'topology.json')
    raw = out/'telemetry'; raw.mkdir()
    journal = raw/'samples.jsonl'
    def command(label, cmd, phase='discovery'):
        start = time.monotonic_ns()
        try:
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            text = run.stdout+'\n'+run.stderr; code = run.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            text = repr(exc); code = -1
        path = raw/f'{label}.txt'; path.write_text(text)
        with journal.open('a') as f:
            f.write(json.dumps(dict(monotonic_ns=start, end_monotonic_ns=time.monotonic_ns(),
                                   phase=phase, command=cmd, returncode=code, raw=str(path)))+'\n')
        return text, code
    overview, _ = command('overview', ['xrt-smi', 'examine'])
    help_text, _ = command('help', ['xrt-smi', 'examine', '--help'])
    bdfs = discover_bdfs(overview)
    device = a.device or (bdfs[0] if len(bdfs)==1 else None)
    if a.device and a.device not in bdfs: p.error('Requested device is not in current enumeration')
    if device:
        specific, _ = command('device-help', ['xrt-smi', 'examine', '-d', device, '--help'])
        help_text += '\n'+specific
    reports = supported_reports(help_text) if device else []
    # JSON requires explicit advertised format and output options.
    json_flags = ['--format','JSON','--output'] if re.search(r'--format\b', help_text) and re.search(r'--output\b',help_text) and re.search(r'\bJSON\b', help_text, re.I) else []
    topology_match = re.search(r'\b(\d+)x(\d+)\b', overview)
    unknown = lambda why: dict(value=None, reason=why)
    topology = dict(status='started', device=device, discovered_bdfs=bdfs,
                    topology=dict(rows=int(topology_match[1]), columns=int(topology_match[2])) if topology_match else None,
                    supported_reports=reports, classification='unknown',
                    allocated_columns=unknown('No validated report schema mapping columns to this worker PID'),
                    active_columns=unknown('Snapshot states do not establish simultaneous MAC utilization'),
                    hw_contexts=unknown('Inspect raw reports and correlate worker PID'),
                    submissions_completions_delta=unknown('Counter schema unavailable; no guessed deltas'),
                    migrations_errors=unknown('Counter schema unavailable'),
                    clock_power_temperature=unknown('Inspect available raw reports'),
                    limitation='Topology is rows x columns, not used cores. No utilization percentages inferred.')
    def save(): (out/'topology.json').write_text(json.dumps(topology,indent=2)+'\n')
    save()
    observations = []
    def sample(index, phase):
        for report in reports:
            cmd = ['xrt-smi','examine','-d',device,'-r',report]
            if json_flags: cmd += json_flags+[str(raw/f'{index}-{report}.json')]
            text, code = command(f'{index}-{report}', cmd, phase)
            if report=='aie-partitions' and code==0:
                json_path=raw/f'{index}-{report}.json'
                payload=json_path.read_text() if json_path.exists() else text
                try:
                    contexts=parse_partitions(payload)
                    observation=dict(monotonic_ns=time.monotonic_ns(),phase=phase,contexts=contexts)
                    observations.append(observation)
                    with (raw/'partitions.jsonl').open('a') as f: f.write(json.dumps(observation)+'\n')
                except (ValueError,KeyError,TypeError) as exc:
                    with (raw/'parse-errors.txt').open('a') as f: f.write(repr(exc)+'\n')
    sample('idle', 'idle')
    import onnxruntime as ort
    manifest = dict(ort=ort.__version__, sdk_setup='external Ryzen AI SDK environment',
                    xrt_driver_firmware_source=str(raw/'overview.txt'), report_help_source=str(raw/'help.txt'),
                    command=sys.argv, python=sys.version, clock='host monotonic_ns', interval_s=a.interval,
                    diagnostic_only=True, device_nodes=[str(n) for n in Path('/dev/accel').glob('accel*')],
                    code={str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__),ROOT/'benchmarks/npu_split.py',ROOT/'benchmarks/audit_npu_compile.py',ROOT/'benchmarks/compare_npu_variants.py',ROOT/'scripts/inspect_npu_runtime.sh',ROOT/'configs/npu-targets-v2.json']},
                    model_sha256=digest(ROOT/'models/npu/yolo26n-640-split-v2-a16w8-int16.onnx'))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not manifest['device_nodes'] or not device:
        topology.update(status='unavailable', reason='NPU device nodes or unique device BDF unavailable; no inference launched')
        save(); return 0
    ack = out/'phase-acks'; ack.mkdir()
    phases = out/'phases.jsonl'
    cmd = [sys.executable,'-B',str(ROOT/'benchmarks/npu_split.py'),'--output',str(out/'diagnostic'),
           '--model',str(ROOT/'models/npu/yolo26n-640-split-v2-a16w8-int16.onnx'),
           '--int16-input','--opt-level','2','--target','X2','--disk-cache','--skip-profile',
           '--seconds',str(a.seconds),'--repeats','1','--phase-file',str(phases),'--phase-ack-dir',str(ack),
           '--compare-npu',str(ROOT/'results/npu-variants-20261007-214005/v2-screen')]
    with (out/'diagnostic.log').open('w') as log:
        proc = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        topology['worker_pid'] = proc.pid; save()
        seen = 0; phase = 'loading'; count = 0; next_sample = time.monotonic(); deadline = next_sample+a.timeout
        try:
            while proc.poll() is None:
                if time.monotonic()>deadline: raise TimeoutError('Diagnostic worker timeout')
                entries = phases.read_text().split('\n')[:-1] if phases.exists() else []
                for line in entries[seen:]:
                    event = json.loads(line); phase = event['phase']
                    sample(f'phase-{phase}', phase+'-checkpoint-paused')
                    (ack/phase).touch(); seen += 1
                if phase=='steady' and time.monotonic()>=next_sample:
                    sample(f'steady-{count:04d}', phase); count += 1
                    next_sample=time.monotonic()+a.interval
                time.sleep(.1)
        except Exception:
            proc.kill(); proc.wait(); raise
    sample('ended', 'ended')
    topology.update(status='complete' if proc.returncode==0 else 'failed', worker_returncode=proc.returncode,
                    steady_samples=count, raw_review_required=True,
                    reason='Allocation remains unknown until raw report schema and context-to-PID correspondence are verified')
    topology.update(summarize_allocation(observations,proc.pid,topology['topology']['columns'] if topology['topology'] else None))
    save()
    return int(proc.returncode!=0)


if __name__=='__main__': sys.exit(main())
