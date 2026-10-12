# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""Read-only audit of requested versus serialized compiler settings."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def audit(context):
    data = json.loads(context.read_text())
    config = data.get('config', {})
    passes = []
    for item in config.get('passes', []):
        attrs = item.get('passDpuParam', {}).get('xcompilerAttrs')
        if attrs:
            passes.append(dict(name=item.get('name'), opt_level=attrs.get('opt_level'),
                               target=attrs.get('target'), attributes=attrs))
    logs = list(context.parents[2].glob('*.log'))
    pdi = []
    for log in logs:
        for match in re.finditer(r'PDI Swap times\s*:\s*(\d+)', log.read_text(errors='replace')):
            pdi.append(dict(log=str(log), count=int(match[1]), meaning='Compiler diagnostic count; NOT runtime reload count'))
    artifacts = [dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p))
                 for p in sorted(context.parent.rglob('*.xmodel'))]
    return dict(context=str(context), context_sha256=digest(context),
                requested=config.get('sessionOptions', {}), selected_target=config.get('target'),
                serialized_passes=passes, artifacts=artifacts, pdi=pdi,
                device_subgraph_count=data.get('deviceSubgraphCount'),
                effective_opt_level=None, effective_setting_reason='Serialized pass may be a template; binary implementation/log proof unavailable. Hash differences do not prove optimization levels.')


def collect(roots):
    rows, errors = [], []
    for root in roots:
        contexts = [root] if root.is_file() else sorted(root.rglob('context.json'))
        for context in contexts:
            try: rows.append(audit(context))
            except Exception as exc: errors.append(dict(path=str(context), error=repr(exc)))
    return dict(status='complete' if not errors else 'partial', rows=rows, errors=errors,
                conclusion='Requested opt_level and serialized pass settings are separately recorded; no claim that provider options were ignored.',
                sdk_audit='Readable voe Python sources contain no integer compiler opt_level application path. Core implementation is binary; no unsupported cache edits or repeat sweep.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--root', action='append', type=Path)
    a = p.parse_args()
    roots = a.root or [ROOT/'results/npu-tuning-20261007-180759', ROOT/'results/npu-variants-20261007-214005']
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = collect(roots)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f"Audited {len(result['rows'])} contexts; errors={len(result['errors'])}: {a.output}")
    if result['errors']: raise SystemExit(1)


if __name__ == '__main__': main()
