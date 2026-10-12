# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""COCO bbox mAP: FP32 CPU vs A16W8 CPU vs distributed INT16 NPU model.

Requires pycocotools and COCO instances annotations. Never infers mAP acceptance
from smoke-test success. Default evaluation uses every annotated image.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52,
            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
            88, 89, 90]
METRIC_NAMES = ['mAP50_95', 'mAP50', 'mAP75', 'AP_small', 'AP_medium',
                'AP_large', 'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']


def restore_box(box, height, width):
    """Invert the actual rounded resize used by npu_common.preprocess."""
    ratio = min(640 / height, 640 / width)
    nh, nw = round(height * ratio), round(width * ratio)
    top, left = (640 - nh) // 2, (640 - nw) // 2
    x1, y1, x2, y2 = box
    x1, x2 = [min(width, max(0, (x - left) * width / nw)) for x in [x1, x2]]
    y1, y2 = [min(height, max(0, (y - top) * height / nh)) for y in [y1, y2]]
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def coco_predictions(detections, info, threshold):
    """Discard nonpositive boxes consistently; retain every exclusion for audit."""
    rows, excluded = [], []
    for row in detections:
        if row[4] < threshold:
            continue
        cls = int(row[5])
        if row[5] != cls or not 0 <= cls < 80 or not 0 <= row[4] <= 1:
            raise ValueError('Invalid class ID or probability')
        reason = None
        if row[2] < row[0] or row[3] < row[1]:
            reason = 'inverted_raw_box'
        box = restore_box(row[:4], info['height'], info['width'])
        if reason is None and (box[2] <= 0 or box[3] <= 0):
            reason = 'nonpositive_box_after_clip'
        if reason:
            excluded.append(dict(reason=reason, detection=[float(v) for v in row]))
            continue
        rows.append(dict(image_id=info['id'], category_id=COCO_IDS[cls],
                         bbox=box, score=float(row[4])))
    return rows, excluded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, default=ROOT/'distribute/yolo26n-640-split-v2-a16w8-int16.onnx')
    p.add_argument('--quantized-cpu-model', type=Path, default=ROOT/'models/npu/yolo26n-640-split-v2-a16w8.onnx')
    p.add_argument('--fp32-model', type=Path, default=ROOT/'models/npu/yolo26n-640-split-fp32-v2.onnx')
    p.add_argument('--image-manifest', type=Path, help='Fixed subset manifest containing image_id and sha256')
    p.add_argument('--limit', type=int, help='Diagnostic subset only; not full validation')
    p.add_argument('--cpu-only', action='store_true', help='Diagnostic CPU evaluation; no NPU claim')
    p.add_argument('--score-threshold', type=float, default=.001)
    p.add_argument('--resume', action='store_true', help='Resume per-image checkpoints in --output')
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error('--limit must be positive')
    if a.limit is not None and a.image_manifest is not None:
        p.error('--limit and --image-manifest are mutually exclusive')
    if not 0 <= a.score_threshold <= 1:
        p.error('score threshold must be in [0,1]')
    if a.output.exists() and not a.resume:
        raise FileExistsError(f'Refusing to overwrite {a.output}')
    if a.resume and not (a.output/'checkpoint-config.json').is_file():
        raise ValueError('No resumable checkpoints in this output; use a new output directory')
    a.output.mkdir(parents=True, exist_ok=a.resume)
    if a.resume and (a.output/'result.json').exists():
        import shutil
        shutil.copy2(a.output/'result.json',a.output/f'result-before-resume-{time.time_ns()}.json')
    report = dict(status='started', accuracy_status='unqualified: no agreed mAP acceptance threshold',
                  command=sys.argv, metrics={}, cpu_only=a.cpu_only,
                  score_threshold=a.score_threshold,
                  invalid_box_policy='Discard inverted/nonpositive boxes equally for all backends; save excluded rows; no coordinate swapping',
                  coco_parameters=dict(iou_thresholds='0.50:0.05:0.95', max_dets=[1, 10, 100]),
                  scope='Independent of quantization calibration only; not necessarily independent of original model training')
    def save():
        temp = a.output/'result.json.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(a.output/'result.json')
    save()
    try:
        if not a.annotations.is_file() or not a.images.is_dir():
            raise RuntimeError('COCO images/instances annotations unavailable')
        if not a.cpu_only and not list(Path('/dev/accel').glob('accel*')):
            raise RuntimeError('NPU device unavailable; no independent NPU mAP evaluated')
        import cv2
        import numpy as np
        import onnxruntime as ort
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from npu_common import cpu_session, preprocess, quantize_input, sha256, write_json
        from npu_postprocess import decode

        annotation = json.loads(a.annotations.read_text())
        images = sorted(annotation['images'], key=lambda x:x['id'])
        if a.limit is None and len(images) != 5000:
            raise ValueError('Full COCO val2017 evaluation requires exactly 5000 images')
        if len({i['id'] for i in images}) != len(images):
            raise ValueError('Duplicate evaluation image IDs')
        if sorted(c['id'] for c in annotation['categories']) != COCO_IDS:
            raise ValueError('Expected standard COCO 80-category instances annotations')
        if a.limit is not None:
            images = images[:a.limit]
        if a.image_manifest is not None:
            selected = json.loads(a.image_manifest.read_text())
            selected = selected['images'] if isinstance(selected, dict) else selected
            ids = [int(v['image_id']) for v in selected]
            if len(ids) != len(set(ids)) or not ids:
                raise ValueError('Duplicate/empty evaluation manifest')
            by_id = {i['id']:i for i in images}
            if any(image_id not in by_id for image_id in ids):
                raise ValueError('Manifest image ID absent from COCO annotations')
            images = [by_id[image_id] for image_id in ids]
        if not images:
            raise ValueError('No images')
        model = a.model.resolve()
        meta = json.loads(model.with_suffix('.json').read_text())
        cpu_quant = a.quantized_cpu_model.resolve()
        fp32 = a.fp32_model.resolve()
        if sha256(model) != meta['model_sha256'] or sha256(cpu_quant) != meta['source_sha256']:
            raise ValueError('Model/metadata hashes do not match')
        cpu_meta = json.loads(cpu_quant.with_suffix('.json').read_text())
        if cpu_meta.get('model_sha256') != sha256(cpu_quant) or cpu_meta.get('source_sha256') != sha256(fp32):
            raise ValueError('Quantized CPU or FP32 source provenance mismatch')
        calibration = cpu_meta['calibration']
        # Check both file bytes and decoded pixels, catching identical recompressed images.
        def pixel_hash(im):
            return hashlib.sha256(str(im.shape).encode()+im.tobytes()).hexdigest()
        cal_files, cal_pixels = set(), set()
        import os
        for record in calibration:
            path = Path(os.path.expandvars(record['path'])).expanduser()
            if not path.is_file() or sha256(path) != record['sha256']:
                raise ValueError(f'Calibration provenance unavailable/changed: {path}')
            im = cv2.imread(str(path))
            if im is None:
                raise ValueError(f'Unreadable calibration image: {path}')
            cal_files.add(record['sha256']); cal_pixels.add(pixel_hash(im))
        manifest, overlaps = [], []
        for info in images:
            path = a.images/info['file_name']
            if not path.is_file():
                raise ValueError(f'Missing evaluation image: {path}')
            im = cv2.imread(str(path))
            if im is None or im.shape[:2] != (info['height'], info['width']):
                raise ValueError(f'Image shape/read failure: {path}')
            digest = sha256(path)
            if digest in cal_files or pixel_hash(im) in cal_pixels:
                overlaps.append(info['id'])
            manifest.append(dict(image_id=info['id'], file_name=info['file_name'], sha256=digest))
        if a.image_manifest is not None:
            expected = {int(v['image_id']):v['sha256'] for v in selected}
            if any(v['sha256'] != expected[v['image_id']] for v in manifest):
                raise ValueError('Evaluation image manifest hash mismatch')
        write_json(a.output/'images-manifest.json', manifest)
        report.update(image_count=len(images), annotation_sha256=sha256(a.annotations),
                      calibration_image_count=len(calibration), calibration_overlap_ids=overlaps,
                      models={k:dict(path=str(v),sha256=sha256(v)) for k,v in
                              [('fp32_cpu',fp32),('quantized_cpu',cpu_quant),('quantized_npu',model)]},
                      ort=ort.__version__, python=sys.version,
                      evaluation_code_sha256=sha256(__file__))
        save()
        if overlaps:
            raise ValueError('Evaluation overlaps calibration; refusing independent mAP claim')
        sessions = dict(fp32_cpu=cpu_session(fp32), quantized_cpu=cpu_session(cpu_quant))
        if not a.cpu_only:
            if 'VitisAIExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('VitisAI EP unavailable')
            options = dict(cache_dir=str(a.output/'cache'), opt_level='2', enable_cache_file_io_in_mem='0')
            npu = ort.InferenceSession(str(model), providers=['VitisAIExecutionProvider','CPUExecutionProvider'],
                                      provider_options=[options,{}])
            npu.disable_fallback()
            if 'VitisAIExecutionProvider' not in npu.get_providers():
                raise RuntimeError('Created session lost VitisAI provider')
            sessions['quantized_npu'] = npu
            report['npu_provider_options'] = options
        predictions = {name:[] for name in sessions}
        excluded = {name:[] for name in sessions}
        checkpoint_config = dict(models=report['models'],annotation_sha256=report['annotation_sha256'],
                                 manifest_sha256=sha256(a.output/'images-manifest.json'),
                                 backends=list(sessions), score_threshold=a.score_threshold,
                                 ort=ort.__version__, policy=report['invalid_box_policy'],
                                 code={f:sha256(ROOT/'benchmarks'/f) for f in
                                       ['evaluate_npu_coco.py','npu_common.py','npu_postprocess.py']})
        if a.resume:
            if checkpoint_config != json.loads((a.output/'checkpoint-config.json').read_text()):
                raise ValueError('Checkpoint models/data/settings/code have changed; use a new output')
        else:
            write_json(a.output/'checkpoint-config.json',checkpoint_config)
        checkpoint_dir = a.output/'predictions-by-image'
        checkpoint_dir.mkdir(exist_ok=True)
        scratch = np.empty((1,3,640,640), np.float32)
        integer = np.empty(scratch.shape, np.int16)
        start = time.perf_counter()
        for index,info in enumerate(images):
            checkpoint_path = checkpoint_dir/f'{info["id"]:012d}.json'
            if checkpoint_path.exists():
                cached = json.loads(checkpoint_path.read_text())
                if cached['image_id'] != info['id'] or set(cached['predictions']) != set(sessions):
                    raise ValueError('Invalid checkpoint')
                for name in sessions:
                    predictions[name].extend(cached['predictions'][name])
                    excluded[name].extend(cached['excluded'][name])
                continue
            im = cv2.imread(str(a.images/info['file_name']))
            x = preprocess(im)
            quantize_input(x,meta['scale'],meta['zero_point'],scratch,integer)
            cached = dict(image_id=info['id'],predictions={},excluded={})
            for name,session in sessions.items():
                raw = session.run(None,{session.get_inputs()[0].name:integer if name=='quantized_npu' else x})
                if len(raw)!=2 or [list(v.shape) for v in raw]!=[[1,4,8400],[1,80,8400]]:
                    raise ValueError(f'Unexpected split outputs: {name}')
                if not all(np.isfinite(v).all() for v in raw):
                    raise ValueError(f'Nonfinite outputs: {name}, image {info["id"]}')
                detections = decode(*raw)
                rows,invalid = coco_predictions(detections,info,a.score_threshold)
                cached['predictions'][name] = rows
                cached['excluded'][name] = [dict(image_id=info['id'],**v) for v in invalid]
                predictions[name].extend(rows)
                excluded[name].extend(cached['excluded'][name])
            # Publish only after every backend for this image has completed.
            write_json(checkpoint_path,cached)
            if (index+1)%100==0 or index==0 or index+1==len(images):
                report['completed_images'] = index+1; save()
                print(f'Inference {index+1}/{len(images)} ({time.perf_counter()-start:.1f}s)',flush=True)
        report['evaluation_wall_s'] = time.perf_counter()-start
        report['completed_images'] = len(images)
        report['excluded_geometry_counts'] = {
            name:dict(total=len(rows),inverted_raw_boxes=sum(v['reason']=='inverted_raw_box' for v in rows),
                      score_ge_025=sum(v['detection'][4]>=.25 for v in rows)) for name,rows in excluded.items()}
        write_json(a.output/'excluded-detections.json',excluded)
        gt = COCO(str(a.annotations))
        for name,rows in predictions.items():
            path = a.output/f'{name}-predictions.json'; write_json(path,rows)
            if rows:
                dt = gt.loadRes(str(path))
            else:
                dt = COCO(); dt.dataset = dict(images=annotation['images'],categories=annotation['categories'],annotations=[])
                dt.createIndex()
            evaluator = COCOeval(gt,dt,'bbox')
            evaluator.params.imgIds = [i['id'] for i in images]
            evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
            report['metrics'][name] = dict(zip(METRIC_NAMES,map(float,evaluator.stats)))
            save()
        base = report['metrics']['fp32_cpu']
        report['delta_vs_fp32_percentage_points'] = {
            name:{k:100*(metrics[k]-base[k]) for k in METRIC_NAMES}
            for name,metrics in report['metrics'].items() if name!='fp32_cpu'}
        report.update(status='complete', independent_of_calibration=True,
                      full_coco_val2017=(len(images)==5000 and a.limit is None and a.image_manifest is None),
                      npu_evaluated=not a.cpu_only,
                      acceptance_passed=None)
        save()
    except Exception as exc:
        report.update(status='failed',error=repr(exc)); save(); raise
    print(f'Results: {a.output}',flush=True)


if __name__=='__main__':
    main()
