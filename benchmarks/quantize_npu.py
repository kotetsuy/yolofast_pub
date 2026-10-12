# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""Quantize an externally supplied ONNX model with AMD Quark's public API."""
import argparse
import copy
from pathlib import Path
import cv2
import numpy as np
from npu_common import preprocess, no_overwrite, sha256, write_json

class CalibrationImages:
    def __init__(self, directory, limit):
        self.paths = sorted(Path(directory).glob('*.jpg'))[:limit]
        if not self.paths:
            raise ValueError('Calibration directory contains no JPG images')
        self.iterator = iter(self.paths)

    def get_next(self):
        path = next(self.iterator, None)
        if path is None:
            return None
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot read calibration image: {path}')
        return {'images': preprocess(image)}

    def rewind(self):
        self.iterator = iter(self.paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calib-dir', type=Path, required=True)
    parser.add_argument('--calib-limit', type=int, default=60)
    parser.add_argument('--per-channel', action='store_true')
    args = parser.parse_args()
    if args.calib_limit < 1:
        parser.error('--calib-limit must be positive')
    no_overwrite(args.output, args.output.with_suffix('.json'))
    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config.config import Config
    from quark.onnx.quantization.config.custom_config import A16W8_CONFIG
    reader = CalibrationImages(args.calib_dir, args.calib_limit)
    settings = copy.deepcopy(A16W8_CONFIG)
    settings.per_channel = args.per_channel
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ModelQuantizer(Config(global_quant_config=settings)).quantize_model(
        str(args.input), str(args.output), reader)
    # Record hashes and ordinal IDs, rather than private image paths.
    write_json(args.output.with_suffix('.json'), {
        'status': 'complete', 'source_sha256': sha256(args.input),
        'model_sha256': sha256(args.output), 'recipe': 'A16W8_CONFIG',
        'per_channel': args.per_channel,
        'calibration': [{'id': i, 'sha256': sha256(path)}
                        for i, path in enumerate(reader.paths)]})

if __name__ == '__main__':
    main()
