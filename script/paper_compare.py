"""Compare fixed old/new exports on identical poses; test data never select weights."""
import csv
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
import torch
import ultralytics

from models import RopeClassifier
from utils import load_and_normalize_csv


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def metrics(labels, predictions):
    matrix = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(labels, predictions):
        matrix[actual, predicted] += 1
    tp, fn, fp, tn = matrix.flatten()
    return {'count': int(matrix.sum()), 'confusion_matrix': matrix.tolist(),
            'accuracy': float((tp + tn) / matrix.sum()),
            'abnormal_precision': float(tp / (tp + fp)) if tp + fp else 0.,
            'abnormal_recall': float(tp / (tp + fn)) if tp + fn else 0.,
            'abnormal_f1': float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.}


def load_model(path):
    package = torch.load(path, map_location='cpu', weights_only=True)
    cfg = package['config']
    model = RopeClassifier(cfg['in_channels'], cfg['num_classes'])
    model.load_state_dict(package['model_state_dict'])
    model.eval()
    return model, package


def predict(model, coords, window, stride):
    starts = list(range(0, len(coords) - window + 1, stride))
    if not starts:
        raise ValueError('No complete inference windows')
    samples = torch.tensor(np.stack([coords[i:i + window].reshape(window, -1) for i in starts]), dtype=torch.float32)
    with torch.inference_mode():
        probabilities = torch.softmax(model(samples), dim=1).numpy()
    mean = probabilities.mean(axis=0)
    return {'window_starts': starts, 'window_probabilities': probabilities.tolist(),
            'window_predictions': probabilities.argmax(1).tolist(), 'num_windows': len(starts),
            'mean_probabilities': mean.tolist(), 'predicted_class': int(mean.argmax()),
            'abnormal_argmax_windows': int((probabilities.argmax(1) == 0).sum()),
            'top3': [{'window_index': i, 'start': starts[i], 'end': starts[i] + window,
                      'prob_abnormal': float(probabilities[i, 0])}
                     for i in np.argsort(-probabilities[:, 0])[:3].tolist()]}


@click.command()
@click.option('--experiment', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--old-model', type=click.Path(exists=True, path_type=Path),
              default='../CascadeMotion-NEOGUI/pretrained/classifiers/model_export.pt')
def compare(experiment, old_model):
    assert Path(sys.prefix).name.lower() == 'sport', 'Use conda sport'
    torch.set_num_threads(1)
    new_model = experiment / 'model_export.pt'
    old, old_package = load_model(old_model)
    new, new_package = load_model(new_model)
    assert old_package['config'] == new_package['config']
    summary = json.loads((experiment / 'summary.json').read_text())
    assert new_package['metadata']['training']['checkpoint_sha256'] == summary['checkpoint_sha256']
    assert new_package['metadata']['training_audit_sha256']['metrics.csv'] == digest(experiment / 'metrics.csv')
    raw = torch.load(experiment / 'paper_20261011.pth', weights_only=True)
    assert all(torch.equal(raw[k], new_package['model_state_dict'][k]) for k in raw)
    spec = importlib.util.spec_from_file_location('neogui_classifier', '../CascadeMotion-NEOGUI/core/classifier.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    deployed = module.Classifier()
    deployed.load_state_dict(new_package['model_state_dict'])
    deployed.eval()
    example = torch.arange(64 * 34, dtype=torch.float32).reshape(1, 64, 34) / 100
    with torch.inference_mode():
        assert torch.equal(new(example), deployed(example))
    manifest = json.loads((experiment / 'manifest.json').read_text())
    cases = [{'subset': subset, 'csv': Path(item['path']), 'label': item['label']}
             for subset, files in manifest.items() for item in files]
    for category, label in [('normal', 1), ('abnormal', 0)]:
        paths = sorted((experiment / 'pose_csv' / category).glob('*.csv'))
        assert len(paths) == 1, 'Expected exactly one reserved test video per class'
        cases += [{'subset': 'test', 'csv': path, 'label': label,
                   'video': next((Path('data/raw/test') / category).glob(path.stem + '.*'))} for path in paths]
    cases += [{'subset': 'neogui_cached', 'csv': Path('../CascadeMotion-NEOGUI/temp/imp_3570084c.csv'),
               'video': Path('../CascadeMotion-NEOGUI/temp/imp_3570084c.mp4'), 'label': None},
              {'subset': 'neogui_reextracted', 'csv': experiment / 'pose_csv/test/imp_3570084c.csv',
               'video': Path('../CascadeMotion-NEOGUI/temp/imp_3570084c.mp4'), 'label': None}]
    training_names = {Path(item['path']).stem for files in manifest.values() for item in files}
    assert not any(case['csv'].stem in training_names for case in cases if case['subset'] == 'test')
    source_videos = []
    for subset, files in manifest.items():
        for item in files:
            category = 'normal' if item['label'] == 1 else 'abnormal'
            video = next((Path('data/raw') / category).glob(Path(item['path']).stem + '.*'))
            source_videos.append({'subset': subset, 'path': os.path.relpath(video), 'sha256': digest(video)})
    source_hashes = {item['sha256'] for item in source_videos}
    assert all(digest(case['video']) not in source_hashes for case in cases if case['subset'] == 'test')
    cfg = new_package['config']
    results, curve_rows = [], []
    for case in cases:
        path, subset = case['csv'], case['subset']
        frame = pd.read_csv(path)
        coords = load_and_normalize_csv(path)
        assert np.isfinite(coords).all()
        record = {'subset': subset, 'csv': os.path.relpath(path), 'csv_sha256': digest(path),
                  'label': case['label'], 'rows': len(coords), 'unique_frames': int(frame.frame_id.nunique()),
                  'duplicate_frame_rows': int(frame.frame_id.duplicated().sum())}
        if 'video' in case:
            cap = cv2.VideoCapture(str(case['video']))
            assert cap.isOpened()
            total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            record.update(video=os.path.relpath(case['video']), video_sha256=digest(case['video']),
                          video_frames=total, fps=fps,
                          row_index_equals_frame_id=bool(np.array_equal(frame.frame_id.to_numpy(), np.arange(total))))
        for name, model in [('old', old), ('new', new)]:
            record[name] = predict(model, coords, cfg['window_size'], cfg['stride'])
        record['window_disagreements'] = sum(a != b for a, b in zip(record['old']['window_predictions'], record['new']['window_predictions']))
        for i, start in enumerate(record['old']['window_starts']):
            curve_rows.append([subset, path.stem, i, start, start + cfg['window_size'],
                               record['old']['window_probabilities'][i][0], record['new']['window_probabilities'][i][0]])
        results.append(record)
    aggregate = {}
    for subset in ['train', 'validation', 'test']:
        group = [r for r in results if r['subset'] == subset]
        aggregate[subset] = {}
        for name in ['old', 'new']:
            labels = [r['label'] for r in group for _ in range(r[name]['num_windows'])]
            predictions = [p for r in group for p in r[name]['window_predictions']]
            aggregate[subset][name] = {'window': metrics(labels, predictions),
                'file': metrics([r['label'] for r in group], [r[name]['predicted_class'] for r in group])}
    output = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'command': [sys.executable, *sys.argv],
              'source_sha256': {p: digest(p) for p in ['script/paper_compare.py', 'script/pose.py']},
              'environment': {'python': sys.executable, 'torch': str(torch.__version__), 'ultralytics': ultralytics.__version__, 'device': 'cpu', 'threads': 1},
              'models': {name: {'path': os.path.relpath(path), 'sha256': digest(path)} for name, path in [('old', old_model), ('new', new_model)]},
              'pose': {'model': 'pretrained/pose/yolo11n-pose.pt', 'sha256': digest('pretrained/pose/yolo11n-pose.pt'), 'imgsz': 640},
              'training_summary': summary, 'source_videos': source_videos, 'metrics': aggregate, 'cases': results,
              'checks': ['disjoint source filenames', 'export audit hashes', 'raw/export identical weights', 'Model/NEOGUI identical outputs'],
              'limitations': ['New model test videos were excluded from training and checkpoint selection.',
                             'Old model training membership is unknown; its scores are historical comparison only.',
                             'Only two labelled test videos; subjects are unidentified and window labels inherit video labels.',
                             'NEOGUI video has no human label; model differences do not establish which is correct.',
                             'New pose CSVs use the existing video2csv CLI; missing/multiple detections can affect row-to-frame alignment.']}
    (experiment / 'comparison.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    with (experiment / 'window_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['subset', 'video', 'window', 'start_row', 'end_row', 'old_p_abnormal', 'new_p_abnormal'])
        writer.writerows(curve_rows)
    print(json.dumps({'metrics': aggregate, 'neogui': [{k: r[k] for k in ['subset', 'rows', 'old', 'new', 'window_disagreements']} for r in results if r['label'] is None]}, indent=2))


if __name__ == '__main__':
    compare()
