"""CLI training with file-disjoint validation, best checkpoints and audit logs."""
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dataset import RopeDataset
from models import RopeClassifier
from utils import load_and_normalize_csv, load_dataset_from_config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files(data, subset):
    files = []
    for category, label in [('normal', 1), ('abnormal', 0)]:
        source = data.get('subsets', {}).get(subset, {}).get(category)
        paths = source if isinstance(source, list) else sorted(Path(source).glob('*.csv')) if source else []
        for path in paths:
            path = Path(path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append({'path': os.path.relpath(path), 'label': label, 'sha256': digest(path),
                          'rows': len(load_and_normalize_csv(path))})
    return files


def check_split(manifest):
    seen_paths, seen_hashes = set(), set()
    for subset, files in manifest.items():
        if not files:
            raise ValueError(f'{subset} has no source files')
        if {item['label'] for item in files} != {0, 1}:
            raise ValueError(f'{subset} must contain both classes')
        for item in files:
            path = str(Path(item['path']).resolve()).lower()
            if path in seen_paths or item['sha256'] in seen_hashes:
                raise ValueError('Duplicate source file across or within splits')
            seen_paths.add(path)
            seen_hashes.add(item['sha256'])


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss_sum += float(criterion(logits, targets)) * len(targets)
            correct += int((logits.argmax(1) == targets).sum())
            count += len(targets)
    return loss_sum / count, correct / count


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='YAML配置文件路径')
@click.option('--name', default=None, help='实验名称，模型保存到 exp/<name>/')
def train(config, name):
    with open(config, encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream)
    if cfg.get('required_environment') and Path(sys.prefix).name.lower() != cfg['required_environment']:
        raise ValueError(f"Use conda {cfg['required_environment']}")
    seed = int(cfg.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(cfg.get('threads', 1)))
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    checkpoint = Path('exp') / name / f'{name}.pth' if name else Path(cfg.get('model_name', 'best_model.pth'))
    output = checkpoint.parent
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.csv').exists():
        raise FileExistsError('Training logs exist; choose a new experiment directory')
    subsets = ['train', 'validation'] if 'validation' in cfg['data'].get('subsets', {}) else ['train']
    manifest = {subset: source_files(cfg['data'], subset) for subset in subsets}
    check_split(manifest)
    loaders, counts = {}, {}
    generator = torch.Generator().manual_seed(seed)
    for subset in subsets:
        segments, labels = load_dataset_from_config(cfg['data'], subset)
        if not all(np.isfinite(x).all() for x in segments):
            raise ValueError('Non-finite normalized coordinates')
        dataset = RopeDataset(segments, labels, cfg['window_size'], cfg['stride'])
        if not len(dataset):
            raise ValueError(f'{subset} has no complete windows')
        loaders[subset] = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=subset == 'train',
                                     generator=generator if subset == 'train' else None)
        counts[subset] = {'files': len(segments), 'windows': len(dataset),
                          'class_windows': {str(k): sum(y == k for _, y in dataset.samples) for k in [0, 1]}}

    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model_cfg = cfg.get('model', {})
    model = RopeClassifier(model_cfg.get('in_channels', 34), model_cfg.get('num_classes', 2)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5,
                patience=cfg.get('lr_patience', 20), min_lr=cfg.get('min_lr', 1e-6)) if 'validation' in loaders else None
    snapshot = output / 'config.yaml'
    snapshot.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    run = {'started_utc': datetime.now(timezone.utc).isoformat(), 'command': [sys.executable, *sys.argv],
           'cwd': str(Path.cwd()), 'seed': seed, 'python': platform.python_version(), 'torch': str(torch.__version__),
           'numpy': np.__version__, 'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
           'deterministic_algorithms': True, 'torch_threads': torch.get_num_threads(), 'source_commit': revision,
           'source_sha256': {p: digest(p) for p in ['script/train.py', 'models/__init__.py', 'dataset/__init__.py', 'utils/__init__.py']},
           'config_sha256': digest(snapshot), 'manifest_sha256': digest(output / 'manifest.json'), 'counts': counts,
           'selection': 'minimum validation cross-entropy' if scheduler else 'minimum training-mode cross-entropy',
           'test_data_used_for_selection': False}
    (output / 'run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    best, best_epoch, best_step, bad_epochs, steps = float('inf'), 0, 0, 0, 0
    progress_best = float('inf')
    fields = ['epoch', 'global_step', 'lr', 'train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy', 'best_epoch']
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            loss_sum, correct, count = 0.0, 0, 0
            lr = optimizer.param_groups[0]['lr']
            for inputs, targets in loaders['train']:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss')
                loss.backward()
                optimizer.step()
                steps += 1
                loss_sum += float(loss.detach()) * len(targets)
                correct += int((logits.argmax(1) == targets).sum())
                count += len(targets)
            train_loss, train_acc = loss_sum / count, correct / count
            val_loss, val_acc = evaluate(model, loaders['validation'], criterion, device) if scheduler else (None, None)
            score = val_loss if scheduler else train_loss
            if score < best:
                best, best_epoch, best_step = score, epoch, steps
                torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, checkpoint)
            if score < progress_best - cfg.get('min_delta', 1e-4):
                progress_best, bad_epochs = score, 0
            else:
                bad_epochs += 1
            if scheduler:
                scheduler.step(val_loss)
            writer.writerow(dict(zip(fields, [epoch, steps, lr, train_loss, train_acc, val_loss, val_acc, best_epoch])))
            stream.flush()
            if epoch == 1 or epoch % cfg.get('log_every', 10) == 0:
                print(f'Epoch {epoch}/{cfg["epochs"]} step={steps} train={train_loss:.6f} '
                      f'validation={val_loss if scheduler else "n/a"} best={best_epoch}', flush=True)
            if scheduler and epoch >= cfg.get('min_epochs', 0) and bad_epochs >= cfg.get('patience', 100):
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    final = {subset: dict(zip(['loss', 'accuracy'], evaluate(model, loader, criterion, device))) for subset, loader in loaders.items()}
    summary = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'epochs_completed': epoch, 'optimizer_steps': steps,
               'maximum_epochs': cfg['epochs'], 'stopped_early': epoch < cfg['epochs'], 'best_epoch': best_epoch,
               'best_optimizer_step': best_step, 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
               'best_checkpoint_metrics': final, 'counts': counts, 'selection': run['selection'],
               'limitations': ['Video-disjoint validation; subject identities are unavailable.',
                               'Directory labels are inherited by overlapping windows.',
                               'Finite validation data and early stopping do not guarantee absence of overfitting.']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    train()
