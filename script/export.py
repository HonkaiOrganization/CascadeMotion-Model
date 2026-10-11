"""
模型导出脚本：将训练好的模型导出为 .pt 文件。

用法:
    python -m script.export --checkpoint exp/my_model/my_model.pth --output exp/my_model/my_model_export.pt --config exp/my_model/config.yaml
"""
import click
import yaml
import torch
import os
import json
from pathlib import Path
import hashlib
from datetime import datetime

from models import RopeClassifier


@click.command()
@click.option('--checkpoint', type=click.Path(exists=True), required=True, help='训练好的模型权重文件路径 (.pth)')
@click.option('--output', type=click.Path(), default='model_export.pt', help='导出的 .pt 文件路径')
@click.option('--config', type=click.Path(exists=True), required=True, help='YAML配置文件路径')
def export(checkpoint, output, config):
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 1. 初始化模型
    model_config = cfg.get('model', {})
    in_channels = model_config.get('in_channels', 34)
    num_classes = model_config.get('num_classes', 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RopeClassifier(
        in_channels=in_channels,
        num_classes=num_classes
    ).to(device)

    # 2. 加载权重
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. 构建导出包
    export_data = {
        'model_state_dict': state_dict,
        'config': {
            'in_channels': in_channels,
            'num_classes': num_classes,
            'window_size': cfg['window_size'],
            'stride': cfg['stride'],
        },
        'metadata': {
            'export_time': datetime.now().isoformat(),
            'checkpoint_path': os.path.abspath(checkpoint),
            'config_path': os.path.abspath(config),
        }
    }

    experiment = Path(checkpoint).parent
    if (experiment / 'summary.json').is_file():
        export_data['metadata']['training'] = json.loads((experiment / 'summary.json').read_text(encoding='utf-8'))
        export_data['metadata']['training_audit_sha256'] = {
            filename: hashlib.sha256((experiment / filename).read_bytes()).hexdigest()
            for filename in ['config.yaml', 'manifest.json', 'run.json', 'metrics.csv', 'summary.json']
        }

    # 4. 保存
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(export_data, output)

    print(f"模型已导出至: {output}")
    print(f"  - in_channels: {in_channels}")
    print(f"  - num_classes: {num_classes}")
    print(f"  - window_size: {cfg['window_size']}")
    print(f"  - stride: {cfg['stride']}")


if __name__ == '__main__':
    export()
