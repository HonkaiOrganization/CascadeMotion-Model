"""
模型导出脚本：将训练好的模型导出为 .pt 文件。

用法:
    python -m script.export --checkpoint best_model.pth --output model_export.pt --config config/config.yaml
"""
import click
import yaml
import torch
import os
import json
from datetime import datetime

from models import JumpRopeClassifier


@click.command()
@click.option('--checkpoint', type=click.Path(exists=True), required=True, help='训练好的模型权重文件路径 (.pth)')
@click.option('--output', type=click.Path(), default='model_export.pt', help='导出的 .pt 文件路径')
@click.option('--config', type=click.Path(exists=True), required=True, help='YAML配置文件路径')
def export(checkpoint, output, config):
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 1. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JumpRopeClassifier(
        in_channels=34,
        num_classes=2
    ).to(device)

    # 2. 加载权重
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. 构建导出包
    export_data = {
        'model_state_dict': state_dict,
        'config': {
            'in_channels': 34,
            'num_classes': 2,
            'window_size': cfg['window_size'],
            'stride': cfg['stride'],
        },
        'metadata': {
            'export_time': datetime.now().isoformat(),
            'checkpoint_path': os.path.abspath(checkpoint),
            'config_path': os.path.abspath(config),
        }
    }

    # 4. 保存
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(export_data, output)

    print(f"模型已导出至: {output}")
    print(f"  - in_channels: 34")
    print(f"  - num_classes: 2")
    print(f"  - window_size: {cfg['window_size']}")
    print(f"  - stride: {cfg['stride']}")


if __name__ == '__main__':
    export()
