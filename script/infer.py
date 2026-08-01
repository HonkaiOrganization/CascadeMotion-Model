import click
import yaml
import torch
import numpy as np
from models import RopeClassifier
from utils import load_and_normalize_csv

@click.command()
@click.option('--csv-path', type=click.Path(exists=True), required=True, help='待推理的CSV文件路径')
@click.option('--model-path', type=click.Path(exists=True), required=True, help='模型权重路径')
@click.option('--config', type=click.Path(exists=True), required=True, help='YAML配置文件路径')
def infer(csv_path, model_path, config):
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    model_config = cfg.get('model', {})
    in_channels = model_config.get('in_channels', 34)
    num_classes = model_config.get('num_classes', 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RopeClassifier(in_channels=in_channels, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    coords = load_and_normalize_csv(csv_path)
    window_size = cfg['window_size']
    stride = cfg.get('stride', window_size // 2)
    
    abnormal_windows = 0
    total_windows = 0
    
    with torch.no_grad():
        for i in range(0, len(coords) - window_size + 1, stride):
            window = coords[i:i+window_size].reshape(window_size, -1)
            input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            
            total_windows += 1
            if pred_class == 0: # 0 代表不规范
                abnormal_windows += 1
                
    abnormal_ratio = (abnormal_windows / total_windows) * 100 if total_windows > 0 else 0
    print(f"推理完成: {csv_path}")
    print(f"总分析窗口数: {total_windows}")
    print(f"判定为'不规范'的比例: {abnormal_ratio:.2f}%")

if __name__ == '__main__':
    infer()
