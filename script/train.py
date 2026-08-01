import click
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from utils import load_dataset_from_config
from dataset import RopeDataset
from models import RopeClassifier

@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='YAML配置文件路径')
@click.option('--name', default=None, help='实验名称，模型保存到 exp/<name>/')
def train(config, name):
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 确定模型保存路径
    model_name = cfg.get('model_name')
    if name:
        exp_dir = os.path.join('exp', name)
        os.makedirs(exp_dir, exist_ok=True)
        model_name = os.path.join(exp_dir, f"{name}.pth")
    elif not model_name:
        model_name = 'best_model.pth'

    data_config = cfg.get('data', {})
    
    train_segments, train_labels = load_dataset_from_config(data_config, subset='train')
    if not train_segments:
        raise ValueError("训练集为空，请检查配置文件中的路径")
    
    print(f"加载训练集: {len(train_segments)} 个文件")

    train_dataset = RopeDataset(train_segments, train_labels, cfg['window_size'], cfg['stride'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    
    model_config = cfg.get('model', {})
    in_channels = model_config.get('in_channels', 34)
    num_classes = model_config.get('num_classes', 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RopeClassifier(in_channels=in_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    
    
    for epoch in range(cfg['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{cfg['epochs']} | Train Loss: {running_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), model_name)


if __name__ == '__main__':
    train()