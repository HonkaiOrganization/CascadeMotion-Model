import click
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

from utils import load_dataset_from_config
from dataset import JumpRopeDataset
from models import JumpRopeClassifier

@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='YAML配置文件路径')
@click.option('--tb-dir', type=click.Path(), default='runs', help='TensorBoard日志目录')
def train(config, tb_dir):
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 1. 从配置文件加载数据集
    data_config = cfg.get('data', {})
    
    # 加载训练集
    train_segments, train_labels = load_dataset_from_config(data_config, subset='train')
    if not train_segments:
        raise ValueError("训练集为空，请检查配置文件中的路径")
    
    # 加载测试集
    test_segments, test_labels = load_dataset_from_config(data_config, subset='test')
    
    print(f"加载训练集: {len(train_segments)} 个文件")
    print(f"加载测试集: {len(test_segments)} 个文件")

    # 2. 构建 DataLoader
    train_dataset = JumpRopeDataset(train_segments, train_labels, cfg['window_size'], cfg['stride'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)

    # 测试集 DataLoader (用于训练过程中监控)
    test_loader = None
    if test_segments:
        test_dataset = JumpRopeDataset(test_segments, test_labels, cfg['window_size'], cfg['stride'])
        test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False)
        print(f"测试集已加载，将记录 Test Loss 到 TensorBoard")
    
    # 3. 初始化模型与优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JumpRopeClassifier(in_channels=34, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    
    writer = SummaryWriter(tb_dir)
    best_val_acc = 0.0
    
    # 4. 训练循环
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
        
        # 保存最新模型
        torch.save(model.state_dict(), 'best_model.pth')
            
    writer.close()

if __name__ == '__main__':
    train()