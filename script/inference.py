"""
推理脚本：从导出的 .pt 文件加载模型并对 CSV 文件或目录进行推理。

用法:
    # 单个文件推理
    python -m script.inference --model model_export.pt --input data/csv/test/normal/test.csv

    # 目录批量推理
    python -m script.inference --model model_export.pt --input data/csv/test/normal/

    # 输出详细结果到 JSON
    python -m script.inference --model model_export.pt --input data/csv/test/ --output results.json
"""
import click
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import glob

from utils import load_and_normalize_csv
from models import JumpRopeClassifier


LABEL_MAP = {0: 'abnormal', 1: 'normal'}


def segment_and_predict(model, coords, window_size, stride, device):
    """
    对单个视频片段进行滑窗推理，返回整体预测结果。

    Args:
        model: 已加载的模型
        coords: numpy array, shape (Frames, 17, 2)
        window_size: 窗口大小
        stride: 滑窗步长
        device: torch device

    Returns:
        dict: 包含预测标签、置信度、各窗口结果
    """
    num_frames = coords.shape[0]
    if num_frames < window_size:
        return {
            'status': 'skip',
            'reason': f'帧数不足: {num_frames} < {window_size}'
        }

    window_probs = []
    for i in range(0, num_frames - window_size + 1, stride):
        window = coords[i:i + window_size]
        # 展平为 (window_size, 34)
        window_flat = window.reshape(window_size, -1)
        input_tensor = torch.tensor(window_flat, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            window_probs.append(probs.cpu().numpy())

    # 平均所有窗口的预测概率
    avg_probs = np.mean(window_probs, axis=0)[0]
    predicted_class = int(np.argmax(avg_probs))
    confidence = float(avg_probs[predicted_class])

    return {
        'status': 'ok',
        'predicted_label': LABEL_MAP[predicted_class],
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': {
            'abnormal': float(avg_probs[0]),
            'normal': float(avg_probs[1])
        },
        'num_windows': len(window_probs),
        'num_frames': num_frames
    }


@click.command()
@click.option('--model', type=click.Path(exists=True), required=True, help='导出的模型文件 (.pt)')
@click.option('--input', 'input_path', type=click.Path(exists=True), required=True, help='输入 CSV 文件或目录')
@click.option('--output', type=click.Path(), default=None, help='输出 JSON 结果文件路径 (可选)')
def inference(model, input_path, output):
    # 1. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    export_data = torch.load(model, map_location=device, weights_only=True)

    model_cfg = export_data['config']
    model_instance = JumpRopeClassifier(
        in_channels=model_cfg['in_channels'],
        num_classes=model_cfg['num_classes']
    ).to(device)
    model_instance.load_state_dict(export_data['model_state_dict'])
    model_instance.eval()

    window_size = model_cfg['window_size']
    stride = model_cfg.get('stride', 32)

    print(f"模型已加载: {model}")
    print(f"  - window_size: {window_size}")
    print(f"  - stride: {stride}")
    print(f"  - device: {device}")
    print()

    # 2. 收集输入文件
    if os.path.isfile(input_path):
        csv_files = [input_path]
    else:
        csv_files = sorted(glob.glob(os.path.join(input_path, '**', '*.csv'), recursive=True))

    if not csv_files:
        print(f"未找到 CSV 文件: {input_path}")
        return

    print(f"找到 {len(csv_files)} 个文件，开始推理...\n")

    # 3. 推理
    results = {}
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        try:
            coords = load_and_normalize_csv(csv_path)
            result = segment_and_predict(model_instance, coords, window_size, stride, device)
            results[filename] = result

            # 打印结果
            if result['status'] == 'ok':
                print(f"[{filename}]")
                print(f"  预测: {result['predicted_label'].upper()}")
                print(f"  置信度: {result['confidence']:.4f}")
                print(f"  概率: abnormal={result['probabilities']['abnormal']:.4f}, normal={result['probabilities']['normal']:.4f}")
                print(f"  窗口数: {result['num_windows']}, 帧数: {result['num_frames']}")
            else:
                print(f"[{filename}] 跳过: {result['reason']}")
            print()

        except Exception as e:
            results[filename] = {'status': 'error', 'error': str(e)}
            print(f"[{filename}] 错误: {e}\n")

    # 4. 输出统计
    ok_results = [r for r in results.values() if r.get('status') == 'ok']
    if ok_results:
        normal_count = sum(1 for r in ok_results if r['predicted_label'] == 'normal')
        abnormal_count = sum(1 for r in ok_results if r['predicted_label'] == 'abnormal')
        print(f"--- 推理完成 ---")
        print(f"总计: {len(ok_results)} 个文件")
        print(f"  normal: {normal_count}")
        print(f"  abnormal: {abnormal_count}")

    # 5. 保存结果
    if output:
        output_dir = os.path.dirname(output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至: {output}")


if __name__ == '__main__':
    inference()
