import os
import glob
import json
import cv2
import numpy as np
import pandas as pd
import click
from ultralytics import YOLO


KPT_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_sho", "R_sho", "L_elb", "R_elb",
    "L_wri", "R_wri", "L_hip", "R_hip",
    "L_kne", "R_kne", "L_ank", "R_ank"
]

VIDEO_EXTENSIONS = ["*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV", "*.mkv", "*.MKV"]


def _build_columns():
    """构建 CSV 列名。"""
    columns = ["frame_id", "person_id"]
    for name in KPT_NAMES:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_conf"])
    return columns


def _find_video_files(input_dir):
    """查找目录下所有视频文件（兼容大小写扩展名）。"""
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    return video_files


def _process_subset(input_dir, output_dir, subset, yolo_model, imgsz):
    """处理单个子集的视频文件。"""
    subset_output_dir = os.path.join(output_dir, subset)
    os.makedirs(subset_output_dir, exist_ok=True)

    video_files = _find_video_files(input_dir)
    if not video_files:
        click.echo(f"未在目录 {input_dir} 中找到视频文件。")
        return

    columns = _build_columns()

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_csv_path = os.path.join(subset_output_dir, f"{video_name}.csv")
        output_json_path = os.path.join(subset_output_dir, f"{video_name}.json")

        # 检查 CSV 和 JSON 是否都已存在
        if os.path.exists(output_csv_path) and os.path.exists(output_json_path):
            click.echo(f"跳过 (已存在): {video_name}")
            continue

        click.echo(f"正在处理视频: {video_name} ...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            click.echo(f"错误：无法打开视频 {video_path}")
            continue

        metadata = {
            "source_video": os.path.abspath(video_path),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

        frame_id = 0
        all_data = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame, verbose=False, imgsz=imgsz)
            result = results[0]

            if result.keypoints is not None and result.keypoints.data.shape[0] > 0:
                keypoints_xy = result.keypoints.xy.cpu().numpy()
                keypoints_conf = result.keypoints.conf.cpu().numpy()
                num_persons = keypoints_xy.shape[0]

                for person_id in range(num_persons):
                    row = [frame_id, person_id]
                    for i in range(17):
                        row.extend([
                            keypoints_xy[person_id, i, 0],
                            keypoints_xy[person_id, i, 1],
                            keypoints_conf[person_id, i]
                        ])
                    all_data.append(row)
            frame_id += 1

        cap.release()

        df = pd.DataFrame(all_data, columns=columns) if all_data else pd.DataFrame(columns=columns)
        df.to_csv(output_csv_path, index=False)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        click.echo(f"完成: {output_csv_path} 与 {output_json_path}")


@click.group()
def cli():
    """YOLO-Pose 数据处理与可视化工具集。"""
    pass


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('output_dir', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--model', default='pretrained/pose/yolo11n-pose.pt', show_default=True, help='YOLO-Pose 模型权重路径或名称。')
@click.option('--imgsz', default=640, show_default=True, type=int, help='推理时的图像尺寸。')
@click.option('--subset', type=click.Choice(['normal', 'abnormal', 'test']), default=None, show_default=True, help='数据集子集类型。')
def video2csv(input_dir, output_dir, model, imgsz, subset):
    click.echo(f"加载模型: {model}")
    yolo_model = YOLO(model)

    subsets = [subset] if subset else ['normal', 'abnormal', 'test']
    for s in subsets:
        _process_subset(input_dir, output_dir, s, yolo_model, imgsz)


if __name__ == '__main__':
    cli()
