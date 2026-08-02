# CascadeMotion-Model

基于 YOLO-Pose 和 1D-CNN 的通用动作分类模型。通过视频提取人体关键点，对动作进行二分类（规范 / 不规范）。适用于各类运动姿态评估场景。

## 项目结构

```
CascadeMotion-Model/
├── config/
│   └── config.yaml              # 训练超参数与数据路径配置（模板）
├── data/
│   ├── csv/                     # 处理后的关键点 CSV 数据
│   │   ├── normal/
│   │   └── abnormal/
│   └── raw/                     # 原始视频数据
│       ├── normal/
│       ├── abnormal/
│       └── test/
├── dataset/
│   └── __init__.py              # RopeDataset (PyTorch Dataset)
├── exp/                         # 实验输出（每次训练一个子目录）
│   └── <model_name>/
│       ├── <name>.pth           # 训练权重
│       ├── <name>_export.pt     # 导出模型包
│       └── config.yaml          # 本次训练的配置快照
├── models/
│   └── __init__.py              # RopeClassifier (1D-CNN 模型)
├── pretrained/
│   ├── pose/                    # YOLO-Pose 预训练权重
│   │   └── yolo11n-pose.pt
│   └── classifiers/             # 已训练好的分类器模型
├── script/
│   ├── record.py                # 实时摄像头录制（30秒分段）
│   ├── pose.py                  # YOLO-Pose 视频→CSV 数据处理
│   ├── train.py                 # 模型训练
│   ├── export.py                # 模型导出 (.pth → .pt)
│   ├── infer.py                 # 从 .pth 权重推理
│   └── export_infer.py          # 从导出 .pt 文件推理（支持批量）
├── utils/
│   └── __init__.py              # 数据加载与归一化工具
├── webui.py                     # Gradio WebUI（一键训练）
├── requirements.txt             # 依赖列表
└── .gitignore
```

## 安装

```bash
git clone <repo-url>
cd CascadeMotion-Model

# 安装依赖
pip install -r requirements.txt

# 下载 YOLO-Pose 预训练权重到 pretrained/pose/
# https://docs.ultralytics.com/tasks/pose/
```

## 使用方式

### WebUI（推荐）

```bash
python webui.py
# 浏览器打开 http://localhost:7860
```

在网页中输入正常/不规范动作的视频目录，设置模型名称，系统将自动完成全流程。

### CLI 手动流程

#### 1. 录制数据

```bash
# 录制规范动作
python -m script.record --mode normal --camera 0

# 录制不规范动作
python -m script.record --mode abnormal --camera 0
```

按 `q` 键停止录制，自动按 30 秒分段保存。

#### 2. 提取关键点

```bash
# 处理指定子集
python -m script.pose video2csv data/raw/normal data/csv --subset normal
python -m script.pose video2csv data/raw/abnormal data/csv --subset abnormal

# 或处理整个目录结构
python -m script.pose video2csv data/raw data/csv
```

#### 3. 训练模型

```bash
python -m script.train --config config/config.yaml --name my_model
# 模型保存到 exp/my_model/my_model.pth
```

#### 4. 导出模型

```bash
python -m script.export \
    --checkpoint exp/my_model/my_model.pth \
    --output exp/my_model/my_model_export.pt \
    --config exp/my_model/config.yaml
```

#### 5. 推理

```bash
# 批量推理（推荐）
python -m script.export_infer --model exp/my_model/my_model_export.pt --input data/csv/test/

# 单文件推理
python -m script.infer --csv-path data/csv/test/test.csv --model-path exp/my_model/my_model.pth --config config/config.yaml
```

## 配置说明

`config/config.yaml` 是训练配置模板：

```yaml
# 模型参数
model:
  in_channels: 34    # 输入通道数 (17 关节 × 2 坐标)
  num_classes: 2     # 分类数

# 训练参数
epochs: 500
batch_size: 32
lr: 0.001
window_size: 64      # 滑窗帧数
stride: 32           # 滑窗步长
```

WebUI 和 `train.py --name` 会自动生成实验配置到 `exp/<name>/`。

## 模型架构

- **输入**：滑窗帧序列 × 17 关节 × 2 坐标 (x, y)，经空间归一化
- **骨干网络**：3 层 Conv1d → BatchNorm → ReLU → Pooling
- **分类器**：Dropout(0.5) → Linear
- **归一化**：以髋关节中点为原点，肩宽为基准尺度

## 许可证

[Apache 2.0](LICENSE)
