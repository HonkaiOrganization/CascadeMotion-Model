import os
import sys
import time
import threading
import datetime
import click
import cv2
import yaml


class CameraRecorder:
    """相机录制服务。

    - 按下录制按键后等待 3 秒预热，再开始正式录制
    - 每 30 秒自动切分保存为一个片段
    - 自动读取相机最大分辨率
    """

    SEGMENT_SECONDS = 30
    WARMUP_SECONDS = 3

    def __init__(self, camera_index: int, mode: str, data_dir: str):
        self.camera_index = camera_index
        self.mode = mode  # "normal" 或 "abnormal"
        self.data_dir = data_dir
        self.cap: cv2.VideoCapture | None = None
        self._stop_event = threading.Event()

    # ---------- 公共方法 ----------

    def run(self):
        """启动录制流程。"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            click.echo(f"错误：无法打开相机 index={self.camera_index}")
            return

        # 获取相机最大分辨率
        max_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        max_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        click.echo(f"相机分辨率: {max_w}x{max_h}")

        # 确保输出目录存在
        subset_dir = os.path.join(self.data_dir, self.mode)
        os.makedirs(subset_dir, exist_ok=True)

        click.echo(f"模式: {self.mode} | 输出目录: {subset_dir}")
        click.echo("按 q 键停止录制")

        # 等待 3 秒预热
        click.echo("预热中... 3 秒后开始录制")
        time.sleep(self.WARMUP_SECONDS)
        click.echo(">>> 开始录制！")

        self._record_loop(subset_dir, max_w, max_h)

    def stop(self):
        """停止录制。"""
        self._stop_event.set()
        if self.cap:
            self.cap.release()

    # ---------- 内部逻辑 ----------

    def _record_loop(self, subset_dir: str, width: int, height: int):
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames_per_segment = int(fps * self.SEGMENT_SECONDS)

        segment_idx = 0
        global_frame_count = 0

        while not self._stop_event.is_set():
            # 为当前片段创建写入器
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.mode}_{timestamp}_seg{segment_idx:04d}.mp4"
            filepath = os.path.join(subset_dir, filename)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

            click.echo(f"[片段 {segment_idx}] 开始 -> {filename}")

            frame_in_segment = 0
            while frame_in_segment < frames_per_segment and not self._stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    click.echo("警告：读取帧失败，结束录制")
                    self._stop_event.set()
                    break

                writer.write(frame)
                frame_in_segment += 1
                global_frame_count += 1

                # 每 60 帧检查一次键盘输入（降低 CPU 开销）
                if frame_in_segment % 60 == 0:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        click.echo("收到 q 键，正在停止录制...")
                        self._stop_event.set()

            writer.release()
            click.echo(f"[片段 {segment_idx}] 已保存 ({frame_in_segment} 帧)")

            segment_idx += 1

        click.echo(f"录制结束，共保存 {segment_idx} 个片段")


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["normal", "abnormal"]),
    required=True,
    help="录制模式：normal（正常）或 abnormal（不规范）",
)
@click.option("--camera", type=int, default=0, show_default=True, help="相机设备索引")
@click.option("--data-dir", type=str, default=None, help="数据输出根目录（默认读取 config/config.yaml 中的 data.raw_dir）")
def record(mode: str, camera: int, data_dir: str | None):
    """数据集录制服务。

    用法示例:
        python script/record.py --mode normal
        python script/record.py --mode abnormal --camera 1
    """
    # 确定输出目录
    if data_dir is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            data_dir = cfg.get("data", {}).get("raw_dir", "data/raw")
        else:
            data_dir = "data/raw"

    recorder = CameraRecorder(camera_index=camera, mode=mode, data_dir=data_dir)

    try:
        recorder.run()
    except KeyboardInterrupt:
        click.echo("\n收到中断信号，正在停止...")
    finally:
        recorder.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    record()
