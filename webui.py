import sys
import os
import subprocess
import yaml
import gradio as gr

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def run_command(cmd):
    """
    执行 CLI 命令，实时 yield 每行输出。
    通过 _rc 列表记录退出码，供调用方检查。
    返回: (_rc, generator) 元组
    """
    _rc = [None]

    def _gen():
        print(f"$ {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        for line in process.stdout:
            yield line.rstrip('\r\n')
        process.wait()
        _rc[0] = process.returncode

    return _rc, _gen()


def pipeline(normal_dir, abnormal_dir, model_name):
    """训练流水线：视频→CSV→训练→导出，全程 CLI 调用，非零退出码立即终止。"""

    try:
        # ---- 参数校验 ----
        normal_dir = normal_dir.strip()
        abnormal_dir = abnormal_dir.strip()
        model_name = model_name.strip()

        if not normal_dir:
            yield "❌ 请输入正常动作视频目录", None
            return
        if not abnormal_dir:
            yield "❌ 请输入不规范动作视频目录", None
            return
        if not model_name:
            yield "❌ 请输入模型名称", None
            return
        if not os.path.isdir(normal_dir):
            yield f"❌ 正常动作目录不存在: {normal_dir}", None
            return
        if not os.path.isdir(abnormal_dir):
            yield f"❌ 不规范动作目录不存在: {abnormal_dir}", None
            return

        # ---- 路径准备 ----
        exp_dir = os.path.join(PROJECT_ROOT, "exp", model_name)
        os.makedirs(exp_dir, exist_ok=True)

        model_path = os.path.join(exp_dir, f"{model_name}.pth")
        export_path = os.path.join(exp_dir, f"{model_name}_export.pt")
        csv_output_dir = os.path.join(PROJECT_ROOT, "data", "csv")
        os.makedirs(csv_output_dir, exist_ok=True)

        # 生成实验配置（基于 config/config.yaml，写入 exp/<name>/）
        base_config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
        with open(base_config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        cfg['model_name'] = model_path  # 绝对路径，train.py 直接保存到 exp/

        exp_config_path = os.path.join(exp_dir, "config.yaml")
        with open(exp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

        output = ""

        # ========== 步骤 1: 正常视频 → CSV ==========
        output += "\n" + "=" * 50 + "\n"
        output += "📹 步骤 1/4: 正常动作视频 → CSV 关键点提取\n"
        output += "=" * 50 + "\n"
        yield output, None

        rc, gen = run_command([
            PYTHON, '-m', 'script.pose',
            'video2csv',
            normal_dir, csv_output_dir,
            '--subset', 'normal',
        ])
        for line in gen:
            output += line + "\n"
            yield output, None

        if rc[0] != 0:
            output += f"\n❌ 步骤 1 失败 (exit code: {rc[0]})，进程已终止\n"
            yield output, None
            return

        output += "\n✅ 步骤 1 完成\n"
        yield output, None

        # ========== 步骤 2: 不规范视频 → CSV ==========
        output += "\n" + "=" * 50 + "\n"
        output += "📹 步骤 2/4: 不规范动作视频 → CSV 关键点提取\n"
        output += "=" * 50 + "\n"
        yield output, None

        rc, gen = run_command([
            PYTHON, '-m', 'script.pose',
            'video2csv',
            abnormal_dir, csv_output_dir,
            '--subset', 'abnormal',
        ])
        for line in gen:
            output += line + "\n"
            yield output, None

        if rc[0] != 0:
            output += f"\n❌ 步骤 2 失败 (exit code: {rc[0]})，进程已终止\n"
            yield output, None
            return

        output += "\n✅ 步骤 2 完成\n"
        yield output, None

        # ========== 步骤 3: 训练模型 ==========
        output += "\n" + "=" * 50 + "\n"
        output += f"🧠 步骤 3/4: 训练模型 → exp/{model_name}/\n"
        output += "=" * 50 + "\n"
        yield output, None

        rc, gen = run_command([
            PYTHON, '-m', 'script.train',
            '--config', exp_config_path,
        ])
        for line in gen:
            output += line + "\n"
            yield output, None

        if rc[0] != 0:
            output += f"\n❌ 步骤 3 失败 (exit code: {rc[0]})，进程已终止\n"
            yield output, None
            return

        output += "\n✅ 步骤 3 完成\n"
        yield output, None

        # ========== 步骤 4: 导出模型 ==========
        output += "\n" + "=" * 50 + "\n"
        output += "📦 步骤 4/4: 导出模型\n"
        output += "=" * 50 + "\n"
        yield output, None

        rc, gen = run_command([
            PYTHON, '-m', 'script.export',
            '--checkpoint', model_path,
            '--output', export_path,
            '--config', exp_config_path,
        ])
        for line in gen:
            output += line + "\n"
            yield output, None

        if rc[0] != 0:
            output += f"\n❌ 步骤 4 失败 (exit code: {rc[0]})，进程已终止\n"
            yield output, None
            return

        if not os.path.exists(export_path):
            output += f"\n❌ 导出文件未找到: {export_path}\n"
            yield output, None
            return

        output += "\n✅ 步骤 4 完成\n"
        output += "\n" + "=" * 50 + "\n"
        output += f"🎉 全部完成！实验输出目录: exp/{model_name}/\n"
        output += "=" * 50 + "\n"
        yield output, export_path

    except KeyboardInterrupt:
        output += "\n⚠️ 用户中断，进程已终止\n"
        yield output, None
        return


# ==================== Gradio UI ====================

def build_ui():
    with gr.Blocks(title="CascadeMotion 动作分类训练系统") as demo:
        gr.Markdown("# 🎬 CascadeMotion 动作分类训练系统")
        gr.Markdown(
            "输入正常/不规范动作的视频目录，设置模型名称，系统将自动完成："
            "关键点提取 → 模型训练 → 模型导出"
        )

        with gr.Row():
            normal_dir = gr.Textbox(
                label="✅ 正常动作视频目录",
                placeholder=r"例如: D:\videos\normal",
            )
            abnormal_dir = gr.Textbox(
                label="❌ 不规范动作视频目录",
                placeholder=r"例如: D:\videos\abnormal",
            )

        model_name = gr.Textbox(
            label="模型名称",
            placeholder="例如: squat_classifier",
        )

        run_btn = gr.Button("▶ 开始训练", variant="primary")

        log_output = gr.Textbox(
            label="训练日志",
            lines=25,
            max_lines=50,
            interactive=False,
            autoscroll=True,
        )

        model_file = gr.File(label="📥 导出模型下载", visible=True)

        run_btn.click(
            fn=pipeline,
            inputs=[normal_dir, abnormal_dir, model_name],
            outputs=[log_output, model_file],
        )

    return demo


if __name__ == '__main__':
    demo = build_ui()
    demo.launch(server_name='0.0.0.0', server_port=7860)
