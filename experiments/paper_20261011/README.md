# Paper retraining, 2026-10-11

Run in the existing conda `sport` environment, from CascadeMotion-Model.

```powershell
conda activate sport
python -B -m unittest discover -s tests -p test_training.py -v
python -u -B -m script.train --config config/paper_20261011.yaml --name paper_20261011
python -B -m script.export --checkpoint exp/paper_20261011/paper_20261011.pth --output exp/paper_20261011/model_export.pt --config exp/paper_20261011/config.yaml
python -u -B -m script.pose video2csv data/raw/test/normal exp/paper_20261011/pose_csv --subset normal --model pretrained/pose/yolo11n-pose.pt --imgsz 640
python -u -B -m script.pose video2csv data/raw/test/abnormal exp/paper_20261011/pose_csv --subset abnormal --model pretrained/pose/yolo11n-pose.pt --imgsz 640
python -u -B -m script.pose video2csv ../CascadeMotion-NEOGUI/temp exp/paper_20261011/pose_csv --subset test --model pretrained/pose/yolo11n-pose.pt --imgsz 640
python -B -m script.paper_compare --experiment exp/paper_20261011
```

Use a fresh experiment name/directory for another run; existing training logs are protected from replacement.

The six training files are from July 2; two validation files are from July 3. Adjacent recorded segments stay together in training. Two labelled videos in `data/raw/test` are reserved for evaluation and are processed after the selected checkpoint is fixed. The NEOGUI temporary video is unlabelled. Files and source videos are checked by SHA-256. This is a video split; subject identities are unavailable.

Training ran for 200 epochs / 2,200 optimizer steps. Minimum validation cross-entropy selected epoch 11 / step 121. The final epoch is not the deployment candidate. Selection uses validation loss, not validation accuracy or test results. Training loss is measured in training mode, with dropout and batch statistics; the summary evaluates the selected weights in evaluation mode.

`metrics.csv` records every epoch. `run.json` records the interpreter, command, GPU, seed, source revision and source hashes. `manifest.json` records the exact CSV split and hashes. The export embeds the training summary and audit-file hashes. The checkpoint and export are included for reproducibility. The existing NEOGUI deployment package is preserved.

Original videos, pose CSVs, credentials and cloud responses are not included. Reproducing inference requires the corresponding local source data and the fixed YOLO11n-Pose package. Directory labels are inherited by windows; scores from two test videos do not establish subject-independent generalization. The old model is a historical comparison, with unknown training membership. NEOGUI prediction differences do not show which model is correct without human labels.

## Observed results

| Set | Old window accuracy | New window accuracy | New abnormal F1 |
| --- | ---: | ---: | ---: |
| Training: 176 windows | 100.00% | 100.00% | 1.0000 |
| Validation: 83 windows | 100.00% | 96.39% | 0.9697 |
| Reserved test: 72 windows | 100.00% | 100.00% | 1.0000 |

Both reserved test videos have correct file-level predictions. Their input poses are identical for both classifiers. This comparison does not establish that the new model is superior.

| NEOGUI pose input | Old mean abnormal probability | New mean abnormal probability | Different window labels |
| --- | ---: | ---: | ---: |
| Existing cached CSV | 0.688283 | 0.814170 | 15/76 |
| Fresh YOLO11n-Pose extraction | 0.633409 | 0.680279 | 5/76 |

Both classifiers predict abnormal for the unlabelled NEOGUI video. Probability changes are not accuracy improvements. All window outputs are retained in `comparison.json` and `window_comparison.csv`.
