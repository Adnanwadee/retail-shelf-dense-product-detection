# Model Weights

Model weights are intentionally not stored in Git history.

Expected local files:

```text
weights/
├── best_yolo.pt
├── best_rtdetr.pt
└── resnet50_best.pth
```

## Why weights are excluded

The trained weights are excluded to keep the repository lightweight and avoid committing large binary files.

## How to use the demo

Place the required weights inside this directory before running:

```bash
streamlit run app/streamlit_app.py
```

## Weight roles

| File | Role |
|---|---|
| `best_yolo.pt` | YOLOv8m dense product detector |
| `best_rtdetr.pt` | RT-DETR-L dense product detector |
| `resnet50_best.pth` | Optional ResNet-50 visual product-group classifier |

If `resnet50_best.pth` is missing, the app can still run in detection-only mode.
