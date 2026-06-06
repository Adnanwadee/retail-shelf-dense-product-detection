# RT-DETR-L Detector Model Card

## Model

RT-DETR-L dense retail shelf product detector.

## Purpose

Evaluate a transformer-based detector for dense supermarket shelf product localization.

## Task Type

Object Detection.

## Dataset Context

The dataset uses shelf images resized to 640 × 640 pixels. The original detection setup contains a single generic product class.

## Best Configuration

| Parameter | Value |
|---|---|
| Architecture | RT-DETR-L |
| Image size | 640 × 640 |
| Optimizer | AdamW |
| Learning rate | 0.0005 |
| Batch size | 8 |
| Final epochs | 25 |

## Test Results

| Metric | Value |
|---|---:|
| mAP@50 | 0.9029 |
| mAP@50-95 | 0.5933 |
| Precision | 0.8793 |
| Recall | 0.8702 |
| F1-score | 0.8748 |
| Average inference | ~70 ms |

## Deployment Notes

RT-DETR-L provides a transformer-based comparison point and performs competitively, but its higher latency makes it less suitable than YOLOv8m for fast shelf-auditing scenarios.

## Limitations

- Higher inference latency than YOLOv8m.
- Larger model footprint.
- Not optimized here for edge deployment.
