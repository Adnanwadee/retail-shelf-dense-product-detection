# YOLOv8m Detector Model Card

## Model

YOLOv8m dense retail shelf product detector.

## Purpose

Detect product instances in dense supermarket shelf images using bounding boxes.

## Task Type

Object Detection.

## Dataset Context

The dataset uses shelf images resized to 640 × 640 pixels. The original detection setup contains a single generic product class.

## Best Configuration

| Parameter | Value |
|---|---|
| Architecture | YOLOv8m |
| Image size | 640 × 640 |
| Optimizer | AdamW |
| Learning rate | 0.0005 |
| Batch size | 8 |
| Final epochs | 25 |

## Test Results

| Metric | Value |
|---|---:|
| mAP@50 | 0.9359 |
| mAP@50-95 | 0.6185 |
| Precision | 0.9079 |
| Recall | 0.8828 |
| F1-score | 0.8952 |
| Average inference | ~38 ms |

## Deployment Notes

YOLOv8m was selected as the preferred deployment-oriented detector due to its balance of localization accuracy, inference speed, model size, and practical usability.

## Limitations

- Dense occlusion may still cause missed detections.
- The detector localizes generic product instances, not verified commercial SKU identities.
- Production use requires validation on store-specific shelf layouts.
