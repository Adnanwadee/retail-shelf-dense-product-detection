"""Utility functions for the Retail Shelf Streamlit demo.

The demo supports:
- YOLOv8m dense product detection.
- RT-DETR-L dense product detection.
- Optional ResNet-50 visual product-group classification.

Model weights are intentionally expected under the local ``weights/`` directory
and are not stored in Git history.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from ultralytics import RTDETR, YOLO


LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "weights"

DETECTION_MODEL_PATHS: dict[str, Path] = {
    "YOLO": WEIGHTS_DIR / "best_yolo.pt",
    "RT-DETR": WEIGHTS_DIR / "best_rtdetr.pt",
}

RESNET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def _load_torch_state_dict(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a PyTorch state dictionary with compatibility across torch versions."""
    try:
        loaded = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location=device)

    if isinstance(loaded, dict) and "state_dict" in loaded:
        state_dict = loaded["state_dict"]
    else:
        state_dict = loaded

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict dictionary from {path}, got {type(state_dict)!r}")

    return state_dict


@st.cache_resource(show_spinner=False)
def load_detection_model(model_name: str) -> YOLO | RTDETR | None:
    """Load a detection model by name.

    Parameters
    ----------
    model_name:
        Either ``YOLO`` or ``RT-DETR``.

    Returns
    -------
    YOLO | RTDETR | None
        Loaded detector if weights exist and loading succeeds; otherwise ``None``.
    """
    normalized_name = model_name.strip().upper()
    if normalized_name == "RT-DETR":
        normalized_name = "RT-DETR"
    elif normalized_name == "YOLO":
        normalized_name = "YOLO"

    model_path = DETECTION_MODEL_PATHS.get(normalized_name)
    if model_path is None:
        st.error(f"Unsupported detection model: {model_name}")
        return None

    if not model_path.exists():
        st.error(
            "Detection weights were not found. "
            f"Expected local file: `{model_path.relative_to(REPO_ROOT)}`"
        )
        st.info(
            "Place the required model weights under the `weights/` directory. "
            "Weights are intentionally excluded from Git history."
        )
        return None

    try:
        if normalized_name == "YOLO":
            return YOLO(str(model_path))
        return RTDETR(str(model_path))
    except Exception as exc:
        LOGGER.exception("Failed to load detection model from %s", model_path)
        st.error(f"Failed to load detection model `{normalized_name}`: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def load_classification_model(
    path: str | Path = WEIGHTS_DIR / "resnet50_best.pth",
    num_classes: int = 91,
) -> tuple[nn.Module | None, torch.device]:
    """Load the optional ResNet-50 visual product-group classifier.

    The classifier is optional. If weights are not available, the Streamlit app
    still runs detection-only inference and labels boxes as detected products.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(path)

    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path

    if not model_path.exists():
        st.warning(
            "Optional ResNet-50 visual group classifier weights were not found. "
            f"Expected local file: `{model_path.relative_to(REPO_ROOT)}`. "
            "The app will run in detection-only mode."
        )
        return None, device

    try:
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

        state_dict = _load_torch_state_dict(model_path, device)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        return model, device
    except Exception as exc:
        LOGGER.exception("Failed to load classification model from %s", model_path)
        st.warning(
            "Failed to load the optional ResNet-50 visual group classifier. "
            f"The app will continue in detection-only mode. Details: {exc}"
        )
        return None, device


def _safe_crop(
    image: Image.Image,
    bbox: list[int],
    min_size: int = 5,
) -> Image.Image | None:
    """Crop a product region safely from an image."""
    width, height = image.size
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None

    return image.crop((x1, y1, x2, y2))


def run_fast_pipeline(
    det_model: YOLO | RTDETR | None,
    clf_model: nn.Module | None,
    device: torch.device,
    image: Image.Image,
    conf_threshold: float,
    class_names: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Run detection and optional visual product-group classification.

    Returns
    -------
    tuple[np.ndarray, list[dict[str, Any]]]
        The plotted detection image and structured detection/grouping results.
    """
    if det_model is None:
        st.error("Detection model is not available. Add weights before running inference.")
        fallback_image = np.array(image.convert("RGB"))
        return fallback_image, []

    try:
        det_results = det_model.predict(image, conf=conf_threshold, verbose=False)
    except Exception as exc:
        LOGGER.exception("Detection inference failed.")
        st.error(f"Detection inference failed: {exc}")
        fallback_image = np.array(image.convert("RGB"))
        return fallback_image, []

    if not det_results:
        fallback_image = np.array(image.convert("RGB"))
        return fallback_image, []

    result = det_results[0]
    boxes = result.boxes
    plotted_image = result.plot(labels=False)

    if boxes is None or len(boxes) == 0:
        return plotted_image, []

    image_rgb = image.convert("RGB")
    crops_tensors: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []

    for box in boxes:
        raw_bbox = box.xyxy[0].tolist()
        bbox = [int(value) for value in raw_bbox]
        crop = _safe_crop(image_rgb, bbox)

        if crop is None:
            continue

        det_conf = float(box.conf.item() * 100.0)

        crops_tensors.append(RESNET_TRANSFORM(crop))
        metadata.append(
            {
                "bbox": bbox,
                "det_conf": det_conf,
            }
        )

    if not metadata:
        return plotted_image, []

    if clf_model is None:
        detection_only_results = [
            {
                "bbox": item["bbox"],
                "label": "Detected Product",
                "confidence": item["det_conf"],
                "det_conf": item["det_conf"],
            }
            for item in metadata
        ]
        return plotted_image, detection_only_results

    batch_tensor = torch.stack(crops_tensors).to(device)
    final_results: list[dict[str, Any]] = []
    batch_size = 32

    with torch.no_grad():
        for start_idx in range(0, len(batch_tensor), batch_size):
            batch = batch_tensor[start_idx : start_idx + batch_size]
            outputs = clf_model(batch)
            probabilities = F.softmax(outputs, dim=1)
            top_probabilities, top_ids = torch.topk(probabilities, 1)

            for local_idx in range(len(batch)):
                global_idx = start_idx + local_idx
                class_id = int(top_ids[local_idx].item())
                confidence = float(top_probabilities[local_idx].item() * 100.0)

                label = (
                    class_names[class_id]
                    if 0 <= class_id < len(class_names)
                    else f"Visual_Group_{class_id}"
                )

                final_results.append(
                    {
                        "bbox": metadata[global_idx]["bbox"],
                        "label": label,
                        "confidence": confidence,
                        "det_conf": metadata[global_idx]["det_conf"],
                    }
                )

    return plotted_image, final_results
