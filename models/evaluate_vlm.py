import re
import torch
import numpy as np
from torchvision.ops import box_iou

def parse_vlm_text_output(text_output):
    """
    Bóc tách tọa độ và hướng từ chuỗi output của mô hình.
    Ví dụ: "<|box_start|>(15,20),(85,90)<|box_end|> Tổn thương nằm bên phải"
    """
    # Lấy tọa độ
    box_pattern = r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>"
    match = re.search(box_pattern, text_output)
    
    bbox = None
    if match:
        bbox = [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))]
        
    # Lấy laterality (hướng)
    text_lower = text_output.lower()
    lat = 1 if "phải" in text_lower or "right" in text_lower else 0 if "trái" in text_lower or "left" in text_lower else -1
    
    return bbox, lat

def filter_best_prediction_per_slice(predictions):
    """
    Lọc trùng lặp để mỗi lát cắt (slice_z) chỉ lấy dự đoán tự tin nhất.
    predictions: list of dict [{'slice_z': int, 'bbox': list, 'confidence': float}]
    """
    best_preds = {}
    for pred in predictions:
        z = pred['slice_z']
        if z not in best_preds or pred.get('confidence', 0) > best_preds[z].get('confidence', 0):
            best_preds[z] = pred
    return list(best_preds.values())

def compute_3d_viou(pred_boxes, gt_boxes):
    """Tính trung bình IoU của các lát cắt (xấp xỉ 3D vIoU)."""
    if not pred_boxes or not gt_boxes: return 0.0
    
    pred_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
    gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
    
    ious = torch.diag(box_iou(pred_tensor, gt_tensor))
    return ious.mean().item()