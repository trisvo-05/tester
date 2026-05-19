import json
import numpy as np
import torchvision.transforms.functional as TF
from sklearn.metrics import cohen_kappa_score

# --- TASK 3: Xử lý Hard Case ---
def apply_hardcase_augmentation(image_tensor, bbox, is_type_I_error=True):
    """Sửa lỗi nhầm Trái/Phải bằng Horizontal Flip (Lật ngang)."""
    if is_type_I_error:
        image_tensor = TF.hflip(image_tensor)
        # Lật tọa độ (Giả sử ảnh 224x224)
        x1, y1, x2, y2 = bbox
        bbox = [224 - x2, y1, 224 - x1, y2]
    return image_tensor, bbox

# --- TASK 4: QC & Threshold Tuning ---
def evaluate_and_rebuild_dataset(report_path="archive/instruction_validate_report.json", train_path="archive/train.jsonl"):
    print("Đang đọc Validation Report để đánh giá chất lượng nhãn...")
    with open(report_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
        
    auto_labels = []
    doctor_labels = []
    valid_samples = []
    
    for item in report_data:
        auto_labels.append(item['auto_lat_pred'])
        doctor_labels.append(item['doctor_lat_gt'])
        valid_samples.append(item)
        
    kappa = cohen_kappa_score(auto_labels, doctor_labels)
    print(f"-> Điểm đồng thuận Cohen's Kappa: {kappa:.4f}")
    
    if kappa < 0.78:
        print("-> Kappa < 0.78. Tiến hành tăng IoU threshold (0.35) và Rebuild train.jsonl...")
        high_quality_data = [s for s in valid_samples if s.get('iou_score', 0) >= 0.35]
        
        with open(train_path, 'w', encoding='utf-8') as f:
            for d in high_quality_data:
                json.dump(d, f, ensure_ascii=False)
                f.write('\n')
        print(f"Đã rebuild {train_path} với {len(high_quality_data)} mẫu chất lượng cao.")
    else:
        print("-> Chất lượng nhãn đạt chuẩn. Giữ nguyên dataset.")
        
    return kappa