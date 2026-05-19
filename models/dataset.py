
import os
import json
import torch
import random
from PIL import Image
from torch.utils.data import Dataset

class MedSGJointDataset(Dataset):
    def __init__(self, root_dir, num_slices=5, transform=None):
        self.root_dir = root_dir
        self.num_slices = num_slices
        self.transform = transform
        self.data = []
        self.k_choices = [1] 
        
        print(f"Dang quet du lieu tai: {root_dir}")
        for i in range(1, 9):
            task_name = f"Task{i}"
            json_path = os.path.join(root_dir, f"{task_name}.json")
            task_dir = os.path.join(root_dir, task_name)
            
            if os.path.exists(json_path) and os.path.exists(task_dir):
                with open(json_path, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
                    for item in task_data:
                        item['source_task_dir'] = task_dir
                    self.data.extend(task_data)
        print(f"Tong cong: {len(self.data)} mau.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item.get('question', '')
        
        raw_bbox = item.get('visual_grounding', [0.0, 0.0, 0.0, 0.0]) 
        x1, y1, x2, y2 = [int(v) for v in raw_bbox]
        target_bbox_str = f"<|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"
        
        q_lower = question.lower()
        lat_str = "bên phải" if "right" in q_lower or "phải" in q_lower else "bên trái" if "left" in q_lower or "trái" in q_lower else ""
        target_answer = f"{target_bbox_str} Tổn thương nằm {lat_str}".strip()
        
        slice_paths = item.get('images', []) 
        task_dir = item['source_task_dir']
        total_slices = len(slice_paths)
        if total_slices == 0: return None

        k = random.choice(self.k_choices)
        center_idx = total_slices // 2
        half = self.num_slices // 2
        
        raw_indices = [center_idx + i * k for i in range(-half, half + 1)]
        indices = [max(0, min(i, total_slices - 1)) for i in raw_indices]
            
        cluster_imgs = []
        for i in indices:
            file_name = os.path.basename(slice_paths[i])
            img_path = os.path.join(task_dir, file_name)
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform: img = self.transform(img)
                cluster_imgs.append(img)
            except Exception:
                return None 
            
        if len(cluster_imgs) < self.num_slices: return None
        cluster_tensor = torch.stack(cluster_imgs) 
        
        return cluster_tensor, question, target_answer

def collate_fn_filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None
    return torch.utils.data.dataloader.default_collate(batch)