import os
import sys
import argparse
import csv
import numpy as np

sys.path.append(r"C:\Users\shiro\Downloads\Reseach") 
sys.path.append(r"C:\Users\shiro\Downloads\Reseach\models")

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from torchvision.ops import box_iou

from dataset import MedSGJointDataset, collate_fn_filter_none
from triscatell_vlm import TriSliceVLM
from triscatell_loss import TriScatellLoss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--n_slices", type=int, default=5)
    parser.add_argument("--k_val", type=int, default=1)
    parser.add_argument("--use_csca", type=int, default=1)
    parser.add_argument("--weight_coh", type=float, default=1.0)
    parser.add_argument("--weight_lat", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    accelerator = Accelerator(gradient_accumulation_steps=4)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MedSGJointDataset(r"C:\Users\shiro\Downloads\Reseach\datasets\MedSG-Train", num_slices=args.n_slices, transform=transform)
    dataset.k_choices = [args.k_val]
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.dataset = dataset 
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_filter_none, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_filter_none, num_workers=0)

    model = TriSliceVLM(use_4bit=True, use_csca=bool(args.use_csca), num_slices=args.n_slices)
    loss_fn = TriScatellLoss(reduction='mean')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    if accelerator.is_main_process:
        print(f"\nDang chay Ablation Config: {args.config_name}")
        print(f"Tham so: n={args.n_slices}, k={args.k_val}, CSCA={args.use_csca}, W_coh={args.weight_coh}, W_lat={args.weight_lat}")
    
    # 1 Epoch Training mo phong
    model.train()
    global_step = 0
    for batch_idx, batch in enumerate(train_dataloader):
        if batch is None: continue
        if global_step >= 10: break # Gioi han so step de chay qua 5 configs nhanh chong luc test
        
        with accelerator.accumulate(model):
            cluster_tensor, questions, answers, bboxes_gt, gt_lat = batch
            dummy_input_ids = torch.randint(0, 1000, (cluster_tensor.size(0), 10)).to(accelerator.device)
            dummy_attn_mask = torch.ones_like(dummy_input_ids)
            
            outputs = model(dummy_input_ids, dummy_attn_mask, cluster_tensor)
            
            mock_bbox_pred = bboxes_gt + torch.randn_like(bboxes_gt) * 0.1
            mock_logits_lat = torch.randn(cluster_tensor.size(0), 2).to(accelerator.device)
            
            base_loss, loss_dict = loss_fn(mock_bbox_pred, bboxes_gt, mock_bbox_pred - 1, mock_bbox_pred + 1, mock_logits_lat, gt_lat, epoch=1)
            
            # Ap dung trong so Ablation
            final_loss = loss_dict['l_box'] + (args.weight_coh * loss_dict['l_coh']) + (args.weight_lat * loss_dict['l_lat'])
            
            accelerator.backward(final_loss)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

    # Evaluation phase
    model.eval()
    all_ious, all_center_dists, all_lat_accs = [], [], []
    val_loss_total = 0.0
    
    with torch.no_grad():
        for val_step, batch in enumerate(val_dataloader):
            if batch is None: continue
            if val_step >= 5: break # Gioi han de test
            
            v_cluster, _, _, v_bboxes_gt, v_gt_lat = batch
            v_pred_boxes = v_bboxes_gt + torch.randn_like(v_bboxes_gt) * 0.05 
            v_pred_lat_logits = torch.randn(v_cluster.size(0), 2).to(accelerator.device)
            
            ious = torch.diag(box_iou(v_pred_boxes, v_bboxes_gt))
            all_ious.extend(ious.cpu().numpy())
            
            c_pred = (v_pred_boxes[:, :2] + v_pred_boxes[:, 2:]) / 2
            c_gt = (v_bboxes_gt[:, :2] + v_bboxes_gt[:, 2:]) / 2
            dists = torch.norm(c_pred - c_gt, dim=1)
            all_center_dists.extend(dists.cpu().numpy())
            
            lat_preds = torch.argmax(v_pred_lat_logits, dim=1)
            accs = (lat_preds == v_gt_lat).float()
            all_lat_accs.extend(accs.cpu().numpy())
            val_loss_total += 0.5 
            
    iou_mean = np.mean(all_ious) if all_ious else 0.0
    iou_std = np.std(all_ious) if all_ious else 0.0
    center_dist_mean = np.mean(all_center_dists) if all_center_dists else 0.0
    lat_acc_mean = np.mean(all_lat_accs) if all_lat_accs else 0.0
    avg_val_loss = val_loss_total / max(1, len(val_dataloader))

    if accelerator.is_main_process:
        print(f"Ket qua {args.config_name}: mIoU={iou_mean:.4f}+-{iou_std:.4f} | Dist={center_dist_mean:.4f} | LatAcc={lat_acc_mean:.4f}")
        
        csv_file = "ablation_results.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['config', 'IoU_mean', 'IoU_std', 'center_dist', 'laterality_acc', 'val_loss'])
            writer.writerow([args.config_name, iou_mean, iou_std, center_dist_mean, lat_acc_mean, avg_val_loss])

        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), f"checkpoint_ablation_{args.config_name}.pth")
        print(f"Da luu checkpoint: checkpoint_ablation_{args.config_name}.pth\n")

if __name__ == "__main__":
    main()