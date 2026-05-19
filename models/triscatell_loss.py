import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class TriScatellLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.lambda_1 = 0.3 

    def get_lambda_2(self, epoch):
        if epoch <= 1: return 0.5
        elif epoch >= 3: return 0.1
        else: return 0.5 - 0.2 * (epoch - 1)

    def compute_coherence_loss(self, bbox_z, bbox_minus, bbox_plus):
        c_z = (bbox_z[:, :2] + bbox_z[:, 2:]) / 2.0
        c_m = (bbox_minus[:, :2] + bbox_minus[:, 2:]) / 2.0
        c_p = (bbox_plus[:, :2] + bbox_plus[:, 2:]) / 2.0

        c_min = torch.min(c_m, c_p)
        c_max = torch.max(c_m, c_p)

        hinge = F.relu(c_min - c_z) + F.relu(c_z - c_max)
        return hinge.sum(dim=1)

    def forward(self, bbox_pred, bbox_gt, bbox_pred_minus, bbox_pred_plus, logits_lat, gt_lat, epoch):
        # 1. Box Loss
        l1_loss = F.l1_loss(bbox_pred, bbox_gt, reduction='none').mean(dim=1)
        giou_loss = generalized_box_iou_loss(bbox_pred, bbox_gt, reduction='none')
        l_box = l1_loss + giou_loss

        # 2. Coherence Loss & Laterality Loss
        l_coh = self.compute_coherence_loss(bbox_pred, bbox_pred_minus, bbox_pred_plus)
        l_lat = F.cross_entropy(logits_lat, gt_lat, reduction='none')

        # 3. Schedule
        lambda_2 = self.get_lambda_2(epoch)
        total_loss = l_box + self.lambda_1 * l_coh + lambda_2 * l_lat

        if self.reduction == 'mean':
            total_loss_out = total_loss.mean()
            l_box_out, l_coh_out, l_lat_out = l_box.mean(), l_coh.mean(), l_lat.mean()
        else:
            total_loss_out = total_loss

        # 4. Log WandB
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "loss/L_box": l_box_out.item(),
                "loss/L_coh": l_coh_out.item(),
                "loss/L_lat": l_lat_out.item(),
                "loss/Total": total_loss_out.item(),
                "schedule/lambda_2": lambda_2
            })

        return total_loss_out, {"l_box": l_box_out, "l_coh": l_coh_out, "l_lat": l_lat_out}