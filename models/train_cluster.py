%%writefile train_cluster.py
import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import get_cosine_schedule_with_warmup
from transformers import Qwen2_5_VLProcessor

from dataset import MedSGJointDataset, collate_fn_filter_none
from triscatell_vlm import TriSliceVLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Đường dẫn đến dataset")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4)
    set_seed(42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = MedSGJointDataset(root_dir=args.data_dir, num_slices=5, transform=transform)
    
    if len(full_dataset) == 0:
        raise ValueError(f"Không tìm thấy dữ liệu nào tại {args.data_dir}. Vui lòng kiểm tra lại đường dẫn!")
        
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset = full_dataset 
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_filter_none, num_workers=2)

    model = TriSliceVLM(use_4bit=True) 
    processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    processor.tokenizer.add_special_tokens({'additional_special_tokens': ['<|box_start|>', '<|box_end|>']})

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 30, 1000)
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)

    model.train()
    global_step = 0
    
    for epoch in range(1, 4):
        if accelerator.is_main_process:
            print(f"\n🚀 BẮT ĐẦU EPOCH {epoch}")
            
        for batch in train_dataloader:
            if batch is None: continue
            
            with accelerator.accumulate(model):
                cluster_tensor, questions, target_answers = batch
                
                text_inputs = processor.tokenizer(
                    list(questions), padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(accelerator.device)
                
                labels = processor.tokenizer(
                    list(target_answers), padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).input_ids.to(accelerator.device)
                labels[labels == processor.tokenizer.pad_token_id] = -100
                
                B, T, C, H, W = cluster_tensor.shape
                pixel_values = cluster_tensor.view(B*T, C, H, W).to(accelerator.device)
                image_grid_thw = torch.tensor([[T, H//14, W//14]] * B, dtype=torch.long).to(accelerator.device)
                
                loss, logits = model(
                    input_ids=text_inputs.input_ids, 
                    attention_mask=text_inputs.attention_mask, 
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels
                )
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    global_step += 1
                    if accelerator.is_main_process and global_step % 10 == 0:
                        print(f"Step {global_step} | Loss: {loss.item():.4f}")
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

if __name__ == "__main__":
    main()