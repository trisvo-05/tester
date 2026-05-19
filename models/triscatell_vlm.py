import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

class CSCAModule(nn.Module):
    def __init__(self, vit_dim=1152, hidden_dim=1024, num_heads=8):
        super().__init__()
        self.in_proj = nn.Linear(vit_dim, hidden_dim) if vit_dim != hidden_dim else nn.Identity()
        self.out_proj = nn.Linear(hidden_dim, vit_dim) if vit_dim != hidden_dim else nn.Identity()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, visual_features, num_slices):
        q = self.in_proj(visual_features)
        attn_out, _ = self.mha(q, q, q)
        out = q + self.gamma * self.norm(attn_out)
        return self.out_proj(out)

class TriSliceVLM(nn.Module):
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", use_4bit=True, use_csca=True, num_slices=5):
        super().__init__()
        self.use_csca = use_csca
        self.num_slices = num_slices
        
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
        else:
            model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}

        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        
        lora_config = LoraConfig(
            r=64, lora_alpha=128, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        
        vit_dim = self.backbone.config.text_config.hidden_size 
        llm_dim = self.backbone.config.text_config.hidden_size 
        
        if self.use_csca:
            self.csca = CSCAModule(vit_dim=vit_dim, hidden_dim=llm_dim, num_heads=8)
        else:
            self.csca = nn.Identity()

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, labels=None):
        visual_inputs = self.backbone.visual(pixel_values, grid_thw=image_grid_thw)
        
        if self.use_csca:
            visual_inputs = self.csca(visual_inputs, self.num_slices)
            
        text_embeds = self.backbone.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat([visual_inputs, text_embeds], dim=1)
        
        batch_size = inputs_embeds.size(0)
        num_vis_tokens = visual_inputs.size(1)
        vis_mask = torch.ones((batch_size, num_vis_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
        full_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
        
        # Xử lý nhãn: Bỏ qua tính Loss trên phần hình ảnh bằng giá trị -100
        if labels is not None:
            vis_labels = torch.full((batch_size, num_vis_tokens), -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([vis_labels, labels], dim=1)
        else:
            full_labels = None
        
        outputs = self.backbone(
            inputs_embeds=inputs_embeds, 
            attention_mask=full_attention_mask,
            labels=full_labels,
            output_hidden_states=False
        )
        
        return outputs.loss, outputs.logits