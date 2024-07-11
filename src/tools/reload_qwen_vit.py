import torch
from transformers import AutoModelForCausalLM

torch.manual_seed(1234)

qwen_model_path = 'pretrained/Qwen-VL-Chat'
save_path = 'pretrained/QwenViT/qwen_vit_G.pt'

model = AutoModelForCausalLM.from_pretrained(qwen_model_path, device_map="cpu", trust_remote_code=True).eval()

visual_encoder = model.transformer.visual
print(visual_encoder)

torch.save(visual_encoder.state_dict(), save_path)
