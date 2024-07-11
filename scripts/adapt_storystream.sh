#ps aux | grep 'src/train/train' | awk '{print $2}' | xargs kill -9

torchrun --nproc_per_node=8 --nnodes=1 --master_port=20007 --node_rank=0 \
   src/train/train_sdxl_img2img_llm.py \
    --image_transform configs/processer/qwen_448_transform_keep_ratio.yaml \
    --sd_image_transform configs/processer/sd_transform_1024.yaml \
    --visual_encoder configs/visual_encoder/qwen_vitg_448.yaml \
    --discrete_model configs/discrete_model/discrete_identity.yaml \
    --adapter configs/detokenizer/detokenizer_sdxl_qwen_vit_pretrained.yaml \
    --train_dataset configs/data/george_sdxl.yaml \
    --tokenizer configs/tokenizer/clm_llama_tokenizer.yaml \
    --llm_model configs/clm_models/llama2chat7b_lora.yaml \
    --agent_model configs/clm_models/agent_7b_sft.yaml \
    --diffusion_model_path pretrained/stable-diffusion-xl-base-1.0 \
    --output_dir train_output_detokenizer/george_sdxl \
    --expr_name 'george_sdxl' \
    --learning_rate 1e-4 \
    --weight_decay 0.03 \
    --mixed_precision bf16 \
    --num_train_epochs 10 \
    --max_steps 1600 \
    --save_steps 1600 \
    --lr_scheduler_type cosine \
    --warmup_steps 500 \
    --min_lr_ratio 0.01 \
    --dataloader_num_workers 2 \
    --gradient_accumulation_steps 4 \
    --deepspeed_plugin configs/accelerate/deepspeed_stage_2.yaml
