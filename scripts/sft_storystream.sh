#ps aux | grep 'src/train/train' | awk '{print $2}' | xargs kill -9
#ps aux | grep 'run' | awk '{print $2}' | xargs kill -9

torchrun --nproc_per_node=8 --nnodes=1 --master_port=29502 \
    src/train/train_clm_sft.py \
    --image_transform configs/processer/qwen_448_transform.yaml \
    --tokenizer configs/tokenizer/clm_llama_tokenizer.yaml \
    --visual_encoder configs/visual_encoder/qwen_vitg_448.yaml \
    --llm_model configs/clm_models/llama2chat7b_lora.yaml \
    --agent_model configs/clm_models/agent_7b_seedx_pretrained.yaml \
    --train_dataset configs/data/george_mid_long30_sft.yaml \
    --output_dir train_output/sft_george \
    --expr_name 'sft_george' \
    --learning_rate 1e-4 \
    --batch_size 16 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --num_train_epochs 10 \
    --max_steps 6000 \
    --save_steps 3000 \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --min_lr_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed_plugin configs/accelerate/deepspeed_stage_2.yaml