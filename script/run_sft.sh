# 运行脚本前请仔细阅读wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_zh)
# Read the wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_zh) carefully before running the script
lr=1e-4
lora_rank=8
lora_alpha=16
lora_trainable="q_proj,v_proj"
lora_dropout=0.05

pretrained_model=/root/autodl-tmp/llama2-chinese
chinese_tokenizer_path=/root/autodl-tmp/peft_chinese_alpaca_lora_7b/tokenizer.model
dataset_dir=/root/autodl-tmp/sft-hc3/train
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=1
max_seq_length=64
output_dir=/root/autodl-tmp/output_dir
validation_file=/root/autodl-tmp/sft-hc3/test.json
peft_model=/root/autodl-tmp/peft_chinese_alpaca_lora_7b
deepspeed_config_file=ds_zero2_no_offload.json

torchrun --nnodes 1 --nproc_per_node 1 run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --validation_file ${validation_file}
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --peft_path ${peft_model} 
    --ddp_find_unused_parameters False
    # --modules_to_save ${modules_to_save} \
