set CUDA_VISIBLE_DEVICES=0 
set WANDB_DISABLED=true

python main.py ^
    --do_train ^
    --train_file train.json ^
    --validation_file test.json ^
    --prompt_column prompt ^
    --response_column response ^
    --overwrite_cache ^
    --model_name_or_path D:\ml\chatglm-6b-int4-qe\ ^
    --output_dir output/roleplaying-chatglm-6b-pt-128-1e-2 ^
    --overwrite_output_dir ^
    --max_source_length 400 ^
    --max_target_length 80 ^
    --pre_seq_len 128 ^
    --per_device_train_batch_size 1 ^
    --per_device_eval_batch_size 1 ^
    --gradient_accumulation_steps 16 ^
    --predict_with_generate ^
    --max_steps 6000 ^
    --logging_steps 10 ^
    --save_steps 50 ^
    --learning_rate 1e-2 ^
    --quantization_bit 4 ^
    --resume_from_checkpoint output/roleplaying-chatglm-6b-pt-128-1e-2/checkpoint-3150

