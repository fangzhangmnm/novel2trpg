set CUDA_VISIBLE_DEVICES=0 
set WANDB_DISABLED=true

python main.py ^
    --do_train ^
    --train_file output_zero/train.json ^
    --validation_file output_zero/test.json ^
    --prompt_column prompt ^
    --response_column response ^
    --overwrite_cache ^
    --model_name_or_path D:\ml\chatglm-6b-int4\ ^
    --output_dir output_zero/llm-checkpoints/ ^
    --overwrite_output_dir ^
    --max_source_length 450 ^
    --max_target_length 50 ^
    --pre_seq_len 100 ^
    --per_device_train_batch_size 1 ^
    --per_device_eval_batch_size 1 ^
    --gradient_accumulation_steps 16 ^
    --predict_with_generate ^
    --max_steps 6000 ^
    --logging_steps 10 ^
    --save_steps 50 ^
    --learning_rate 1e-2 ^
    --quantization_bit 4 
@REM --resume_from_checkpoint output_zero/llm-checkpoints/checkpoint-####

