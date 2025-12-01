export PYTHONHASHSEED=5112
export output_dir="./seed5112_rank8/000111111000/qqp"
python3 run_glue_deberta_roberta.py \
--model_name_or_path roberta-base \
--task_name qqp \
--do_train \
--do_eval \
--max_seq_length 512 \
--per_device_train_batch_size 16 \
--learning_rate 5e-4 \
--num_train_epochs 25 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_layers 000111111000 \
--seed 5112 \
--weight_decay 0.1 \
--fp16 True \
--lr_scheduler_type linear
