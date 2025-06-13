deepspeed --num_gpus 1 \
    ds_train.py \
    --deepspeed \
    --deepspeed_config ds_config.yml \
    --block_size 1024 \
    --pipeline_parallel_size 1 \
    --eval_steps 10
