deepspeed --num_gpus 1 \
    ds_train.py \
    --deepspeed \
    --deepspeed_config ds_config.yml
