deepspeed --num_gpus 1 \
    train_with_ds.py \
    --deepspeed \
    --deepspeed_config ds_config.yaml
