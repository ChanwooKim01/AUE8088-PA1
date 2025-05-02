#!/bin/bash

# Model list
model_list=("MyNetwork" "MyNetwork1" "MyNetwork2" "MyNetwork3") 
# model_list=("resnet18") 
# torch.optim list
optimizer_list=("SGD")

scheduler_list=("MultiStepLR")
batch_list=(512)
num_classes=200
num_epochs=40

for model_name in "${model_list[@]}"
do
  for optimizer_name in "${optimizer_list[@]}"
  do
    for scheduler_name in "${scheduler_list[@]}"
    do
      for batch_size in "${batch_list[@]}"
      do
        echo "Model name: ${model_name}, Optimizer: ${optimizer_name}, Scheduler: ${scheduler_name}"

        wandb_name="${model_name}-B${batch_size}-${optimizer_name}-${scheduler_name}"

        MODEL_NAME=${model_name} \
        OPTIMIZER_TYPE=${optimizer_name} \
        SCHEDULER_TYPE=${scheduler_name} \
        WANDB_NAME=${wandb_name} \
        BATCH_SIZE=${batch_size} \
        python train.py
      done
    done
  done
done
