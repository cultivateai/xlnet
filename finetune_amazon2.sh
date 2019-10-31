#!/bin/bash
export GS_ROOT='gs://sentiment-datasets'
export LARGE_DIR='xlnet_cased_L-24_H-1024_A-16'
export AMAZON_DIR='/home/andyhorng/xlnet/amazon_dataset/amazon_review_polarity_csv/'
export TPU_NAME='sentiment'
python run_classifier.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=amazon2 \
  --data_dir=${AMAZON_DIR} \
  --output_dir=${GS_ROOT}/proc_data/amazon2 \
  --model_dir=${GS_ROOT}/exp/amazon2 \
  --uncased=False \
  --spiece_model_file=${LARGE_DIR}/spiece.model \
  --model_config_path=${GS_ROOT}/${LARGE_DIR}/xlnet_config.json \
  --init_checkpoint=${GS_ROOT}/${LARGE_DIR}/xlnet_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=2e-5 \
  --train_steps=8000 \
  --warmup_steps=500 \
  --save_steps=500 \
  --iterations=500


# Performance: Error rate 32.26 (amazon5), 2.40 (amazon2)


# ctpu up --tf-version 1.14.1.dev20190518 --zone=us-central1-b --name=sentiment --tpu-size v3-8
# git clone https://github.com/cultivateai/xlnet.git
# gsutil -m cp -r gs://sentiment-datasets/yelp_dataset ./
# gsutil -m cp -r gs://sentiment-datasets/amazon_dataset ./
# gsutil -m cp -r gs://sentiment-datasets/aclImdb ./
# gsutil -m cp -r gs://sentiment-datasets/xlnet_cased_L-24_H-1024_A-16 ./


# to view tensorboard (run in cloud shell)
export STORAGE_BUCKET=gs://sentiment-datasets
export MODEL_DIR=${STORAGE_BUCKET}/exp/amazon2
export TPU_IP=10.240.1.2
tensorboard --logdir=${MODEL_DIR} --master_tpu_unsecure_channel=${TPU_IP} &
