#!/bin/bash
export GS_ROOT='gs://sentiment-datasets'
export LARGE_DIR='/home/andyhorng/xlnet/xlnet_cased_L-24_H-1024_A-16'
export IMDB_DIR='/home/andyhorng/xlnet/aclImdb'
export TPU_NAME='andy'
python run_classifier.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=imdb \
  --data_dir=${IMDB_DIR} \
  --output_dir=${GS_ROOT}/proc_data/imdb \
  --model_dir=${GS_ROOT}/exp/imdb \
  --uncased=False \
  --spiece_model_file=${LARGE_DIR}/spiece.model \
  --model_config_path=${LARGE_DIR}/xlnet_config.json \
  --init_checkpoint=${LARGE_DIR}/xlnet_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=2e-5 \
  --train_steps=4000 \
  --warmup_steps=500 \
  --save_steps=500 \
  --iterations=500


# Expected performance: "eval_accuracy 0.962+ "
