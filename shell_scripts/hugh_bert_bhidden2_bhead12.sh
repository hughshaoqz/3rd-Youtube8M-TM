#!/bin/bash

echo $(pwd)
cd ../
echo $(pwd)

BUCKET_NAME=YOUR_BUCKET_NAME

MODEL=Bert
bert_hidden=2
bert_head=12
pool='none'
ckpt=200000

train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord'
valid_data_pattern='gs://youtube8m-ml-us-east1/3/frame/validate/validate*.tfrecord'
train_valid_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord'
test_data_pattern='gs://youtube8m-ml/3/frame/test/test*.tfrecord'

TRAIN_DIR="${MODEL}_bhidden${bert_hidden}_bhead${bert_head}_${pool}"

#'''
echo "============== start pre-training train* =============="

JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); 
gcloud --verbosity=debug ai-platform jobs \
submit training $JOB_NAME \
--package-path=src --module-name=src.train \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=src/cloudml-gpu.yaml \
-- --train_data_pattern=${train_data_pattern} --train_all --train_valid_data_pattern=${train_valid_data_pattern} --google_cloud=True \
--model=${MODEL} \
--bert_hidden_layer=${bert_hidden} --bert_attention_heads=${bert_head} --pooling_strategy=${pool} --sample_random_frames=False \
--frame_features --feature_names="rgb,audio" --feature_sizes='1024,128' \
--batch_size=128 --base_learning_rate=0.0001 --moe_l2=1e-6 --iterations=5 --learning_rate_decay=0.9 \
--export_model_steps=1000 --num_epochs=10000000 --max_steps=200000 \
--bert_config_file=$BUCKET_NAME/bert_config.json \
--train_dir=$BUCKET_NAME/${TRAIN_DIR}
#--start_new_model
#'''
'''
echo "============== start fine-tuning* =============="
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); 
gcloud --verbosity=debug ai-platform jobs \
submit training $JOB_NAME \
--package-path=src --module-name=src.train \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=src/cloudml-gpu.yaml \
-- --train_data_pattern=${valid_data_pattern} \
--train_dir=$BUCKET_NAME/${TRAIN_DIR} --google_cloud=True \
--model=${MODEL} --bert_hidden_layer=${bert_hidden} --bert_attention_heads=${bert_head} --bert_config_file=$BUCKET_NAME/bert_config.json --segment_labels=True \
--pooling_strategy=${pool} --sample_random_frames=False \
--frame_features --feature_names="rgb,audio" --feature_sizes='1024,128' \
--batch_size=128 --base_learning_rate=0.0001 --moe_l2=1e-6 --iterations=5 --learning_rate_decay=0.9 \
--export_model_steps=200 --num_epochs=10000000 --max_steps=220000
'''
'''
echo "============== start evaluating =============="
JOB_TO_EVAL=${TRAIN_DIR}
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S); 
gcloud --verbosity=debug ai-platform jobs \
submit training $JOB_NAME \
--package-path=src --module-name=src.eval \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=src/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://youtube8m-ml-us-east1/3/frame/validate/validate*.tfrecord' --train_dir=$BUCKET_NAME/${JOB_TO_EVAL} --google_cloud=True \
--model=${MODEL} --bert_hidden_layer=${bert_hidden} --bert_attention_heads=${bert_head} --bert_config_file=$BUCKET_NAME/bert_config.json --segment_labels \
--pooling_strategy=${pool} --sample_random_frames=False \
--norun_once --top_k=1000 --batch_size=512 --moe_l2=1e-6 --iterations=5 --checkpoint_file=${ckpt}
'''

#####################################################################################
'''
echo "============== start evaluating for inference =============="
JOB_TO_EVAL=${TRAIN_DIR}
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S); 
gcloud --verbosity=debug ai-platform jobs \
submit training $JOB_NAME \
--package-path=src --module-name=src.eval \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=src/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://youtube8m-ml-us-east1/3/frame/validate/validate*.tfrecord' --train_dir=$BUCKET_NAME/${JOB_TO_EVAL} --google_cloud=True \
--model=${MODEL} --bert_hidden_layer=${bert_hidden} --bert_attention_heads=${bert_head} --bert_config_file=$BUCKET_NAME/bert_config.json --segment_labels \
--pooling_strategy=${pool} --sample_random_frames=False \
--run_once=True --top_k=1000 --batch_size=512 --moe_l2=1e-6 --iterations=5 --checkpoint_file=${ckpt}
'''
'''
echo "============== start inference oof =============="
JOB_TO_EVAL=${TRAIN_DIR} \
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S); \
gcloud --verbosity=debug ai-platform jobs \
submit training $JOB_NAME \
--package-path=src --module-name=src.inference \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=src/cloudml-gpu.yaml \
-- --input_data_pattern=${valid_data_pattern} \
--google_cloud=True \
--segment_labels \
--segment_label_ids_file=$BUCKET_NAME/segment_label_ids.csv \
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 --shift=1 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_oof_shift1.csv

echo "============== start inference oof =============="
JOB_TO_EVAL=${TRAIN_DIR} \
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S); \
gcloud --verbosity=debug ai-platform jobs \
submit training $JOB_NAME \
--package-path=src --module-name=src.inference \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=src/cloudml-gpu.yaml \
-- --input_data_pattern=${valid_data_pattern} \
--google_cloud=True \
--segment_labels \
--segment_label_ids_file=$BUCKET_NAME/segment_label_ids.csv \
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 --shift=-1 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_oof_shift-1.csv

echo "============== start inference oof =============="
JOB_TO_EVAL=${TRAIN_DIR} \
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S); \
gcloud --verbosity=debug ai-platform jobs \
submit training $JOB_NAME \
--package-path=src --module-name=src.inference \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=src/cloudml-gpu.yaml \
-- --input_data_pattern=${valid_data_pattern} \
--google_cloud=True \
--segment_labels \
--segment_label_ids_file=$BUCKET_NAME/segment_label_ids.csv \
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_oof.csv
'''
'''
echo "============== start inference test =============="
JOB_TO_EVAL=${TRAIN_DIR} \
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S); \
gcloud --verbosity=debug ai-platform jobs \
submit training $JOB_NAME \
--package-path=src --module-name=src.inference \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=src/cloudml-gpu.yaml \
-- --input_data_pattern=${test_data_pattern} \
--google_cloud=True \
--segment_labels \
--segment_label_ids_file=$BUCKET_NAME/segment_label_ids.csv \
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 --shift=1 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_shift1.csv


echo "============== start inference test =============="
JOB_TO_EVAL=${TRAIN_DIR} \
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S); \
gcloud --verbosity=debug ai-platform jobs \
submit training $JOB_NAME \
--package-path=src --module-name=src.inference \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=src/cloudml-gpu.yaml \
-- --input_data_pattern=${test_data_pattern} \
--google_cloud=True \
--segment_labels \
--segment_label_ids_file=$BUCKET_NAME/segment_label_ids.csv \
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 --shift=-1 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_shift-1.csv
'''
'''
echo "============== start inference test =============="
JOB_TO_EVAL=${TRAIN_DIR} \
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S); \
gcloud --verbosity=debug ai-platform jobs \
submit training $JOB_NAME \
--package-path=src --module-name=src.inference \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=src/cloudml-gpu.yaml \
-- --input_data_pattern=${test_data_pattern} \
--google_cloud=True \
--segment_labels \
--segment_label_ids_file=$BUCKET_NAME/segment_label_ids.csv \
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune.csv
'''
