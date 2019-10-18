#!/bin/bash

echo $(pwd)
cd ../
echo $(pwd)

BUCKET_NAME=YOUR_BUCKET_NAME

MODEL=MixNeXtVladModel
iter=60
ckpt=200000

train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord'
valid_data_pattern='gs://youtube8m-ml-us-east1/3/frame/validate/validate*.tfrecord'
train_valid_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord'
test_data_pattern='gs://youtube8m-ml/3/frame/test/test*.tfrecord'

TRAIN_DIR="${MODEL}_iter${iter}"

#'''
echo "============== start training train* =============="

JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); 
gcloud --verbosity=debug ai-platform jobs \
submit training $JOB_NAME \
--package-path=src --module-name=src.train \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=src/cloudml-gpu.yaml \
-- --train_data_pattern=${train_data_pattern} --train_all --train_valid_data_pattern=${train_valid_data_pattern} --google_cloud=True \
--model=${MODEL} \
--train_dir=$BUCKET_NAME/${TRAIN_DIR} --google_cloud=True \
--model=${MODEL} \
--groups=8 --nextvlad_cluster_size=112 --nextvlad_hidden_size=2048 \
--expansion=2 --gating_reduction=16  --drop_rate=0.5 \
--mix_number=3 --cl_temperature=3 --cl_lambda=9 \
--video_level_classifier_model=LogisticModel --l2_penalty=1e-5 --label_loss=CrossEntropyLoss \
--frame_features --feature_names="rgb,audio" --feature_sizes='1024,128' \
--batch_size=128 --base_learning_rate=0.0001 --moe_l2=1e-6 --iterations=${iter} --learning_rate_decay=0.9 \
--export_model_steps=1000 --num_epochs=10000000 --max_steps=200000 \
--train_dir=$BUCKET_NAME/${TRAIN_DIR}
#--start_new_model
#'''
'''
echo "============== start finetuning train* =============="
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); 
gcloud --verbosity=debug ai-platform jobs \
submit training $JOB_NAME \
--package-path=src --module-name=src.train \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=src/cloudml-gpu.yaml \
-- --train_data_pattern=${valid_data_pattern} \
--train_dir=$BUCKET_NAME/${TRAIN_DIR} --google_cloud=True \
--model=${MODEL} --segment_labels=True \
--groups=8 --nextvlad_cluster_size=112 --nextvlad_hidden_size=2048 \
--expansion=2 --gating_reduction=16  --drop_rate=0.5 \
--mix_number=3 --cl_temperature=3 --cl_lambda=9 \
--video_level_classifier_model=LogisticModel --l2_penalty=1e-5 --label_loss=CrossEntropyLoss --iterations=${iter} \
--frame_features --feature_names="rgb,audio" --feature_sizes='1024,128' \
--batch_size=128 --base_learning_rate=0.0001 --moe_l2=1e-6 --iterations=${iter} --learning_rate_decay=0.9 \
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
--model=${MODEL} --segment_labels \
--groups=8 --nextvlad_cluster_size=112 --nextvlad_hidden_size=2048 \
--expansion=2 --gating_reduction=16  --drop_rate=0.5 \
--mix_number=3 --cl_temperature=3 --cl_lambda=9 \
--video_level_classifier_model=LogisticModel --l2_penalty=1e-5 --label_loss=CrossEntropyLoss --iterations=${iter} \
--norun_once --top_k=1000 --batch_size=128 --moe_l2=1e-6 --iterations=${iter} --checkpoint_file=${ckpt}
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
--model=${MODEL} --segment_labels \
--groups=8 --nextvlad_cluster_size=112 --nextvlad_hidden_size=2048 \
--expansion=2 --gating_reduction=16  --drop_rate=0.5 \
--mix_number=3 --cl_temperature=3 --cl_lambda=9 \
--video_level_classifier_model=LogisticModel --l2_penalty=1e-5 --label_loss=CrossEntropyLoss --iterations=${iter} \
--run_once=True --top_k=1000 --batch_size=128 --checkpoint_file=${ckpt}
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
--top_k 1000 --num_readers 10 --batch_size=128 --segment_max_pred=100000 --shift=1 \
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
--top_k 1000 --num_readers 10 --batch_size=128 --segment_max_pred=100000 --shift=2 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_oof_shift2.csv

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
--top_k 1000 --num_readers 10 --batch_size=128 --segment_max_pred=100000 \
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
--top_k 1000 --num_readers 10 --batch_size=128 --segment_max_pred=100000 --shift=1 \
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
--top_k 1000 --num_readers 10 --batch_size=128 --segment_max_pred=100000 --shift=2 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_shift2.csv
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
--top_k 1000 --num_readers 10 --batch_size=128 --segment_max_pred=100000 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune.csv
'''
