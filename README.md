# BERT and Frame-Level Models for Large-scale Video Segment Classificationwith Test-time Augmentation

This repo contains the code of the 9th position at [The 3rd YouTube-8M Video Understanding Challenge](https://www.kaggle.com/c/youtube8m-2019/overview) on Kaggle.

Our best final submission mode is a weighted ensemble of 9 models, which reaches 0.7871 on the private leaderboard. The weights are tuned by Baysian Optimization. For the code of Baysian Optimization and 
ensemble method, please refer to Ensemble.ipynb file.
The details of each model can  be found below:

| Model  | Weight (%) | Public MAP | Private MAP | 
| ------------- | ------------- | ------------- | ------------- |
|GatedNetVLAD(H1024C16) TTA[-1, 1]|0.22|0.7629|0.7556|
|BERT(L2H12) TTA[-1, 1]|2.81|0.7729|0.7680|
|BERTCrossMean(L2H8) TTA[-1, 1]|4.90|0.7748|0.7680|
|BERT(L3H12) TTA[-1, 1]|13.88|0.7751|0.7688|
|BERTCrossAttn(L2H8) TTA[-1, 1]|5.97|0.7758|0.7692
|BERTFirst(L2H12) TTA[-2, 1]|19.67|0.7792|0.7707|
|MixNeXtVLAD(Iter 300) TTA[-1, 1]|20.03|0.7809|0.7711|
|MixNeXtVLAD(Iter 60) TTA[0, 2]|17.26|0.7796|0.7721|
|BERTMean(L2H12) TTA[-2, 1]|15.46|0.7802|0.7725|
|Ensemble of all above|100.00|0.7944|0.7871|

The code is based on the [Google Youtube-8M starter code](https://github.com/google/youtube-8m). Please refer to their README to see the needed dependencies in detail. 
In general, you need tensorflow 1.4, numpy 1.14 and pandas 0.24 installed. the model
is trained and evaluated on [YouTube-8M](https://research.google.com/youtube8m/) dataset.

Hardware Requirement: The model has been run on a single NVIDIA RTX 2080Ti 11 GB GPU locally, and standard_p100 (NVIDIA TESLA P100 16 GB GPU) on Google Cloud AI Platform.

## Train a Single Model

The following are the examples of training a single model locally or on the Google Cloud AI Platform. The model we show below is BERTMean(L2H12) TTA[-2, 1], which
got MAP 0.7725 on the final private leaderboard. For the training details of other models, please refer to shell_script folder.

There mainly 4 steps for the whole process to get final prediction of a single model: Pre-training, Fine-tuning, Evaluation, and Inference.

We first pre-train the model on total 6M video-level label training partition and almost all validation partition data except the 300 holdout files for 
200k steps with batch size 128 and initial learning rate 1e-4. Learning rate is exponentially decreased by a factor of 0.9 every 2M examples. 
After the pre-training, we fine-tune the model on validation partition files with segment-level label except the pre-holdout 300 files for local validation. We fine-tune the model for another 20k steps 
and pick the checkpoint with the highest MAP on local validation data. Finally, we evaluate the picked checkpoint, and inference prediction based on it.

### Google Cloud AI Platform
We recommend using Google Cloud AI Platform to run the code due to the large size of the dataset. If use Google Cloud AI Platform, you could access the dataset
in the google public bucket. You can initialize the parameters as followes:
```sh
# Since this example is for BERTMean(L2H12) TTA[-2, 1], we choose MODEL=Bert, bert_hidden=2, bert_head=12, pooling=mean
BUCKET_NAME=YOUR_BUCKET_NAME

MODEL=Bert
bert_hidden=2
bert_head=12
pool='mean'
ckpt=200000

train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord'
valid_data_pattern='gs://youtube8m-ml-us-east1/3/frame/validate/validate*.tfrecord'
train_valid_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord'
test_data_pattern='gs://youtube8m-ml/3/frame/test/test*.tfrecord'

TRAIN_DIR="${MODEL}_bhidden${bert_hidden}_bhead${bert_head}_${pool}"
```
1 Pre-Training
```sh
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
```

2.a Fine-tuning
```sh
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
```

2.b Evaluation when Fine-tuning
```sh
ckpt=200000
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
```
3 Evaluation for Inference
```sh
ckpt=YOUR_BEST_CKPT
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
```

4.a Inference for OOF
```sh
# Since we use TTA, it is required to inference couple times with different shifts. For example, if model is TTA[-2,1], you need to inference 4 times, one for --shift=-2, one for --shift=-1, one for --shift=0, one for --shift=1 
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
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 --shift=-2 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_oof_shift-2.csv

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
```

4.b Inference for Test
```sh
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
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 --shift=-2 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_shift-2.csv

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
```

### Locally

In order to train the model locally, you need to download all required files from [YouTube-8M](https://research.google.com/youtube8m/). You can initialize the parameters as followes:
```sh
# Since this example is for BERTMean(L2H12) TTA[-2, 1], we choose MODEL=Bert, bert_hidden=2, bert_head=12, pooling=mean
BUCKET_NAME=YOUR_BUCKET_NAME
MODEL=Bert
bert_hidden=2
bert_head=12
pooling=mean

# Specify your data path. We use both train and valid for pre-train, so you also need to specify the train_valid_data_pattern, which is the validation data we used for training.
train_data_pattern=./inputs/data/frame/2/train/train*.tfrecord
train_valid_data_pattern=./inputs/data/frame/2/validate/validate*.tfrecord
valid_data_pattern=./inputs/data/frame/3/validate_data/validate*.tfrecord
test_data_pattern=./inputs/data/frame/3/test/test*.tfrecord

TRAIN_DIR=${MODEL}_${bert_hidden}_${bert_head}_${polling}
```

1 Pre-Training
```sh
python src/train.py --train_data_pattern=${train_data_pattern} --train_all --train_valid_data_pattern=${train_valid_data_pattern} --google_cloud=True \
--model=${MODEL} \
--bert_hidden_layer=${bert_hidden} --bert_attention_heads=${bert_head} --pooling_strategy=${pool} --sample_random_frames=False \
--frame_features --feature_names="rgb,audio" --feature_sizes='1024,128' \
--batch_size=128 --base_learning_rate=0.0001 --moe_l2=1e-6 --iterations=5 --learning_rate_decay=0.9 \
--export_model_steps=1000 --num_epochs=10000000 --max_steps=200000 \
--bert_config_file=$BUCKET_NAME/bert_config.json \
--train_dir=$BUCKET_NAME/${TRAIN_DIR}
```

2.a Fine-tuning
```sh
python src/train.py --train_data_pattern=${valid_data_pattern} \
--train_dir=$BUCKET_NAME/${TRAIN_DIR} --google_cloud=True \
--model=${MODEL} --bert_hidden_layer=${bert_hidden} --bert_attention_heads=${bert_head} --bert_config_file=$BUCKET_NAME/bert_config.json --segment_labels=True \
--pooling_strategy=${pool} --sample_random_frames=False \
--frame_features --feature_names="rgb,audio" --feature_sizes='1024,128' \
--batch_size=128 --base_learning_rate=0.0001 --moe_l2=1e-6 --iterations=5 --learning_rate_decay=0.9 \
--export_model_steps=200 --num_epochs=10000000 --max_steps=220000
```

2.b Evaluation when Fine-tuning
```sh
ckpt=200000
python src/eval.py --eval_data_pattern='gs://youtube8m-ml-us-east1/3/frame/validate/validate*.tfrecord' --train_dir=$BUCKET_NAME/${JOB_TO_EVAL} --google_cloud=True \
--model=${MODEL} --bert_hidden_layer=${bert_hidden} --bert_attention_heads=${bert_head} --bert_config_file=$BUCKET_NAME/bert_config.json --segment_labels \
--pooling_strategy=${pool} --sample_random_frames=False \
--norun_once --top_k=1000 --batch_size=512 --moe_l2=1e-6 --iterations=5 --checkpoint_file=${ckpt}
```

3 Evaluation for Inference
```sh
ckpt=YOUR_BEST_CKPT
python src/eval.py --eval_data_pattern='gs://youtube8m-ml-us-east1/3/frame/validate/validate*.tfrecord' --train_dir=$BUCKET_NAME/${JOB_TO_EVAL} --google_cloud=True \
--model=${MODEL} --bert_hidden_layer=${bert_hidden} --bert_attention_heads=${bert_head} --bert_config_file=$BUCKET_NAME/bert_config.json --segment_labels \
--pooling_strategy=${pool} --sample_random_frames=False \
--run_once=True --top_k=1000 --batch_size=512 --moe_l2=1e-6 --iterations=5 --checkpoint_file=${ckpt}
```
4.a Inference for OOF
```sh
# Since we use TTA, it is required to inference couple times with different shifts. For example, if model is TTA[-2,1], you need to inference 4 times, one for --shift=-2, one for --shift=-1, one for --shift=0, one for --shift=1 
 
python src/inference.py --input_data_pattern=${valid_data_pattern} \
--google_cloud=True \
--segment_labels \
--segment_label_ids_file=$BUCKET_NAME/segment_label_ids.csv \
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 --shift=-2 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_oof_shift-2.csv

python src/inference.py --input_data_pattern=${valid_data_pattern} \
--google_cloud=True \
--segment_labels \
--segment_label_ids_file=$BUCKET_NAME/segment_label_ids.csv \
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 --shift=-1 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_oof_shift-1.csv

python src/inference.py --input_data_pattern=${valid_data_pattern} \
--google_cloud=True \
--segment_labels \
--segment_label_ids_file=$BUCKET_NAME/segment_label_ids.csv \
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 --shift=1 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_oof_shift1.csv

python src/inference.py --input_data_pattern=${valid_data_pattern} \
--google_cloud=True \
--segment_labels \
--segment_label_ids_file=$BUCKET_NAME/segment_label_ids.csv \
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_oof.csv
```

4.b Inference for Test
```sh
python src/inference.py --input_data_pattern=${test_data_pattern} \
--google_cloud=True \
--segment_labels \
--segment_label_ids_file=$BUCKET_NAME/segment_label_ids.csv \
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 --shift=-2 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_shift-2.csv

python src/inference.py --input_data_pattern=${test_data_pattern} \
--google_cloud=True \
--segment_labels \
--segment_label_ids_file=$BUCKET_NAME/segment_label_ids.csv \
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 --shift=-1 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_shift-1.csv

python src/inference.py --input_data_pattern=${test_data_pattern} \
--google_cloud=True \
--segment_labels \
--segment_label_ids_file=$BUCKET_NAME/segment_label_ids.csv \
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 --shift=1 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune_shift1.csv

python src/inference.py --input_data_pattern=${test_data_pattern} \
--google_cloud=True \
--segment_labels \
--segment_label_ids_file=$BUCKET_NAME/segment_label_ids.csv \
--top_k 1000 --num_readers 10 --batch_size=512 --segment_max_pred=100000 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_finetune.csv
```