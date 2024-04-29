# EMPIAR-10045

python preprocess.py \
--pre_configs '/mnt/data1/ET/DeepETPicker_test/SampleDatasets/EMPIAR_10045/configs/EMPIAR_10045_preprocess.py'

python generate_train_config.py \
--pre_configs '/mnt/data1/ET/DeepETPicker_test/SampleDatasets/EMPIAR_10045/configs/EMPIAR_10045_preprocess.py' \
--dset_name '10045_train_tmp' \
--cfg_save_path '/mnt/data1/ET/DeepETPicker_test/SampleDatasets/EMPIAR_10045/configs' \
--train_set_ids '0' \
--val_set_ids '0' \
--batch_size 4 \
--block_size 72 \
--pad_size 12 \
--learning_rate 0.001 \
--max_epoch 60 \
--threshold 0.5 \
--gpu_id 0


python train_bash.py \
--train_configs '/mnt/data1/ET/DeepETPicker_test/SampleDatasets/EMPIAR_10045/configs/10045_train_tmp.py' \


python test_bash.py \
--train_configs '/mnt/data1/ET/DeepETPicker_test/SampleDatasets/EMPIAR_10045/configs/EMPIAR_10045_train.py' \
--checkpoints '/mnt/data1/ET/DeepETPicker_test/SampleDatasets/EMPIAR_10045/runs/EMPIAR_10045_train/EMPIAR_10045_train_ResUNet_BlockSize72_DiceLoss_MaxEpoch60_bs8_lr0.001_IP1_bg1_coord1_Softmax0_bn__TNNone/version_2/checkpoints/epoch=41-step=797.ckpt'



# SHREC2021
python preprocess.py \
--pre_configs '/mnt/data1/ET/DeepETPicker_test/SampleDatasets/SHREC_2021/configs/SHREC_2021_preprocess.py'

python generate_train_config.py \
--pre_configs '/mnt/data1/ET/DeepETPicker_test/SampleDatasets/SHREC_2021/configs/SHREC_2021_preprocess.py' \
--dset_name 'shrec2021_train_tmp' \
--cfg_save_path '/mnt/data1/ET/DeepETPicker_test/SampleDatasets/SHREC_2021/configs' \
--train_set_ids '0-1' \
--val_set_ids '2' \
--batch_size 4 \
--block_size 72 \
--pad_size 12 \
--learning_rate 0.001 \
--max_epoch 60 \
--threshold 0.5 \
--gpu_id 0


python train_bash.py \
--train_configs '/mnt/data1/ET/DeepETPicker_test/SampleDatasets/SHREC_2021/configs/shrec2021_train_tmp.py' \


python test_bash.py \
--train_configs '/mnt/data1/ET/DeepETPicker_test/SampleDatasets/SHREC_2021/configs/shrec2021_train_tmp.py' \
--checkpoints '/mnt/data1/ET/DeepETPicker_test/SampleDatasets/SHREC_2021/runs/shrec2021_train_tmp/shrec2021_train_tmp_ResUNet_BlockSize72_DiceLoss_MaxEpoch60_bs4_lr0.001_IP1_bg1_coord1_Softmax1_bn__TNNone/version_4/checkpoints/epoch=1-step=1551.ckpt'
