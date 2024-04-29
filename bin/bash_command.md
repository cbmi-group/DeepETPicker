
In addition to the GUI version of DeepETPicker, we also provide a non-GUI version of DeepETPicker for people who understand python and deep-learning. It consists of four processes, including `preprocessing`, `train config generation`, `training` and `testing`.

Note: If you are not familiar with python and deep learning, the GUI version is recommended.

##  Preprocessing

```bash
python preprocess.py \
--pre_configs 'PATH_TO_Preprocess_CONFIG'
```

where `pre_configs` corresponds to the configuration file of preprocessing. We have provided a sample for EMPIAR-10045 dataset in `cofigs/EMPIAR_10045_preprocess.py`. The items of configuration file is the same as the generated file of `preprocessing` panel of GUI version. More details can be found in section `Preprocessing` of https://github.com/cbmi-group/DeepETPicker/tree/main.

Note: `pre_configs` could also directly load the configuration file generated by `preprocessing` panel of GUI DeepETPicker.

## Generate configuration file for training and testing

```bash
python generate_train_config.py \
--pre_configs 'PATH_TO_Preprocess_CONFIG' \
--dset_name 'Train_Config_Name' \
--cfg_save_path 'Save_Path_for_Config_Name' \
--train_set_ids '0' \
--val_set_ids '0' \
--batch_size 8 \
--block_size 72 \
--pad_size 12 \
--learning_rate 0.001 \
--max_epoch 60 \
--threshold 0.5 \
--gpu_id 0
```

where `dset_name` and `cfg_save_path` are the name and save path of training configuration file, respectively. Other parameters are the same as the input of `Training` panel of GUI version. More details can be found in section `Training of DeepETPicker` of https://github.com/cbmi-group/DeepETPicker/tree/main.

## Training

```bash
python train_bash.py \
--train_configs 'Train_Config_Name'
```

where `train_configs` is the train configuration file generated by `generate_train_config.py`. 

Note: `train_configs` could also directly load the configuration file generated by `Training` panel of GUI DeepETPicker.

## Testing

```bash
python test_bash.py \
--train_configs 'Train_Config_Name' \
--checkpoints 'Checkpoint_Path'
```




