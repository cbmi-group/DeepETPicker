import sys
import os
import json
from os.path import dirname, abspath
import importlib

DeepETPickerHome = dirname(abspath(__file__))
DeepETPickerHome = os.path.split(DeepETPickerHome)[0]
sys.path.append(DeepETPickerHome)
sys.path.append(os.path.split(DeepETPickerHome)[0])
option = importlib.import_module(f".options.option", package=os.path.split(DeepETPickerHome)[1])

if __name__ == "__main__":
    options = option.BaseOptions()
    args = options.gather_options()

    with open(args.pre_configs, 'r') as f:
        train_config = json.loads(''.join(f.readlines()).lstrip('pre_config='))
    train_config['dset_name'] = args.dset_name
    train_config['coord_path'] = train_config['base_path'] + '/coords'
    train_config['tomo_path'] = train_config['base_path'] + '/data_std'
    train_config['label_name'] = train_config['label_type'] + f"{train_config['label_diameter']}"
    train_config['label_path'] = train_config['base_path'] + f"/{train_config['label_name']}"
    train_config['ocp_name'] = 'data_ocp'
    train_config['ocp_path'] = train_config['base_path'] + '/data_ocp'
    train_config['model_name'] = 'ResUNet'
    train_config['train_set_ids'] = args.train_set_ids
    train_config['val_set_ids'] = args.val_set_ids
    train_config['batch_size'] = args.batch_size
    train_config['patch_size'] = args.block_size
    train_config['padding_size'] = args.pad_size[0]
    train_config['lr'] = args.learning_rate
    train_config['max_epochs'] = args.max_epoch
    train_config['seg_thresh'] = args.threshold
    train_config['gpu_ids'] = ''.join([str(i) for i in args.gpu_id])


    with open(f"{args.cfg_save_path}/{args.dset_name}.py", 'w') as f:
        f.write("train_configs=")
        json.dump(train_config, f, separators=(',\n' + ' ' * len('train_configs={'), ': '))




