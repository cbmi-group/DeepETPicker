import json
import sys
import os
import importlib
from os.path import dirname, abspath
import numpy as np
DeepETPickerHome = dirname(abspath(__file__))
DeepETPickerHome = os.path.split(DeepETPickerHome)[0]
sys.path.append(DeepETPickerHome)
sys.path.append(os.path.split(DeepETPickerHome)[0])
test = importlib.import_module(".test", package=os.path.split(DeepETPickerHome)[1])
option = importlib.import_module(f".options.option", package=os.path.split(DeepETPickerHome)[1])

if __name__ == '__main__':
    options = option.BaseOptions()
    args = options.gather_options()

    # cofig
    with open(args.train_configs, 'r') as f:
        cfg = json.loads(''.join(f.readlines()).lstrip('train_configs='))

    # parameters
    args.use_bg = True
    args.use_IP = True
    args.use_coord = True
    args.test_use_pad = True
    args.use_seg = True
    args.meanPool_NMS = True
    args.f_maps = [24, 48, 72, 108]
    args.num_classes = cfg['num_cls']
    train_cls_num = cfg['num_cls']
    if args.num_classes == 1:
        args.use_sigmoid = True
        args.use_softmax = False
    else:
        train_cls_num = train_cls_num + 1
        args.use_sigmoid = False
        args.use_softmax = True
    args.batch_size = cfg['batch_size']
    args.block_size = cfg['patch_size']
    args.val_batch_size = args.batch_size
    args.val_block_size = args.block_size
    args.pad_size = [cfg['padding_size']]
    args.learning_rate = cfg['lr']
    args.max_epoch = cfg['max_epochs']
    args.threshold = cfg['seg_thresh']
    args.gpu_id = [int(i) for i in cfg['gpu_ids'].split(',')]
    args.test_mode = 'test_only'
    args.out_name = 'PredictedLabels'
    args.de_duplication = True
    args.de_dup_fmt = 'fmt4'
    args.mini_dist = sorted([int(i) // 2 + 1 for i in cfg['ocp_diameter'].split(',')])[0]
    args.data_split = [0, 1, 0, 1, 0, 1]
    args.configs = args.train_configs
    args.num_classes = train_cls_num

    # test_idxs
    dset_list = np.array(
        [i[:-(len(i.split('.')[-1]) + 1)] for i in os.listdir(cfg['tomo_path']) if cfg['tomo_format'] in i])
    dset_num = dset_list.shape[0]
    num_name = np.concatenate([np.arange(dset_num).reshape(-1, 1), dset_list.reshape(-1, 1)], axis=1)
    np.savetxt(os.path.join(cfg['tomo_path'], 'num_name.csv'),
               num_name,
               delimiter='\t',
               fmt='%s',
               newline='\n')

    # tomo_list = [i for i in os.listdir(cfg[f"{cfg['base_path']}/data_std"]) if cfg['tomo_format'] in i]
    tomo_list = np.loadtxt(f"{cfg['base_path']}/data_std/num_name.csv",
                           delimiter='\t',
                           dtype=str)
    args.test_idxs = np.arange(len(tomo_list))

    for k, v in sorted(vars(args).items()):
        print(k, '=', v)

    # Testing
    test.test_func(args, stdout=None)