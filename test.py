import numpy as np
import os
import time
import torch
from torch import nn
import mrcfile
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import sys
# from dataset.dataloader import Dataset_ClsBased
import pandas as pd
import importlib
from glob import glob
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from options.option import BaseOptions
from model.model_loader import get_model
from utils.misc import combine, get_centroids, de_dup, cal_metrics_NMS_OneCls
import json
from dataset.dataloader_DynamicLoad import Dataset_ClsBased


def test_func(args, stdout=None):
    if stdout is not None:
        save_stdout = sys.stdout
        save_stderr = sys.stderr
        sys.stdout = stdout
        sys.stderr = stdout
    for test_idx in args.test_idxs:
        model_name = args.checkpoints.split('/')[-4] + '_' + args.checkpoints.split('/')[-1].split('-')[0]
        # load config parameters
        if len(args.configs) > 0:
            with open(args.configs, 'r') as f:
                cfg = json.loads(f.readline().lstrip('train_configs='))

        start_time = time.time()
        args.data_split[-2] = test_idx
        args.data_split[-1] = test_idx + 1

        num_name = pd.read_csv(os.path.join(cfg["tomo_path"], 'num_name.csv'), sep='\t', header=None)
        dir_list = num_name.iloc[:, 1]
        dir_name = dir_list[args.data_split[-2]]
        print(dir_name)

        tomo_file = glob(cfg["tomo_path"] + "/*%s" % cfg["tomo_format"])[0]
        data_file = mrcfile.open(tomo_file, permissive=True)
        data_shape = data_file.data.shape
        print(data_shape)
        dataset = cfg["dset_name"]

        if args.use_seg:
            for pad_size in args.pad_size:

                class UNetTest(pl.LightningModule):
                    def __init__(self):
                        super(UNetTest, self).__init__()
                        self.model = get_model(args)
                        self.partical_volume = 4 / 3 * np.pi * (cfg["label_diameter"] / 2) ** 3
                        self.num_classes = args.num_classes

                    def forward(self, x):
                        return self.model(x)

                    def test_step(self, test_batch, batch_idx):
                        with torch.no_grad():
                            img, label, index = test_batch
                            index = torch.cat([i.view(1, -1) for i in index], dim=0).permute(1, 0)
                            if args.use_paf:
                                seg_output, paf_output, logsigma1 = self.forward(img)
                            else:
                                seg_output = self.forward(img)

                            if args.test_use_pad:
                                mp_num = int(sorted([int(i) for i in cfg["ocp_diameter"].split(',')])[-1] / (args.meanPool_kernel - 1) + 1)
                                if args.num_classes > 1:
                                    return self._nms_v2(seg_output[:, 1:], kernel=args.meanPool_kernel,
                                                        mp_num=mp_num, positions=index)
                                else:
                                    return self._nms_v2(seg_output[:, :], kernel=args.meanPool_kernel,
                                                        mp_num=mp_num, positions=index)

                    def test_step_end(self, outputs):
                        return outputs

                    def test_epoch_end(self, epoch_output):
                        with torch.no_grad():
                            if args.meanPool_NMS:
                                coords_out = torch.cat(epoch_output, dim=0).detach().cpu().numpy()
                                print('coords_out:', coords_out.shape)
                                if args.de_duplication:
                                    centroids = de_dup(coords_out, args)
                                out_dir = '/'.join(args.checkpoints.split('/')[:-2]) + f'/{args.out_name}'
                                os.makedirs(os.path.join(out_dir, 'Coords_withArea'), exist_ok=True)
                                np.savetxt(os.path.join(out_dir, 'Coords_withArea', dir_name + '.coords'),
                                           centroids.astype(float),
                                           fmt='%s',
                                           delimiter='\t')

                                coords = centroids[:, 0:4]
                                os.makedirs(os.path.join(out_dir, 'Coords_All'), exist_ok=True)
                                np.savetxt(os.path.join(out_dir, 'Coords_All', dir_name + '.coords'),
                                           coords.astype(int),
                                           fmt='%s',
                                           delimiter='\t')


                    def test_dataloader(self):
                        if args.test_mode == 'test':
                            test_dataset = Dataset_ClsBased(mode='test',
                                                            block_size=args.block_size,
                                                            num_class=args.num_classes,
                                                            random_num=args.random_num,
                                                            use_bg=args.use_bg,
                                                            data_split=args.data_split,
                                                            test_use_pad=args.test_use_pad,
                                                            pad_size=pad_size,
                                                            cfg=cfg,
                                                            args=args)
                            test_dataloader = DataLoader(test_dataset,
                                                         shuffle=False,
                                                         batch_size=args.batch_size,
                                                         num_workers=8 if args.batch_size >= 32 else 4,
                                                         pin_memory=False)

                            self.len_block = test_dataset.test_len
                            self.data_shape = test_dataset.data_shape
                            self.occupancy_map = test_dataset.occupancy_map
                            self.gt_coords = test_dataset.gt_coords
                            self.dir_name = test_dataset.dir_name
                            return test_dataloader
                        elif args.test_mode == 'test_only':
                            test_dataset = Dataset_ClsBased(mode='test_only',
                                                            block_size=args.block_size,
                                                            num_class=args.num_classes,
                                                            random_num=args.random_num,
                                                            use_bg=args.use_bg,
                                                            data_split=args.data_split,
                                                            test_use_pad=args.test_use_pad,
                                                            pad_size=pad_size,
                                                            cfg=cfg,
                                                            args=args)
                            if args.batch_size <= 32:
                                num_work = 4
                            elif args.batch_size <= 64:
                                num_work = 8
                            elif args.batch_size <= 128:
                                num_work = 8
                            else:
                                num_work = 16
                            test_dataloader = DataLoader(test_dataset,
                                                         shuffle=False,
                                                         batch_size=args.batch_size,
                                                         num_workers=num_work,
                                                         pin_memory=False)
                            self.len_block = test_dataset.test_len
                            self.data_shape = test_dataset.data_shape
                            self.dir_name = test_dataset.dir_name
                            return test_dataloader

                    def _nms_v2(self, pred, kernel=3, mp_num=5, positions=None):
                        pred = torch.where(pred > 0.5, 1, 0)
                        meanPool = nn.AvgPool3d(kernel, 1, kernel // 2).cuda()
                        maxPool = nn.MaxPool3d(kernel, 1, kernel // 2).cuda()
                        hmax = pred.clone().float()
                        for _ in range(mp_num):
                            hmax = meanPool(hmax)
                        pred = hmax.clone()
                        hmax = maxPool(hmax)
                        keep = ((hmax == pred).float()) * ((pred > 0.1).float())
                        coords = keep.nonzero()  # [N, 5]
                        coords = coords[coords[:, 2] >= args.pad_size[0]]
                        coords = coords[coords[:, 2] <= args.block_size - args.pad_size[0]]
                        coords = coords[coords[:, 3] >= args.pad_size[0]]
                        coords = coords[coords[:, 3] <= args.block_size - args.pad_size[0]]
                        coords = coords[coords[:, 4] >= args.pad_size[0]]
                        coords = coords[coords[:, 4] <= args.block_size - args.pad_size[0]]

                        try:
                            h_val = torch.cat(
                                [hmax[item[0], item[1], item[2], item[3]:item[3] + 1, item[4]:item[4] + 1] for item in
                                 coords], dim=0)
                            leftTop_coords = positions[coords[:, 0]] - (args.block_size // 2) - args.pad_size[0]
                            coords[:, 2:5] = coords[:, 2:5] + leftTop_coords

                            pred_final = torch.cat(
                                [coords[:, 1:2] + 1, coords[:, 4:5], coords[:, 3:4], coords[:, 2:3], h_val],
                                dim=1)

                            return pred_final
                        except:
                            # print('haha')
                            return torch.zeros([0, 5]).cuda()

                # load trained checkpoints to model
                model = UNetTest.load_from_checkpoint(args.checkpoints)

                # model = UNetTest().model
                model.eval()
                runner = Trainer(gpus=args.gpu_id, #
                                 accelerator='dp'
                                 )
                os.makedirs(f'result/{dataset}/{model_name}/', exist_ok=True)

                runner.test(model=model)

        end_time = time.time()
        used_time = end_time - start_time
        save_path = '/'.join(args.checkpoints.split('/')[:-2]) + f'/{args.out_name}'
        os.makedirs(save_path, exist_ok=True)
        pad_size = args.pad_size[0]

    print('*' * 100)
    print('Testing Finished!')
    print('*' * 100)
    sys.stdout = save_stdout
    sys.stderr = save_stderr

