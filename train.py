import numpy as np
import torch
import torch.nn as nn
import os
import sys
from dataset.dataloader_DynamicLoad import Dataset_ClsBased
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils.loss import DiceLoss
from utils.metrics import seg_metrics
from utils.colors import COLORS
from model.model_loader import get_model
from utils.misc import combine, cal_metrics_NMS_OneCls, get_centroids, cal_metrics_MultiCls, combine_torch
from sklearn.metrics import precision_recall_fscore_support
import time
import json

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


class UNetExperiment(pl.LightningModule):
    def __init__(self, args):
        if args.f_maps is None:
            args.f_maps = [32, 64, 128, 256]
        print(args.pad_size)

        if len(args.configs) > 0:
            with open(args.configs, 'r') as f:
                self.cfg = json.loads(''.join(f.readlines()).lstrip('train_configs='))
        else:
            self.cfg = {}
        if len(args.train_configs) > 0:
            with open(args.train_configs, 'r') as f:
                self.train_cfg = json.loads(''.join(f.readlines()).lstrip('train_configs='))
        else:
            self.train_cfg = self.cfg

        if len(args.val_configs) > 0:
            with open(args.val_configs, 'r') as f:
                self.val_config = json.loads(''.join(f.readlines()).lstrip('train_configs='))
        else:
            self.val_cfg = self.cfg

        super(UNetExperiment, self).__init__()
        self.save_hyperparameters()
        self.model = get_model(args)
        print(self.model)

        if args.loss_func_seg == 'Dice':
            self.loss_function_seg = DiceLoss(args=args)

        if 'gaussian' in self.val_cfg["label_type"]:
            self.thresholds = np.linspace(0.15, 0.45, 7)
        elif 'sphere' in self.val_cfg["label_type"]:
            self.thresholds = np.linspace(0.2, 0.80, 13)
        self.partical_volume = 4 / 3 * np.pi * (self.val_cfg["label_diameter"] / 2) ** 3
        self.args = args

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        args = self.args
        img, label, index = train_batch
        img = img.to(torch.float32)
        seg_output = self.forward(img)
        if args.use_mask:
            mask = label.clone().detach()
            mask[mask > 0] = 1
            label[label < 255] = 0
            label[label > 0] = 1

            # update label and mask according to label-threshold
            label[seg_output > args.seg_tau] = 1
            mask[seg_output > args.seg_tau] = 1
            mask[seg_output < (1 - args.seg_tau)] = 1

            seg_output = seg_output * mask
        loss_seg = self.loss_function_seg(seg_output, label)
        self.log('train_loss', loss_seg, on_step=False, on_epoch=True)
        return loss_seg

    def validation_step(self, val_batch, batch_idx):
        args = self.args
        with torch.no_grad():
            img, label, index = val_batch
            index = torch.cat([i.view(1, -1) for i in index], dim=0).permute(1, 0)
            img = img.to(torch.float32)
            self.seg_output = self.forward(img)

            if (batch_idx >= self.len_block // args.batch_size and args.test_mode == "test_val") or \
                    args.test_mode == "test" or args.test_mode == "val" or args.test_mode == "val_v1":
                loss_seg = self.loss_function_seg(self.seg_output, label)

                precision, recall, f1_score, iou = seg_metrics(self.seg_output, label, threshold=args.threshold)

                self.log('val_loss', loss_seg, on_step=False, on_epoch=True)
                self.log('val_precision', precision, on_step=False, on_epoch=True)
                self.log('val_recall', recall, on_step=False, on_epoch=True)
                self.log('val_f1', f1_score, on_step=False, on_epoch=True)
                self.log('val_iou', iou, on_step=False, on_epoch=True)

                # return loss_seg
                tensorboard = self.logger.experiment

                if (batch_idx == (self.len_block // args.batch_size + 1) and args.test_mode == 'test_val') or \
                        (batch_idx == 0 and args.test_mode == 'test') or \
                        (batch_idx == 0 and args.test_mode == 'val') or \
                        (
                                batch_idx == 0 and args.test_mode == 'val_v1'):  # and True == False  and self.current_epoch % 1 == 0
                    img /= img.abs().max()  # [-1,1]
                    img = img * 0.5 + 0.5  # [0, 1]
                    img_ = img[0, :, 0:(args.block_size - 1):5, :, :].permute(1, 0, 2, 3).repeat(
                        (1, 3, 1, 1))  # sample0: [5, 3, y, x]

                    label_ = label[0, :, 0:(args.block_size - 1):5, :, :]  # sample0 [15, y, x]
                    temp = torch.zeros(
                        (len(np.arange(0, args.block_size - 1, 5)), args.block_size, args.block_size, 3)).float()
                    # print(label.shape, temp.shape)
                    for idx in np.arange(label_.shape[0]):
                        temp[label_[idx] > 0.5] = torch.tensor(
                            COLORS[(idx + 1) if (args.num_classes == 1
                                                 or args.use_paf or
                                                 label_.shape[0] == 1) else idx]).float()
                    label__ = temp.permute(0, 3, 1, 2).contiguous().cuda()  # [15, 3, y, x]

                    seg_output_ = self.seg_output[0, :, 0:(args.block_size - 1):5, :, :]  # sample0 [15, y, x]
                    seg_threshes = [0.5, 0.3, 0.2, 0.15, 0.1, 0.05]
                    seg_preds = []
                    for thresh in seg_threshes:
                        temp = torch.zeros(
                            (len(np.arange(0, args.block_size - 1, 5)), args.block_size, args.block_size, 3)).float()
                        for idx in np.arange(seg_output_.shape[0]):
                            temp[seg_output_[idx] > thresh] = torch.tensor(
                                COLORS[(idx + 1) if (args.num_classes == 1
                                                     or args.use_paf or
                                                     seg_output_.shape[0] == 1) else idx]).float()
                        seg_preds.append(temp.permute(0, 3, 1, 2).contiguous().cuda())  # [15, 3, y, x]

                    seg_preds = torch.cat(seg_preds, dim=0)

                    img_label_seg = torch.cat([img_, label__, seg_preds], dim=0)
                    img_label_seg = make_grid(img_label_seg, (args.block_size - 1) // 5 + 1, padding=2, pad_value=120)

                    tensorboard.add_image('img_label_seg', img_label_seg, self.current_epoch, dataformats="CHW")

            if args.num_classes > 1:
                return self._nms_v2(self.seg_output[:, 1:], kernel=args.meanPool_kernel, mp_num=6, positions=index)
            else:
                return self._nms_v2(self.seg_output[:, :], kernel=args.meanPool_kernel, mp_num=6, positions=index)

    def validation_step_end(self, outputs):
        args = self.args
        if 'test' in args.test_mode:
            return outputs

    def validation_epoch_end(self, epoch_output):
        args = self.args
        with torch.no_grad():
            if 'test' in args.test_mode:
                if args.meanPool_NMS:
                    if args.num_classes == 1:
                        # coords_out: [N, 5]
                        coords_out = torch.cat(epoch_output, dim=0).detach().cpu().numpy()
                        if coords_out.shape[0] > 50000:
                            loc_p, loc_r, loc_f1, avg_dist = 1e-10, 1e-10, 1e-10, 100
                        else:
                            loc_p, loc_r, loc_f1, avg_dist = \
                                cal_metrics_NMS_OneCls(coords_out,
                                                       self.gt_coords,
                                                       self.occupancy_map,
                                                       self.cfg,
                                                       )
                        print("*" * 100)
                        print(f"Precision:{loc_p}")
                        print(f"Recall:{loc_r}")
                        print(f"F1-score:{loc_f1}")
                        print(f"Avg-dist:{avg_dist}")
                        print("*" * 100)
                        self.log('cls_precision', loc_p, on_step=False, on_epoch=True)
                        self.log('cls_recall', loc_r, on_step=False, on_epoch=True)
                        self.log('cls_f1', loc_f1, on_step=False, on_epoch=True)
                        self.log('cls_dist', avg_dist, on_step=False, on_epoch=True)
                        pr = (loc_p * (loc_r ** args.prf1_alpha)) / (loc_p + (loc_r ** args.prf1_alpha) + 1e-10)
                        self.log(f'cls_pr_alpha{args.prf1_alpha:.1f}', pr, on_step=False, on_epoch=True)
                        time.sleep(0.5)
                    else:
                        coords_out = torch.cat(epoch_output, dim=0).detach().cpu().numpy()
                        loc_p, loc_r, loc_f1, loc_miss, avg_dist, gt_classes, pred_classes, self.num2pdb, cls_f1 = \
                            cal_metrics_MultiCls(coords_out, self.gt_coords, self.occupancy_map, self.cfg, args,
                                                 args.pad_size, self.dir_name, self.partical_volume)
                        self.log('cls_f1', cls_f1, on_step=False, on_epoch=True)

    def train_dataloader(self):
        args = self.args
        train_dataset = Dataset_ClsBased(mode=args.train_mode,
                                         block_size=args.block_size,
                                         num_class=args.num_classes,
                                         random_num=args.random_num,
                                         use_bg=args.use_bg,
                                         data_split=args.data_split,
                                         use_paf=args.use_paf,
                                         cfg=self.train_cfg,
                                         args=args)
        return DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          num_workers=8 if args.batch_size >= 32 else 4,
                          shuffle=True,
                          pin_memory=False)

    def val_dataloader(self):
        args = self.args
        val_dataset = Dataset_ClsBased(mode=args.test_mode,
                                       block_size=args.val_block_size,
                                       num_class=args.num_classes,
                                       random_num=args.random_num,
                                       use_bg=args.use_bg,
                                       data_split=args.data_split,
                                       test_use_pad=args.test_use_pad,
                                       pad_size=args.pad_size,
                                       use_paf=args.use_paf,
                                       cfg=self.val_cfg,
                                       args=args)

        self.len_block = val_dataset.test_len
        if 'test' in args.test_mode:
            self.data_shape = val_dataset.data_shape
            self.occupancy_map = val_dataset.occupancy_map
            self.gt_coords = val_dataset.gt_coords
            self.dir_name = val_dataset.dir_name

        val_dataloader1 = DataLoader(val_dataset,
                                     batch_size=args.val_batch_size,
                                     num_workers=8 if args.batch_size >= 32 else 4,
                                     shuffle=False,
                                     pin_memory=False)
        return val_dataloader1

    def _nms_v2(self, pred, kernel=3, mp_num=5, positions=None):
        args = self.args
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
        if coords.shape[0] > 2000:
            return torch.zeros([1, 5]).cuda()
        coords = coords[coords[:, 2] >= args.pad_size]
        coords = coords[coords[:, 2] < args.block_size - args.pad_size]
        coords = coords[coords[:, 3] >= args.pad_size]
        coords = coords[coords[:, 3] < args.block_size - args.pad_size]
        coords = coords[coords[:, 4] >= args.pad_size]
        coords = coords[coords[:, 4] < args.block_size - args.pad_size]

        try:
            h_val = torch.cat(
                [hmax[item[0], item[1], item[2], item[3]:item[3] + 1, item[4]:item[4] + 1] for item in
                 coords], dim=0)
            leftTop_coords = positions[coords[:, 0]] - (args.block_size // 2) - args.pad_size
            coords[:, 2:5] = coords[:, 2:5] + leftTop_coords

            pred_final = torch.cat(
                [coords[:, 1:2] + 1, coords[:, 4:5], coords[:, 3:4], coords[:, 2:3], h_val],
                dim=1)

            return pred_final
        except:
            return torch.zeros([0, 5]).cuda()

    def configure_optimizers(self):
        args = self.args
        if args.optim == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=args.learning_rate,
                                        momentum=0.9, weight_decay=0.001
                                        )
        elif args.optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=args.learning_rate,
                                         betas=(0.9, 0.99)
                                         )
        elif args.optim == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=args.learning_rate,
                                          betas=(0.9, 0.99),
                                          weight_decay=args.weight_decay
                                          )

        if args.scheduler == 'OneCycleLR':
            sched = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=args.learning_rate,
                                                        total_steps=args.max_epoch,
                                                        pct_start=0.1,
                                                        anneal_strategy='cos',
                                                        div_factor=30,
                                                        final_div_factor=100)
            lr_dict = {
                "scheduler": sched,
                "interval": "epoch",
                "frequency": 1
            }

        if args.scheduler is None:
            return [optimizer]
        else:
            return [optimizer], [lr_dict]


def train_func(args, stdout=None):
    if stdout is not None:
        save_stdout = sys.stdout
        save_stderr = sys.stderr
        sys.stdout = stdout
        sys.stderr = stdout

    args.pad_size = args.pad_size[0]
    if 'test' in args.test_mode:
        checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                              monitor=f'cls_pr_alpha{args.prf1_alpha:.1f}' if args.num_classes == 1 else 'cls_f1',
                                              mode='max')
    else:
        checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                              monitor='val_loss',
                                              mode='min')

    model = UNetExperiment(args)
    logger_name = "{}_{}_BlockSize{}_{}Loss_MaxEpoch{}_bs{}_lr{}_IP{}_bg{}_coord{}_Softmax{}_{}_{}_TN{}".format(
        model.train_cfg["dset_name"], args.network, args.block_size, args.loss_func_seg, args.max_epoch,
        args.batch_size,
        args.learning_rate,
        int(args.use_IP), int(args.use_bg), int(args.use_coord),
        int(args.use_softmax), args.norm, args.others, args.sel_train_num)

    os.makedirs(f"{model.train_cfg['base_path']}/runs/{model.train_cfg['dset_name']}", exist_ok=True)
    tb_logger = loggers.TensorBoardLogger(f"{model.train_cfg['base_path']}/runs/{model.train_cfg['dset_name']}",
                                          name=logger_name)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    runner = Trainer(min_epochs=min(50, args.max_epoch),
                     max_epochs=args.max_epoch,
                     logger=tb_logger,
                     gpus=args.gpu_id,
                     checkpoint_callback=checkpoint_callback,
                     callbacks=[lr_monitor],
                     accelerator='dp',
                     precision=32,
                     profiler=True,
                     sync_batchnorm=True,
                     resume_from_checkpoint=args.checkpoints)


    try:
        runner.fit(model)
        print('*' * 100)
        print('Training Finished')
        print(f'Training pid:{os.getpid()}')
        print('*' * 100)
        torch.cuda.empty_cache()
        if stdout is not None:
            sys.stderr = save_stderr
            sys.stdout = save_stdout
        return os.getpid()
    except:
        torch.cuda.empty_cache()
        if stdout is not None:
            stdout.flush()
            stdout.write('Training Exception!')
            sys.stderr = save_stderr
            sys.stdout = save_stdout
        return os.getpid()