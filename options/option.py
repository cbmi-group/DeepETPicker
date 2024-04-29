import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Parameters for 3D particle picking')

        # dataloader parameters
        self.parser.add_argument('--block_size', help='block size', type=int, default=72)
        self.parser.add_argument('--val_block_size', help='block size', type=int, default=0)
        self.parser.add_argument('--random_num', help='random number', type=int, default=0)
        self.parser.add_argument('--num_classes', help='number of classes', type=int, default=13)
        self.parser.add_argument('--use_bg', type=str2bool, help='whether use batch generator', default=False)
        self.parser.add_argument('--test_use_pad', type=str2bool, help='whether use coord conv', default=False)
        self.parser.add_argument('--pad_size', nargs='+', type=int, default=[12])
        self.parser.add_argument('--data_split', nargs='+', type=int, default=[0, 1, 0, 1, 0, 1])
        self.parser.add_argument('--configs', type=str, default='')
        self.parser.add_argument('--pre_configs', type=str, default='')
        self.parser.add_argument('--train_configs', type=str, default='')
        self.parser.add_argument('--ck_mode', type=str, default='')
        self.parser.add_argument('--val_configs', type=str, default='')
        self.parser.add_argument('--loader_type', type=str, default='dataloader_DynamicLoad', help="whether use DynamicLoad",
                                 # choices=["dataloader", "dataloader_DynamicLoad",
                                 #          'dataloader_DynamicLoad_CellSeg',
                                 #          "dataloader_DynamicLoad_Semi"]
                                 )
        self.parser.add_argument('--sel_train_num', nargs='+', type=int)
        self.parser.add_argument('--rand_min', type=int, default=0)
        self.parser.add_argument('--rand_max', type=int, default=99)
        self.parser.add_argument('--rand_scale', type=int, default=100)
        self.parser.add_argument('--input_cat', type=str2bool, help='whether use input cat', default=False)
        self.parser.add_argument('--input_cat_items', nargs='+', type=str, default='None')

        # model parameters
        self.parser.add_argument('--network', help='network type', type=str, default='ResUNet',
                                 # choices=['unet', 'UMC', 'ResUnet', 'DoubleUnet', 'MFNet', 'DMFNet', 'DMFNet_down3',
                                 #          'NestUnet', 'VoxResNet', 'HighRes3DNet', 'HRNetv1']
                                 )
        self.parser.add_argument('--in_channels', help='input channels of the network', type=int, default=1)
        self.parser.add_argument('--f_maps', nargs='+', type=int, help="Feature numbers of ResUnet")
        self.parser.add_argument('--use_LAM', type=str2bool, help='whether use LAM', default=False)
        self.parser.add_argument('--use_IP', type=str2bool, help='whether use image pyramid', default=False)
        self.parser.add_argument('--use_DS', type=str2bool, help='whether use deep supervision', default=False)
        self.parser.add_argument('--use_wDS', type=str2bool, help='whether use deep supervision', default=False)
        self.parser.add_argument('--use_Res', type=str2bool, help='whether use residual connectivity', default=False)
        self.parser.add_argument('--use_coord', type=str2bool, help='whether use coord conv', default=False)
        self.parser.add_argument('--use_softmax', type=str2bool, help='whether use softmax', default=False)
        self.parser.add_argument('--use_sigmoid', type=str2bool, help='whether use sigmoid', default=False)
        self.parser.add_argument('--use_tanh', type=str2bool, help='whether use tanh', default=False)
        self.parser.add_argument('--use_softpool', type=str2bool, help='whether use softpool', default=False)
        self.parser.add_argument('--use_aspp', type=str2bool, help='whether use aspp', default=False)
        self.parser.add_argument('--use_se_loss', type=str2bool, help='whether use SE loss', default=False)
        self.parser.add_argument('--use_att', type=str2bool, help='whether use aspp', default=False)
        self.parser.add_argument('--initial_channels', help='initial_channels of NestUnet', type=int, default=16)
        self.parser.add_argument('--mf_groups', help='number of groups', type=int, default=16)
        self.parser.add_argument('--norm', help='type of normalization', type=str, default='bn',
                                 choices=['bn', 'gn', 'in', 'sync_bn'])
        self.parser.add_argument('--act', help='type of activation function', type=str, default='relu',
                                 choices=['relu', 'lrelu', 'elu', 'gelu'])
        self.parser.add_argument('--use_wds', type=str2bool, help='whether use weighted deep supervision',
                                 default=False)
        self.parser.add_argument('--add_dropout_layer', type=str2bool, help='whether use dropout layer in HighResNet',
                                 default=False)
        self.parser.add_argument('--dimensions', type=int, default=3, help='Dimensions of HighResNet')
        self.parser.add_argument('--use_paf', type=str2bool, help='PostFusion_orit: whether use part affinity field',
                                 default=False)
        self.parser.add_argument('--paf_sigmoid', type=str2bool, help='whether use sigmoid for the branch of '
                                                                      'part affinity field', default=False)
        self.parser.add_argument('--pif_sigmoid', type=str2bool, help='whether use sigmoid for the branch of '
                                                                      'part intensity field', default=False)
        self.parser.add_argument('--final_double', type=str2bool, help='whether use sigmoid for the branch of '
                                                                       'part affinity field', default=False)
        self.parser.add_argument('--HRNet_c', type=int, default=12)
        self.parser.add_argument('--n_block', type=int, default=2)
        self.parser.add_argument('--reduce_ratio', type=int, default=1)
        self.parser.add_argument('--n_stages', nargs='+', type=int, default=[1, 1, 1, 1])
        self.parser.add_argument('--use_uncert', type=str2bool, help='whether use uncert for loss weights',
                                 default=False)
        self.parser.add_argument('--Gau_num', type=int, default=2)
        self.parser.add_argument('--use_seg_gau', type=str2bool, help='whether use seg and gau', default=False)
        self.parser.add_argument('--gau_thresh', type=float, default=0.5)
        self.parser.add_argument('--use_lw', type=str2bool, help='whether use lightweight', default=False)
        self.parser.add_argument('--lw_kernel', type=int, default=3)

        # training hyper-parameters
        self.parser.add_argument('--learning_rate', type=float, default=5e-5)
        self.parser.add_argument('--batch_size', help='batch size', type=int, default=32)
        self.parser.add_argument('--val_batch_size', help='batch size', type=int, default=0)
        self.parser.add_argument('--max_epoch', help='number of epochs', type=int, default=100)
        self.parser.add_argument('--loss_func_seg', help='seg loss function type', type=str, default='Dice')
        self.parser.add_argument('--loss_func_dn', help='denoising loss function type', type=str, default='MSE')
        self.parser.add_argument('--loss_func_paf', help='paf loss function type', type=str, default='MSE')
        self.parser.add_argument('--pred2d_3d', type=str2bool, help='whether use LAM', default=False)
        self.parser.add_argument('--threshold', type=float, default=0.5, help="calculate seg_metrics")
        self.parser.add_argument('--others', help='others', type=str, default='')
        self.parser.add_argument('--paf_weight', type=int, default=1, help='Weight for Paf branch')
        self.parser.add_argument('--border_value', type=int, help='border width', default=0)
        self.parser.add_argument('--dset_name', type=str, help="the name of dataset")
        self.parser.add_argument('--train_mode', type=str, default='train', help='train mode')
        self.parser.add_argument('--gpu_id', nargs='+', type=int, default=[0, 1, 2, 3], help='gpu id')
        self.parser.add_argument('--prf1_alpha', type=float, default=3, help="calculate seg_metrics")
        self.parser.add_argument('--running', type=str2bool, help='whether use LAM', default=False)

        # Contrastive Learning hyper-parameters
        self.parser.add_argument('--checkpoints', type=str, help='Checkpoint directory',
                                 default=None)
        self.parser.add_argument('--checkpoints_version', type=str, help='Checkpoint directory',
                                 default=None)
        self.parser.add_argument('--cent_feats', type=str, help='Checkpoint directory',
                                 default=None)
        self.parser.add_argument('--particle_idx', type=int, default=70, help='Index of reference particle')
        self.parser.add_argument('--sel_particle_num', type=int, default=100, help='Index of reference particle')
        self.parser.add_argument('--iteration_idx', type=int, default=0, help='Iteration index')
        self.parser.add_argument('--cent_kernel', type=int, default=1, help='Iteration index')
        self.parser.add_argument('--Sel_Referance', type=str2bool, default=False, help='Select Reference Particle')
        self.parser.add_argument('--step1', type=str2bool, default=False, help='Select Reference Particle')
        self.parser.add_argument('--step2', type=str2bool, default=False, help='Select Reference Particle')
        self.parser.add_argument('--dir_name', type=str, help='Directory name',
                                 default=None)
        self.parser.add_argument('--stride', type=int, default=8, help='Select Reference Particle')
        self.parser.add_argument('--seg_tau', type=float, default=0.95, help='Segmentation threshold')
        self.parser.add_argument('--use_mask', type=str2bool, help='use mask to cal loss for SSL',
                                 default=False)
        self.parser.add_argument('--use_ema', type=str2bool, default=False,
                                 help='use EMA model')
        self.parser.add_argument('--use_bg_part', type=str2bool, default=False,
                                 help='use background particle')
        self.parser.add_argument('--use_ice_part', type=str2bool, default=False,
                                 help='use ice bg')
        self.parser.add_argument('--use_SimSeg_iteration', type=str2bool, default=False,
                                 help='use SimSeg_iteration')
        self.parser.add_argument('--ema_decay', default=0.999, type=float,
                                 help='EMA decay rate')
        self.parser.add_argument('--T', type=float, default=0.5, help='Segmentation threshold')
        self.parser.add_argument('--coord_path', type=str, help='Coordiate path name',
                                 default=None)

        # test_parameters
        self.parser.add_argument('--test_idxs', nargs='+', type=int, default=[0])
        self.parser.add_argument('--save_pred', type=str2bool, help='whether use segmentation', default=False)
        self.parser.add_argument('--max_pxs', type=int, help='dilation pixel numbers', default=18)
        self.parser.add_argument('--de_duplication', type=str2bool, default=False, help='Whether use dilation')
        self.parser.add_argument('--test_mode', type=str, default='test_val', help='test mode')
        self.parser.add_argument('--paf_connect', type=str2bool, default=False, help='Whether use dilation')
        self.parser.add_argument('--Gau_nms', type=str2bool, default=False, help='Whether use gaussian NMS')
        self.parser.add_argument('--save_mrc', type=str2bool, default=False, help='Whether save .mrc file')
        self.parser.add_argument('--nms_kernel', type=int, help='kernel size for Gaussian NMS', default=3)
        self.parser.add_argument('--nms_topK', type=int, help='topK for Gaussian NMS', default=3)
        self.parser.add_argument('--pif_model', type=str, default='', help='pif model for paf-connect')
        self.parser.add_argument('--first_idx', type=int, help='first_idx', default=0)
        self.parser.add_argument('--use_CL', type=str2bool, default=False, help='Whether use Contrastive Learning')
        self.parser.add_argument('--use_cluster', type=str2bool, default=False, help='Whether use Contrastive Learning')
        self.parser.add_argument('--use_CL_DA', type=str2bool, default=False,
                                 help='Whether use DA for reference particle of Contrastive Learning')
        self.parser.add_argument('--CL_DA_fmt', type=str, default='mean',
                                 help='format of calculating similarity map under CL_DA')
        self.parser.add_argument('--ResearchTitle', type=str, default='None',
                                 help='format of calculating similarity map under CL_DA')
        self.parser.add_argument('--skip_4v94', type=bool, default=False,
                                 help='Whether to skip 4V94 evaluation or not. True in SHREC Cryo-ET 2021 results.')
        self.parser.add_argument('--skip_vesicles', type=bool, default=False,
                                 help='Whether to skip vesicles or not. True in SHREC Cryo-ET 2021 results.')
        self.parser.add_argument('--out_name', type=str, default='TestRes',
                                 help='file name for saving the predicted coordinates')

        self.parser.add_argument('--train_set_ids', type=str, default="0")
        self.parser.add_argument('--val_set_ids', type=str, default="0")
        self.parser.add_argument('--cfg_save_path', type=str, default=".")
        # optim parameters
        self.parser.add_argument('--optim', type=str, default='AdamW')
        self.parser.add_argument('--scheduler', type=str, default='OneCycleLR')
        self.parser.add_argument('--weight_decay', type=float, default=0.01, help="torch.optim: weight decay")
        self.parser.add_argument('--use_dilation', type=str2bool, default=False, help='Whether use dilation')
        self.parser.add_argument('--use_seg', type=str2bool, default=False, help='Whether use dilation')
        self.parser.add_argument('--use_eval', type=str2bool, default=False, help='Whether use dilation')

        # loss parameters
        self.parser.add_argument('--use_weight', type=str2bool, help='whether use different weights for cls losses',
                                 default=False)
        self.parser.add_argument('--NoBG', type=str2bool,
                                 help='whether calculate BG loss (the 0th dim of softmax outputs)',
                                 default=False)
        self.parser.add_argument('--pad_loss', type=str2bool, help='whether use padding loss', default=False)
        self.parser.add_argument('--alpha', type=float, default=0.7, help="Focal Tversky Loss: alpha * FP")
        self.parser.add_argument('--beta', type=float, default=0.3, help="Focal Tversky Loss: beta * FN")
        self.parser.add_argument('--gamma', type=float, default=0.75, help="Focal Tversky Loss: focal gamma")
        self.parser.add_argument('--eta', type=float, default=0.3, help="Dice_SE_Loss: weight of SE loss")
        self.parser.add_argument('--FL_a0', type=float, default=0.1, help="Soft_FL Loss: weight of a0")
        self.parser.add_argument('--FL_a1', type=float, default=0.9, help="Soft_FL Loss: weight of a1")

        # eval parameters
        self.parser.add_argument('--JudgeInDilation', type=str2bool, default=False)
        self.parser.add_argument('--save_FPsTPs', type=str2bool, default=False,
                                 help='Whether save the results of FP and TP')
        self.parser.add_argument('--de_dup_fmt', type=str, default='fmt4', help='de-duplication format')
        self.parser.add_argument('--eval_str', type=str, default='class', help='Whether use dilation')
        self.parser.add_argument('--min_vol', type=int, default=100, help='Minimum volume')
        self.parser.add_argument('--mini_dist', type=int, default=10, help='Minimum volume')
        self.parser.add_argument('--min_dist', type=int, default=10, help='Minimum volume')
        self.parser.add_argument('--eval_cls', type=str2bool, default=False, help='Minimum volume')
        self.parser.add_argument('--class_checkpoints', type=str, help='Checkpoint directory',
                                 default=None)
        self.parser.add_argument('--meanPool_NMS', type=str2bool, default=False, help='mean_pool NMS')
        self.parser.add_argument('--meanPool_kernel', type=int, default=5, help='mean_pool NMS')

    def gather_options(self):
        args = self.parser.parse_args()
        return args
