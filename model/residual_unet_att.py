import torch.nn as nn
from torch.nn import functional as F
import torch
from functools import partial
import sys
from torch.nn import Conv3d, Module, Linear, BatchNorm3d, ReLU
from torch.nn.modules.utils import _pair, _triple

sys.path.append("..")
from utils.coordconv_torch import AddCoords

# from SoftPool import SoftPool3d
try:
    from model.sync_batchnorm import SynchronizedBatchNorm3d
except:
    pass


def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    elif norm == 'sync_bn':
        m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


# Residual 3D UNet
class ResidualUNet3D(nn.Module):
    def __init__(self, f_maps=[32, 64, 128, 256], in_channels=1, out_channels=13,
                 args=None, use_att=False, use_paf=None, use_uncert=None):
        super(ResidualUNet3D, self).__init__()
        if use_att:
            norm = BatchNorm3d
        else:
            norm = args.norm
        act = args.act
        use_lw = args.use_lw
        lw_kernel = args.lw_kernel

        self.use_aspp = args.use_aspp
        self.pif_sigmoid = args.pif_sigmoid
        self.paf_sigmoid = args.paf_sigmoid
        self.use_tanh = args.use_tanh
        self.use_IP = args.use_IP
        self.out_channels = out_channels
        if self.out_channels > 1:
            self.use_softmax = args.use_softmax
        else:
            self.use_sigmoid = args.use_sigmoid
        self.use_coord = args.use_coord

        self.use_softpool = args.use_softpool

        self.use_paf = use_paf
        self.use_uncert = use_uncert

        if self.use_softpool:
            # pool_layer = SoftPool3d
            pass
        else:
            pool_layer = nn.AvgPool3d

        if self.use_IP:
            pools = []
            for _ in range(len(f_maps) - 1):
                pools.append(pool_layer(2))
            self.pools = nn.ModuleList(pools)

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, use_IP=False,
                                  use_coord=self.use_coord,
                                  pool_layer=pool_layer, norm=norm, act=act, use_att=use_att,
                                  use_lw=use_lw, lw_kernel=lw_kernel)
            else:
                # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
                encoder = Encoder(f_maps[i - 1], out_feature_num, use_IP=self.use_IP, use_coord=self.use_coord,
                                  pool_layer=pool_layer, norm=norm, act=act, use_att=use_att,
                                  use_lw=use_lw, lw_kernel=lw_kernel)

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # 使用aspp进一步提取特征
        if self.use_aspp:
            self.aspp = ASPP(in_channels=f_maps[-1], inter_channels=f_maps[-1], out_channels=f_maps[-1])

        self.se_loss = args.use_se_loss
        if self.se_loss:
            self.avgpool = nn.AdaptiveAvgPool3d(1)
            self.fc1 = nn.Linear(f_maps[-1], f_maps[-1])
            self.fc2 = nn.Linear(f_maps[-1], out_channels)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i]
            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (2, 2, 2)
            decoder = Decoder(in_feature_num, out_feature_num, use_coord=self.use_coord, norm=norm, act=act,
                              use_att=use_att, use_lw=use_lw, lw_kernel=lw_kernel)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        if args.final_double:
            self.final_conv = nn.Sequential(
                nn.Conv3d(f_maps[0], f_maps[0] // 2, kernel_size=1),
                nn.Conv3d(f_maps[0] // 2, out_channels, 1)
            )
            if self.use_paf:
                self.paf_conv = nn.Sequential(
                    nn.Conv3d(f_maps[0], f_maps[0] // 2, kernel_size=1),
                    nn.Conv3d(f_maps[0] // 2, 1, 1)
                )
            self.dropout = nn.Dropout3d
        else:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
            if self.use_paf:
                self.paf_conv = nn.Conv3d(f_maps[0], 1, 1)
            self.dropout = nn.Dropout3d

        if self.use_paf:
            if self.use_uncert:
                self.logsigma = nn.Parameter(torch.FloatTensor([0.5] * 2))
            else:
                self.logsigma = torch.FloatTensor([0.5] * 2)

    def forward(self, x):
        if self.use_IP:
            img_pyramid = []
            img_d = x
            for pool in self.pools:
                img_d = pool(img_d)
                img_pyramid.append(img_d)

        encoders_features = []
        for idx, encoder in enumerate(self.encoders):
            if self.use_IP and idx > 0:
                x = encoder(x, img_pyramid[idx - 1])
            else:
                x = encoder(x)
            encoders_features.insert(0, x)

        if self.use_aspp:
            x = self.aspp(x)
        # remove last
        encoders_features = encoders_features[1:]

        if self.se_loss:
            se_out = self.avgpool(x)
            se_out = se_out.view(se_out.size(0), -1)
            se_out = self.fc1(se_out)
            se_out = self.fc2(se_out)

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        out = self.final_conv(x)

        if self.out_channels > 1:
            if self.use_softmax:
                out = torch.softmax(out, dim=1)
            elif self.pif_sigmoid:
                out = torch.sigmoid(out)
            elif self.use_tanh:
                out = torch.tanh(out)
        else:
            if self.use_sigmoid:
                out = torch.sigmoid(out)
            elif self.use_tanh:
                out = torch.tanh(out)

        if self.use_paf:
            paf_out = self.paf_conv(x)
            if self.paf_sigmoid:
                paf_out = torch.sigmoid(paf_out)

        if self.se_loss:
            return [out, se_out]
        else:
            if self.use_paf:
                return [out, paf_out, self.logsigma]
            else:
                return out


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, apply_pooling=True, use_IP=False, use_coord=False,
                 pool_layer=nn.MaxPool3d, norm='bn', act='relu', use_att=False,
                 use_lw=False, lw_kernel=3, input_channels=1):
        super(Encoder, self).__init__()
        if apply_pooling:
            self.pooling = pool_layer(kernel_size=2)
        else:
            self.pooling = None

        self.use_IP = use_IP
        self.use_coord = use_coord
        inplaces = in_channels + input_channels if self.use_IP else in_channels
        inplaces = inplaces + 3 if self.use_coord else inplaces

        if use_att:
            self.basic_module = ExtResNetBlock_att(inplaces, out_channels, norm=norm, act=act)
        else:
            if use_lw:
                self.basic_module = ExtResNetBlock_lightWeight(inplaces, out_channels, lw_kernel=lw_kernel)
            else:
                self.basic_module = ExtResNetBlock(inplaces, out_channels, norm=norm, act=act)
        if self.use_coord:
            self.coord_conv = AddCoords(rank=3, with_r=False)

    def forward(self, x, scaled_img=None):
        if self.pooling is not None:
            x = self.pooling(x)
        if self.use_IP:
            x = torch.cat([x, scaled_img], dim=1)
        if self.use_coord:
            x = self.coord_conv(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=(2, 2, 2), mode='nearest',
                 padding=1, use_coord=False, norm='bn', act='relu', use_att=False,
                 use_lw=False, lw_kernel=3):
        super(Decoder, self).__init__()
        self.use_coord = use_coord
        if self.use_coord:
            self.coord_conv = AddCoords(rank=3, with_r=False)

        # if basic_module=ExtResNetBlock use transposed convolution upsampling and summation joining
        self.upsampling = Upsampling(transposed_conv=True, in_channels=in_channels, out_channels=out_channels,
                                     scale_factor=scale_factor, mode=mode)
        # sum joining
        self.joining = partial(self._joining, concat=False)
        # adapt the number of in_channels for the ExtResNetBlock
        in_channels = out_channels + 3 if self.use_coord else out_channels

        if use_att:
            self.basic_module = ExtResNetBlock_att(in_channels, out_channels, norm=norm, act=act)
        else:
            if use_lw:
                self.basic_module = ExtResNetBlock_lightWeight(in_channels, out_channels, lw_kernel=lw_kernel)
            else:
                self.basic_module = ExtResNetBlock(in_channels, out_channels, norm='bn', act=act)

    def forward(self, encoder_features, x, ReturnInput=False):
        x = self.upsampling(encoder_features, x)
        x = self.joining(encoder_features, x)
        if self.use_coord:
            x = self.coord_conv(x)
        if ReturnInput:
            x1 = self.basic_module(x)
            return x1, x
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class ExtResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm='bn', act='relu'):
        super(ExtResNetBlock, self).__init__()
        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, norm=norm, act=act)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, norm=norm, act=act)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        self.conv3 = SingleConv(out_channels, out_channels, norm=norm, act=act)
        self.non_linearity = nn.ELU(inplace=False)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out
        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)
        return out


class ExtResNetBlock_att(nn.Module):
    def __init__(self, in_channels, out_channels, norm='bn', act='relu'):
        super(ExtResNetBlock_att, self).__init__()
        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, norm=norm, act=act)
        # residual block
        self.conv2 = SplAtConv3d(out_channels, out_channels // 2, norm_layer=norm)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        self.conv3 = SplAtConv3d(out_channels, out_channels // 2, norm_layer=norm)
        self.non_linearity = nn.ELU(inplace=False)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out
        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)
        return out


class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm='bn', act='relu'):
        super(SingleConv, self).__init__()
        self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('batchnorm', normalization(out_channels, norm=norm))
        if act == 'relu':
            self.add_module('relu', nn.ReLU(inplace=False))
        elif act == 'lrelu':
            self.add_module('lrelu', nn.LeakyReLU(negative_slope=0.1, inplace=False))
        elif act == 'elu':
            self.add_module('elu', nn.ELU(inplace=False))
        elif act == 'gelu':
            self.add_module('elu', nn.GELU(inplace=False))


class ExtResNetBlock_lightWeight(nn.Module):
    def __init__(self, in_channels, out_channels, lw_kernel=3):
        super(ExtResNetBlock_lightWeight, self).__init__()
        # first convolution
        self.conv1 = SingleConv_lightWeight(in_channels, out_channels, lw_kernel=lw_kernel)
        # residual block
        self.conv2 = SingleConv_lightWeight(out_channels, out_channels, lw_kernel=lw_kernel)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        self.conv3 = SingleConv_lightWeight(out_channels, out_channels, lw_kernel=lw_kernel)
        self.non_linearity = nn.ELU(inplace=False)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out
        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)
        return out


class SingleConv_lightWeight(nn.Sequential):
    def __init__(self, in_channels, out_channels, lw_kernel=3, layer_scale_init_value=1e-6):
        super(SingleConv_lightWeight, self).__init__()

        self.dwconv = nn.Conv3d(in_channels, in_channels, kernel_size=lw_kernel, padding=lw_kernel//2, groups=in_channels)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels, 2 * in_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * in_channels, out_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, D, H, W) -> (N, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)

        x = x + (input if self.in_channels == self.out_channels else self.skip(input))
        return x


class Upsampling(nn.Module):
    def __init__(self, transposed_conv, in_channels=None, out_channels=None, scale_factor=(2, 2, 2), mode='nearest'):
        super(Upsampling, self).__init__()

        if transposed_conv:
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=scale_factor, padding=1)
        else:
            self.upsample = partial(self._interpolate, mode=mode)

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class SplAtConv3d(Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                 dilation=(1, 1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=BatchNorm3d,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv3d, self).__init__()
        padding = _triple(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            pass
            # from rfconv import RFConv2d
            # self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
            #                      groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv3d(in_channels, channels * radix, kernel_size, stride, padding, dilation,
                               groups=groups * radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels * radix)
        self.relu = ReLU(inplace=False)
        self.fc1 = Conv3d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv3d(inter_channels, channels * radix, 1, groups=self.cardinality)
        # if dropblock_prob > 0.0:
        #     self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)
        self.conv3 = Conv3d(channels, channels * radix, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.bn3 = BatchNorm3d(channels * radix, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        # if self.dropblock_prob > 0.0:
        #     x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel // self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool3d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1, 1)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel // self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x

        out = self.relu3(self.bn3(self.conv3(out)))
        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Training for 3D U-Net models')
    parser.add_argument('--use_IP', type=bool, help='whether use image pyramid', default=False)
    parser.add_argument('--use_DS', type=bool, help='whether use deep supervision', default=False)
    parser.add_argument('--use_Res', type=bool, help='whether use residual connectivity', default=False)
    parser.add_argument('--use_bg', type=bool, help='whether use batch generator', default=False)
    parser.add_argument('--use_coord', type=bool, help='whether use coord conv', default=False)
    parser.add_argument('--use_softmax', type=bool, help='whether use softmax', default=False)
    parser.add_argument('--use_softpool', type=bool, help='whether use softpool', default=False)
    parser.add_argument('--use_aspp', type=bool, help='whether use aspp', default=False)
    parser.add_argument('--use_att', type=bool, help='whether use aspp', default=False)
    parser.add_argument('--use_se_loss', type=bool, help='whether use aspp', default=False)
    parser.add_argument('--pif_sigmoid', type=bool, help='whether use aspp', default=False)
    parser.add_argument('--paf_sigmoid', type=bool, help='whether use aspp', default=False)
    parser.add_argument('--final_double', type=bool, help='whether use aspp', default=False)
    parser.add_argument('--use_tanh', type=bool, help='whether use aspp', default=False)
    parser.add_argument('--norm', help='type of normalization', type=str, default='sync_bn',
                        choices=['bn', 'gn', 'in', 'sync_bn'])
    parser.add_argument('--use_lw', type=bool, help='whether use lightweight', default=True)
    parser.add_argument('--lw_kernel', type=int, default=5)
    parser.add_argument('--act', help='type of activation function', type=str, default='relu',
                        choices=['relu', 'lrelu', 'elu', 'gelu'])
    args = parser.parse_args()

    net = ResidualUNet3D(args=args, use_att=args.use_att, f_maps=[24, 48, 72, 108])
    print(net)

    # conv = SplAtConv3d(64, 32, 3)
    # print(conv)
    data = torch.rand([2, 1, 56, 56, 56])
    out = net(data)
    print(out.shape)
