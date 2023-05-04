import torch
import torch.nn as nn
import torch.nn.modules.conv as conv


class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            x_range = torch.linspace(-1, 1, dim_x, device=input_tensor.device)
            xx_channel = x_range.expand([batch_size_shape, 1, -1])

            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            x_range = torch.linspace(-1, 1, dim_x, device=input_tensor.device)
            y_range = torch.linspace(-1, 1, dim_y, device=input_tensor.device)
            yy_channel, xx_channel = torch.meshgrid(y_range, x_range)
            yy_channel = yy_channel.expand([batch_size_shape, 1, -1, -1])
            xx_channel = xx_channel.expand([batch_size_shape, 1, -1, -1])

            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = input_tensor.shape
            x_range = torch.linspace(-1, 1, dim_x, device=input_tensor.device)
            y_range = torch.linspace(-1, 1, dim_y, device=input_tensor.device)
            z_range = torch.linspace(-1, 1, dim_z, device=input_tensor.device)
            zz_channel, yy_channel, xx_channel = torch.meshgrid(z_range, y_range, x_range)
            zz_channel = zz_channel.expand([batch_size_shape, 1, -1, -1, -1])
            yy_channel = yy_channel.expand([batch_size_shape, 1, -1, -1, -1])
            xx_channel = xx_channel.expand([batch_size_shape, 1, -1, -1, -1])
            out = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                                torch.pow(yy_channel - 0.5, 2) +
                                torch.pow(zz_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out


class CoordConv1d(conv.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 1
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv1d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class CoordConv3d(conv.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv3d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 3
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv3d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out