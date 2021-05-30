from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        x = F.elu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2):
        super(EncoderBlock, self).__init__()
        self.root_feat_maps = 16
        self.num_conv_blocks = 2
        self.model_depth = model_depth
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            for i in range(self.num_conv_blocks):
                self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                # if depth == 0:
                #     self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                #     self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                #     in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                # else:
                #     self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                #     self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                #     in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        features = []
        for k, op in self.module_dict.items():
            if k.startswith("conv"):
                x = op(x)
            else:
                features.append(x)
                x = op(x)
        return x, features