from torch import nn
from .encoder import EncoderBlock
from .decoder import DecoderBlock

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, model_depth=4, pool_size=2):
        super(UNet, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth, pool_size=pool_size)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)
    
    def forward(self, x):
        x, features = self.encoder(x)
        out = self.decoder(x, features)
        return out