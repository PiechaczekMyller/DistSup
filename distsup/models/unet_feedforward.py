import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import itertools
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn

from distsup.models.losses import DiceLoss, ComposedLoss, BCEWithLogitsLoss
from distsup.utils import construct_from_kwargs
from distsup.models import base


class UNet3DBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels,
                               out_channels=mid_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn1 = nn.InstanceNorm3d(out_channels)
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(in_channels=mid_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.LeakyReLU()

    def forward(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        return out


class UNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__()

        features = init_features
        self.encoder1 = UNet3DBlock(in_channels, features, features)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.encoder2 = UNet3DBlock(features, features * 2, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.encoder3 = UNet3DBlock(features * 2, features * 4, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.encoder4 = UNet3DBlock(features * 4, features * 8, features * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.bottleneck = UNet3DBlock(features * 8, features * 16, features * 16)

        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.decoder4 = UNet3DBlock((features * 8) * 2, features * 8, features * 8)
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.decoder3 = UNet3DBlock((features * 4) * 2, features * 4, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.decoder2 = UNet3DBlock((features * 2) * 2, features * 2, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.decoder1 = UNet3DBlock(features * 2, features, features)
        self.mapper = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, input):
        enc1 = self.encoder1(input)
        pooled_enc1 = self.pool1(enc1)
        enc2 = self.encoder2(pooled_enc1)
        pooled_enc2 = self.pool2(enc2)
        enc3 = self.encoder3(pooled_enc2)
        pooled_enc3 = self.pool3(enc3)
        enc4 = self.encoder4(pooled_enc3)
        pooled_enc4 = self.pool4(enc4)

        bottleneck = self.bottleneck(pooled_enc4)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        maps = self.mapper(dec1)
        return maps, bottleneck


class UNetFeedForwardLearner(base.Model):
    def __init__(self,
                 unet=dict(
                     class_name='UNet3D',
                     in_channels=4,
                     out_channels=4,
                     init_features=32),
                 **kwargs):
        super(UNetFeedForwardLearner, self).__init__(**kwargs)

        self.unet = construct_from_kwargs(unet)
        self.N = None
        remove_background_label = lambda x, y: (x[:, 1:, ...], y[:, 1:, ...])
        self.dice_loss = DiceLoss(inputs_preprocess_fn=self.dice_preprocess)
        self.bce_loss = BCEWithLogitsLoss(inputs_preprocess_fn=remove_background_label)

        self.compound_loss = ComposedLoss([self.dice_loss, self.bce_loss])

    @staticmethod
    def dice_preprocess(prediction: torch.Tensor, target: torch.Tensor):
        prediction = torch.sigmoid(prediction)
        return prediction[:, 1:, ...], target[:, 1:, ...]

    def minibatch_loss(self, batch):
        images = batch['features']
        targets = batch['targets']

        N = np.prod(images.size()[1:])
        if self.N is None:
            self.N = N
        else:
            assert N == self.N

        x, bottleneck = self.unet(images)

        loss = self.compound_loss(x, targets)
        stats = {'loss': loss}

        return stats['loss'], stats

    def evaluate(self, batches):
        was_training = self.training
        self.eval()
        stats = None

        for i, batch in enumerate(batches):
            images = batch['features']
            # targets = ((images + 0.5) * 255).long()
            with torch.no_grad():
                _, batch_stats = self.minibatch_loss(batch)
            stats = self.update_stats(stats, batch_stats, i)

        if was_training:
            self.train()

        return stats

    def update_stats(self, epoch_stats, batch_stats, it):
        if epoch_stats is None:
            return batch_stats
        else:
            for key, val in batch_stats.items():
                if type(val) is torch.Tensor:
                    val = val.item()
                epoch_stats[key] += (val - epoch_stats[key]) / it
            return epoch_stats

    # XXX Unused ?
    def parameters(self, with_codebook=True):
        if with_codebook:
            return super(UNetFeedForwardLearner, self).parameters()
        else:
            return itertools.chain(self.encoder.parameters(),
                                   self.decoder.parameters())
