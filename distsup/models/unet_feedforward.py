import itertools
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from distsup.utils import construct_from_kwargs
from distsup.models import base


class Residual(nn.Module):
    def __init__(self, in_ch, out_ch=None, ksp1=(3, 1, 1), ksp2=(1, 1, 0),
                 batch_norm=False):
        """
            ksp (tuple): kernel size, stride, padding
        """
        super(Residual, self).__init__()
        out_ch = out_ch or in_ch
        layers = [
            nn.ReLU(True),
            nn.Conv2d(in_ch, out_ch, *ksp1, bias=False),
            nn.BatchNorm2d(out_ch) if batch_norm else None,
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, *ksp2, bias=False),
            nn.BatchNorm2d(out_ch) if batch_norm else None]
        self.block = nn.Sequential(*[l for l in layers if l is not None])
        if out_ch != in_ch:
            stride = ksp1[1]
            self.short = nn.Conv2d(in_ch, out_ch, 1, stride, 0)

    def forward(self, x):
        return self.block(x) + (self.short(x) if hasattr(self, 'short') else x)


class ConvEncoder(nn.Module):
    def __init__(self, channels, latent_dim, embedding_dim, batch_norm=False):
        super(ConvEncoder, self).__init__()
        layers = [
            nn.Conv2d(3, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels) if batch_norm else None,
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels) if batch_norm else None,
            Residual(channels, batch_norm=batch_norm),
            Residual(channels, batch_norm=batch_norm),
            nn.Conv2d(channels, latent_dim * embedding_dim, 1)]
        self.encoder = nn.Sequential(*[l for l in layers if l is not None])

    def forward(self, x):
        return self.encoder(x)


class ConvDecoder(nn.Module):
    def __init__(self, channels, latent_dim, embedding_dim, batch_norm=False):
        super(ConvDecoder, self).__init__()
        layers = [
            nn.Conv2d(latent_dim * embedding_dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels) if batch_norm else None,
            Residual(channels, batch_norm=batch_norm),
            Residual(channels, batch_norm=batch_norm),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels) if batch_norm else None,
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels) if batch_norm else None,
            nn.ReLU(True),
            nn.Conv2d(channels, 3 * 256, 1)]
        self.decoder = nn.Sequential(*[l for l in layers if l is not None])

    def forward(self, x):
        x = self.decoder(x)
        B, _, H, W = x.size()
        x = x.view(B, 3, 256, H, W).permute(0, 1, 3, 4, 2)
        dist = Categorical(logits=x)
        return dist


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

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
        conv = self.conv(dec1)
        return conv

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class UNetFeedForwardLearner(base.Model):
    def __init__(self,
                 unet=dict(
                     class_name='UNet',
                     in_channels=3,
                     out_channels=1,
                     init_features=32),
                 **kwargs):
        super(UNetFeedForwardLearner, self).__init__(**kwargs)

        self.unet = construct_from_kwargs(unet)
        self.N = None

    def minibatch_loss(self, batch):
        images = batch['features']
        targets = images.clone()

        N = np.prod(images.size()[1:])
        if self.N is None:
            self.N = N
        else:
            assert N == self.N

        x = self.unet(images)
        targets = nn.Softmax()(targets)
        mse = nn.MSELoss()
        loss = mse(images, targets)
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


class ResNetN(nn.Module):
    def __init__(self, args, channels, latent_dim, num_embeddings,
                 embedding_dim, bottleneck='VQVAE',
                 batch_norm=False, n=12, num_classes=10):
        super(ResNetN, self).__init__()
        assert n % 6 == 0
        l = [nn.Conv2d(3, 64, 3, 1, 1)]
        if batch_norm:
            l += [nn.BatchNorm2d(64)]
        self.conv = nn.Sequential(*l)
        self.block1 = nn.Sequential(
            *[Residual(64, 64, (3, 1, 1), batch_norm=batch_norm)
              for i in range(n // 6)])
        self.block2 = nn.Sequential(
            *[Residual(64, 128, (3, 2, 1), (3, 1, 1), batch_norm=batch_norm)] + [
                Residual(128, 128, (3, 1, 1), batch_norm=batch_norm)
                for i in range(n // 6 - 1)])
        self.block3 = nn.Sequential(
            *[Residual(128, 256, (3, 2, 1), (3, 1, 1), batch_norm=batch_norm)] + [
                Residual(256, 256, (3, 1, 1), batch_norm=batch_norm)
                for i in range(n // 6 - 2)] + [
                 Residual(256, 256, (3, 1, 1), batch_norm=batch_norm)])
        self.fc = nn.Linear(256, num_classes)

        self.bottleneck_after_block = args.bottleneck_after_block
        if bottleneck is None or bottleneck.lower() == "none":
            self.codebook = None
        elif bottleneck == 'VQVAE':
            self.codebook = VQEmbedding(
                latent_dim, num_embeddings, embedding_dim,
                codebook_cost=args.codebook_cost,
                reestimation_reservoir_size=args.reestimation_reservoir_size,
                reestimate_every_epochs=args.reestimate_every_epochs,
                reestimate_every_iters=args.reestimate_every_iters,
                reestimate_every_iters_expansion=args.reestimate_every_iters_expansion,
                reestimate_max_iters=args.reestimate_max_iters,
                reestimate_max_epochs=args.reestimate_max_epochs,
                bottleneck_enforce_from_epoch=args.bottleneck_enforce_from_epoch, )
        elif bottleneck == 'EMA':
            self.codebook = VQEmbeddingEMA(
                latent_dim, num_embeddings, embedding_dim, decay=args.ema_decay)
        else:
            raise ValueError

    def forward(self, x, labels, N):
        stats = {'enc_mean': 0, 'enc_std': 0}  # Dummy
        x = self.conv(x)
        for idx in range(1, 4):
            x = getattr(self, f'block{idx}')(x)
            if self.bottleneck_after_block == idx and self.codebook is not None:
                x, vq_loss, stats = self.codebook(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        stats['loss'] = F.cross_entropy(x, labels, reduction='mean')
        stats['acc'] = torch.mean((labels == torch.argmax(x, dim=1)).float()) * 100.0
        if self.codebook is not None:
            stats['loss'] += vq_loss
        return None, stats['loss'], stats

    def parameters(self, with_codebook=True):
        if with_codebook:
            return super(ResNetN, self).parameters()
        else:
            return itertools.chain(self.conv.parameters(),
                                   self.block1.parameters(),
                                   self.block2.parameters(),
                                   self.block3.parameters(),
                                   self.fc.parameters())


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

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
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
