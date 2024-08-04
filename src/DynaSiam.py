import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynasiam.src.encoder_res2net50 import res2net50
from dynasiam.src.decoders import Decoder_sep, Decoder_fuse
from dynasiam.src.dynamic_mlp import Dynamic_mlp_sep, Dynamic_mlp_fuse

transformer_basic_dims = 512

class AttentionGate(nn.Module):
    def __init__(self, in_channels):
        super(AttentionGate, self).__init__()

        # 注意力机制的权重学习
        self.theta = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # x是输入特征图，g是门控信号
        x = x.permute(0, 3, 1, 2)
        g = g.permute(0, 3, 1, 2)
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        psi_sum = self.psi(self.sigmoid(theta_x + phi_g))

        # 使用门控信号调整输入特征图
        result = x * psi_sum
        result = result.permute(0, 2, 3, 1)

        return result

class mmNet(nn.Module):
    def __init__(self, is_training=True):
        super(mmNet, self).__init__()

        self.is_training = is_training

        self.normal_encoder = res2net50()
        self.surface_encoder = res2net50()
        self.mucosal_encoder = res2net50()
        self.tone_encoder = res2net50()

        ########### IntraFormer 模态内交互
        self.normal_encoder_conv = nn.Sequential(
            nn.Conv2d(1024, transformer_basic_dims, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(transformer_basic_dims),
            nn.ReLU(inplace=True),
        )
        self.surface_encoder_conv = nn.Sequential(
            nn.Conv2d(1024, transformer_basic_dims, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(transformer_basic_dims),
            nn.ReLU(inplace=True),
        )
        self.mucosal_encoder_conv = nn.Sequential(
            nn.Conv2d(1024, transformer_basic_dims, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(transformer_basic_dims),
            nn.ReLU(inplace=True),
        )
        self.tone_encoder_conv = nn.Sequential(
            nn.Conv2d(1024, transformer_basic_dims, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(transformer_basic_dims),
            nn.ReLU(inplace=True),
        )

        self.normal_transformer = Dynamic_mlp_sep(in_channels=transformer_basic_dims, hidden_channels=128)
        self.surface_transformer = Dynamic_mlp_sep(in_channels=transformer_basic_dims, hidden_channels=128)
        self.mucosal_transformer = Dynamic_mlp_sep(in_channels=transformer_basic_dims, hidden_channels=128)
        self.tone_transformer = Dynamic_mlp_sep(in_channels=transformer_basic_dims, hidden_channels=128)

        self.normal_decoder_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.surface_decoder_conv = (
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.mucosal_decoder_conv = (
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),
        )
        self.tone_decoder_conv = (
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        ########### InterFormer 模态间交互
        self.multimodal_transformer = Dynamic_mlp_fuse(in_channels=transformer_basic_dims*8, hidden_channels=128)
        self.multimodal_transformer_decoder = nn.Sequential(
            nn.Conv2d(1024*4, 1024*4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024*4),
            nn.ReLU(inplace=True),
        )

        ########### Decoder
        self.decoder_sep = Decoder_sep()
        self.decoder_fuse = Decoder_fuse()

        ########### Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, normal, surface, mucosal, tone):
        # x1: torch.Size([1, 128, 96, 96])
        # x2: torch.Size([1, 256, 48, 48])
        # x3: torch.Size([1, 512, 24, 24])
        # x4: torch.Size([1, 1024, 12, 12])
        normal_x1, normal_x2, normal_x3, normal_x4 = self.normal_encoder(normal)
        surface_x1, surface_x2, surface_x3, surface_x4 = self.surface_encoder(surface)
        mucosal_x1, mucosal_x2, mucosal_x3, mucosal_x4 = self.mucosal_encoder(mucosal)
        tone_x1, tone_x2, tone_x3, tone_x4 = self.tone_encoder(tone)

        assert normal_x4.size() == surface_x4.size() == mucosal_x4.size() == tone_x4.size()

        # 辅助正则器 其实就是深度监督
        if self.is_training:
            normal_pred = self.decoder_sep(normal_x1, normal_x2, normal_x3, normal_x4)
            surface_pred = self.decoder_sep(surface_x1, surface_x2, surface_x3, surface_x4)
            mucosal_pred = self.decoder_sep(mucosal_x1, mucosal_x2, mucosal_x3, mucosal_x4)
            tone_pred = self.decoder_sep(tone_x1, tone_x2, tone_x3, tone_x4)

        ########### IntraFormer 模态内交互
        normal_token_x4 = self.normal_encoder_conv(normal_x4)
        surface_token_x4 = self.surface_encoder_conv(surface_x4)
        mucosal_token_x4 = self.mucosal_encoder_conv(mucosal_x4)
        tone_token_x4 = self.tone_encoder_conv(tone_x4)

        normal_intra_x4 = self.normal_transformer(normal_token_x4)
        surface_intra_x4 = self.surface_transformer(surface_token_x4)
        mucosal_intra_x4 = self.mucosal_transformer(mucosal_token_x4)
        tone_intra_x4 = self.tone_transformer(tone_token_x4)

        normal_intra_x4 = self.normal_decoder_conv(normal_intra_x4)
        surface_intra_x4 = self.normal_decoder_conv(surface_intra_x4)
        mucosal_intra_x4 = self.normal_decoder_conv(mucosal_intra_x4)
        tone_intra_x4 = self.normal_decoder_conv(tone_intra_x4)

        if self.is_training:
            normal_intra_pred = self.decoder_sep(normal_x1, normal_x2, normal_x3, normal_intra_x4)
            surface_intra_pred = self.decoder_sep(surface_x1, surface_x2, surface_x3, surface_intra_x4)
            mucosal_intra_pred = self.decoder_sep(mucosal_x1, mucosal_x2, mucosal_x3, mucosal_intra_x4)
            tone_intra_pred = self.decoder_sep(tone_x1, tone_x2, tone_x3, tone_intra_x4)

        #### 模态间交互
        x1 = torch.stack((normal_x1, surface_x1, mucosal_x1, tone_x1), dim=1)  # （B， 4， 256, 96, 96）
        x1 = x1.reshape(x1.shape[0], -1, x1.shape[3], x1.shape[4])

        x2 = torch.stack((normal_x2, surface_x2, mucosal_x2, tone_x2), dim=1)
        x2 = x2.reshape(x2.shape[0], -1, x2.shape[3], x2.shape[4])

        x3 = torch.stack((normal_x3, surface_x3, mucosal_x3, tone_x3), dim=1)
        x3 = x3.reshape(x3.shape[0], -1, x3.shape[3], x3.shape[4])

        multimodal_token_x4 = torch.cat((normal_intra_x4, surface_intra_x4, mucosal_intra_x4, tone_intra_x4), dim=1)

        multimodal_inter_token_x4 = self.multimodal_transformer(multimodal_token_x4)
        multimodal_inter_x4 = self.multimodal_transformer_decoder(multimodal_inter_token_x4)
        x4_inter = multimodal_inter_x4

        fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4_inter)

        if self.is_training:
            return fuse_pred, (normal_pred, surface_pred, mucosal_pred, tone_pred), \
                   (normal_intra_pred, surface_intra_pred, mucosal_intra_pred, tone_intra_pred), preds


        return fuse_pred


if __name__ == '__main__':
    normal = torch.rand(2, 3, 224, 224)  # (B, D, C, H, W)
    surface = torch.rand(2, 3, 224, 224)
    mucosal = torch.rand(2, 3, 224, 224)
    tone = torch.rand(2, 3, 224, 224)
    model = mmNet(is_training=False)
    # fuse_pred, modal_preds, intra_preds, preds = model(normal, surface, mucosal, tone)
    fuse_pred= model(normal, surface, mucosal, tone)
    print(fuse_pred.shape)

