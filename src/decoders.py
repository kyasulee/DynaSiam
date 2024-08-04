import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder_sep(nn.Module):
    def __init__(self):
        super(Decoder_sep, self).__init__()

        self.dup_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d3_l1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.d3_l2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.d3_out = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.dup_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d2_l1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.d2_l2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.d2_out = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.dup_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d1_l1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.d1_l2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.d1_out = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.sag_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.seg_layer = nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x1, x2, x3, x4):
        de_x4 = self.dup_3(x4)
        de_x4 = self.d3_l1(de_x4)

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_l2(cat_x3)
        de_x3 = self.d3_out(de_x3)
        de_x3 = self.dup_2(de_x3) # (b 1024 48 48)
        de_x3 = self.d2_l1(de_x3)

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_l2(cat_x2)
        de_x2 = self.d2_out(de_x2)
        de_x2 = self.dup_1(de_x2)
        de_x2 = self.d1_l1(de_x2)

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_l2(cat_x1)
        de_x1 = self.d1_out(de_x1)

        logits = self.sag_up(de_x1)
        logits = self.seg_layer(logits)

        return logits


class Decoder_fuse(nn.Module):
    def __init__(self):
        super(Decoder_fuse, self).__init__()

        self.fuse_layer4 = nn.Sequential(
            nn.Conv2d(1024*4, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.fuse_layer3 = nn.Sequential(
            nn.Conv2d(512 * 4, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.fuse_layer2 = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.fuse_layer1 = nn.Sequential(
            nn.Conv2d(128 * 4, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.seg_d4 = nn.Conv2d(1024, 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0, bias=True)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.d3_c1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.d3_c2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.d3_out = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.d2_c1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.d2_c2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.d2_out = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.d1_c1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.d1_c2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.d1_out = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )


    def forward(self, x1, x2, x3, x4):
        # x1: torch.Size([1, 128, 96, 96])
        # x2: torch.Size([1, 256, 48, 48])
        # x3: torch.Size([1, 512, 24, 24])
        # x4: torch.Size([1, 1024, 12, 12])
        de_x4 = self.fuse_layer4(x4)
        pred4 = self.seg_d4(de_x4)

        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.fuse_layer3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred3 = self.seg_d3(de_x3)

        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.fuse_layer2(x2)
        de_x2 = torch.cat((de_x3, de_x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred2 = self.seg_d2(de_x2)

        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.fuse_layer1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_d1(de_x1)
        # pred = self.softmax(logits)

        return self.up4(logits), (self.up8(pred2), self.up16(pred3), self.up32(pred4))

if __name__ == "__main__":
    x1 = torch.randn(1, 128*4, 96, 96)
    x2 = torch.randn(1, 256*4, 48, 48)
    x3 = torch.randn(1, 512*4, 24, 24)
    x4 = torch.randn(1, 1024*4, 12, 12)

    # decoder_sep = Decoder_sep()
    # out1 = decoder_sep(x1, x2, x3, x4)

    decoder_fuse = Decoder_fuse()
    out2 = decoder_fuse(x1, x2, x3, x4)