import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynaMixerOp(nn.Module):
    def __init__(self, dim, seq_len, num_head, reduced_dim=2, area="H"):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_head = num_head
        self.reduced_dim = reduced_dim
        self.area = area

        self.out = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.compress = nn.Sequential(
            nn.Conv2d(dim, num_head * reduced_dim, kernel_size=1),
            nn.BatchNorm2d(num_head * reduced_dim),
            nn.ReLU(inplace=True),
        )
        self.generate = nn.Sequential(
            nn.Conv2d(seq_len * reduced_dim, seq_len * seq_len, kernel_size=1),
            nn.BatchNorm2d(seq_len * seq_len),
            nn.ReLU(inplace=True)
        )
        self.activation = nn.Softmax(dim=-2)

    def forward(self, x):
        B, L, C = x.shape
        x = x.reshape(-1, C, L, L)
        b, c, h, w = x.shape

        if self.area == 'H':
            weights = self.compress(x) # B, num_head * reduced_dim, H, W
            weights = weights.reshape(-1, h, self.num_head, self.reduced_dim)   # B*W，H, self.num_head, self.reduced_dim
            N = weights.shape[0]
            weights = weights.permute(0, 2, 1, 3)   # B*W, self.num_head, H, self.reduced_dim
            weights = weights.reshape(N, self.num_head, -1)
            T = weights.shape[2] # == H*self.reduced_dim
            weights = weights.reshape(B, T, self.num_head, -1)  # B, H*self.reduced_dim, self.num_head, W

            weights = self.generate(weights)    #  B, seq_len * seq_len, self.num_head, W
            weights = weights.reshape(-1, self.num_head, h, w)
            weights = self.activation(weights)  # (B*W, self.num_head, H, H)

            x = x.reshape(-1, h, self.num_head, C // self.num_head) # (B*W, H, self.num_head, C // self.num_head)
            x = x.permute(0, 2, 3, 1)   # (B*W, self.num_head,C // self.num_head, H)
            x = torch.matmul(x, weights)
            x = x.reshape(b, c, h, w)
            x = self.out(x)
            x = x.reshape(B, L, C)

        elif self.area == 'W':
            weights = self.compress(x) # B, num_head * reduced_dim, H, W
            weights = weights.reshape(-1, w, self.num_head, self.reduced_dim)  # B*H，H, self.num_head, self.reduced_dim
            N = weights.shape[0]
            weights = weights.permute(0, 2, 1, 3)  # B*H, self.num_head, W, self.reduced_dim
            weights = weights.reshape(N, self.num_head, -1)
            T = weights.shape[2]  # == W*self.reduced_dim
            weights = weights.reshape(B, T, self.num_head, -1)  # B, W*self.reduced_dim, self.num_head, H

            weights = self.generate(weights)  # B, seq_len * seq_len, self.num_head, H
            weights = weights.reshape(-1, self.num_head, h, w)
            weights = self.activation(weights)  # (B*H, self.num_head, W, W)

            x = x.reshape(-1, h, self.num_head, C // self.num_head)  # (B*H, W, self.num_head, C // self.num_head)
            x = x.permute(0, 2, 3, 1)  # (B*H, self.num_head,C // self.num_head, W)
            x = torch.matmul(x, weights)
            x = x.reshape(b, c, h, w)
            x = self.out(x)
            x = x.reshape(B, L, C)

        return x


class DynaMixerBlock(nn.Module):
    def __init__(self, dim, resolution=7, num_head=4,
                 reduced_dim=2, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.resolution = resolution
        self.num_head = num_head
        self.mix_h = DynaMixerOp(dim=dim, seq_len=resolution, num_head=self.num_head, reduced_dim=reduced_dim, area='H')
        self.mix_w = DynaMixerOp(dim=dim, seq_len=resolution, num_head=self.num_head, reduced_dim=reduced_dim, area='W')
        self.mlp_c = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        # self.proj = nn.Linear(dim, dim)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)

        B, H, W, C = x.shape
        # h = self.mix_h(x.permute(0, 2, 1, 3).reshape(-1, H, C)).reshape(B, W, H, C).permute(0, 2, 1, 3)
        # w = self.mix_w(x.reshape(-1, W, C)).reshape(B, H, W, C)
        # c = self.mlp_c(x)

        h = self.mix_h(x.reshape(-1, H, C)).reshape(B, W, H, C).permute(0, 2, 1, 3)
        w = self.mix_w(x.reshape(-1, W, C)).reshape(B, H, W, C)
        c = self.mlp_c(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Dynamic_mlp_sep(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels:int,
                 num_layers: int = 1,
                 ):
        super().__init__()

        self.in_channels = in_channels # 512
        self.hidden_channel = hidden_channels # 128

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0), # 512 -> 128
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        conv2 = []
        if num_layers == 1:
            conv2.append(DynaMixerBlock(hidden_channels))
        else:
            conv2.append(DynaMixerBlock(hidden_channels))
            for _ in range(1, num_layers-1):
                conv2.append(DynaMixerBlock(hidden_channels))
        self.conv2 = nn.ModuleList(conv2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=0)
        )
        self.norm3 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x
        x = self.conv1(x)

        for m in self.conv2:
            x = m(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = x + identity

        return x



class Dynamic_mlp_fuse(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels:int,
                 num_layers: int = 2,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channel = hidden_channels

        self.conv1 =  nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        conv2 = []
        if num_layers == 1:
            conv2.append(DynaMixerBlock(hidden_channels))
        else:
            conv2.append(DynaMixerBlock(hidden_channels))
            for _ in range(1, num_layers-1):
                conv2.append(DynaMixerBlock(hidden_channels))
        self.conv2 = nn.ModuleList(conv2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=0)
        )
        self.norm3 = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x
        x = self.conv1(x)

        for m in self.conv2:
            x = m(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = x + identity

        return x



if __name__ == "__main__":
    input = torch.randn(2, 512, 7, 7)
    model_sep = Dynamic_mlp_sep(in_channels=512, hidden_channels=128)
    out = model_sep(input)
    print(out.shape)

