import torch.nn as nn
from torch.nn.functional import normalize
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),  # 改
            nn.ReLU(),
            nn.Linear(2000, feature_dim),  # 改
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),  # 改
            nn.ReLU(),
            nn.Linear(2000, 500),  # 改
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, views, input_size, feature_dim):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.views = views
        for v in range(views):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, xs):
        zs = []
        xrs = []
        for v in range(self.views):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            zs.append(z)
            xrs.append(xr)
        return zs, xrs
