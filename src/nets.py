# causal 1d cnn for the flare windows. one shared conv trunk feeding two heads
# (detection + warning) since both read the same soft/hard rise. built so the
# jump to a TCN is a config change, not a rewrite: the conv block already takes a
# dilation and an optional residual, so a plain cnn is just dilation=1 no skip.

import torch.nn as nn


class CausalConv1d(nn.Module):
    # left-pad by (k-1)*d then chop the right overhang, so output[t] never sees a
    # future step. this is the load-bearing bit for warning - a normal conv would
    # leak the very peak it's meant to predict.
    def __init__(self, c_in, c_out, k, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, padding=self.pad, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x[..., :-self.pad] if self.pad else x


class ConvBlock(nn.Module):
    # causal conv -> norm -> relu -> dropout, optional residual. residual off is
    # the baseline cnn; flip it on with dilations and this block is a TCN block.
    def __init__(self, c_in, c_out, k, dilation=1, dropout=0.1, residual=False):
        super().__init__()
        self.conv = CausalConv1d(c_in, c_out, k, dilation)
        self.norm = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.res = None
        if residual:
            # 1x1 to match channels for the skip, identity if they already match
            self.res = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1)

    def forward(self, x):
        y = self.drop(self.act(self.norm(self.conv(x))))
        if self.res is not None:
            y = y + self.res(x)
        return y


class FlareCNN(nn.Module):
    # in (B,T,F). conv over time, take the last (causal) step as the "now"
    # representation, two logit heads off the same trunk.
    def __init__(self, n_feat=6, channels=(32, 64, 64), k=5,
                 dilations=None, dropout=0.1, residual=False, head_hidden=32):
        super().__init__()
        if dilations is None:
            dilations = [1] * len(channels)     # plain cnn; TCN would be [1,2,4,...]
        assert len(dilations) == len(channels)

        blocks = []
        c_prev = n_feat
        for c, d in zip(channels, dilations):
            blocks.append(ConvBlock(c_prev, c, k, dilation=d,
                                    dropout=dropout, residual=residual))
            c_prev = c
        self.trunk = nn.Sequential(*blocks)

        def head():
            return nn.Sequential(nn.Linear(c_prev, head_hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(head_hidden, 1))
        self.det_head = head()
        self.warn_head = head()

    def forward(self, x):
        z = self.trunk(x.transpose(1, 2))       # (B,T,F) -> (B,F,T) -> (B,C,T)
        z = z[..., -1]                          # last timestep = the labelled "now"
        return self.det_head(z).squeeze(-1), self.warn_head(z).squeeze(-1)
