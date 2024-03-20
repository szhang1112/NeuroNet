import torch
import numpy as np
from torch import nn, einsum
from einops.layers.torch import Rearrange
import torch.nn.functional as F
# from .mtl_utils import MMOE
from .mmoe import MMOE
from .layer_utils import Lambda, AttentionPool

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__() 
    
    def forward(self, x):
        return x.squeeze()
        

class Beluga(nn.Module):
    def __init__(self, num_tasks=2002):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,num_tasks)),
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

class BelugaResume(nn.Module):
    def __init__(self, num_tasks=2002):
        super(BelugaResume, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                # nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                # nn.ReLU(),
                # nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,num_tasks)),
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

# class DeepSEA(nn.Module):
#     def __init__(self):
#         super(Beluga, self).__init__()
#         self.model = nn.Sequential(
#             nn.Sequential(
#                 nn.Conv1d(4,64,8,2),
#                 nn.ReLU(),
#                 nn.MaxPool1d(4,4),
#                 nn.Dropout(0.2),
#                 nn.Conv2d(320,480,(1, 8)),
#                 nn.ReLU(),
#                 nn.MaxPool2d((1, 4),(1, 4)),
#                 nn.Dropout(0.2),
#                 nn.Conv2d(480,960,(1, 8)),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(1,2002)
#             ),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         return self.model(x)


class ConvNet(nn.Module):
    def __init__(self, num_tasks=2002):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv1d(4,320,8),
                nn.ReLU(),
                nn.Conv1d(320,320,8),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool1d(4),
                nn.Conv1d(320,480,8),
                nn.ReLU(),
                nn.Conv1d(480,480,8),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool1d(4),
                nn.Conv1d(480,640,8),
                nn.ReLU(),
                nn.Conv1d(640,640,8),
                nn.ReLU(),
            )
        self.linear = nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,num_tasks)),
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x

class QueryNet(nn.Module):
    def __init__(self):
        super(QueryNet, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv1d(4,320,8),
                nn.ReLU(),
                nn.Conv1d(320,320,8),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool1d(4),
                nn.Conv1d(320,480,8),
                nn.ReLU(),
                nn.Conv1d(480,480,8),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool1d(4),
                nn.Conv1d(480,640,8),
                nn.ReLU(),
                nn.Conv1d(640,640,8),
                nn.ReLU(),
            )
        self.linear = nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,512)),
            )
        wv = np.load('/data3/yuanenming/project/deepsea/data/word2vec.npy')
        self.register_buffer('query', torch.from_numpy(wv))
        self.map = nn.Linear(256, 512)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        wv = self.map(self.query)
        x = torch.einsum('bd, cd -> bc', x, wv)
        return x

class EnformerConvMMOE(nn.Module):
    def __init__(self,
            dim=1024,
            num_downsamples=8,
            dropout=0.2,
            num_tasks = 2002,
            ) -> None:
        super().__init__()
        self.dim = dim
        init_dim = dim // 8

        self.stem = nn.Sequential(
            nn.Conv1d(4, init_dim, 15, padding = 7),
            Residual(ConvBlock(init_dim)),
            AttentionPool(init_dim, pool_size=2),
        )

        filter_list = exponential_linspace_int(init_dim, dim, num = (num_downsamples - 1))
        filter_list = [init_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size = 5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size=2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)

        self.final = nn.Sequential(
            nn.Conv1d(self.dim, 2048, 8),
            Squeeze(),
            MMOE(2048, dropout = dropout, num_tasks=num_tasks)
            )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.conv_tower(x)
        x = self.final(x)
        return x