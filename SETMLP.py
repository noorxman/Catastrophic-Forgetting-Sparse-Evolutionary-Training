from functools import reduce
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import utils
import model
import sparse_utils as sp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SETMLP(model.MLP):
    def __init__(self, input_size, output_size, hidden_size=400, hidden_layer_num=2,
                 hidden_dropout_prob=.5,
                 input_dropout_prob=.2, lamb_func=40, zeta=0.1, theta=1, epsilonHid1=50, epsilonHid2=50, ascTopologyChangePeriod=200, earlyStopTopologyChangeIteration = 1e9, affix=''):
        super().__init__(input_size, output_size, hidden_size=hidden_size, hidden_layer_num=hidden_layer_num,
                         hidden_dropout_prob=hidden_dropout_prob,
                         input_dropout_prob=input_dropout_prob, lamb_func=lamb_func)
        self.lastTopologyChangeCritic = False
        self.zeta = zeta
        self.theta = theta
        self.epsilon = epsilonHid1
        self.ascTopologyChangePeriod = ascTopologyChangePeriod
        self.earlyStopTopologyChangeIteration = earlyStopTopologyChangeIteration
        self.affix = affix
        # Layers
        self.l1 = nn.Linear(input_size, hidden_size)
        [self.noPar1, self.mask1] = sp.initializeEpsilonWeightsMask("actor first layer", epsilonHid1, input_size,
                                                                    hidden_size)
        self.torchMask1 = torch.from_numpy(self.mask1).float().to(device)
        self.l1.weight.data.mul_(torch.from_numpy(self.mask1).float())

        self.l2 = nn.Linear(hidden_size, hidden_size)
        [self.noPar2, self.mask2] = sp.initializeEpsilonWeightsMask("actor second layer", epsilonHid2, hidden_size,
                                                                    hidden_size)
        self.torchMask2 = torch.from_numpy(self.mask2).float().to(device)
        self.l2.weight.data.mul_(torch.from_numpy(self.mask2).float())

        self.l3 = nn.Linear(hidden_size, hidden_size)
        [self.noPar3, self.mask3] = sp.initializeEpsilonWeightsMask("actor second layer", epsilonHid2, hidden_size,
                                                                    hidden_size)
        self.torchMask3 = torch.from_numpy(self.mask3).float().to(device)
        self.l3.weight.data.mul_(torch.from_numpy(self.mask3).float())

        self.l4 = nn.Linear(hidden_size, output_size)

    @property
    def name(self):
        return (
            'SET-MLP'
            '-lambda{lamda}'
            '-in{input_size}-out{output_size}'
            '-h{hidden_size}x{hidden_layer_num}'
            '-dropout_in{input_dropout_prob}_hidden{hidden_dropout_prob}'
            '-ascTopologyChangePeriod{ascTopologyChangePeriod}'
            '-epsilon{epsilon}'
            '-zeta{zeta}'
            '-{affix}'

        ).format(
            lamda=self.lamb_func,
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            hidden_layer_num=self.hidden_layer_num,
            input_dropout_prob=self.input_dropout_prob,
            hidden_dropout_prob=self.hidden_dropout_prob,
            ascTopologyChangePeriod=self.ascTopologyChangePeriod,
            epsilon=self.epsilon,
            zeta=self.zeta,
            affix=self.affix,
        )

    def forward(self, x):
        self.layers = nn.ModuleList([
            self.l1, nn.ReLU(), #nn.Dropout(self.input_dropout_prob),
            self.l2, nn.ReLU(), #nn.Dropout(self.input_dropout_prob),
            self.l3, nn.ReLU(), #nn.Dropout(self.input_dropout_prob),
            self.l4
        ])
        return super().forward(x)

    def adapt_connectivity(self, total_it):
        # Adapt the sparse connectivity
        if total_it % self.ascTopologyChangePeriod == 0:
            if total_it > self.earlyStopTopologyChangeIteration:
                self.lastTopologyChangeCritic = True
            [self.mask1, ascStats1] = sp.changeConnectivitySET(self.l1.weight.data.cpu().numpy(),
                                                               self.noPar1, self.mask1,
                                                               self.zeta, self.theta,
                                                               self.lastTopologyChangeCritic, total_it)
            self.torchMask1 = torch.from_numpy(self.mask1).float().to(device)
            [self.mask2, ascStats2] = sp.changeConnectivitySET(self.l2.weight.data.cpu().numpy(),
                                                               self.noPar2, self.mask2,
                                                               self.zeta, self.theta,
                                                               self.lastTopologyChangeCritic, total_it)
            self.torchMask2 = torch.from_numpy(self.mask2).float().to(device)
            [self.mask3, ascStats3] = sp.changeConnectivitySET(self.l3.weight.data.cpu().numpy(),
                                                               self.noPar3, self.mask3,
                                                               self.zeta, self.theta,
                                                               self.lastTopologyChangeCritic, total_it)
            self.torchMask3 = torch.from_numpy(self.mask3).float().to(device)
            # self.ascStatsCritic.append([ascStats1, ascStats2, ascStats3])

        # Maintain the same sparse connectivity for critic
        self.l1.weight.data.mul_(self.torchMask1)
        self.l2.weight.data.mul_(self.torchMask2)
        self.l3.weight.data.mul_(self.torchMask3)
