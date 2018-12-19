import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Conv1(nn.Module):
    def __init__(self, nin, nout, affine=True):
        super().__init__()
        self.conv1 = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.conv2 = nn.Conv2d(nin, nout, kernel_size=1)
        self.bn = nn.BatchNorm2d(nout, affine=affine)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv1(x) 
        x = self.conv2(x)
        x = self.bn(x)
        return x
    

class Conv2(nn.Module):
    def __init__(self, nin, nout, affine=True):
        super().__init__()
        self.conv1 = nn.Conv2d(nin, nin, kernel_size=5, padding=2, groups=nin)
        self.conv2 = nn.Conv2d(nin, nout, kernel_size=1)
        self.bn = nn.BatchNorm2d(nout, affine=affine)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv1(x) 
        x = self.conv2(x)
        x = self.bn(x)
        return x


# def max_pool():
#     max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#     return max_pool


# def avg_pool():
#     avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
#     return avg_pool


class ReductionBlock(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.out1 = nn.Conv2d(nin, nout, kernel_size=3, stride = 2, padding=1)
        self.out2 = nn.Conv2d(nin, nout, kernel_size=1, stride = 2)
    
    def forward(self, x):
        x1 = self.out1(x)
        x2 = self.out2(x)
        x_out = torch.add(x1, x2)
        return x_out

# class ConvBlock(nn.Module):
#     def __init__(self, x, M, N, ch):
#         super().__init__()
#         self.x = x
#         self.edge = {_: [[] for t in range(N)] for _ in range(M)}  # lamda
#         self.node = {_: [[] for t in range(N)] for _ in range(M)}  # operation output
#         self.out_edge = []
#         self.out_node = []
#         self.level_input = {_: [] for _ in range(M)}  # 해당 level의 operation마다 들어가는 input
#         self.level_op = {_: [conv_1(ch, ch).to(device), 
#                             conv_2(ch, ch).to(device), 
#                             conv_1(ch, ch).to(device),
#                             conv_2(ch, ch).to(device)] for _ in range(M)}
# #         self.node, self.edge = self._input(self.node, self.edge)
#         self._init_input(self.node, self.edge, M, N, self.x)
#         self.out_tensor = 0
#         self.out_tensor = self._cal_out(M,N,
#                                         self.level_op,
#                                         self.level_input, 
#                                         self.node, 
#                                         self.edge, 
#                                         self.out_node, 
#                                         self.out_edge, 
#                                         self.out_tensor)
#         self.out = torch.add(self.out_tensor, self.x)  # reduce_sum 괜찮은가..
#         # 먼저 input을 모든 레이어 입력으로 
#     def _init_input(self, node, edge, M, N, input_x):
#         for p in range(N):
#             for n in range(M):    
#                 node[n][p].append(input_x)
#                 edge[n][p].append(edge_weight)
#         #return node, edge
    

#     def make_level_input(self, node, edge):
#         sum_tensor = torch.mul(node[0], edge[0]).unsqueeze(0)
#         for i in range(1, len(edge)):
#             tmp = torch.mul(node[i], edge[i]).unsqueeze(0)
#             sum_tensor = torch.cat((sum_tensor, tmp), 0)
#         return torch.sum(sum_tensor, 0) 
    

#     def _cal_out(self, M,N,level_op, level_input, node, edge, out_node, out_edge, out_tensor):
#         for m in range(M):  # level마다
#             for i in range(N):  # operation마다
#                 # node와 edge 곱해서 합쳐주는 작업
#                 op_input = self.make_level_input(node[m][i], edge[m][i])
#                 level_input[m].append(op_input)

#                 node_input = level_input[m][i]  # operation에 들어갈 list
#                 node_op = level_op[m][i]  # 해당 operation
#                 node_out = node_op(node_input)

#                 for _ in range(N):  # 남은 edge, node    
#                     for j in range(m+1, M):
#                         node[j][i].append(node_out)
#                         edge[j][i].append(edge_weight)

#                 out_node.append(node_out)
#                 out_edge.append(edge_weight)
#         out_tensor = self.make_level_input(out_node, out_edge) # 마지막 것만 생성..output
#         return out_tensor

