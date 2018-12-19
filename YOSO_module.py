import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() 
                                                     else torch.FloatTensor)

def make_block_edge(edge_weight, S, B, M, N):
    edge = [[[] for t in range(N)] for _ in range(M)]
    for m in range(M):  # level마다
        for i in range(N):  # operation마다
            edge[m][i].append(edge_weight)
            for _ in range(N):  # 남은 edge, node    
                for j in range(m+1, M):
                    edge[j][i].append(edge_weight)
    return [[edge for _ in range(B)] for _ in range(S)]


def set_params(S, B, M, N, op, edge, out_edge, reduction_block, input_conv, fc):
    op_params = []
    edge_params = []
    for i in range(S): 
        for j in range(B): 
            for k in range(M): 
                for l in range(N):
                    op_params += op[i][j][k][l].parameters()
                    for m in range(len(edge[i][j][k][l])):
                        edge_params += [edge[i][j][k][l][m]]

            for m in range(M*N):
                edge_params += [out_edge[i][j][m]]

    for i in range(S-1):
        op_params += reduction_block[i].parameters()
    op_params += input_conv.parameters()
    op_params += fc.parameters()

    return op_params, edge_params


def apg_updater(weight, lr, grad, mom, gamma=0.01):
    z = weight - lr * grad
    def soft_thresholding(x, gamma):
        y = torch.max(torch.tensor(0.), torch.abs(x) - torch.tensor(gamma))
        return torch.sign(x) * y   
    z = soft_thresholding(z, lr * gamma)
    mom = z - weight + 0.9 * mom
    weight = z + 0.9 * mom
    return weight


def make_level_input(node, edge):
    sum_tensor = torch.mul(node[0], edge[0]).unsqueeze(0)
    for i in range(1, len(edge)):
        tmp = torch.mul(node[i], edge[i]).unsqueeze(0)
        sum_tensor = torch.cat((sum_tensor, tmp), 0)
    return torch.sum(sum_tensor, 0) 


def conv_block(inputs, M, N, block_op, block_edge, block_out_edge):
    block_node = [[[] for t in range(N)] for _ in range(M)]
    block_out_node = []
    block_input = [[] for _ in range(M)]  # 해당 level의 operation마다 들어가는 input
    
    for p in range(N):
        for n in range(M):    
            block_node[n][p].append(inputs)
    
    for m in range(M):  # level마다
        for i in range(N):  # operation마다
            # node와 edge 곱해서 합쳐주는 작업
            op_input = make_level_input(block_node[m][i], block_edge[m][i])
            block_input[m].append(op_input)

            node_input = block_input[m][i]  # operation에 들어갈 list
            node_out = block_op[m][i](node_input)

            for _ in range(N):  # 남은 edge, node    
                for j in range(m+1, M):
                    block_node[j][i].append(node_out)
            block_out_node.append(node_out)

    out_tensor = make_level_input(block_out_node, block_out_edge) # 마지막 것만 생성..output
    out = torch.add(out_tensor, inputs)
    return out


def model_forward(images, input_conv, S, B, M, N, op, edge, out_edge, reduction_block, fc):
    input_x = input_conv(images)
    inputs = input_x
    for i in range(S):
        for j in range(B):
            block = conv_block(inputs=inputs, 
                               M=M, 
                               N=N, 
                               block_op=op[i][j],
                               block_edge=edge[i][j],
                               block_out_edge=out_edge[i][j])
            inputs = block
        if i < S-1:
            reduction = reduction_block[i]
            inputs = reduction(inputs)
    out=inputs
    last = nn.MaxPool2d(kernel_size=out.size(-1))(out)
    flat = last.view(last.size(0), -1)  
    logit = fc(flat)
    return logit



'''
예전 작업
# For one block
M = 3  # levels
N = 2  # operations
ch = 32

edge = {_: [[] for t in range(N)] for _ in range(M)}  # lamda
node = {_: [[] for t in range(N)] for _ in range(M)}  # operation output
out_edge = []
out_node = []
level_input = {_: [] for _ in range(M)}  # 해당 level의 operation마다 들어가는 input
level_op = {_: [conv_1(ch, ch), conv_2(ch, ch), max_pool(), avg_pool()] for _ in range(M)} # operaion 2개 기준

# out_edge = [torch.tensor(1).float() for _ in range(M*N)]
# out_node = []

# 먼저 input을 모든 레이어 입력으로 
for p in range(N):
    for n in range(M):    
        node[n][p].append(input_x)
        edge[n][p].append(torch.tensor(1).float())


def make_level_input(node, edge):
    sum_tensor = torch.mul(node[0], edge[0]).unsqueeze(0)
    for i in range(1, len(edge)):
        tmp = torch.mul(node[i], edge[i]).unsqueeze(0)
        sum_tensor = torch.cat((sum_tensor, tmp), 0)
    return torch.sum(sum_tensor, 0)
        
    
for m in range(M):
    for i in range(N):  # operation마다
        # node와 edge 곱해서 합쳐주는 작업
        op_input = make_level_input(node[m][i], edge[m][i])
        level_input[m].append(op_input)
        
        node_input = level_input[m][i]  # operation에 들어갈 list
        node_op = level_op[m][i]  # 해당 operation
        node_out = node_op(node_input)
        
        for _ in range(N):  # 남은 edge, node    
            for j in range(m+1, M):
                node[j][i].append(node_out)
                edge[j][i].append(torch.tensor(1).float())
        
        out_node.append(node_out)
        out_edge.append(torch.tensor(1).float())
        
out_tensor = make_level_input(out_node, out_edge)


class conv_2(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=5, padding=2, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class conv_2(nn.Module):
    def __init__(self, nin, nout, affine=True):
        super().__init__()
        self.op = nn.Sequential(nn.Conv2d(nin, nin, kernel_size=5, padding=2, groups=nin),
                                nn.Conv2d(nin, nout, kernel_size=1),
                                nn.BatchNorm2d(nout, affine=affine),
                                nn.ReLU(inplace=False)
                                )

    def forward(self, x):
        return self.op(x)


'''