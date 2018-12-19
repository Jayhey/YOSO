from ConvBlock import Conv1, Conv2, ReductionBlock
from YOSO_module import *
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys

S, B, M, N = 4, 8, 4, 2
filters = [32, 64, 128, 256, 512]
batch_size = 64
op_epochs = 120
edge_epochs = 120

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() 
                                                     else torch.FloatTensor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
op_train_dataset, edge_train_dataset = torch.utils.data.random_split(trainset, [25000, 25000])
op_trainloader = torch.utils.data.DataLoader(op_train_dataset, batch_size=batch_size,shuffle=True)
edge_trainloader = torch.utils.data.DataLoader(edge_train_dataset, batch_size=batch_size,shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)


edge_weight = Variable(torch.ones(1), requires_grad=True).to(device)

op = [[[[Conv1(filters[i],filters[i]).to(device), Conv2(filters[i],filters[i]).to(device)] 
        for _ in range(M)] 
       for _ in range(B)]
      for i in range(S)]
reduction_block = [ReductionBlock(filters[i], filters[i+1]).to(device) for i in range(S-1)]
input_conv = nn.Conv2d(3, 32, 3, padding=1).to(device)
fc = nn.Linear(filters[S-1], 10).to(device)

edge = make_block_edge(edge_weight, S, B, M, N)
out_edge = [[[edge_weight] * (M * N) for _ in range(B)] for _ in range(S)] 


op_params, edge_params = set_params(S, B, M, N, op, edge, out_edge, 
                                    reduction_block, input_conv, fc)

criterion = nn.CrossEntropyLoss()
op_optimizer = optim.SGD(op_params, lr=0.0001, momentum=0.9, weight_decay=4e-05)
edge_optimizer = optim.SGD(edge_params, lr=0.00001, momentum=0.9, weight_decay=0.0001)

for epoch in range(op_epochs):
    print("\r-------------------- Epoch {} --------------------".format(epoch+1))
    for l, data in enumerate(op_trainloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        logit = model_forward(images, input_conv, S, B, M, N, 
                              op, edge, out_edge, reduction_block, fc)

        # y_prob = F.softmax(logit)
        op_optimizer.zero_grad()
        edge_optimizer.zero_grad()
        loss = criterion(logit, labels)
        loss.backward()
        op_optimizer.step()
        sys.stdout.write("\rTraining Iter : {}, Loss : {:.4f}".format(l ,loss.item()))
        sys.stdout.flush()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            logit = model_forward(images, input_conv, S, B, M, N, 
                                  op, edge, out_edge, reduction_block, fc)
            _, predicted = torch.max(logit, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('\rTotal test accuracy: %d %%' % (100 * correct / total))
print("")

lr = edge_optimizer.defaults['lr']
print("\rTraining lamda ")
for epoch in range(edge_epochs):
    print("\r-------------------- Epoch {} --------------------".format(epoch+1))
    mom = [torch.tensor(1.)] * len(edge_params)
    for l, data in enumerate(edge_trainloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        logit = model_forward(images, input_conv, S, B, M, N, 
                              op, edge, out_edge, reduction_block, fc)

        op_optimizer.zero_grad()
        edge_optimizer.zero_grad()
        loss = criterion(logit, labels)
        loss.backward()
        for i, param in enumerate(edge_params):
            param = apg_updater(param, lr, param.grad, mom[i], gamma=0.1)
            mom[i] = param
        
        
        edge_optimizer.step()
        sys.stdout.write("\rTraining Iter : {}, Loss : {:.4f}".format(l ,loss.item()))
        sys.stdout.flush()
