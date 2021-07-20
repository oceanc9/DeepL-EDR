import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import csv
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
import torch, time, csv, random, os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from torch.optim import lr_scheduler
import seaborn as sns

def setup_seed(seed):       # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SKConv(nn.Module):   # SKNet 加权多种卷积核
    def __init__(self, features,planes, WH, M, G, r, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(features,
                          planes,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          groups=G),
                nn.ReLU(inplace=False)))
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(features,
                          planes,
                          kernel_size=(3,1),
                          stride=stride,
                          padding=(1,0),
                          groups=G),
                nn.ReLU(inplace=False)))
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(features,
                          planes,
                          kernel_size=1,
                          stride=stride,
                          groups=G),
                nn.ReLU(inplace=False)))
        self.fc = nn.Linear(planes, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

class ChannelAttention(nn.Module):   # 通道注意力机制
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
class SpatialAttention1(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention1, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
class BasicBlock1(nn.Module):  # 第一部分结构（1×1卷积核，ResNet结构）
    expansion = 1
    def __init__(self, in_planes, planes, stride):
        super(BasicBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):

        out = torch.sigmoid(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out) * out
        s_a = self.sa(out)
        out = self.sa(out) * out  # 空间广播机制
        out += self.shortcut(x)
        out = torch.sigmoid(out)

        return out,s_a
class BasicBlock3(nn.Module):  # 第一部分结构（1×1卷积核，ResNet结构）
    expansion = 1
    def __init__(self, in_planes, planes, stride):
        super(BasicBlock3, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention1()
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):

        out = torch.sigmoid(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out) * out
        out = self.sa(out) * out  # 空间广播机制
        out += self.shortcut(x)
        out = torch.sigmoid(out)

        return out
class BasicBlock0(nn.Module):  # 第一部分结构（1×1卷积核，ResNet结构）
    expansion = 1
    def __init__(self, in_planes, planes, stride):
        super(BasicBlock0, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ca = ChannelAttention(53)
        self.sa = SpatialAttention()
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.ca(x) * x
        out = torch.sigmoid(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        s_a = self.sa(out)
        out = self.sa(out) * out  # 空间广播机制
        out += self.shortcut(x)
        out = torch.sigmoid(out)

        return out, s_a
# 第二部分结构（提取相邻牙位信息）
class BasicBlock2(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride):
        super(BasicBlock2, self).__init__()
        self.conv1 = SKConv(in_planes,planes, None, 3, 8, 2)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = torch.sigmoid(self.bn1(self.conv1(x)))
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 53
        self.layer1 = self._make_layer(block[3], 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block[1], 32, num_blocks[1], stride=1)
        self.in_planes = 53
        self.layer3 = self._make_layer(block[2], 64, num_blocks[0], stride=1)
        self.layer4 = self._make_layer(block[1], 32, num_blocks[2], stride=1)
        self.in_planes = 53
        self.layer5 = self._make_layer(block[0], 64, num_blocks[1], stride=1)
        self.layer6 = self._make_layer(block[1], 32, num_blocks[1], stride=1)
        self.in_planes = 53
        self.layer7 = self._make_layer(block[0], 64, num_blocks[1], stride=1)
        self.layer8 = self._make_layer(block[1], 32, num_blocks[1], stride=1)
        self.in_planes = 53
        self.layer9 = self._make_layer(block[3], 64, num_blocks[0], stride=1)
        self.layer10 = self._make_layer(block[1], 32, num_blocks[1], stride=1)



        self.feature0 = nn.AvgPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.feature4 = nn.AvgPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.lstm = nn.GRU(32, 16, 1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Sequential(nn.Linear( 32, 2))
        self.fc2 = nn.Sequential(nn.Linear( 32, 2))
        self.fc3 = nn.Sequential(nn.Linear( 32, 2))
        self.embed0 = nn.Sequential(torch.nn.Linear(2, 2),nn.Sigmoid())
        self.embed4 = nn.Sequential(torch.nn.Linear(2, 2),nn.Sigmoid())
        self.fc0 = nn.Sequential(nn.Linear(258, 32),nn.Sigmoid())
        self.fc4 = nn.Sequential(nn.Linear(258, 32),nn.Sigmoid())
        self.fc00 = nn.Linear(32, 2)
        self.fc44 = nn.Linear(32, 2)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, unrela_x):
        x = x.permute(0, 3, 2, 1)
        out1 = self.layer2(self.layer1(x)).permute(0, 3, 2, 1).reshape(-1, 32)
        out2, sa_1 = self.layer3(x)
        out2 = self.layer4(out2).permute(0, 3, 2, 1).reshape(-1, 32)
        out3, sa_2 = self.layer5(x)
        out3 = self.layer6(out3).permute(0, 3, 2, 1).reshape(-1, 32)
        out4, sa_3 = self.layer7(x)
        out4 = self.layer8(out4).permute(0, 3, 2, 1).reshape(-1, 32)
        out5 = self.layer10(self.layer9(x)).permute(0, 3, 2, 1).reshape(-1, 32)
        output = torch.stack((out1, out2, out3, out4, out5))
        output = output.permute(1, 0, 2)
        h0 = torch.rand(1*2, output.size(0), 16).cuda()  # 同样考虑向前层和向后层
        # c0 = torch.rand(1, output.size(0), 2).cuda()
        output, _ = self.lstm(output,  h0)  # LSTM输出大小为 (batch_size, seq_length, hidden_size*2)
        output = output.permute(1, 0, 2)
        # 牙周治疗
        output0 = output[0].reshape(x.shape[0], 2, 16, 32).permute(0, 3, 2, 1)
        output0 = self.feature0(output0).reshape(output0.shape[0], 32 * 4 *2)
        embedding = self.embed0(unrela_x[:,:2])
        output0 = self.fc00(self.fc0(torch.cat((output0, embedding), 1)))
        # 拆除旧修复体
        output1 = self.fc1(output[0])
        # 拔除
        output2 = self.fc2(output[1])
        # 牙体牙髓
        output3 = self.fc3(output[2])
        # 修复科复诊
        output4 = output[4].reshape(x.shape[0], 2, 16, 32).permute(0, 3, 2, 1)
        output4 =self.feature4(output4).reshape(output4.shape[0], 32 * 4 *2)
        embedding = self.embed4(unrela_x[:,:2])
        output4 = self.fc44(self.fc4(torch.cat((output4, embedding), 1)))
        # return output0, output1, output2, output3, output4
        if self.training is False:
            return sa_1, sa_2, sa_3
        else:
            return output0, output1, output2, output3, output4

def CNN_RNN():
    return ResNet([BasicBlock1,BasicBlock2,BasicBlock0,BasicBlock3], [1, 1, 3])



def load_train_data(train_file,label, reprepare=False):
    data_path = 'DATA/'
    if os.path.exists(data_path + label[0]+ '_' + 'train.pth') and not reprepare:
        x_train,unrela_x_train, y_train, valid_train, case_train = torch.load( data_path + label[0]+ '_' + 'train.pth')
    else:
        x_train,unrela_x_train, y_train, valid_train, case_train = prepare_data(train_file,label)
        torch.save([x_train,unrela_x_train, y_train, valid_train, case_train], data_path + label[0]+ '_' + 'train.pth')
    return x_train,unrela_x_train, y_train, valid_train, case_train

def load_test_data(test_file,label, reprepare=False):
    data_path = 'DATA/'
    if os.path.exists( data_path + label[0]+ '_' + 'test.pth') and not reprepare:
        x_test, unrela_x_test, y_test, valid_test, case_test = torch.load( data_path + label[0]+ '_' + 'test.pth')
    else:
        x_test, unrela_x_test, y_test, valid_test, case_test = prepare_data(test_file,label)
        torch.save([x_test, unrela_x_test, y_test, valid_test, case_test],  data_path + label[0]+ '_' + 'test.pth')
    return x_test, unrela_x_test, y_test, valid_test, case_test

train_file = 'train_NEW_All_4.csv'
test_file = 'test_NEW_All_4.csv'
data_czx = pd.read_csv(train_file)
head_czx = list(data_czx.columns)

label0 = ['牙周治疗']
label1 = ['拆除旧修复体']
label2 = ['拔除']  #
label3 = ['牙体牙髓科治疗']
label4 = ['修复科复诊']

y_train0 = load_train_data(train_file,label0)[2]
y_test0 = load_test_data(test_file,label0)[2]
x_train,unrela_x_train, y_train1, valid_train, case_train = load_train_data(train_file,label1)
x_test,unrela_x_test,  y_test1, valid_test, case_test = load_test_data(test_file,label1)
y_train2 = load_train_data(train_file,label2)[2]
print(y_train2.shape)
for i in range(y_train2.shape[0]):
    for j in range(y_train2.shape[1]):
        for h in  range(y_train2.shape[2]):
            if y_train2[i,j,h] > 1:
                y_train2[i,j,h] = 1
y_test2 = load_test_data(test_file,label2)[2]
for i in range(y_test2.shape[0]):
    for j in range(y_test2.shape[1]):
        for h in  range(y_test2.shape[2]):
            if y_test2[i,j,h] > 1:
                y_test2[i,j,h] = 1

y_train3 = load_train_data(train_file,label3)[2]
y_test3 = load_test_data(test_file,label3)[2]
for i in range(y_train3.shape[0]):
    for j in range(y_train3.shape[1]):
        for h in  range(y_train3.shape[2]):
            if y_train3[i,j,h] > 1:
                y_train3[i,j,h] = 1
for i in range(y_test3.shape[0]):
    for j in range(y_test3.shape[1]):
        for h in  range(y_test3.shape[2]):
            if y_test3[i,j,h] > 1:
                y_test3[i,j,h] = 1
y_train4 = load_train_data(train_file,label4)[2]
y_test4 = load_test_data(test_file,label4)[2]

train_size = x_train.shape[0]
test_size = x_test.shape[0]
# print("这个tensor是什么样子的呢",x_train.shape)
# print("这个tensor是什么样子的呢",x_train[1][1][1,:].shape)
# print("这个tensor是什么样子的呢",x_train[:,:,:,1].shape)
# print("这个tensor是什么样子的呢",x_train[:,:,:,3].shape)

# print(x_train.shape)
h_state = None
n_label = 2
def penalty_l1(model, beta):
    l1_loss = torch.tensor(0.0, requires_grad=True).float().cuda()
    for name, param in model.named_parameters():
        if param.requires_grad and 'bias' not in name:
            l1_loss = l1_loss + beta * torch.sum(torch.abs(param))
    return l1_loss
idx_train = []
for j, j_tuple in enumerate(valid_train):
    if len(j_tuple) > 0: idx_train.extend([j * 32 + k for k in j_tuple])
idx_test = []
for j, j_tuple in enumerate(valid_test):
    if len(j_tuple) > 0: idx_test.extend([j * 32 + k for k in j_tuple])

setup_seed(0)


model = CNN_RNN().cuda()
model_dict = torch.load('cnn_rnn.params')

model.load_state_dict(model_dict, strict=False)
xianzhuxing_0 = []
xianzhuxing_1 = []
xianzhuxing_2 = []
xianzhuxing_3 = []
xianzhuxing_4 = []
# print("牙位无关的特征",unrela_x_train.shape)
# model.eval()
with torch.no_grad():
    for i in range(53):  ###到底有多少个特征呢？这个输入是不是何之前的不一样？
    #  if head_czx[i+3] == "缺失(is missing)":
        x_train1 = x_train  + torch.zeros_like(x_train)
        x_train2 = x_train - torch.zeros_like(x_train)
        x_train1[:,:,:,i] = x_train[:,:,:,i] - 0.1*x_train[:,:,:,i]
        x_train2[:,:,:,i] = x_train[:,:,:,i] + 0.1*x_train[:,:,:,i]
        # for m in range()
        
        # x_train1[:,:,:,i] = x_train[:,:,:,i] 
        # x_train2[:,:,:,i] = x_train[:,:,:,i] + 1   

        # print("训练集的shape是什么",x_train.shape)
        print("没有哪里出错吧",x_train2[:,:,:,i].shape)
        print("到底结果错哪里了",torch.sum(x_train2 - x_train1))
    # train
        
        output0, output1, output2, output3, output4 = model(x_train1,unrela_x_train)# output shape [size,32, 1]
        
        ##用max得到的结果。
        # output0s = torch.max(output0, dim=1)[1].data.cpu().numpy()
        # output1s = torch.max(output1, dim=1)[1].data.cpu().numpy()
        # output2s = torch.max(output2, dim=1)[1].data.cpu().numpy()
        # output3s = torch.max(output3, dim=1)[1].data.cpu().numpy()
        # output4s = torch.max(output4, dim=1)[1].data.cpu().numpy()
        # output1s = torch.softmax(output1, dim=1)[:, 1].data.cpu().numpy()

        #用softmax得到的结果。
        output0s = torch.softmax(output0, dim=1)[:,1].data.cpu().numpy()
        output1s = torch.softmax(output1, dim=1)[:,1].data.cpu().numpy()
        output2s = torch.softmax(output2, dim=1)[:,1].data.cpu().numpy()
        output3s = torch.softmax(output3, dim=1)[:,1].data.cpu().numpy()
        output4s = torch.softmax(output4, dim=1)[:,1].data.cpu().numpy()

        # print("output1s的形式",output1s.shape)
        # print("output1s的样子",output1s)
        # output2s = torch.softmax(output2, dim=1)[:, 1].data.cpu().numpy()
        # print("output2s的形式",output2s.shape)
        

        
        
        
        _output0, _output1, _output2, _output3, _output4 = model(x_train2,unrela_x_train)
        
        ###用max得到的结果
        # _output0s = torch.max(_output0, dim=1)[1].data.cpu().numpy()
        # _output1s = torch.max(_output1, dim=1)[1].data.cpu().numpy()
        # _output2s = torch.max(_output2, dim=1)[1].data.cpu().numpy()
        # _output3s = torch.max(_output3, dim=1)[1].data.cpu().numpy()
        # _output4s = torch.max(_output4, dim=1)[1].data.cpu().numpy()


        ##用softmax得到的结果
        _output0s = torch.softmax(_output0, dim=1)[:,1].data.cpu().numpy()
        _output1s = torch.softmax(_output1, dim=1)[:,1].data.cpu().numpy()
        _output2s = torch.softmax(_output2, dim=1)[:,1].data.cpu().numpy()
        _output3s = torch.softmax(_output3, dim=1)[:,1].data.cpu().numpy()
        _output4s = torch.softmax(_output4, dim=1)[:,1].data.cpu().numpy()
        
        # if head_czx[i+3] == "缺失(is missing)":
        #     for k in range(len(output0s)):
                    
        #             # print("看看有什么变化罗",i,output2s[k],_output2s[k])
        #             if (output2s[k] != _output2s[k]):
        #                 print("这个有变化哟",k,output2s[k],_output2s[k])
        #     print("所以最后缺失的影响到底是正还是负：",sum(_output2s-output2s)/len(output2s))
        
        
        # _output1s = torch.softmax(_output1, dim=1)[:, 1].data.cpu().numpy()
        # print("softmax之后到底是什么啊",_output1s[20:40])
        
        # print("第",i,"个特征对",head_czx[i+3],"第5个节点的输出的影响值",(sum(_output1s-output1s)/len(output1s)))
        temp_0 = [head_czx[i+3],(sum(_output0s-output0s)/len(output0s))]
        xianzhuxing_0.append(temp_0)

        temp_1 = [head_czx[i+3],(sum(_output1s-output1s)/len(output1s))]
        xianzhuxing_1.append(temp_1)

        temp_2 = [head_czx[i+3],(sum(_output2s-output2s)/len(output2s))]
        xianzhuxing_2.append(temp_2)

        temp_3 = [head_czx[i+3],(sum(_output3s-output3s)/len(output3s))]
        xianzhuxing_3.append(temp_3)

        temp_4 = [head_czx[i+3],(sum(_output4s-output4s)/len(output4s))]
        xianzhuxing_4.append(temp_4)
    for j in range(2):
        unrela_x_train1 = unrela_x_train + torch.zeros_like(unrela_x_train)
        unrela_x_train2 = unrela_x_train - torch.zeros_like(unrela_x_train)
        unrela_x_train1[:,j] = unrela_x_train1[:,j] - 0.1*unrela_x_train1[:,j]
        unrela_x_train2[:,j] = unrela_x_train2[:,j] + 0.1*unrela_x_train2[:,j]

        output0, output1, output2, output3, output4 = model(x_train,unrela_x_train1)# output shape [size,32, 1]
        
        #max对应的结果
        # output0s = torch.max(output0, dim=1)[1].data.cpu().numpy()
        # output1s = torch.max(output1, dim=1)[1].data.cpu().numpy()
        # output2s = torch.max(output2, dim=1)[1].data.cpu().numpy()
        # output3s = torch.max(output3, dim=1)[1].data.cpu().numpy()
        # output4s = torch.max(output4, dim=1)[1].data.cpu().numpy()

        #softmax对应的结果
        output0s = torch.softmax(output0, dim=1)[:,1].data.cpu().numpy()
        output1s = torch.softmax(output1, dim=1)[:,1].data.cpu().numpy()
        output2s = torch.softmax(output2, dim=1)[:,1].data.cpu().numpy()
        output3s = torch.softmax(output3, dim=1)[:,1].data.cpu().numpy()
        output4s = torch.softmax(output4, dim=1)[:,1].data.cpu().numpy()

        # output1s = torch.softmax(output1, dim=1)[:, 1].data.cpu().numpy()
        # print("output1s的形式",output1s.shape)
        # print("output1s的样子",output1s)
        # output2s = torch.softmax(output2, dim=1)[:, 1].data.cpu().numpy()
        # print("output2s的形式",output2s.shape)
        

        
        
        
        _output0, _output1, _output2, _output3, _output4 = model(x_train,unrela_x_train2)
       
       #max对应的结果
        # _output0s = torch.max(_output0, dim=1)[1].data.cpu().numpy()
        # _output1s = torch.max(_output1, dim=1)[1].data.cpu().numpy()
        # _output2s = torch.max(_output2, dim=1)[1].data.cpu().numpy()
        # _output3s = torch.max(_output3, dim=1)[1].data.cpu().numpy()
        # _output4s = torch.max(_output4, dim=1)[1].data.cpu().numpy()_output0s = torch.softmax(_output0, dim=1)[:,1].data.cpu().numpy()
        
        #softmax对应的结果
        _output0s = torch.softmax(_output0, dim=1)[:,1].data.cpu().numpy()
        _output1s = torch.softmax(_output1, dim=1)[:,1].data.cpu().numpy()
        _output2s = torch.softmax(_output2, dim=1)[:,1].data.cpu().numpy()
        _output3s = torch.softmax(_output3, dim=1)[:,1].data.cpu().numpy()
        _output4s = torch.softmax(_output4, dim=1)[:,1].data.cpu().numpy()


        
        ##softmax对应的结果

        # print("第",i,"个特征对","第5个节点的输出的影响值",(sum(_output4s-output4s)/len(output4s)))
        temp_0 = [j,(sum(_output0s-output0s)/len(output0s))]
        xianzhuxing_0.append(temp_0)

        temp_1 = [j,(sum(_output1s-output1s)/len(output1s))]
        xianzhuxing_1.append(temp_1)

        temp_2 = [j,(sum(_output2s-output2s)/len(output2s))]
        xianzhuxing_2.append(temp_2)

        temp_3 = [j,(sum(_output3s-output3s)/len(output3s))]
        xianzhuxing_3.append(temp_3)

        temp_4 = [j,(sum(_output4s-output4s)/len(output4s))]
        xianzhuxing_4.append(temp_4)

#依次是牙周治疗，拆除旧修复体，拔除，牙体牙髓，修复科复诊
xianzhuxing_0 = pd.DataFrame(xianzhuxing_0,index = None)
xianzhuxing_0.to_csv("显著性分析_牙周治疗_临时2.csv")

xianzhuxing_1 = pd.DataFrame(xianzhuxing_1,index = None)
xianzhuxing_1.to_csv("显著性分析_拆除旧修复体_临时2.csv")

xianzhuxing_2 = pd.DataFrame(xianzhuxing_2,index = None)
xianzhuxing_2.to_csv("显著性分析_拔除_临时2.csv")

xianzhuxing_3 = pd.DataFrame(xianzhuxing_3,index = None)
xianzhuxing_3.to_csv("显著性分析_牙体牙髓_临时2.csv")

xianzhuxing_4 = pd.DataFrame(xianzhuxing_4,index = None)
xianzhuxing_4.to_csv("显著性分析_修复科复诊_临时2.csv")






        # print("output0是什么形式的？",output0.shape)
        # print("output1是什么形式的？",output1.shape)
        # # outputs1 = outputs.reshape(-1, 2)
        # # print("outputs1是什么形式的？",outputs1)
        # print("变身前的label0是什么形式的？",y_train0.shape)
        # print("变身前的label1是什么形式的？",y_train1.shape)
        # labels1 = y_train.reshape(-1)  # [batch_size*32]  ##这个label是不一样的哦！
        # print("变身后的labels1的形式是什么样子的？",labels1.shape)
        # outputs_v1 = outputs1[idx_train, :]
        # labels_v1 = labels1[idx_train]
        # pred_y_train = torch.max(outputs_v1, dim=1)[1].data.cpu().numpy()
        # target_y_train = labels_v1.data.cpu().numpy()
        # pred_y_train1 = torch.softmax(outputs_v1, dim=1)[:, 1].data.cpu().numpy()
        # print("第一个已经好了")
        
        # outputs22 = model(x_train2,unrela_x_train)# output shape [batch_size*32, 2]
        # outputs2 = outputs22.reshape(-1, 2)
        # labels2 = y_train.reshape(-1)  # [batch_size*32]
        # outputs_v2 = outputs2[idx_train, :]
        # labels_v2 = labels2[idx_train]
        # pred_y_train = torch.max(outputs_v2, dim=1)[1].data.cpu().numpy()
        # target_y_train2 = labels_v2.data.cpu().numpy()
        # pred_y_train2 = torch.softmax(outputs_v2, dim=1)[:, 1].data.cpu().numpy()
        
        # cha_i = sum(pred_y_train2 - pred_y_train1)/len(pred_y_train1)
        # print("第",i,"个特征的显著性是",cha_i)
        # temp = [head_czx[i+3],cha_i]
        # xianzhuxing.append(temp)
# xianzhuxing = pd.DataFrame(xianzhuxing,index = None)
# xianzhuxing.to_csv("显著性分析_拔除_加减0.1.csv")


# spatialatt1,spatialatt2,spatialatt3 =  model(x_train,unrela_x_train) # 依次代表三个label的空间注意力系数

# print("spatialatt1",spatialatt1.shape,type(spatialatt1))
# print("end")
# plt.ion()

# for i in range(len(spatialatt1)):
#     spatial1_heatmap1 = spatialatt1[i].squeeze(0).permute(1,0).cpu().detach().numpy()  #[2,16] ##将x_train的第1份病例得到对应的spatial注意力机制
#     plt.figure(figsize=(16,6))
#     plt.subplot(321)
#     sns.heatmap(spatial1_heatmap1, vmax=1, vmin=0, linewidths=1, cmap='viridis', alpha=1,square=True)  #coolwarm hot viridis
#     plt.xlabel('Spatial Attention of Y1')

#     y1 = y_train1[i].reshape(2, 16).cpu().numpy()
#     # plt.figure(figsize=(16, 4))
#     plt.subplot(322)
#     sns.heatmap(y1, linewidths=1, vmax=1, vmin=0, cmap='viridis', alpha=0.8, square=True)
#     plt.xlabel('Y1')

#     spatial1_heatmap2 = spatialatt2[i].squeeze(0).permute(1, 0).cpu().detach().numpy()
#     plt.subplot(323)
#     sns.heatmap(spatial1_heatmap2, vmax=1, vmin=0, linewidths=1, cmap='viridis', alpha=1,square=True)  #coolwarm hot viridis
#     plt.xlabel('Spatial Attention of Y2')

#     y2 = y_train2[i].reshape(2, 16).cpu().numpy()
#     # plt.figure(figsize=(16, 4))
#     plt.subplot(324)
#     sns.heatmap(y2, linewidths=1, vmax=1, vmin=0, cmap='viridis', alpha=0.8, square=True)
#     plt.xlabel('Y2')

#     spatial1_heatmap3 = spatialatt3[i].squeeze(0).permute(1, 0).cpu().detach().numpy()
#     plt.subplot(325)
#     sns.heatmap(spatial1_heatmap3, vmax=1, vmin=0, linewidths=1, cmap='viridis', alpha=1,square=True)  #coolwarm hot viridis
#     plt.xlabel('Spatial Attention of Y3')

#     y3 = y_train3[i].reshape(2, 16).cpu().numpy()
#     # plt.figure(figsize=(16, 4))
#     plt.subplot(326)
#     sns.heatmap(y3, linewidths=1, vmax=1, vmin=0, cmap='viridis', alpha=0.8, square=True)
#     plt.xlabel('Y3')

#     plt.savefig('特征热力图//空间注意力机制热力图//第' + str(i) + '个病例的空间注意力机制热力图.czx.pdf')
#     plt.show()
#     plt.close()
#     #
#     # spatial2_heatmap = spatialatt2[i].squeeze(0).permute(1,0).detach().numpy()  #[2,16]
#     # plt.subplot(212)
#     # sns.heatmap(spatial2_heatmap, vmax=1, vmin=0, linewidths=1, cmap='viridis', alpha=1, square=True)
#     # plt.xlabel('Spatial Attention of Y2')
#     # plt.show()
#     # #
#     #

# plt.ioff()
#     # y0 = y_test[i].reshape(2, 16).cpu().numpy()
#     # plt.figure(figsize=(16,4))
#     # plt.subplot(211)
#     # sns.heatmap(y0, linewidths=1, vmax=1, vmin=0, cmap='viridis', alpha=0.8, square=True)
#     # plt.xlabel('Y1')
#     # plt.show()
# #
# # label2 = y_train1[i][:,0].reshape(2, 16).numpy()
# # label3 = y_train1[i][:,1].reshape(2, 16).numpy()
# # y1 = label2+label3
# # plt.subplot(212)
# # sns.heatmap(y1, linewidths=1, vmax=1, vmin=0, cmap='viridis', alpha=0.8,square=True)
# # plt.xlabel('Y2')
# # plt.show()



