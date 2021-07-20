# # -*- coding: UTF-8 -*-
import torch
import torch, time, csv, random, os
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
import torch.nn as nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import csv
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
'''Train CIFAR10/CIFAR100 with PyTorch.'''
import argparse
import os
from optimizers import (KFACOptimizer, EKFACOptimizer)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.network_utils import get_network
from utils.data_utils import get_dataloader



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
        out = self.sa(out) * out  # 空间广播机制
        out += self.shortcut(x)
        out = torch.sigmoid(out)

        return out
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
        out = self.sa(out) * out  # 空间广播机制
        out += self.shortcut(x)
        out = torch.sigmoid(out)

        return out
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
        self.layer9 = self._make_layer(block[3], 64, num_blocks[0], stride=1)
        self.layer10 = self._make_layer(block[1], 32, num_blocks[1], stride=1)
        for p in self.parameters():
            p.requires_grad = False
        self.in_planes = 53
        self.layer3 = self._make_layer(block[2], 64, num_blocks[0], stride=1)
        self.layer4 = self._make_layer(block[1], 32, num_blocks[2], stride=1)
        self.in_planes = 53
        self.layer5 = self._make_layer(block[0], 64, num_blocks[1], stride=1)
        self.layer6 = self._make_layer(block[1], 32, num_blocks[1], stride=1)
        self.in_planes = 53
        self.layer7 = self._make_layer(block[0], 64, num_blocks[1], stride=1)
        self.layer8 = self._make_layer(block[1], 32, num_blocks[1], stride=1)


        self.lstm = nn.GRU(32, 16, 1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Sequential(nn.Linear( 32, 2))
        self.fc2 = nn.Sequential(nn.Linear( 32, 2))
        self.fc3 = nn.Sequential(nn.Linear( 32, 2))
        self.embed0 = nn.Sequential(torch.nn.Linear(2, 2),nn.Sigmoid())
        self.embed4 = nn.Sequential(torch.nn.Linear(2, 2),nn.Sigmoid())
        self.fc0 = nn.Sequential(nn.Linear(1026, 2))
        self.fc4 = nn.Sequential(nn.Linear(1026, 2))

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
        out2 = self.layer4(self.layer3(x)).permute(0, 3, 2, 1).reshape(-1, 32)
        out3 = self.layer6(self.layer5(x)).permute(0, 3, 2, 1).reshape(-1, 32)
        out4 = self.layer8(self.layer7(x)).permute(0, 3, 2, 1).reshape(-1, 32)
        out5 = self.layer10(self.layer9(x)).permute(0, 3, 2, 1).reshape(-1, 32)
        output = torch.stack((out1, out2, out3, out4, out5))
        output = output.permute(1, 0, 2)
        h0 = torch.rand(1*2, output.size(0), 16).cuda()  # 同样考虑向前层和向后层
        # c0 = torch.rand(1, output.size(0), 2).cuda()
        output, _ = self.lstm(output,  h0)  # LSTM输出大小为 (batch_size, seq_length, hidden_size*2)
        output = output.permute(1, 0, 2)
        # 牙周治疗
        output0 = output[0].reshape(x.shape[0], 2, 16, 32).permute(0, 3, 2, 1)
        output0 = output0.reshape(output0.shape[0], 32 * 16 *2)
        embedding = self.embed0(unrela_x[:,:2])
        output0 = self.fc0(torch.cat((output0, embedding), 1))
        # 拆除旧修复体
        output1 = self.fc1(output[0])
        # 拔除
        output2 = self.fc2(output[1])
        # 牙体牙髓
        output3 = self.fc3(output[2])
        # 修复科复诊
        output4 = output[4].reshape(x.shape[0], 2, 16, 32).permute(0, 3, 2, 1)
        output4 = output4.reshape(output4.shape[0], 32 * 16 *2)
        embedding = self.embed4(unrela_x[:,:2])
        output4 = self.fc4(torch.cat((output4, embedding), 1))
        return output0, output1, output2, output3, output4

def ResNet50():
    return ResNet([BasicBlock1,BasicBlock2,BasicBlock0,BasicBlock3], [1, 1, 3])

# 准备数据
def prepare_data(file_path,label):               # 预处理
    df = pd.read_csv(file_path, header=0)
    df.fillna(0, inplace=True)
    rela_treatment = ['拔除', '牙体牙髓科治疗', '可摘局部义齿修复', '种植修复', '即刻义齿修复',
       '桩核冠修复', '固定义齿修复', '拆除旧修复体', '旧义齿修理', '停戴旧义齿', '组织调整材料重衬', '截冠覆盖',
       '全口义齿修复', '系带修整术', '冠修复', '磁性附着体修复', '修复设计', '试保留但预后差', '可摘局部义齿调改',
       '制作诊断蜡型', '重新修复', '冠延长', '种植手术', '上颌窦内提升', '粘接脱落修复体', '拍片', '调合',
       '外科治疗', '软合垫治疗', '取研究模型'] # '拔除',
    unrela_treatment = ['牙周治疗''修复科复诊' ]
    attr_case = ['病例号', '就诊id', '牙位']
    rela_features = ['合龈间隙的距离(occluso gingival space)_离散',
       '近远中间隙的距离(mesio distal space distance)_离散',
       '近远中间隙的距离(mesio distal space distance)_连续',
       '龈退缩(gingival recession)_离散', '龈退缩(gingival recession)_连续',
       '覆盖(上颌牙切缘到下牙唇面的水平距离不大于3mm覆盖正常.大于3mm为深覆盖(overjet))_离散',
       '牙体断面与牙龈相对位置(gingival position of fracture)_离散',
       '牙体断面与牙龈相对位置(gingival position of fracture)_连续',
       '根分歧处透射影(furcation imaging)_离散',
       '与合平面的关系(relationship with the occlusal plane)_离散', '缺失(is missing)',
       '牙体缺损(tooth defect)', '残冠(residual crown)', '残根(residual root)',
       '叩诊(percussion)', '牙体断面位置(position of fracture)', '龋损(caries)',
       '充填体是否存在(filling existance)', '充填体材料种类(filling material)',
       '修复体种类(restoration classification)', '修复体材料(restoration material)',
       '松动度(mobility)', '牙的位置(tooth position)',
       '牙相关影像-牙槽骨吸收情况(teeth related imaging)', '根尖周影像(apical root imaging)',
       '根充影像(root canal filling imaging)', '修复体边缘(restoration margins)',
       '探诊(exploration probing)', '楔状缺损(wedge shaped defect)',
       '充填体边缘(filling margins)', '继发龋(secondary caries)', '悬突(overhang)',
       '崩瓷(porcelain fracture)', '牙龈红肿(gingival swollen)',
       '牙龈瘘管(gingival fistula)', '覆合(上牙切缘盖过下切牙切三分之一以内为正常.超过此范围为深覆合(overbite))',
       '反合(下牙位于上牙唇侧的为前牙反合(crossbite))', '对刃', '咬合紧(tight bite)',
       '牙间隙(diastema)', '近中根牙槽骨牙槽骨吸收(absorption near mesial root)',
       '远中根牙槽骨牙槽骨吸收(absorption near distal root)', '桩核影像学(post imaging)',
       '根管影像(root canal imaging)', '根周膜影像(periodontal membrane imaging)',
       '根长影像(length of root imaging)', '阻生齿(impaction)',
       '固位(retention of existed fixed restorations)',
       '缺牙部位剩余牙槽嵴情况(residual ridge conditions)', '松软牙槽嵴(soft flabby ridges)',
       '黏膜疼痛 增生 肿胀 溃疡(pain hyperemization swollen and ucler)', '骨突 骨尖(torus)',
       '旧义齿人工牙合面磨耗情况(artificial tooth wear)']
    unrela_features = ['口腔卫生情况(oral hygiene)', '牙石情况(oral dental calculus)',
       '下颌相对于上颌左右方向颌骨关系(side to side jaw relationship)',
       '下颌相对于上颌前后方向颌骨关系(anterorposterior jaw relationship)',
       '固位(retention of existed removable restorations)',
       '舌系带附着点距牙槽骨上缘的距离(lingual frena attachment position)',
       '影像学检查(roentgenograph of jaw bones and arches)',
       '开口度检查(mandibular opening)', '开口型检查(mandibular opening patterns)',
       '疼痛(pain of TMJ )', '弹响(click on TMJ)',
       '面部外形(maxillofacial appearance)', '上唇丰满度(upper lip fullness)',
       '面下1/3垂直距离(vertical dimension)',
       '上/下颌义齿基托部位有无裂纹/折断(conditions of resin base of existed restorations)',
       '基托与组织不贴合(unfit of tissue and base)',
       '上颌义齿部位边缘伸展(maxillary border extension)',
       '下颌义齿部位边缘伸展(mandibular border extension)',
       '摘戴困难(difficulty in delivery)', '人工牙缺损']
    label_default = torch.tensor([0] * len(label), dtype=torch.long)
    features_default = torch.tensor([0] * len(rela_features), dtype=torch.float)  # shape [n_feat]
    unrela_x_default = torch.tensor([0] * len(unrela_features), dtype=torch.float)
    # 分组
    grouped = df.groupby(['病例号', '就诊id'])
    n_sample = len(grouped)
    n_feature = len(rela_features)
    n_label = len(label)
    print(n_label)
    x = torch.tensor(np.array([]), dtype=torch.float)  # [N, 32, n_feat]
    unrela_x = torch.tensor(np.array([]), dtype=torch.float)  # [N, 32, n_feat]
    y = torch.tensor(np.array([]), dtype=torch.long)  # [N, 32, 1]
    valid = []
    case = []
    if label[0] in rela_treatment:
        for idx, group in tqdm(grouped):
            group['tooth_id'] = group['牙位'].map(int)
            _x = torch.empty(size=(32, n_feature))
            _y = torch.empty(size=(32, n_label), dtype=torch.long)
            _valid = []
            if 0 in list(group['tooth_id']):
                _unrela_x = torch.from_numpy(group.loc[group.tooth_id == 0, unrela_features].values[0]).type(
                    torch.FloatTensor)
            else:
                _unrela_x = unrela_x_default
            for t in range(1, 33, 1):
                _t = int(t) - 1 if int(t) <= 16 else 48 - int(t)
                if t in list(group['tooth_id']):
                    tmp_x = torch.from_numpy(group.loc[group.tooth_id == t, rela_features].values[0]).type(
                        torch.FloatTensor)
                    tmp_y = torch.from_numpy(group.loc[group.tooth_id == t, label].values[0]).type(torch.LongTensor)
                    _valid.append(_t)
                else:
                    tmp_x = features_default
                    tmp_y = label_default
                _x[_t, :] = tmp_x
                _y[_t, :] = tmp_y
            x = torch.cat((x, _x.unsqueeze(0)), dim=0)
            unrela_x = torch.cat((unrela_x, _unrela_x.unsqueeze(0)), dim=0)
            y = torch.cat((y, _y.unsqueeze(0)), dim=0)
            _valid.sort()  # 升序
            valid.append(_valid)
            case.append([group['病例号'].values[0], group['就诊id'].values[0]])
    else:
        for idx, group in tqdm(grouped):
            group['tooth_id'] = group['牙位'].map(int)
            _x = torch.empty(size=(32, n_feature))
            _valid = []
            if 0 in list(group['tooth_id']):
                _unrela_x = torch.from_numpy(group.loc[group.tooth_id == 0, unrela_features].values[0]).type(torch.FloatTensor)
                _y = torch.from_numpy(group.loc[group.tooth_id == 0, label].values[0]).type(torch.LongTensor)
            else:
                _unrela_x = unrela_x_default
                _y = label_default
            for t in range(1, 33, 1):
                _t = int(t) - 1 if int(t) <= 16 else 48 - int(t)
                if t in list(group['tooth_id']):
                    tmp_x = torch.from_numpy(group.loc[group.tooth_id == t, rela_features].values[0]).type(torch.FloatTensor)
                    _valid.append(_t)
                else:
                    tmp_x = features_default
                _x[_t, :] = tmp_x
            x = torch.cat((x, _x.unsqueeze(0)), dim=0)
            unrela_x = torch.cat((unrela_x, _unrela_x.unsqueeze(0)), dim=0)
            y = torch.cat((y, _y.unsqueeze(0)), dim=0)
            _valid.sort()  # 升序
            valid.append(_valid)
            case.append([group['病例号'].values[0], group['就诊id'].values[0]])
    x = x.reshape(n_sample,  2, 16, n_feature)
    x = x.cuda()
    unrela_x = unrela_x.cuda()
    y = y.cuda()
    return x,unrela_x , y, valid, case

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
print(x_train.shape)
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


# beta = 1e-4
idx_shuffle = [i for i in range(train_size)]
epochs = 200

LR = 1e-1
loss_func0 = nn.CrossEntropyLoss()   # the target label is not one-hotted
b = []
b1 = []
lamda = 1e-4
top = 0
batch_loss_ls = []
loss_ls = {'train': [], 'test': []}
acc_ls = {'train': [], 'test': []}
setup_seed(0)
beta = 1e-4

# batch_size = [32+i for i in range

Para = [[1.3,0.9,0.8],[1 ,1.2 ,0.8],[1.3,0.8,0.8],[1.3,1.0,0.8],[1.3,1.0,0.9],[1.3,1.0,1.0],[1.3,0.9,0.8],
        [1.3,0.9,0.9],[1.3,0.9,1.0]]
PARA = [[1.3,1.0,1.0]]
para = PARA[0]
batch_size = 64
# BATCH = [2+i for i in range(56)]
# for batch_size in BATCH:
#     print(batch_size)
#     # for i in A:
#     #     print(i)
model = ResNet50().cuda()
model_dict1 = torch.load('m_1.params')
model_dict2 = torch.load('m_2.params')
model_dict3 = torch.load('m_3.params')
model_dict0 = torch.load('m_0.params')
model_dict4 = torch.load('m_4.params')
# del_key1 = []
# del_key2 = []
# del_key3 = []
del_key0 = []
del_key4 = []
# for key, _ in model_dict1.items():
#     if "fc" in key:
#         del_key1.append(key)
#
# for key, _ in model_dict2.items():
#     if "fc" in key:
#         del_key2.append(key)
#
# for key, _ in model_dict3.items():
#     if "fc" in key:
#         del_key3.append(key)

for key, _ in model_dict0.items():
    if "fc" in key or 'embed' in key or 'layer3' in key:
        del_key0.append(key)
#
for key, _ in model_dict4.items():
    if "fc" in key or 'embed' in key or 'feature0' in key:
        del_key4.append(key)

# for key in del_key1:
#     del model_dict1[key]
#
# for key in del_key2:
#     del model_dict2[key]
#
# for key in del_key3:
#     del model_dict3[key]

for key in del_key0:
    del model_dict0[key]
#
for key in del_key4:
    del model_dict4[key]

model.load_state_dict(model_dict0, strict=False)
model.load_state_dict(model_dict1, strict=False)
model.load_state_dict(model_dict2, strict=False)
model.load_state_dict(model_dict3, strict=False)
model.load_state_dict(model_dict4, strict=False)

params_1x = [
    param for name, param in model.named_parameters()
    if "layer" not in name]
params_0x = [
    param for name, param in model.named_parameters()
    if "layer" in name]
optimizer = torch.optim.Adam([{
    'params': params_1x}, {
    'params': params_0x,
    'lr': 0.00001}], lr=0.001)

# model.load_state_dict(model_dict0, strict=False)
# model.load_state_dict(model_dict4, strict=False)

# params_1x = [
#     param for name, param in model.named_parameters()
#     if "layer" not in name]
# params_0x = [
#     param for name, param in model.named_parameters()
#     if "layer" in name]


beta = 1e-4
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

for epoch in range(epochs):
    random.shuffle(idx_shuffle)
    x_train, unrela_x_train, y_train0, y_train1, y_train2, y_train3, y_train4 = x_train[idx_shuffle], \
                                                                                unrela_x_train[idx_shuffle], \
                                                                                y_train0[idx_shuffle], y_train1[
                                                                                    idx_shuffle], \
                                                                                y_train2[idx_shuffle], y_train3[
                                                                                    idx_shuffle], y_train4[
                                                                                    idx_shuffle]
    valid_train = [valid_train[i] for i in idx_shuffle]
    case_train = [case_train[i] for i in idx_shuffle]
    idx_train = []
    for j, j_tuple in enumerate(valid_train):
        if len(j_tuple) > 0: idx_train.extend([j * 32 + k for k in j_tuple])
    model.train()
    for i in range(train_size // batch_size + 1):
        if i == train_size // batch_size:
            inputs, aux, labels1, valid, case = x_train[i * batch_size: i * batch_size + train_size % batch_size], \
                                                unrela_x_train[
                                                i * batch_size: i * batch_size + train_size % batch_size], \
                                                y_train1[i * batch_size: i * batch_size + train_size % batch_size], \
                                                valid_train[
                                                i * batch_size: i * batch_size + train_size % batch_size], \
                                                case_train[i * batch_size: i * batch_size + train_size % batch_size]
            labels0 = y_train0[batch_size * i:batch_size * i + train_size % batch_size]
            labels2 = y_train2[batch_size * i:batch_size * i + train_size % batch_size]
            labels3 = y_train3[batch_size * i:batch_size * i + train_size % batch_size]
            labels4 = y_train4[batch_size * i:batch_size * i + train_size % batch_size]
        else:
            inputs, aux, labels1, valid, case = x_train[i * batch_size: i * batch_size + batch_size], \
                                                unrela_x_train[i * batch_size: i * batch_size + batch_size], \
                                                y_train1[i * batch_size: i * batch_size + batch_size], \
                                                valid_train[i * batch_size: i * batch_size + batch_size], \
                                                case_train[i * batch_size: i * batch_size + batch_size]
            labels0 = y_train0[batch_size * i:batch_size * i + batch_size]
            labels2 = y_train2[batch_size * i:batch_size * i + batch_size]
            labels3 = y_train3[batch_size * i:batch_size * i + batch_size]
            labels4 = y_train4[batch_size * i:batch_size * i + batch_size]

        outputs0, outputs1, outputs2, outputs3, outputs4 = model(inputs, aux)

        outputs1 = outputs1.reshape(-1, n_label)  # [batch_size*32, n_label]
        outputs2 = outputs2.reshape(-1, n_label)  # [batch_size*32, n_label]
        outputs3 = outputs3.reshape(-1, n_label)  # [batch_size*32, n_label]

        labels0 = labels0.reshape(-1)
        labels1 = labels1.reshape(-1)  # [batch_size*32, n_label]
        labels2 = labels2.reshape(-1)  # [batch_size*32, n_label]
        labels3 = labels3.reshape(-1)  # [batch_size*32, n_label]
        labels4 = labels4.reshape(-1)  # [batch_size*32, n_label]
        # 提取有效牙位的预测与标签
        idx = []
        for j, j_tuple in enumerate(valid):
            if len(j_tuple) > 0: idx.extend([j * 32 + k for k in j_tuple])
        idx = torch.tensor(idx)
        outputs_v1 = outputs1[idx, :]
        labels_v1 = labels1[idx]
        outputs_v2 = outputs2[idx, :]
        labels_v2 = labels2[idx]
        outputs_v3 = outputs3[idx, :]
        labels_v3 = labels3[idx]
        loss = loss_func0(outputs0, labels0) +1.3* loss_func0(outputs_v1, labels_v1) + \
               loss_func0(outputs_v2, labels_v2) + loss_func0(outputs_v3, labels_v3) + 1.5*loss_func0(outputs4, labels4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # scheduler.step()  # learning e decay
    model.eval()
    with torch.no_grad():
        # Train
        outputs0, outputs1, outputs2, outputs3, outputs4 = model(x_train, unrela_x_train)

        labels0 = y_train0.reshape(-1)  # [batch_size*32]
        pred_y_train0 = torch.max(outputs0, dim=1)[1].data.cpu().numpy()
        target_y_train0 = labels0.data.cpu().numpy()
        pred_y_train00 = torch.softmax(outputs0, dim=1)[:, 1].data.cpu().numpy()

        outputs1 = outputs1.reshape(-1, n_label)  # output shape [batch_size*32, 2]
        labels1 = y_train1.reshape(-1)  # [batch_size*32]
        outputs_v1 = outputs1[idx_train, :]
        labels_v1 = labels1[idx_train]
        pred_y_train1 = torch.max(outputs_v1, dim=1)[1].data.cpu().numpy()
        target_y_train1 = labels_v1.data.cpu().numpy()
        pred_y_train11 = torch.softmax(outputs_v1, dim=1)[:, 1].data.cpu().numpy()

        outputs2 = outputs2.reshape(-1, n_label)  # output shape [batch_size*32, 2]
        labels2 = y_train2.reshape(-1)  # [batch_size*32]
        outputs_v2 = outputs2[idx_train, :]
        labels_v2 = labels2[idx_train]
        pred_y_train2 = torch.max(outputs_v2, dim=1)[1].data.cpu().numpy()
        target_y_train2 = labels_v2.data.cpu().numpy()
        pred_y_train22 = torch.softmax(outputs_v2, dim=1)[:, 1].data.cpu().numpy()

        outputs3 = outputs3.reshape(-1, n_label)  # output shape [batch_size*32, 2]
        labels3 = y_train3.reshape(-1)  # [batch_size*32]
        outputs_v3 = outputs3[idx_train, :]
        labels_v3 = labels3[idx_train]
        pred_y_train3 = torch.max(outputs_v3, dim=1)[1].data.cpu().numpy()
        target_y_train3 = labels_v3.data.cpu().numpy()
        pred_y_train33 = torch.softmax(outputs_v3, dim=1)[:, 1].data.cpu().numpy()

        labels4 = y_train4.reshape(-1)  # [batch_size*32]
        pred_y_train4 = torch.max(outputs4, dim=1)[1].data.cpu().numpy()
        target_y_train4 = labels4.data.cpu().numpy()
        pred_y_train44 = torch.softmax(outputs4, dim=1)[:, 1].data.cpu().numpy()

        print(loss_func0(outputs0, labels0), loss_func0(outputs_v1, labels_v1), \
              loss_func0(outputs_v2, labels_v2), loss_func0(outputs_v3, labels_v3), loss_func0(outputs4, labels4))
        # Test
        outputs0, outputs1, outputs2, outputs3, outputs4 = model(x_test, unrela_x_test)

        labels0 = y_test0.reshape(-1)  # [batch_size*32]
        pred_y_test0 = torch.max(outputs0, dim=1)[1].data.cpu().numpy()
        target_y_test0 = labels0.data.cpu().numpy()
        pred_y_test00 = torch.softmax(outputs0, dim=1)[:, 1].data.cpu().numpy()

        outputs1 = outputs1.reshape(-1, n_label)  # output shape [batch_size*32, 2]
        labels1 = y_test1.reshape(-1)  # [batch_size*32]
        outputs_v1 = outputs1[idx_test, :]
        labels_v1 = labels1[idx_test]
        pred_y_test1 = torch.max(outputs_v1, dim=1)[1].data.cpu().numpy()
        target_y_test1 = labels_v1.data.cpu().numpy()
        pred_y_test11 = torch.softmax(outputs_v1, dim=1)[:, 1].data.cpu().numpy()

        outputs2 = outputs2.reshape(-1, n_label)  # output shape [batch_size*32, 2]
        labels2 = y_test2.reshape(-1)  # [batch_size*32]
        outputs_v2 = outputs2[idx_test, :]
        labels_v2 = labels2[idx_test]
        pred_y_test2 = torch.max(outputs_v2, dim=1)[1].data.cpu().numpy()
        target_y_test2 = labels_v2.data.cpu().numpy()
        pred_y_test22 = torch.softmax(outputs_v2, dim=1)[:, 1].data.cpu().numpy()

        outputs3 = outputs3.reshape(-1, n_label)  # output shape [batch_size*32, 2]
        labels3 = y_test3.reshape(-1)  # [batch_size*32]
        outputs_v3 = outputs3[idx_test, :]
        labels_v3 = labels3[idx_test]
        pred_y_test3 = torch.max(outputs_v3, dim=1)[1].data.cpu().numpy()
        target_y_test3 = labels_v3.data.cpu().numpy()
        pred_y_test33 = torch.softmax(outputs_v3, dim=1)[:, 1].data.cpu().numpy()

        labels4 = y_test4.reshape(-1)  # [batch_size*32]
        pred_y_test4 = torch.max(outputs4, dim=1)[1].data.cpu().numpy()
        target_y_test4 = labels4.data.cpu().numpy()
        pred_y_test44 = torch.softmax(outputs4, dim=1)[:, 1].data.cpu().numpy()
    # if metrics.accuracy_score(target_y_test0, pred_y_test0)>0.84 and metrics.f1_score(target_y_test1, pred_y_test1)>0.76 \
    #         and metrics.f1_score(target_y_test2, pred_y_test2)>0.85 and metrics.f1_score(target_y_test3, pred_y_test3) > 0.79 and metrics.f1_score(target_y_test4, pred_y_test4)>0.83:
        print('牙周治疗')
        print(
            "Train/Test  Acc {:.4f} / {:.4f}, Recall {:.4f} / {:.4f}, Precision {:.4f} / {:.4f}, F1 {:.4f} / {:.4f}, AUC {:.4f} / {:.4f}".format(
                metrics.accuracy_score(target_y_train0, pred_y_train0),
                metrics.accuracy_score(target_y_test0, pred_y_test0),
                metrics.recall_score(target_y_train0, pred_y_train0),
                metrics.recall_score(target_y_test0, pred_y_test0),
                metrics.precision_score(target_y_train0, pred_y_train0),
                metrics.precision_score(target_y_test0, pred_y_test0),
                metrics.f1_score(target_y_train0, pred_y_train0),
                metrics.f1_score(target_y_test0, pred_y_test0),
                metrics.roc_auc_score(target_y_train0, pred_y_train00),
                metrics.roc_auc_score(target_y_test0, pred_y_test00)
            ))

        print('拆除旧修复体')
        print(
            "Train/Test  Acc {:.4f} / {:.4f}, Recall {:.4f} / {:.4f}, Precision {:.4f} / {:.4f}, F1 {:.4f} / {:.4f}, AUC {:.4f} / {:.4f}".format(
                metrics.accuracy_score(target_y_train1, pred_y_train1),
                metrics.accuracy_score(target_y_test1, pred_y_test1),
                metrics.recall_score(target_y_train1, pred_y_train1),
                metrics.recall_score(target_y_test1, pred_y_test1),
                metrics.precision_score(target_y_train1, pred_y_train1),
                metrics.precision_score(target_y_test1, pred_y_test1),
                metrics.f1_score(target_y_train1, pred_y_train1),
                metrics.f1_score(target_y_test1, pred_y_test1),
                metrics.roc_auc_score(target_y_train1, pred_y_train11),
                metrics.roc_auc_score(target_y_test1, pred_y_test11)
            ))
        # “第 2 步结果”
        print('拔除')
        print(
            "Train/Test  Acc {:.4f} / {:.4f}, Recall {:.4f} / {:.4f}, Precision {:.4f} / {:.4f}, F1 {:.4f} / {:.4f}, AUC {:.4f} / {:.4f}".format(
                metrics.accuracy_score(target_y_train2, pred_y_train2),
                metrics.accuracy_score(target_y_test2, pred_y_test2),
                metrics.recall_score(target_y_train2, pred_y_train2),
                metrics.recall_score(target_y_test2, pred_y_test2),
                metrics.precision_score(target_y_train2, pred_y_train2),
                metrics.precision_score(target_y_test2, pred_y_test2),
                metrics.f1_score(target_y_train2, pred_y_train2),
                metrics.f1_score(target_y_test2, pred_y_test2),
                metrics.roc_auc_score(target_y_train2, pred_y_train22),
                metrics.roc_auc_score(target_y_test2, pred_y_test22)
            ))
        # “第 2 步结果”
        print('牙体牙髓科治疗')
        print(
            "Train/Test  Acc {:.4f} / {:.4f}, Recall {:.4f} / {:.4f}, Precision {:.4f} / {:.4f}, F1 {:.4f} / {:.4f}, AUC {:.4f} / {:.4f}".format(
                metrics.accuracy_score(target_y_train3, pred_y_train3),
                metrics.accuracy_score(target_y_test3, pred_y_test3),
                metrics.recall_score(target_y_train3, pred_y_train3),
                metrics.recall_score(target_y_test3, pred_y_test3),
                metrics.precision_score(target_y_train3, pred_y_train3),
                metrics.precision_score(target_y_test3, pred_y_test3),
                metrics.f1_score(target_y_train3, pred_y_train3),
                metrics.f1_score(target_y_test3, pred_y_test3),
                metrics.roc_auc_score(target_y_train3, pred_y_train33),
                metrics.roc_auc_score(target_y_test3, pred_y_test33)
            ))

        print('修复科复诊')
        print(
            "Train/Test  Acc {:.4f} / {:.4f}, Recall {:.4f} / {:.4f}, Precision {:.4f} / {:.4f}, F1 {:.4f} / {:.4f}, AUC {:.4f} / {:.4f}".format(
                metrics.accuracy_score(target_y_train4, pred_y_train4),
                metrics.accuracy_score(target_y_test4, pred_y_test4),
                metrics.recall_score(target_y_train4, pred_y_train4),
                metrics.recall_score(target_y_test4, pred_y_test4),
                metrics.precision_score(target_y_train4, pred_y_train4),
                metrics.precision_score(target_y_test4, pred_y_test4),
                metrics.f1_score(target_y_train4, pred_y_train4),
                metrics.f1_score(target_y_test4, pred_y_test4),
                metrics.roc_auc_score(target_y_train4, pred_y_train44),
                metrics.roc_auc_score(target_y_test4, pred_y_test44)
            ))
        



