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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm




def setup_seed(seed):     # 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

'NEW_All_删除空牙位与多方案_删掉错误标注_牙位相关的特征只有60个_head'   # 数据版本

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
       '与合平面的关系(relationship with the occlusal plane)_离散',
       '探诊深度(probing depth)_连续', '冠根比例(crown root ratio)_离散', '缺失(is missing)',
       '牙体缺损(tooth defect)', '残冠(residual crown)', '残根(residual root)',
       '叩诊(percussion)', '牙体断面位置(position of fracture)', '龋损(caries)',
       '充填体是否存在(filling existance)', '充填体材料种类(filling material)',
       '修复体种类(restoration classification)', '修复体材料(restoration material)',
       '松动度(mobility)', '牙的位置(tooth position)',
       '牙相关影像-牙槽骨吸收情况(teeth related imaging)', '根尖周影像(apical root imaging)',
       '根充影像(root canal filling imaging)', '修复体边缘(restoration margins)',
       '探诊(exploration probing)', '楔状缺损(wedge shaped defect)',
       '充填体边缘(filling margins)', '继发龋(secondary caries)',
       '修复体食物嵌塞(restoration food impact)', '悬突(overhang)',
       '崩瓷(porcelain fracture)', '牙齿牙石情况(tooth dental calculus)',
       '牙龈红肿(gingival swollen)', '牙龈瘘管(gingival fistula)',
       '根分叉病变(furcation involvement)',
       '覆合(上牙切缘盖过下切牙切三分之一以内为正常.超过此范围为深覆合(overbite))',
       '反合(下牙位于上牙唇侧的为前牙反合(crossbite))', '对刃', '咬合紧(tight bite)',
       '牙排列(arrangement)', '牙间隙(diastema)', '合曲线(occlusal curve)',
       '近中根牙槽骨牙槽骨吸收(absorption near mesial root)',
       '远中根牙槽骨牙槽骨吸收(absorption near distal root)', '桩核影像学(post imaging)',
       '根管影像(root canal imaging)', '根周膜影像(periodontal membrane imaging)',
       '根长影像(length of root imaging)', '阻生齿(impaction)',
       '固位(retention of existed fixed restorations)',
       '缺牙部位剩余牙槽嵴情况(residual ridge conditions)', '松软牙槽嵴(soft flabby ridges)',
       '黏膜疼痛 增生 肿胀 溃疡(pain hyperemization swollen and ucler)', '骨突 骨尖(torus)',
       '旧义齿人工牙合面磨耗情况(artificial tooth wear)']
    unrela_features = ['口腔卫生情况(oral hygiene)', '牙石情况(oral dental calculus)',
       '余留牙牙龈情况(oral gingival conditions)',
       '下颌相对于上颌左右方向颌骨关系(side to side jaw relationship)',
       '下颌相对于上颌前后方向颌骨关系(anterorposterior jaw relationship)',
       '固位(retention of existed removable restorations)',
       '舌系带附着点距牙槽骨上缘的距离(lingual frena attachment position)',
       '唾液分泌(saliva secretion)',
       '影像学检查(roentgenograph of jaw bones and arches)',
       '开口度检查(mandibular opening)', '开口型检查(mandibular opening patterns)',
       '疼痛(pain of TMJ )', '弹响(click on TMJ)',
       '面部外形(maxillofacial appearance)', '上唇丰满度(upper lip fullness)',
       '面下1/3垂直距离(vertical dimension)',
       '上/下颌义齿基托部位有无裂纹/折断(conditions of resin base of existed restorations)',
       '义齿松动(denture loose)', '基托与组织不贴合(unfit of tissue and base)',
       '上颌义齿部位边缘伸展(maxillary border extension)',
       '下颌义齿部位边缘伸展(mandibular border extension)',
       '咀嚼问题(bad behavior in mastication)', '摘戴困难(difficulty in delivery)',
       '下颌舌骨后窝凹陷', '人工牙缺损']
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
label0 = ['拔除']

x_train, unrela_x_train, y_train, valid_train, case_train = load_train_data(train_file,label0)

x_test, unrela_x_test, y_test, valid_test, case_test = load_test_data(test_file,label0)
for i in range(y_train.shape[0]):
    for j in range(y_train.shape[1]):
        for h in  range(y_train.shape[2]):
            if y_train[i,j,h] > 1:
                y_train[i,j,h] = 1
for i in range(y_test.shape[0]):
    for j in range(y_test.shape[1]):
        for h in  range(y_test.shape[2]):
            if y_test[i,j,h] > 1:
                y_test[i,j,h] = 1
# a = x_train.reshape(-1,53).cpu().numpy()
# b = x_test.reshape(-1,53).cpu().numpy()
#
# d=np.append(a,b,axis=0)
#
# c = [i for i in range(53)]
# c.remove(2)
# c.remove(4)
# c.remove(7)
# ct = ColumnTransformer(
#     [('one_hot_encoder', OneHotEncoder(categories='auto'), c)],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
#     remainder='passthrough'                                         # Leave the rest of the columns untouched
# )
#
# ct.fit(d)
# x_train = torch.from_numpy(ct.transform(a).reshape(x_train.shape[0], 2, 16, 156)).type(torch.FloatTensor).cuda()
# x_test = torch.from_numpy(ct.transform(b).reshape(x_test.shape[0], 2, 16, 156)).type(torch.FloatTensor).cuda()
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
# 第二部分结构（提取相邻牙位信息）
class BasicBlock2(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride):
        super(BasicBlock2, self).__init__()
        self.conv1 = SKConv(in_planes,planes, None, 3, 8, 2)
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = torch.sigmoid(self.bn1(self.conv1(x)))
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 53
        self.layer5 = self._make_layer(block[0], 64, num_blocks[0], stride=1)
        self.layer6 = self._make_layer(block[1], 32, num_blocks[1], stride=1)
        self.fc2 = nn.Linear(32, 2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, unrela_x):
        x = x.permute(0, 3, 2, 1)
        out = self.layer6(self.layer5(x)).permute(0, 3, 2, 1).reshape(-1, 32)
        out = self.fc2(out)
        return out

def ResNet50():
    return ResNet([BasicBlock1,BasicBlock2], [3, 1])

def penalty_l1(model, beta=1e-4):
    l1_loss = torch.tensor(0.0, requires_grad=True).float().cuda()
    for name, param in model.named_parameters():
        if 'bias' not in name:  # 不对bias进行惩罚
            l1_loss = l1_loss + beta * torch.sum(torch.abs(param))
    return l1_loss

idx_train = []
for j, j_tuple in enumerate(valid_train):
    if len(j_tuple) > 0: idx_train.extend([j * 32 + k for k in j_tuple])
idx_test = []
for j, j_tuple in enumerate(valid_test):
    if len(j_tuple) > 0: idx_test.extend([j * 32 + k for k in j_tuple])
train_size = x_train.shape[0]
test_size = x_test.shape[0]
print(train_size)
print(test_size)
epochs = 200
batch_size = 64
beta = 1e-4  # L1Loss


idx_shuffle = [i for i in range(train_size)]

#
batch_loss_ls = []
loss_ls = {'train': [], 'test': []}
acc_ls = {'train': [], 'test': []}
f1_ls = []

top = 0
start = time.time()
LR = 1e-1

# loss_func = focal_loss()  # nn.CrossEntropyLoss(), Focal_loss(), nn.BCE
loss_func = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss(), Focal_loss(), nn.BCE
setup_seed(0)
model = ResNet50().cuda()
model_dict = torch.load('m_2.params')

model.load_state_dict(model_dict, strict=False)
model.eval()

# train
outputs = model(x_train, unrela_x_train)  # output shape [batch_size*32, 2]
outputs1 = outputs.reshape(-1, 2)
labels1 = y_train.reshape(-1)  # [batch_size*32]
outputs_v1 = outputs1[idx_train, :]
labels_v1 = labels1[idx_train]
pred_y_train = torch.max(outputs_v1, dim=1)[1].data.cpu().numpy()
target_y_train = labels_v1.data.cpu().numpy()
pred_y_train1 = torch.softmax(outputs_v1, dim=1)[:, 1].data.cpu().numpy()

outputs1 = model(x_test, unrela_x_test)  # output shape [batch_size*32, 2]
outputs1 = outputs1.reshape(-1, 2)
labels1 = y_test.reshape(-1)  # [batch_size*32]
outputs_v1 = outputs1[idx_test, :]
labels_v1 = labels1[idx_test]
pred_y_test = torch.max(outputs_v1, dim=1)[1].data.cpu().numpy()
target_y_test = labels_v1.data.cpu().numpy()
pred_y_test1 = torch.softmax(outputs_v1, dim=1)[:, 1].data.cpu().numpy()

print("训练数据")
print('F1: %.4f' % metrics.f1_score(target_y_train, pred_y_train))
print('Accuracy: %.4f' % metrics.accuracy_score(target_y_train, pred_y_train))
print('AUC: %.4f' % metrics.roc_auc_score(target_y_train, pred_y_train1))
print('Sensitivity: %.4f' % metrics.recall_score(target_y_train, pred_y_train))
tn1, fp1, fn1, tp1 = metrics.confusion_matrix(target_y_train, pred_y_train).ravel()
print('Specificity' )
print(tn1/(tn1+fp1))
print('PPV: %.4f' % metrics.precision_score(target_y_train, pred_y_train))
print('NPV' )
print(tn1/(tn1+fn1) )

print("测试数据")
print('F1: %.4f' % metrics.f1_score(target_y_test, pred_y_test))
print('Accuracy: %.4f' % metrics.accuracy_score(target_y_test, pred_y_test))
print('AUC: %.4f' % metrics.roc_auc_score(target_y_test, pred_y_test1))
print('Sensitivity: %.4f' % metrics.recall_score(target_y_test, pred_y_test))
tn2, fp2, fn2, tp2 = metrics.confusion_matrix(target_y_test, pred_y_test).ravel()
print('Specificity')
print( tn2/(tn2+fp2))
print('PPV: %.4f' % metrics.precision_score(target_y_test, pred_y_test))
print('NPV')
print(tn2/(tn2+fn2))
#
# print(
#     "Train/Test  Acc {:.4f} / {:.4f}, Recall {:.4f} / {:.4f}, Precision {:.4f} / {:.4f}, F1 {:.4f} / {:.4f}, AUC {:.4f} / {:.4f}".format(
#         metrics.accuracy_score(target_y_train, pred_y_train),
#         metrics.accuracy_score(target_y_test, pred_y_test),
#         metrics.recall_score(target_y_train, pred_y_train),
#         metrics.recall_score(target_y_test, pred_y_test),
#         metrics.precision_score(target_y_train, pred_y_train),
#         metrics.precision_score(target_y_test, pred_y_test),
#         metrics.f1_score(target_y_train, pred_y_train),
#         metrics.f1_score(target_y_test, pred_y_test),
#         metrics.roc_auc_score(target_y_train, pred_y_train1),
#         metrics.roc_auc_score(target_y_test, pred_y_test1)
#     ))

# optimizer = torch.optim.SGD(model.parameters(), lr=LR,momentum=0.9, weight_decay=1e-4)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 70, 120], gamma=0.1, last_epoch=-1)
# for epoch in range(epochs):
#     print(epoch)
#     random.shuffle(idx_shuffle)
#     x_train, unrela_x_train, y_train = x_train[idx_shuffle],unrela_x_train[idx_shuffle], y_train[idx_shuffle]
#     valid_train = [valid_train[i] for i in idx_shuffle]
#     case_train = [case_train[i] for i in idx_shuffle]
#     idx_train = []
#     for j, j_tuple in enumerate(valid_train):
#         if len(j_tuple) > 0: idx_train.extend([j * 32 + k for k in j_tuple])
#     model.train()
#     for i in range(train_size // batch_size + 1):
#         if i == train_size // batch_size:
#             inputs, embed, labels, valid, case = x_train[i * batch_size: i * batch_size + train_size % batch_size], \
#                                           unrela_x_train[i * batch_size: i * batch_size + train_size % batch_size],\
#                                           y_train[i * batch_size: i * batch_size + train_size % batch_size], \
#                                           valid_train[i * batch_size: i * batch_size + train_size % batch_size], \
#                                           case_train[i * batch_size: i * batch_size + train_size % batch_size]
#         else:
#             inputs, embed, labels, valid, case = x_train[i * batch_size: i * batch_size + batch_size], \
#                                           unrela_x_train[i * batch_size: i * batch_size + batch_size], \
#                                           y_train[i * batch_size: i * batch_size + batch_size], \
#                                           valid_train[i * batch_size: i * batch_size + batch_size], \
#                                           case_train[i * batch_size: i * batch_size + batch_size]
#
#         outputs = model(inputs,embed)  # output shape [batch_size, 32, n_label]
#         outputs = outputs.reshape(-1, 2)
#         labels = labels.reshape(-1)  # [batch_size*32, n_label]
#         idx = []
#         for j, j_tuple in enumerate(valid):
#             if len(j_tuple) > 0: idx.extend([j * 32 + k for k in j_tuple])
#         idx = torch.tensor(idx)
#         outputs_v1 = outputs[idx, :]
#         labels_v1 = labels[idx]
#         loss = loss_func(outputs_v1, labels_v1)
#         # + penalty_l1(model, beta=beta)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     scheduler.step()  # learning e decay
#     model.eval()
#     # train
#     outputs = model(x_train, unrela_x_train)  # output shape [batch_size*32, 2]
#     outputs1 = outputs.reshape(-1, 2)
#     labels1 = y_train.reshape(-1)  # [batch_size*32]
#     outputs_v1 = outputs1[idx_train, :]
#     labels_v1 = labels1[idx_train]
#     pred_y_train = torch.max(outputs_v1, dim=1)[1].data.cpu().numpy()
#     target_y_train = labels_v1.data.cpu().numpy()
#     pred_y_train1 = torch.softmax(outputs_v1, dim=1)[:, 1].data.cpu().numpy()
#
#     outputs1 = model(x_test, unrela_x_test)  # output shape [batch_size*32, 2]
#     outputs1 = outputs1.reshape(-1, 2)
#     labels1 = y_test.reshape(-1)  # [batch_size*32]
#     outputs_v1 = outputs1[idx_test, :]
#     labels_v1 = labels1[idx_test]
#     pred_y_test = torch.max(outputs_v1, dim=1)[1].data.cpu().numpy()
#     target_y_test = labels_v1.data.cpu().numpy()
#     pred_y_test1 = torch.softmax(outputs_v1, dim=1)[:, 1].data.cpu().numpy()
#
#     print(
#         "Train/Test  Acc {:.4f} / {:.4f}, Recall {:.4f} / {:.4f}, Precision {:.4f} / {:.4f}, F1 {:.4f} / {:.4f}, AUC {:.4f} / {:.4f}".format(
#             metrics.accuracy_score(target_y_train, pred_y_train),
#             metrics.accuracy_score(target_y_test, pred_y_test),
#             metrics.recall_score(target_y_train, pred_y_train),
#             metrics.recall_score(target_y_test, pred_y_test),
#             metrics.precision_score(target_y_train, pred_y_train),
#             metrics.precision_score(target_y_test, pred_y_test),
#             metrics.f1_score(target_y_train, pred_y_train),
#             metrics.f1_score(target_y_test, pred_y_test),
#             metrics.roc_auc_score(target_y_train, pred_y_train1),
#             metrics.roc_auc_score(target_y_test, pred_y_test1)
#         ))
#     if epoch == 56:
#         torch.save(model.state_dict(), 'm_3.params')


