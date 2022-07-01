# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch从深度学习到图神经网络>配套代码
@配套代码技术支持：bbs.aianaconda.com
Created on Fri Feb  1 00:07:25 2019
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

#继承nn.Module类，构建网络模型
import distance


class LogicNet(nn.Module):
    def __init__(self,inputdim,hiddendim,outputdim):#初始化网络结构
        super(LogicNet,self).__init__()
        self.Linear1 = nn.Linear(inputdim,hiddendim) #定义全连接层
        self.Linear2 = nn.Linear(hiddendim,outputdim)#定义全连接层
        self.criterion = nn.CrossEntropyLoss() #定义交叉熵函数

    def forward(self,x): #搭建用两层全连接组成的网络模型
        x = self.Linear1(x)#将输入数据传入第1层
        x = torch.tanh(x)#对第一层的结果进行非线性变换
        #-------------------------------------
        # 如果想加入dropout则
        #x = nn.functional.dropout(x, p=0.07, training=self.training)
        #-------------------------------------
        fea = x
        x = self.Linear2(x)#再将数据传入第2层
#        print("LogicNet")
        return fea, x

    def predict(self,x):#实现LogicNet类的预测接口
        #调用自身网络模型，并对结果进行softmax处理,分别得出预测数据属于每一类的概率
        _, y_pred = self.forward(x)
        pred = torch.softmax(y_pred, dim=1)
        return torch.argmax(pred,dim=1)  #返回每组预测概率中最大的索引

    def getloss(self,x,y): #实现LogicNet类的损失值计算接口
        _, y_pred = self.forward(x)
        loss = self.criterion(y_pred,y)#计算损失值得交叉熵
        return loss

    def fea_out(self,x,y):
        fea, _ = self.forward(x)
        return fea, y

class Multi(nn.Module):
    def __init__(self,inputdim,hiddendim,outputdim):#初始化网络结构
        super(Multi,self).__init__()
        self.Linear1_1 = nn.Linear(inputdim,hiddendim) #定义全连接层
        self.Linear1_2 = nn.Linear(hiddendim,outputdim)#定义全连接层
        self.Linear2_1 = nn.Linear(inputdim, hiddendim)  # 定义全连接层
        self.Linear2_2 = nn.Linear(hiddendim, outputdim)  # 定义全连接层
        self.criterion = nn.CrossEntropyLoss() #定义交叉熵函数

    def forward(self,x1,x2): #搭建用两层全连接组成的网络模型
        # moon data
        x1 = self.Linear1_1(x1)#将输入数据传入第1层
        x1 = torch.tanh(x1)#对第一层的结果进行非线性变换
        fea1 = x1
        x1 = self.Linear1_2(x1)#再将数据传入第2层
        # circle data
        x2 = self.Linear2_1(x2)  # 将输入数据传入第1层
        x2 = torch.tanh(x2)  # 对第一层的结果进行非线性变换
        fea2 = x2
        x2 = self.Linear2_2(x2)  # 再将数据传入第2层
        return fea1, fea2, x1, x2

    def predict(self, x1, x2):#实现LogicNet类的预测接口
        #调用自身网络模型，并对结果进行softmax处理,分别得出预测数据属于每一类的概率
        _, _, y_pred1, y_pred2 = self.forward(x1,x2)
        pred1 = torch.softmax(y_pred1, dim=1)
        pred2 = torch.softmax(y_pred2, dim=1)
        pred_all = torch.softmax(y_pred1 + y_pred2, dim=1)
        return torch.argmax(pred1,dim=1), torch.argmax(pred2,dim=1), torch.argmax(pred_all,dim=1)  #返回每组预测概率中最大的索引

    def getloss(self,x1,y1,x2,y2): #实现LogicNet类的损失值计算接口
        _, _, y_pred1, y_pred2 = self.forward(x1,x2)
        loss1 = self.criterion(y_pred1, y1)#计算损失值得交叉熵
        loss2 = self.criterion(y_pred2, y2)

        return loss1 + loss2

    def fea_out(self,x1,y1,x2,y2):
        fea1, fea2, _, _ = self.forward(x1,x2)
        return fea1, fea2, y1, y2


class Hybrid(nn.Module):
    def __init__(self,inputdim,hiddendim,outputdim):#初始化网络结构
        super(Hybrid,self).__init__()
        self.Linear1_1 = nn.Linear(inputdim,hiddendim) #定义全连接层1
        self.Linear1_2 = nn.Linear(hiddendim,outputdim)#定义全连接层private1
        self.Linear1_3 = nn.Linear(inputdim,hiddendim)  # 定义全连接层share1

        self.Linear2_1 = nn.Linear(inputdim, hiddendim)  # 定义全连接层2
        self.Linear2_2 = nn.Linear(hiddendim, outputdim)  # 定义全连接层private2
        self.Linear2_3 = nn.Linear(inputdim,hiddendim)  # 定义全连接层share2
        self.fc = nn.Linear(2*hiddendim, outputdim)
        self.criterion = nn.CrossEntropyLoss() #定义交叉熵函数
        self.loss_diff = DiffLoss()
        self.loss_smi = distance.CMD()


    def forward(self,x1,x2): #搭建用两层全连接组成的网络模型
        # moon data
        x = self.Linear1_1(x1)#将输入数据传入第1层
        x = torch.tanh(x)#对第一层的结果进行非线性变换
        fea_p1 = x
        pre1 = self.Linear1_2(x)#再将数据传入第2层
        # circle data
        x = self.Linear2_1(x2)  # 将输入数据传入第1层
        x = torch.tanh(x)  # 对第一层的结果进行非线性变换
        fea_p2 = x
        pre2 = self.Linear2_2(x)  # 再将数据传入第2层
        #share
        fea_s1 = self.Linear1_3(x1)
        fea_s2 = self.Linear2_3(x2)
        fea_share = torch.cat((fea_s1, fea_s2),1)
        fea_share = torch.tanh(fea_share)
        pre_share = self.fc(fea_share)
        return fea_p1, fea_p2, fea_s1, fea_s2, fea_share, pre1, pre2, pre_share

    def predict(self, x1, x2):#实现LogicNet类的预测接口
        #调用自身网络模型，并对结果进行softmax处理,分别得出预测数据属于每一类的概率
        _, _, _, _, _, y_pred1, y_pred2, y_preds = self.forward(x1,x2)
        pred1 = torch.softmax(y_pred1, dim=1)
        pred2 = torch.softmax(y_pred2, dim=1)
        pred_all = torch.softmax(y_pred1 + y_pred2 + y_preds, dim=1)
        return torch.argmax(pred1,dim=1), torch.argmax(pred2,dim=1), torch.argmax(pred_all,dim=1)  #返回每组预测概率中最大的索引

    def getloss(self,x1,y1,x2,y2): #实现LogicNet类的损失值计算接口
        fea_p1, fea_p2, fea_s1, fea_s2, _, y_pred1, y_pred2, y_preds = self.forward(x1,x2)
        loss1 = self.criterion(y_pred1, y1)#计算损失值得交叉熵
        loss2 = self.criterion(y_pred2, y2)
        loss3 = self.criterion(y_preds, y2)

        # loss_smi = ((fea_s1 - fea_s2) ** 2).sum(1).sqrt().mean()
        loss_smi = distance.mmd(fea_s1, fea_s2)
        # loss_smi = self.loss_smi(fea_s1, fea_s2, 1)
        loss_diff = self.loss_diff(fea_p1, fea_s1) + self.loss_diff(fea_p2, fea_s2)

        return loss1 + loss2 + loss3 + 0.001 * loss_smi + 0.001 * loss_diff

    def fea_out(self,x1,y1,x2,y2):
        fea_p1, fea_p2, _, _, fea_share, _, _, _ = self.forward(x1,x2)
        return fea_p1, fea_p2, fea_share, y1, y2

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

def moving_average(a, w=10):#定义函数计算移动平均损失值
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

def plot_losses(losses):
    avgloss= moving_average(losses) #获得损失值的移动平均值
    plt.figure(1)
    plt.subplot(211)
    plt.plot(range(len(avgloss)), avgloss, 'b--')
    plt.xlabel('step number')
    plt.ylabel('Training loss')
    plt.title('step number vs. Training loss')
    plt.show()

def predict(model,x):   #封装支持Numpy的预测接口
    x = torch.from_numpy(x).type(torch.FloatTensor)
    ans = model.predict(x)
    return ans.numpy()

def plot_decision_boundary(pred_func,X,Y):#在直角坐标系中可视化模型能力
    #计算取值范围
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    #在坐标系中采用数据，生成网格矩阵，用于输入模型
    xx,yy=np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #将数据输入并进行预测
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #将预测的结果可视化
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.title("Linear predict")
    arg = np.squeeze(np.argwhere(Y==0),axis = 1)
    arg2 = np.squeeze(np.argwhere(Y==1),axis = 1)
    plt.scatter(X[arg,0], X[arg,1], s=100,c='b',marker='+')
    plt.scatter(X[arg2,0], X[arg2,1],s=40, c='r',marker='o')
    plt.show()
