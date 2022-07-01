# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sklearn.datasets     #引入数据集
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import LogicNet,plot_losses,predict,plot_decision_boundary, Multi, Hybrid
from sklearn.metrics import accuracy_score
import pylab
from sklearn.manifold import TSNE
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)           #设置随机数种子
X1, Y1 = sklearn.datasets.make_moons(200,noise=0.2) #生成2组半圆形数据
arg = np.squeeze(np.argwhere(Y1==0),axis = 1)     #获取第1组数据索引
arg2 = np.squeeze(np.argwhere(Y1==1),axis = 1)#获取第2组数据索引

plt.title("moons data")
plt.scatter(X1[arg,0], X1[arg,1], s=100,c='b',marker='+',label='data1')
plt.scatter(X1[arg2,0], X1[arg2,1],s=40, c='r',marker='o',label='data2')
plt.legend()
plt.show()

np.random.seed(0)           #设置随机数种子
X2, Y2 = sklearn.datasets.make_circles(200,noise=0.1) #生成2组半圆形数据
arg = np.squeeze(np.argwhere(Y2==0),axis = 1)     #获取第1组数据索引
arg2 = np.squeeze(np.argwhere(Y2==1),axis = 1)#获取第2组数据索引

plt.title("circles data")
plt.scatter(X2[arg,0], X2[arg,1], s=100,c='b',marker='+',label='data1')
plt.scatter(X2[arg2,0], X2[arg2,1],s=40, c='r',marker='o',label='data2')
plt.legend()
plt.show()



model = Hybrid(inputdim=2,hiddendim=3,outputdim=2)#
# 初始化模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#定义优化器

xt1 = torch.from_numpy(X1).type(torch.FloatTensor)#将Numpy数据转化为张量
yt1 = torch.from_numpy(Y1).type(torch.LongTensor)
xt2 = torch.from_numpy(X2).type(torch.FloatTensor)#将Numpy数据转化为张量
yt2 = torch.from_numpy(Y2).type(torch.LongTensor)
epochs = 3000#定义迭代次数
losses = []#定义列表，用于接收每一步的损失值
for i in range(epochs):
    loss = model.getloss(xt1, yt1, xt2, yt2)
    losses.append(loss.item())
    optimizer.zero_grad()#清空之前的梯度
    loss.backward()#反向传播损失值
    optimizer.step()#更新参数



plot_losses(losses)

pre1, pre2, pre_all = model.predict(xt1, xt2)
print(accuracy_score(pre1,yt1), accuracy_score(pre2,yt2), accuracy_score(pre_all,yt1))

# plot_decision_boundary(lambda x : predict(model,x) ,xt.numpy(), yt.numpy())

src_features1, src_features2, feature_all, l1, l2 = model.fea_out(xt1, yt1, xt2, yt2)

src_features1 = src_features1.data.numpy()
l = l1.data.numpy()

fea0 = np.squeeze(src_features1[np.argwhere(l<1)])
fea1 = np.squeeze(src_features1[np.argwhere(l>0)])
fea0 = TSNE(n_components= 2).fit_transform(fea0)
fea1 = TSNE(n_components= 2).fit_transform(fea1)
plt.scatter(fea0[:, 0], fea0[:, 1], color = 'b')
plt.scatter(fea1[:, 0], fea1[:, 1], color = 'r')
plt.title('moon')
pylab.show()

src_features2 = src_features2.data.numpy()
l = l2.data.numpy()

fea0 = np.squeeze(src_features2[np.argwhere(l<1)])
fea1 = np.squeeze(src_features2[np.argwhere(l>0)])
fea0 = TSNE(n_components= 2).fit_transform(fea0)
fea1 = TSNE(n_components= 2).fit_transform(fea1)
plt.scatter(fea0[:, 0], fea0[:, 1], color = 'b')
plt.scatter(fea1[:, 0], fea1[:, 1], color = 'r')
plt.title('circle')
pylab.show()

src_features3 = feature_all.data.numpy()
l = l2.data.numpy()

fea0 = np.squeeze(src_features3[np.argwhere(l<1)])
fea1 = np.squeeze(src_features3[np.argwhere(l>0)])
fea0 = TSNE(n_components= 2).fit_transform(fea0)
fea1 = TSNE(n_components= 2).fit_transform(fea1)
plt.scatter(fea0[:, 0], fea0[:, 1], color = 'b')
plt.scatter(fea1[:, 0], fea1[:, 1], color = 'r')
plt.title('fea_fusion')
pylab.show()

