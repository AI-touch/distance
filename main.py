# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sklearn.datasets     #引入数据集
from sklearn.manifold import TSNE
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import LogicNet,plot_losses,predict,plot_decision_boundary
from sklearn.metrics import accuracy_score
import pylab
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)           #设置随机数种子
X, Y = sklearn.datasets.make_moons(200,noise=0.2) #生成2组半圆形数据
arg = np.squeeze(np.argwhere(Y==0),axis = 1)     #获取第1组数据索引
arg2 = np.squeeze(np.argwhere(Y==1),axis = 1)#获取第2组数据索引

plt.title("moons data")
plt.scatter(X[arg,0], X[arg,1], s=100,c='b',marker='+',label='data1')
plt.scatter(X[arg2,0], X[arg2,1],s=40, c='r',marker='o',label='data2')
plt.legend()
plt.show()


model = LogicNet(inputdim=2,hiddendim=3,outputdim=2)#
# 初始化模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#定义优化器

xt = torch.from_numpy(X).type(torch.FloatTensor)#将Numpy数据转化为张量
yt = torch.from_numpy(Y).type(torch.LongTensor)
epochs = 1000#定义迭代次数
losses = []#定义列表，用于接收每一步的损失值
for i in range(epochs):
    loss = model.getloss(xt,yt)
    losses.append(loss.item())
    optimizer.zero_grad()#清空之前的梯度
    loss.backward()#反向传播损失值
    optimizer.step()#更新参数



plot_losses(losses)


print(accuracy_score(model.predict(xt),yt))

plot_decision_boundary(lambda x : predict(model,x) ,xt.numpy(), yt.numpy())

src_features, l = model.fea_out(xt, yt)
src_features = src_features.data.numpy()
l = l.data.numpy()

fea0 = np.squeeze(src_features[np.argwhere(l<1)])
fea1 = np.squeeze(src_features[np.argwhere(l>0)])
fea0 = TSNE(n_components= 2).fit_transform(fea0)
fea1 = TSNE(n_components= 2).fit_transform(fea1)
plt.scatter(fea0[:, 0], fea0[:, 1], color = 'b')
plt.scatter(fea1[:, 0], fea1[:, 1], color = 'r')

plt.title('Adapted')
pylab.show()



