#!/usr/bin/env python
# coding: utf-8

# In[1]:


import timeit
import torch
import torch.nn as nn
import torch.nn.init as init
import random
import scipy.io
import numpy as np
from pyDOE import lhs
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from collections import OrderedDict

from math import exp

if torch.cuda.is_available():
    """ Cuda support """
    print('cuda available')
    device = torch.device('cuda')
else:
    print('cuda not avail')
    device = torch.device('cpu')
# device = torch.device('cpu')    
np.random.seed(1234)

def grad(u, x):
    """ Get grad """
    gradient = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]
    return gradient


# # $ -0.01(u_{xx} + u_{yy}) + u_x = 0 $
# # $ u(x,0) = u(x,1) = 0, u(0,y) = sin(\pi y), u(1,y) = 2sin(\pi y) $
# ## 选点：均匀抽样20000个残差点，4个边界条件各随机取200个点
# ## 架构：$ self.net1(x,y) + self.net2(x,y) * exp((x - 1) / 0.01) $

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from pyDOE import lhs
from numpy import pi, sin, sinh, sqrt
import scipy.io

epochs = 100001

Diff = 1e-3

nPt = 1001
x0 = 0.
x1 = 1.
Re = 100

data = np.loadtxt("4—0.001.csv", delimiter=",", skiprows=1)  # 跳过标题行

x1 = data[:, 0].flatten()[:, None]
y1 = data[:, 1].flatten()[:, None]
u = data[:, 2].flatten()[:, None]

x = np.linspace(0, 1, nPt).reshape(-1, 1)
y = np.linspace(0, 1, nPt).reshape(-1, 1)

N_b = 100
N_f = 10000

lb = rb = np.array([0.0])
ub = lftb = np.array([1.0])

X_f = lb + (ub - lb) * lhs(2, N_f)  # 生成N_f个随机点

idx_y = np.random.choice(y.shape[0], N_b, replace=False)
yb = y[idx_y,:]

X_lb = np.concatenate((yb, 0 * yb + lb[0]), 1) # u(x, 0)
X_ub = np.concatenate((yb, 0 * yb + ub[0]), 1) # u(x, 1)
X_rb = np.concatenate((0 * yb + rb[0], yb), 1) # u(0, y)
X_lftb = np.concatenate((0 * yb + lftb[0], yb), 1) # u(1, y)

u_rb = np.array(sin(pi * X_rb[:, 1:2]))
u_lftb = np.array(2 * sin(pi * X_lftb[:, 1:2]))

X, Y = np.meshgrid(x, y)
u_star = u

X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

u.shape


# In[3]:


class DNN(torch.nn.Module):
    """ DNN Class """
    
    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential(nn.Linear(1, 100), nn.GELU(),
                                 nn.Linear(100, 100), nn.GELU(),
                                 nn.Linear(100, 100), nn.GELU(),
                                 nn.Linear(100, 1))
        
    def forward(self, x):
        out = self.net(x)
        return out
    
class GKNN1(torch.nn.Module):
    """ DNN Class """
    
    def __init__(self):
        super(GKNN1, self).__init__()
        self.net1 = nn.Sequential(nn.Linear(2, 100), nn.Tanh(),
                                 nn.Linear(100, 100), nn.Tanh(),
                                 nn.Linear(100, 100), nn.Tanh(),
                                 nn.Linear(100, 1))
        
        
        
    def forward(self, x):
        
        return self.net1(x)  

    # def _initialize_weights(self):
    #     """ 初始化权重和偏置 """
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             # 使用Xavier/Glorot初始化权重
    #             init.xavier_uniform_(m.weight, gain=1.0)
    #             # 初始化偏置为0
    #             if m.bias is not None:
    #                 init.zeros_(m.bias)

class GKNN2(torch.nn.Module):
    """ DNN Class """
    
    def __init__(self):
        super(GKNN2, self).__init__()
        self.net1 = nn.Sequential(nn.Linear(2, 100), nn.Tanh(),
                                 nn.Linear(100, 100), nn.Tanh(),
                                 nn.Linear(100, 100), nn.Tanh(),
                                 nn.Linear(100, 1))
        
        
        
    def forward(self, x):
        
        return self.net1(x)  

    # def _initialize_weights(self):
    #     """ 初始化权重和偏置 """
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             # 使用Xavier/Glorot初始化权重
    #             init.xavier_uniform_(m.weight, gain=1.0)
    #             # 初始化偏置为0
    #             if m.bias is not None:
    #                 init.zeros_(m.bias)
                    
class GKNN3(torch.nn.Module):
    """ DNN Class """
    
    def __init__(self):
        super(GKNN3, self).__init__()
        self.net1 = nn.Sequential(nn.Linear(2, 100), nn.Tanh(),
                                 nn.Linear(100, 100), nn.Tanh(),
                                 nn.Linear(100, 100), nn.Tanh(),
                                 nn.Linear(100, 100), nn.Tanh(),
                                 nn.Linear(100, 100), nn.Tanh(),
                                 nn.Linear(100, 1))
        
        
        
    def forward(self, x):
        
        return self.net1(x)  

    def _initialize_weights(self):
        """ 初始化权重和偏置 """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Xavier/Glorot初始化权重
                init.xavier_uniform_(m.weight, gain=1.0)
                # 初始化偏置为0
                if m.bias is not None:
                    init.zeros_(m.bias)
    
class GKNN4(torch.nn.Module):
    """ DNN Class """
    
    def __init__(self):
        super(GKNN4, self).__init__()
        self.net1 = nn.Sequential(nn.Linear(2, 100), nn.Tanh(),
                                 nn.Linear(100, 100), nn.Tanh(),
                                 nn.Linear(100, 100), nn.Tanh(),
                                 nn.Linear(100, 100), nn.Tanh(),
                                 nn.Linear(100, 100), nn.Tanh(),
                                 nn.Linear(100, 1))
        
        
        
    def forward(self, x):
        
        return self.net1(x)  
    
    def _initialize_weights(self):
        """ 初始化权重和偏置 """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Xavier/Glorot初始化权重
                init.xavier_uniform_(m.weight, gain=1.0)
                # 初始化偏置为0
                if m.bias is not None:
                    init.zeros_(m.bias)
                    


# In[4]:


class PINN():
    def __init__(self, X_f, X_lb, X_ub, X_rb, X_lftb, u_lb, u_ub, epochs, Diff, X_star, u_star):
        
        self.rba = 1
        if self.rba == 1:
            self.rsum = 0
            self.eta = 0.001
            self.gamma = 0.999
        self.epochs = epochs
        self.Diff = Diff
        
        self.best_model_params = []
        self.min_l2_error = float('inf')
        
        self.X_star = X_star
        self.u_star = u_star
        
        self.iter = 0
        self.exec_time = 0
        
        self.x_f = torch.tensor(X_f[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_f = torch.tensor(X_f[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_lb = torch.tensor(X_lb[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_lb = torch.tensor(X_lb[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_ub = torch.tensor(X_ub[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_ub = torch.tensor(X_ub[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_rb = torch.tensor(X_rb[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_rb = torch.tensor(X_rb[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.u_rb = torch.tensor(u_rb, dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_lftb = torch.tensor(X_lftb[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_lftb = torch.tensor(X_lftb[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.u_lftb = torch.tensor(u_lftb, dtype=torch.float32, requires_grad=True).to(device)
        
        self.gknn1 = GKNN1().to(device)
        
        self.it = []; self.l2 = []; self.loss_u = []; self.loss_r = []; self.losses = []; self.lx = []
        
        self.optimizer1 = torch.optim.Adam(self.gknn1.parameters(), lr=1e-3, betas=(0.9, 0.999))
        
        
        
    def net_u(self, x, y):
        a = torch.cat((x, y), 1)
        
        u = self.gknn1(a)
        return u
    
 
    
    def net_r(self, x, y):
        u = self.net_u(x, y)
        
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        u_y = grad(u, y)
        u_yy = grad(u_y, y)
        r = - self.Diff * (u_xx + u_yy) + u_x
        return r
    
   
    
    
    def train(self):
        self.gknn1.train()
        
        a = timeit.default_timer()
        for epoch in range(self.epochs):
            start_time = timeit.default_timer()
            
            u_lb_pred = self.net_u(self.x_lb, self.y_lb)
            u_ub_pred = self.net_u(self.x_ub, self.y_ub)
            u_rb_pred = self.net_u(self.x_rb, self.y_rb)
            u_lftb_pred = self.net_u(self.x_lftb, self.y_lftb)
            
            loss_u = torch.mean((u_lb_pred - 0) ** 2) + \
                     torch.mean((u_ub_pred - 0) ** 2) + \
                     torch.mean((u_rb_pred - self.u_rb) ** 2) + \
                     torch.mean((u_lftb_pred - self.u_lftb) ** 2)
            
            r_pred = self.net_r(self.x_f, self.y_f)
            
            r_norm = self.eta * torch.abs(r_pred) / torch.max(torch.abs(r_pred))
            self.rsum = (self.rsum * self.gamma + r_norm).detach()
            loss_r = torch.mean((r_pred) ** 2)
            loss = loss_u + loss_r 
            
            self.optimizer1.zero_grad()
            loss.backward()
            self.optimizer1.step()
            
            end_time = timeit.default_timer()
            self.exec_time = end_time - start_time
            
            if epoch % 100 ==0:
                u_pred = self.predict(self.X_star)
                l2_error = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
                if loss.item() < self.min_l2_error:
                    self.min_l2_error = loss.item()
                    self.best_u_pred = u_pred
                
                self.loss_u.append(loss_u.item())
                self.loss_r.append(loss_r.item())
                self.losses.append(loss.item())
                self.l2.append(l2_error)
                
                print('It %d, Loss %.3e, min_Loss %.3e, l2_error %.3e, min_l2 %.3e, Time %.2f s' % (epoch, loss.item(), min(self.losses), l2_error, min(self.l2), self.exec_time))
                
                # if loss.item() < 3 * 1e-7:
                #     b = timeit.default_timer() - a
                #     print('Time for stop epochs %.2f s' % (b))
                #     break
                if epoch % 100000 ==0:
                    b = timeit.default_timer() - a
                    print('Time for 100000 epochs %.2f s' % (b))
                    
    def predict(self, X):
        x = torch.tensor(X[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        y = torch.tensor(X[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.gknn1.eval()
        
        u = self.net_u(x, y)
        u = u.detach().cpu().numpy()
        return u  

    def get_best_u_pred(self):
        return self.best_u_pred

class RBAGKPINN():
    def __init__(self, X_f, X_lb, X_ub, X_rb, X_lftb, u_lb, u_ub, epochs, Diff, X_star, u_star):
        
        self.rba = 1
        if self.rba == 1:
            self.rsum = 0
            self.eta = 0.001
            self.gamma = 0.999
        self.epochs = epochs
        self.Diff = Diff
        
        self.best_model_params = []
        self.min_l2_error = float('inf')
        
        self.X_star = X_star
        self.u_star = u_star
        
        self.iter = 0
        self.exec_time = 0
        
        self.x_f = torch.tensor(X_f[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_f = torch.tensor(X_f[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_lb = torch.tensor(X_lb[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_lb = torch.tensor(X_lb[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_ub = torch.tensor(X_ub[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_ub = torch.tensor(X_ub[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_rb = torch.tensor(X_rb[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_rb = torch.tensor(X_rb[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.u_rb = torch.tensor(u_rb, dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_lftb = torch.tensor(X_lftb[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_lftb = torch.tensor(X_lftb[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.u_lftb = torch.tensor(u_lftb, dtype=torch.float32, requires_grad=True).to(device)
        
        self.gknn1 = GKNN1().to(device)
        self.gknn2 = GKNN2().to(device)
        # self.gknn3 = GKNN3().to(device) 
        
        self.it = []; self.l2 = []; self.loss_u = []; self.loss_r = []; self.losses = []; self.lx = []
        
        self.optimizer1 = torch.optim.Adam(self.gknn1.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.optimizer2 = torch.optim.Adam(self.gknn2.parameters(), lr=1e-3, betas=(0.9, 0.999))
        # self.optimizer3 = torch.optim.Adam(self.gknn3.parameters(), lr=1e-3, betas=(0.9, 0.999))
        # self.optimizer4 = torch.optim.Adam(self.gknn4.parameters(), lr=1e-3, betas=(0.9, 0.999))
        
        
    def net_u(self, x, y):
        a = torch.cat((x, y), 1)
        

        # 假设 self.gknn1 是您的神经网络模型
        
        u = self.gknn1(a) + self.gknn2(a) * torch.exp((x - 1) / self.Diff) 
        return u
    
 
    
    def net_r(self, x, y):
        u = self.net_u(x, y)
        
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        u_y = grad(u, y)
        u_yy = grad(u_y, y)
        r = - self.Diff * (u_xx + u_yy) + u_x
        return r
    
   
    
    
    def train(self):
        self.gknn1.train()
        self.gknn2.train()
        # self.gknn3.train()
        
        a = timeit.default_timer()
        for epoch in range(self.epochs):
            start_time = timeit.default_timer()
            
            u_lb_pred = self.net_u(self.x_lb, self.y_lb)
            u_ub_pred = self.net_u(self.x_ub, self.y_ub)
            u_rb_pred = self.net_u(self.x_rb, self.y_rb)
            u_lftb_pred = self.net_u(self.x_lftb, self.y_lftb)
            
            loss_u = torch.mean((u_lb_pred - 0) ** 2) + \
                     torch.mean((u_ub_pred - 0) ** 2) + \
                     torch.mean((u_rb_pred - self.u_rb) ** 2) + \
                     torch.mean((u_lftb_pred - self.u_lftb) ** 2)
            
            r_pred = self.net_r(self.x_f, self.y_f)
            
            r_norm = self.eta * torch.abs(r_pred) / torch.max(torch.abs(r_pred))
            self.rsum = (self.rsum * self.gamma + r_norm).detach()
            loss_r = torch.mean((self.rsum * r_pred) ** 2)
            loss = loss_u + loss_r 
            
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            # self.optimizer3.zero_grad()
            
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()
            # self.optimizer3.step()
            
            end_time = timeit.default_timer()
            self.exec_time = end_time - start_time
            
            if epoch % 100 ==0:
                u_pred = self.predict(self.X_star)
                l2_error = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
                if l2_error < self.min_l2_error:
                    self.min_l2_error = l2_error
                    self.best_u_pred = u_pred
                
                self.loss_u.append(loss_u.item())
                self.loss_r.append(loss_r.item())
                self.losses.append(loss.item())
                self.l2.append(l2_error)
                
                print('It %d, Loss %.3e, min_Loss %.3e, l2_error %.3e, min_l2 %.3e, Time %.2f s' % (epoch, loss.item(), min(self.losses), l2_error, min(self.l2), self.exec_time))
                
                # if loss.item() < 3 * 1e-7:
                #     b = timeit.default_timer() - a
                #     print('Time for stop epochs %.2f s' % (b))
                #     break
                if epoch % 100000 ==0:
                    b = timeit.default_timer() - a
                    print('Time for 100000 epochs %.2f s' % (b))
                    
    def predict(self, X):
        x = torch.tensor(X[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        y = torch.tensor(X[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.gknn1.eval()
        self.gknn2.eval()
       
        u = self.net_u(x, y)
        u = u.detach().cpu().numpy()
        return u        
    
    def get_best_u_pred(self):
        return self.best_u_pred


# In[5]:


#model = RBAGKPINN(x_in, u_in, epochs, Diff)
# model1 = RBAGKPINN(X_f, X_lb, X_ub, X_rb, X_lftb, u_rb, u_lftb, epochs, Diff, X_star, u_star)
# model1.train()
# u_pred_pinn = model1.predict(X_star)


model2 = RBAGKPINN(X_f, X_lb, X_ub, X_rb, X_lftb, u_rb, u_lftb, epochs, Diff, X_star, u_star)
model2.train()
u_pred_gkpinn = model2.get_best_u_pred()


# In[6]:


# import numpy as np
# u_pred1 = np.array(u_pred_pinn)  # 将结果转换为NumPy数组
# u_pred2 = np.array(u_pred_gkpinn)
# # 或者
# np.savetxt('u_pred_pinn.csv', u_pred1, delimiter=',')  # 保存为CSV文件
# np.savetxt('u_pred_gkpinn.csv', u_pred2, delimiter=',')  # 保存为CSV文件


# In[7]:


# a = model1.loss_u.index(min(model1.loss_u))
# b = model1.loss_r.index(min(model1.loss_r))
# a * 100, min(model1.loss_u) , b * 100, min(model1.loss_r)


# In[8]:


a = model2.loss_u.index(min(model2.loss_u))
b = model2.loss_r.index(min(model2.loss_r))
a * 100, min(model2.loss_u) , b * 100, min(model2.loss_r)


# In[9]:


# fig = plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
plt.plot(model2.loss_u, label='loss_u')
plt.plot(model2.loss_r, '--', label='loss_r')
plt.yscale('log')
plt.legend(loc='lower right')
plt.show()


# In[10]:


plt.plot(model2.l2)
plt.xscale('log')
plt.yscale('log')
plt.show()
c = model2.l2.index(min(model2.l2))
c, "{:.3e}".format(min(model2.l2)), "{:.3e}".format(model2.losses[c]), "{:.3e}".format(model2.loss_u[c]), "{:.3e}".format(model2.loss_r[c])


# In[11]:


import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter

# error_U1 = np.abs(u_star - u_pred_pinn).reshape(X.shape)
error_U2 = np.abs(u_star - u_pred_gkpinn).reshape(X.shape)

# vmin = min(error_U1.min(), error_U2.min())
# vmax = max(error_U1.max(), error_U2.max())

# fig1, ax1 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
# surf1 = ax1.plot_surface(X, Y, error_U1, cmap='jet', edgecolor='none')
# norm1 = mcolors.LogNorm(vmin=1e-4, vmax=1e-3)
# cbar1 = plt.colorbar(surf1, ax=ax1, norm=norm1)

# cbar1.set_label('Normalized Error Scale (0 to 5)')

# # 设置坐标轴标签
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_zlabel('Error U1')

# bins = [0.0005, 0.0015, 0.0025, 0.25, 1, 1.75]
# nbin = len(bins) - 1
# 绘制第二个模型的误差图
Z = u_pred_gkpinn.reshape(X.shape)

fig2, ax2 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
surf2 = ax2.plot_surface(X, Y, Z, cmap='jet', edgecolor='none')
norm2 = mcolors.LogNorm(vmin=1e-4, vmax=1e-3)
cbar2 = plt.colorbar(surf2, ax=ax2, norm=norm2)

# cbar2.set_label('Normalized Error Scale (0 to 5)')

# 设置坐标轴标签
ax2.set_xlabel('x')
ax2.set_ylabel('y')
# ax2.set_zlabel('Error U2')

ax2.tick_params(axis='z', pad=4)
ax2.tick_params(axis='x', pad=-5)
ax2.tick_params(axis='y', pad=2)
ax2.tick_params(which='major', labelsize=12)
ax2.locator_params(axis='z', nbins=5)
# 显示图表
plt.show()


# In[12]:


import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter

# error_U1 = np.abs(u_star - u_pred_pinn).reshape(X.shape)
error_U2 = np.abs(u_star - u_pred_gkpinn).reshape(X.shape)

# vmin = min(error_U1.min(), error_U2.min())
# vmax = max(error_U1.max(), error_U2.max())

# fig1, ax1 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
# surf1 = ax1.plot_surface(X, Y, error_U1, cmap='jet', edgecolor='none')
# norm1 = mcolors.LogNorm(vmin=1e-4, vmax=1e-3)
# cbar1 = plt.colorbar(surf1, ax=ax1, norm=norm1)

# cbar1.set_label('Normalized Error Scale (0 to 5)')

# # 设置坐标轴标签
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_zlabel('Error U1')

# bins = [0.0005, 0.0015, 0.0025, 0.25, 1, 1.75]
# nbin = len(bins) - 1
# 绘制第二个模型的误差图
fig2, ax2 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
surf2 = ax2.plot_surface(X, Y, error_U2, cmap='jet', edgecolor='none')
norm2 = mcolors.LogNorm(vmin=1e-4, vmax=1e-3)
cbar2 = plt.colorbar(surf2, ax=ax2, norm=norm2)

# cbar2.set_label('Normalized Error Scale (0 to 5)')

# 设置坐标轴标签
ax2.set_xlabel('x')
ax2.set_ylabel('y')
# ax2.set_zlabel('Error U2')

# 显示图表
plt.show()


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties

# 假设 u_star, u_pred_pinn, u_pred_gkpinn, X, Y 已经定义

# 计算误差
error_U1 = np.abs(u_star - u_pred_pinn).reshape(X.shape)
error_U2 = np.abs(u_pred_gkpinn - u_star).reshape(X.shape)

# 计算误差范围
vmin = min(error_U1.min(), error_U2.min())
vmax = max(error_U1.max(), error_U2.max())

font = FontProperties(size=16, family='serif')

cmap = (mpl.colors.ListedColormap(['red', 'green', 'blue', 'cyan'])
        .with_extremes(over='0.25', under='0.75'))
bounds = [0.0005, 0.0015,0.0025, 0.25, 1, 1.75]
# 创建第一个图和颜色条
fig1, ax1 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
surf1 = ax1.plot_surface(X, Y, error_U1, cmap='jet', edgecolor='none')
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
norm = mcolors.Normalize(vmin=vmin, vmax=10)
cbar1 = plt.colorbar(surf1, ax=ax1, orientation='vertical', norm=norm, ticks=bounds)
# formatter = ticker.LogFormatter(10, labelOnlyBase=False)  # labelOnlyBase=True 表示只显示底数的指数部分
cbar1.ax.set_yticklabels(cbar1.ax.get_yticklabels(), fontproperties=font)

# 创建第二个图和颜色条
fig2, ax2 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
surf2 = ax2.plot_surface(X, Y, error_U2, cmap='jet', edgecolor='none')
cbar2 = plt.colorbar(surf2, ax=ax2, orientation='vertical', norm=norm, ticks=bounds)
cbar2.ax.set_yticklabels(cbar2.ax.get_yticklabels(), fontproperties=font)


# 显示图表
plt.show()
vmin, vmax 


# In[ ]:


# import numpy as np
# import scipy.sparse as sp
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as mcolors
# from matplotlib.ticker import ScalarFormatter

# # u_pred = model.get_best_u_pred()
# # # u_pred = model.predict(X_star)
# # l2_error = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
# # print('l2_error: %e' % (l2_error))
# vmin = min(np.abs(error_U1).min(), np.abs(error_U2).min())
# vmax = max(np.abs(error_U1).max(), np.abs(error_U2).max())
# # error_u = u_star - u_pred
# # error_U = error_u.reshape(X.shape) 
# # Z = u_pred.reshape(X.shape)
# # E = error_U.reshape(X.shape)
# # fig = plt.figure(figsize=(24, 6))

# # ax1 = fig.add_subplot(131, projection='3d')
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# error_U1 = np.abs(error_U1)
# # 绘制3D曲面图
# surf = ax.plot_surface(X, Y, error_U1, cmap='jet', edgecolor='none')
# norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
# fig.colorbar(surf, norm=norm)
# # 设置坐标轴标签
# ax.set_xlabel('x', labelpad=-2, fontsize=12)
# ax.set_ylabel('y', labelpad=8, fontsize=12)
# # ax.set_zlabel('u', labelpad=6, fontsize=12)

# # 设置视角
# ax.view_init(elev=30, azim=-70)

# # 设置坐标轴刻度参数
# ax.tick_params(axis='z', pad=4)
# ax.tick_params(axis='x', pad=-5)
# ax.tick_params(axis='y', pad=2)
# ax.tick_params(which='major', labelsize=12)
# ax.locator_params(axis='z', nbins=5)
# plt.show()
# vmin,vmax


# In[ ]:


# import numpy as np
# import scipy.sparse as sp
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as mcolors
# from matplotlib.ticker import ScalarFormatter


# vmin = min(np.abs(error_U1).min(), np.abs(error_U2).min())
# vmax = max(np.abs(error_U1).max(), np.abs(error_U2).max())

# common_vmin = 0
# common_vmax = 5

# # 绘制第一个模型的误差图
# fig1, ax1 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
# surf1 = ax1.plot_surface(X, Y, error_U1, cmap='jet', edgecolor='none')
# norm1 = mcolors.Normalize(vmin=common_vmin, vmax=common_vmax)
# cbar1 = plt.colorbar(surf1, ax=ax1, norm=norm1)
# cbar1.set_label('Normalized Error Scale (0 to 5)')

# # 设置坐标轴标签
# ax1.set_xlabel('x')
# ax1.set_ylabel('t')
# ax1.set_zlabel('Error U1')

# # 绘制第二个模型的误差图
# fig2, ax2 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
# surf2 = ax2.plot_surface(X, Y, error_U2, cmap='jet', edgecolor='none')
# norm2 = mcolors.Normalize(vmin=0.03, vmax=0.05)
# cbar2 = plt.colorbar(surf2, ax=ax2, norm=norm2)
# cbar2.set_label('Normalized Error Scale (0 to 5)')

# # 设置坐标轴标签
# ax2.set_xlabel('x')
# ax2.set_ylabel('t')
# ax2.set_zlabel('Error U2')

# # 显示图表
# plt.show()
# vmin,vmax


# In[ ]:


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as mcolors

# # 假设 X 和 T 是你的数据网格
# X, T = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))

# # 假设 error_U1 和 error_U2 是你的两个模型的误差数据
# error_U1 = np.sin(X) * np.cos(T)  # 模拟数据1
# error_U2 = 0.003 * np.cos(X) * np.sin(T)  # 模拟数据2

# # 计算两个误差数据的最小值和最大值
# vmin = min(error_U1.min(), error_U2.min())
# vmax = max(error_U1.max(), error_U2.max())

# # 创建一个图表和两个子图
# fig = plt.figure(figsize=(14, 6))

# # 第一个子图
# ax1 = fig.add_subplot(121, projection='3d')
# surf1 = ax1.plot_surface(X, T, error_U1, cmap='jet', edgecolor='none')
# norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
# cbar1 = fig.colorbar(surf1, ax=ax1, norm=norm, pad=0.1)
# cbar1.set_label('Error Scale (%.3f to %.3f)' % (vmin, vmax))

# # 第二个子图
# ax2 = fig.add_subplot(122, projection='3d')
# surf2 = ax2.plot_surface(X, T, error_U2, cmap='jet', edgecolor='none')
# cbar2 = fig.colorbar(surf2, ax=ax2, norm=norm, pad=0.1)
# cbar2.set_label('Error Scale (%.3f to %.3f)' % (vmin, vmax))

# # 设置坐标轴标签
# for ax in [ax1, ax2]:
#     ax.set_xlabel('x')
#     ax.set_ylabel('t')
#     ax.set_zlabel('Error')

# # 显示图表
# plt.show()


# In[ ]:


# import numpy as np
# import scipy.sparse as sp
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as mcolors
# from matplotlib.ticker import ScalarFormatter

# u_pred = model.get_best_u_pred()
# # u_pred = model.predict(X_star)
# l2_error = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
# print('l2_error: %e' % (l2_error))

# error_u = u_star - u_pred
# error_U = error_u.reshape(X.shape) 
# Z = u_pred.reshape(X.shape)

# fig = plt.figure(figsize=(24, 6))

# ax1 = fig.add_subplot(131, projection='3d')
# surf = ax1.plot_surface(X, Y, Z, cmap='jet', edgecolor='none')

# norm = mcolors.Normalize(vmin=Z.min(), vmax=Z.max())
# cbar = fig.colorbar(surf, norm=norm, location='left', pad=0)
# cbar.ax.tick_params(labelsize=16)

# ax1.set_xlabel('x', labelpad=-2, fontsize=16)
# ax1.set_ylabel('y', labelpad=8, fontsize=16)
# # ax1.set_zlabel('u', labelpad=12, fontsize=16, fontweight='bold')

# ax1.view_init(elev=30, azim=-65)  

# ax1.tick_params(axis='z', pad=8)
# ax1.tick_params(axis='x', pad=-5)
# ax1.tick_params(axis='y', pad=2)
# ax1.tick_params(axis='both', which='major', labelsize=14)

# ax2 = fig.add_subplot(132, projection='3d')
# surf = ax2.plot_surface(X, Y, error_U, cmap='jet', edgecolor='none')

# norm = mcolors.Normalize(vmin=error_U.min(), vmax=error_U.max())
# cbar = fig.colorbar(surf, norm=norm, location='left', pad=0)
# cbar.ax.tick_params(labelsize=16)

# ax2.set_xlabel('x', labelpad=-2, fontsize=16)
# ax2.set_ylabel('y', labelpad=8, fontsize=16)
# # ax2.set_zlabel('u', labelpad=12, fontsize=16, fontweight='bold')

# ax2.view_init(elev=30, azim=-65)  

# ax2.tick_params(axis='z', pad=8)
# ax2.tick_params(axis='x', pad=-5)
# ax2.tick_params(axis='y', pad=2)
# ax2.tick_params(axis='both', which='major', labelsize=14)

# ax3 = fig.add_subplot(133)
# ax3.tick_params(axis='both', which='major', labelsize=16)
# ax3.plot(model.loss_u, label='loss_u')
# ax3.plot(model.loss_r, '--', label='loss_r')
# ax3.set_yscale('log')
# ax3.legend(loc='upper right')
# plt.show()

