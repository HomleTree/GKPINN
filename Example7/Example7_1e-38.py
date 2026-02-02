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
import os

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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.cuda.empty_cache()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

epochs = 100001

Diff = 1 * 1e-38

nPt = 400

N0 = 100
N_b = 100
N_f = 10000

# data = np.loadtxt("u_data2.csv", delimiter=",", skiprows=0)  # 跳过标题行

# x1 = data[:, 0].flatten()[:, None]
# t1 = data[:, 1].flatten()[:, None]
# u = data[:, 2].flatten()[:, None]

lb = np.array([0.0])
ub = np.array([1.0])

x = np.linspace(0, 1, 1001).reshape(-1, 1)
t = np.linspace(0, 1, 5001).reshape(-1, 1)

X_f = lb + (ub - lb) * lhs(2, N_f)

idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x,:]
u0 = np.sin(2 * np.pi * x0)

idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t,:]

X0 = np.concatenate((x0, 0 * x0), 1) # u(x,0)
X_lb = np.concatenate((0 * tb + lb[0], tb), 1) # u(0, t)
X_ub = np.concatenate((0 * tb + ub[0], tb), 1) # u(1, t)

X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

X_star.shape


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

    def _initialize_weights(self):
        """ 初始化权重和偏置 """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Xavier/Glorot初始化权重
                init.xavier_uniform_(m.weight, gain=1.0)
                # 初始化偏置为0
                if m.bias is not None:
                    init.zeros_(m.bias)

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

    def _initialize_weights(self):
        """ 初始化权重和偏置 """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Xavier/Glorot初始化权重
                init.xavier_uniform_(m.weight, gain=1.0)
                # 初始化偏置为0
                if m.bias is not None:
                    init.zeros_(m.bias)
                    
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
    def __init__(self, u0, X0, X_f, X_lb, X_ub, epochs, Diff, X_star):
        
        self.rba = 1
        if self.rba == 1:
            self.rsum = 0
            self.eta = 0.001
            self.gamma = 0.999
        
        self.min_l2_error = float('inf')
        
        self.epochs = epochs
        self.Diff = Diff
        
        self.X_star = X_star
        # self.u_star = u_star
        
        self.iter = 0
        self.exec_time = 0
        
        self.x0 = torch.tensor(X0[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t0 = torch.tensor(X0[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.u0 = torch.tensor(u0, dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_f = torch.tensor(X_f[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_lb = torch.tensor(X_lb[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_lb = torch.tensor(X_lb[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_ub = torch.tensor(X_ub[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_ub = torch.tensor(X_ub[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.gknn1 = GKNN1().to(device)
        
        self.it = []; self.l2 = []; self.loss_u = []; self.loss_r = []; self.losses = []
        self.loss_0 = []; self.min_loss = float('inf')
        
        self.optimizer1 = torch.optim.Adam(self.gknn1.parameters(), lr=1e-3, betas=(0.9, 0.999))
        
    def net_u(self, x, t):
        a = torch.cat((x, t), 1)
        u = self.gknn1(a)
        return u
    
    def net_r(self, x, t):
        u = self.net_u(x, t)
        
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        u_t = grad(u, t)
        
        r = u_t - self.Diff * u_xx - u_x - u
        return r
    
    
    def train(self):
        self.gknn1.train()
         
        a = timeit.default_timer()
        for epoch in range(self.epochs):
            start_time = timeit.default_timer()
            
            u_0_pred = self.net_u(self.x0, self.t0)
            loss_0 = torch.mean((u_0_pred - self.u0) ** 2)
            
            u_lb_pred = self.net_u(self.x_lb, self.t_lb)
            u_ub_pred = self.net_u(self.x_ub, self.t_ub)
            loss_u = torch.mean((u_lb_pred - 0) ** 2) + \
                     torch.mean((u_ub_pred - 1) ** 2)
            
            r_pred = self.net_r(self.x_f, self.t_f)
            
            r_norm = self.eta * torch.abs(r_pred) / torch.max(torch.abs(r_pred))
            self.rsum = (self.rsum * self.gamma + r_norm).detach()
            loss_r = torch.mean((r_pred) ** 2)
            
            loss = loss_0 + loss_u + loss_r 
            
            self.optimizer1.zero_grad()
            
            loss.backward()
            
            self.optimizer1.step()
           
            
            end_time = timeit.default_timer()
            self.exec_time = end_time - start_time
            if epoch % 100 ==0:
                u_pred = self.predict(self.X_star)
                l2_error = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
                if l2_error < self.min_l2_error:
                    self.min_l2_error = l2_error
                    self.best_u_pred = u_pred
                
                self.loss_0.append(loss_0.item())
                self.loss_u.append(loss_u.item())
                self.loss_r.append(loss_r.item())
                self.losses.append(loss.item())
                self.l2.append(l2_error)
                
                print('It %d, Loss %.3e, min_Loss %.3e, l2_error %.3e, min_l2 %.3e, Time %.2f s' % (epoch, loss.item(), min(self.losses), l2_error, min(self.l2), self.exec_time))
                
                # if loss.item() < 1 * 1e-14:
                #     b = timeit.default_timer() - a
                #     print('Time for stop epochs %.2f s' % (b))
                #     break
                if epoch % 100000 ==0:
                    b = timeit.default_timer() - a
                    print('Time for 100000 epochs %.2f s' % (b))
    
    def predict(self, X):
        x = torch.tensor(X[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        t = torch.tensor(X[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.gknn1.eval()
      
        u = self.net_u(x, t)
        u = u.detach().cpu().numpy()
        return u   
    
    def get_best_u_pred(self):
        return self.best_u_pred         



        

class RBAGKPINN():
    def __init__(self, u0, X0, X_f, X_lb, X_ub, epochs, Diff, X_star):
        
        self.rba = 1
        if self.rba == 1:
            self.rsum = 0
            self.eta = 0.001
            self.gamma = 0.999
        
        self.min_l2_error = float('inf')
        
        self.epochs = epochs
        self.Diff = Diff
        
        self.X_star = X_star
        # self.u_star = u_star
        
        self.iter = 0
        self.exec_time = 0
        
        self.x0 = torch.tensor(X0[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t0 = torch.tensor(X0[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.u0 = torch.tensor(u0, dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_f = torch.tensor(X_f[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_lb = torch.tensor(X_lb[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_lb = torch.tensor(X_lb[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_ub = torch.tensor(X_ub[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_ub = torch.tensor(X_ub[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.gknn1 = GKNN1().to(device)
        self.gknn2 = GKNN2().to(device)
        
        self.it = []; self.l2 = []; self.loss_u = []; self.loss_r = []; self.losses = []
        self.loss_0 = []; self.min_loss = float('inf')
        
        self.optimizer1 = torch.optim.Adam(self.gknn1.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.optimizer2 = torch.optim.Adam(self.gknn2.parameters(), lr=1e-3, betas=(0.9, 0.999))
        
    def net_u(self, x, t):
        a = torch.cat((x, t), 1)
        u = self.gknn1(a) + self.gknn2(a) * torch.exp(-x / self.Diff) 
        return u
    
    def net_r(self, x, t):
        u = self.net_u(x, t)
        
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        u_t = grad(u, t)
        
        r = u_t - self.Diff * u_xx - u_x - u
        return r
    
    
    def train(self):
        self.gknn1.train()
        self.gknn2.train()
        
        a = timeit.default_timer()
        for epoch in range(self.epochs):
            start_time = timeit.default_timer()
            
            u_0_pred = self.net_u(self.x0, self.t0)
            loss_0 = torch.mean((u_0_pred - self.u0) ** 2)
            
            u_lb_pred = self.net_u(self.x_lb, self.t_lb)
            u_ub_pred = self.net_u(self.x_ub, self.t_ub)
            loss_u = torch.mean((u_lb_pred - 0) ** 2) + \
                     torch.mean((u_ub_pred - 1) ** 2)
            
            r_pred = self.net_r(self.x_f, self.t_f)
            
            r_norm = self.eta * torch.abs(r_pred) / torch.max(torch.abs(r_pred))
            self.rsum = (self.rsum * self.gamma + r_norm).detach()
            loss_r = torch.mean((self.rsum * r_pred) ** 2)
            
            loss = loss_0 + loss_u + loss_r 
            
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            
            loss.backward()
            
            self.optimizer1.step()
            self.optimizer2.step()
            
            end_time = timeit.default_timer()
            self.exec_time = end_time - start_time
            if epoch % 100 ==0:
                u_pred = self.predict(self.X_star)
                # l2_error = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
                if loss.item() < self.min_l2_error:
                    self.min_l2_error = loss.item()
                    self.best_u_pred = u_pred
                
                self.loss_0.append(loss_0.item())
                self.loss_u.append(loss_u.item())
                self.loss_r.append(loss_r.item())
                self.losses.append(loss.item())
                # self.l2.append(l2_error)
                
                print('It %d, Loss %.3e, min_Loss %.3e, Time %.2f s' % (epoch, loss.item(), min(self.losses), self.exec_time))
                
                # if loss.item() < 1 * 1e-14:
                #     b = timeit.default_timer() - a
                #     print('Time for stop epochs %.2f s' % (b))
                #     break
                if epoch % 100000 ==0:
                    b = timeit.default_timer() - a
                    print('Time for 100000 epochs %.2f s' % (b))
    
    def predict(self, X):
        x = torch.tensor(X[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        t = torch.tensor(X[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.gknn1.eval()
        self.gknn2.eval()
        
        u = self.net_u(x, t)
        u = u.detach().cpu().numpy()
        return u   
    
    def get_best_u_pred(self):
        return self.best_u_pred         


# In[5]:


#model = PINN(x_in, u_in, epochs, Diff)
model = RBAGKPINN(u0, X0 ,X_f, X_lb, X_ub, epochs, Diff, X_star)
#model = RBAGKPINN(x_in, u_in, epochs, Diff)
model.train()


# In[6]:


a = model.loss_u.index(min(model.loss_u))
b = model.loss_r.index(min(model.loss_r))
a * 100, min(model.loss_u), b * 100, min(model.loss_r)


# In[7]:


fig = plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(model.loss_u, label='loss_u')
plt.plot(model.loss_r, '--', label='loss_r')
plt.yscale('log')
plt.legend(loc='lower right')
plt.show()


# In[8]:


# plt.plot(model.l2)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
# c = model.l2.index(min(model.l2))
# c, "{:.3e}".format(min(model.l2)), "{:.3e}".format(model.losses[c]), "{:.3e}".format(model.loss_u[c]), "{:.3e}".format(model.loss_r[c])


# In[9]:


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as mcolors

# u_pred = model.get_best_u_pred()
# min_Loss = model.min_l2_error
# # l2_error = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
# # print('l2_error: %e' % (l2_error))

# # error_u = u_star - u_pred
# # error_U = error_u.reshape(X.shape) 
# Z = u_pred.reshape(X.shape)

# fig = plt.figure(figsize=(24, 6))

# ax1 = fig.add_subplot(131, projection='3d')
# surf = ax1.plot_surface(X, T, Z, cmap='jet', edgecolor='none')

# norm = mcolors.Normalize(vmin=Z.min(), vmax=Z.max())
# cbar = fig.colorbar(surf, norm=norm, location='left', pad=0)
# cbar.ax.tick_params(labelsize=16)

# ax1.set_xlabel('x', labelpad=-2)
# ax1.set_ylabel('t', labelpad=8)
# # ax1.set_zlabel('u', labelpad=12, fontsize=16, fontweight='bold')

# ax1.view_init(elev=30, azim=-65)  

# ax1.tick_params(axis='z', pad=8)
# ax1.tick_params(axis='x', pad=-5)
# ax1.tick_params(axis='y', pad=2)
# ax1.tick_params(axis='both', which='major', labelsize=14)

# # ax2 = fig.add_subplot(132, projection='3d')
# # surf = ax2.plot_surface(X, T, error_U, cmap='jet', edgecolor='none')

# # norm = mcolors.Normalize(vmin=error_U.min(), vmax=error_U.max())
# # cbar = fig.colorbar(surf, norm=norm, location='left', pad=0)
# # cbar.ax.tick_params(labelsize=16)

# # ax2.set_xlabel('x', labelpad=-2, fontsize=16)
# # ax2.set_ylabel('t', labelpad=8, fontsize=16)
# # ax2.set_zlabel('u', labelpad=12, fontsize=16, fontweight='bold')

# # ax2.view_init(elev=30, azim=-75)  

# # ax2.tick_params(axis='z', pad=8)
# # ax2.tick_params(axis='x', pad=-5)
# # ax2.tick_params(axis='y', pad=2)
# # ax2.tick_params(axis='both', which='major', labelsize=14)

# ax3 = fig.add_subplot(132)
# ax3.tick_params(axis='both', which='major', labelsize=16)
# ax3.plot(model.loss_u, label='loss_u')
# ax3.plot(model.loss_r, '--', label='loss_r')
# ax3.set_yscale('log')
# ax3.legend(loc='upper right')
# plt.show()


# In[12]:


import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter

u_pred = model.get_best_u_pred()
# u_pred = model.predict(X_star)
# l2_error = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
# print('l2_error: %e' % (l2_error))

# error_u = u_star - u_pred
# error_U = error_u.reshape(X.shape) 
Z = u_pred.reshape(X.shape)

# fig = plt.figure(figsize=(24, 6))

# ax1 = fig.add_subplot(131, projection='3d')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制3D曲面图
surf = ax.plot_surface(X, T, Z, cmap='jet', edgecolor='none')
norm = mcolors.Normalize(vmin=Z.min(), vmax=Z.max())
fig.colorbar(surf, norm=norm)
# 设置坐标轴标签
ax.set_xlabel('x', labelpad=-2, fontsize=12)
ax.set_ylabel('t', labelpad=8, fontsize=12)
# ax.set_zlabel('u', labelpad=6, fontsize=12)

# 设置视角
ax.view_init(elev=30, azim=-135)

# 设置坐标轴刻度参数
ax.tick_params(axis='z', pad=4)
ax.tick_params(axis='x', pad=-5)
ax.tick_params(axis='y', pad=2)
ax.tick_params(which='major', labelsize=12)
ax.locator_params(axis='z', nbins=5)
# ax.set_zlim(bottom=-1, top=Z.max())
plt.show()

