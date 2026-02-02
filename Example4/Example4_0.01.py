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


# In[2]:


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

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from pyDOE import lhs
from numpy import pi, sin, sinh, sqrt
import scipy.io

epochs = 100001

Diff = 1e-2

nPt = 1001
x0 = 0.
x1 = 1.
Re = 100

data = np.loadtxt("4—0.01.csv", delimiter=",", skiprows=1)  # 跳过标题行

x1 = data[:, 0].flatten()[:, None]
y1 = data[:, 1].flatten()[:, None]
u = data[:, 2].flatten()[:, None]

x = np.linspace(0, 1, nPt).reshape(-1, 1)
y = np.linspace(0, 1, nPt).reshape(-1, 1)

N_b = 200
N_f = 20000

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



# In[4]:


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
                    


# In[5]:


class PINN():
    def __init__(self, X_f, X_lb, X_ub, X_rb, X_lftb, u_lb, u_ub, epochs, Diff, X_star, u_star):
        
        self.rba = 1
        if self.rba == 1:
            self.rsum = 0
            self.eta = 0.001
            self.gamma = 0.999
        self.epochs = epochs
        self.Diff = Diff
        
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
                
                self.loss_u.append(loss_u.item())
                self.loss_r.append(loss_r.item())
                self.losses.append(loss.item())
                self.l2.append(l2_error)
                
                print('It %d, Loss %.3e, min_Loss %.3e, l2_error %.3e, min_l2 %.3e, Time %.2f s' % (epoch, loss.item(), min(self.losses), l2_error, min(self.l2), self.exec_time))
               
                # if loss.item() < 5 * 1e-7:
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


class RBAGKPINN():
    def __init__(self, X_f, X_lb, X_ub, X_rb, X_lftb, u_lb, u_ub, epochs, Diff, X_star, u_star):
        
        self.rba = 1
        if self.rba == 1:
            self.rsum = 0
            self.eta = 0.001
            self.gamma = 0.999
        self.epochs = epochs
        self.Diff = Diff
        
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
                
                self.loss_u.append(loss_u.item())
                self.loss_r.append(loss_r.item())
                self.losses.append(loss.item())
                self.l2.append(l2_error)
                
                print('It %d, Loss %.3e, min_Loss %.3e, l2_error %.3e, min_l2 %.3e, Time %.2f s' % (epoch, loss.item(), min(self.losses), l2_error, min(self.l2), self.exec_time))
               
                # if loss.item() < 5 * 1e-7:
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


# In[6]:


#model = RBAGKPINN(x_in, u_in, epochs, Diff)
model = RBAGKPINN(X_f, X_lb, X_ub, X_rb, X_lftb, u_rb, u_lftb, epochs, Diff, X_star, u_star)
model.train()


# In[ ]:


a = model.loss_u.index(min(model.loss_u))
b = model.loss_r.index(min(model.loss_r))
a * 100, min(model.loss_u) , b * 100, min(model.loss_r)


# In[ ]:


fig = plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(model.loss_u)
plt.plot(model.loss_r)
plt.yscale('log')
plt.show()


# In[ ]:


plt.plot(model.l2)
plt.xscale('log')
plt.yscale('log')
plt.show()
c = model.l2.index(min(model.l2))
c, "{:.3e}".format(min(model.l2)), "{:.3e}".format(model.losses[c])


# In[ ]:


import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

u_pred = model.predict(X_star)

l2_error = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
print('l2_error: %e' % (l2_error))

error_u = u_star - u_pred
error_U = error_u.reshape(X.shape) 
Z = u_pred.reshape(X.shape)

fig = plt.figure(figsize=(30, 14))
ax1 = fig.add_subplot(211, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='jet', edgecolor='none')
norm = mcolors.Normalize(vmin=Z.min(), vmax=Z.max())
cbar = fig.colorbar(surf, norm=norm)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u(x, y)')
ax1.set_title('3D Surface plot of u(x, y)')
ax1.view_init(elev=15, azim=-65)  

ax2 = fig.add_subplot(212, projection='3d')
surf = ax2.plot_surface(X, Y, error_U, cmap='jet', edgecolor='none')
norm = mcolors.Normalize(vmin=error_U.min(), vmax=error_U.max())
cbar = fig.colorbar(surf, norm=norm)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('u(x, y)')
ax2.set_title('error_u')
ax2.view_init(elev=15, azim=-65)  

plt.show()


