#!/usr/bin/env python
# coding: utf-8

# In[25]:


import timeit
import torch
import torch.nn as nn
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


# In[26]:


import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

epochs = 10001

Diff = 1 * 1e-3

nPt = 400
x0 = 0.
x1 = 1.

x_in = np.linspace(x0, x1, nPt).reshape(-1, 1)

N_b = 50
N_f = 1000

lb = np.array([0.0])
ub = np.array([1.0])

X_f = lb + (ub - lb) * lhs(1, N_f)

def u(x):
    return np.exp(-x) + np.exp((1 + Diff) * (x - 1) / Diff)

idx_x = np.random.choice(x_in.shape[0], N_b, replace=False)
xb = x_in[idx_x,:]

X_lb = 0 * xb + lb[0]
X_ub = 0 * xb + ub[0]

u_in = u(x_in)

print('shape of x',x_in.shape)
print('shape of u',u_in.shape)

X_star = np.linspace(0, 1, nPt).reshape(-1, 1)
u_star = u(X_star)

plt.plot(X_star, u_star, label='Exact')
plt.legend(loc='upper right')
plt.show()


# In[27]:


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
        self.net1 = nn.Sequential(nn.Linear(1, 100), nn.Sigmoid(),
                                 nn.Linear(100, 100), nn.Sigmoid(),
                                 nn.Linear(100, 100), nn.Sigmoid(),
                                 nn.Linear(100, 1))
        
        self.net2 = nn.Sequential(nn.Linear(1, 100), nn.Sigmoid(),
                                  nn.Linear(100, 100), nn.Sigmoid(),
                                  nn.Linear(100, 1))
        
        self.net3 = nn.Sequential(nn.Linear(1, 100), nn.Sigmoid(),
                                  nn.Linear(100, 100), nn.Sigmoid(),
                                  nn.Linear(100, 100), nn.Sigmoid(),
                                  nn.Linear(100, 100), nn.Sigmoid(),
                                  nn.Linear(100, 1))
        
    def forward(self, x):
        
        return self.net1(x)  


class GKNN2(torch.nn.Module):
    """ DNN Class """
    
    def __init__(self):
        super(GKNN2, self).__init__()
        self.net1 = nn.Sequential(nn.Linear(1, 100), nn.Sigmoid(),
                                 nn.Linear(100, 100), nn.Sigmoid(),
                                 nn.Linear(100, 100), nn.Sigmoid(),
                                 nn.Linear(100, 1))
        
        self.net2 = nn.Sequential(nn.Linear(1, 100), nn.Sigmoid(),
                                  nn.Linear(100, 100), nn.Sigmoid(),
                                  nn.Linear(100, 1))
        
        self.net3 = nn.Sequential(nn.Linear(1, 100), nn.Sigmoid(),
                                  nn.Linear(100, 100), nn.Sigmoid(),
                                  nn.Linear(100, 1))
        
    def forward(self, x):
        # a = torch.ones(nPt, 1).to(device)
        return self.net3(x)


# In[28]:


class PINN():
    def __init__(self, X_f, X_lb, X_ub, epochs, Diff, X_star, u_star):
        
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
        
        self.x_f = torch.tensor(X_f, dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_lb = torch.tensor(X_lb, dtype=torch.float32, requires_grad=True).to(device)
        self.x_ub = torch.tensor(X_ub, dtype=torch.float32, requires_grad=True).to(device)
        
        self.gknn1 = GKNN1().to(device)
        
        self.it = []; self.l2 = []; self.loss_u = []; self.loss_r = []; self.losses = []; self.lx = []
        
        self.optimizer1 = torch.optim.Adam(self.gknn1.parameters(), lr=1e-3, betas=(0.9, 0.999))
        
    def net_u(self, x):
        u = self.gknn1(x)
        return u
    
    def net_r(self, x):
        u = self.net_u(x)
        
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        r =  - self.Diff * u_xx + u_x + (1 + self.Diff) * u
        return r
    
    
    def train(self):
        self.gknn1.train()
       
        a = timeit.default_timer()
        for epoch in range(self.epochs):
            
            e = torch.tensor(-(1 + self.Diff) / self.Diff)
            f = torch.tensor(-1)
            start_time = timeit.default_timer()
            u_lb_pred = self.net_u(self.x_lb)
            u_ub_pred = self.net_u(self.x_ub)
            loss_u = torch.mean((u_lb_pred - (torch.tensor(1) + torch.exp(e))) ** 2) + \
                     torch.mean((u_ub_pred - (torch.tensor(1) + torch.exp(f))) ** 2)
            
            r_pred = self.net_r(self.x_f)
            
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
                u_pred, u_x_pred = self.predict(self.X_star)
                l2_error = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
                if l2_error < self.min_l2_error:
                    self.min_l2_error = l2_error
                    self.best_u_pred = u_pred
                    self.best_u_x_pred = u_x_pred
                
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
        x = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
        
        self.gknn1.eval()
    
        u = self.net_u(x)
        u_x = grad(u, x)
        u = u.detach().cpu().numpy()
        u_x = u_x.detach().cpu().numpy()
        return u, u_x     
    
    def get_best_u_pred(self):
        return self.best_u_pred, self.best_u_x_pred



        

class RBAGKPINN():
    def __init__(self, X_f, X_lb, X_ub, epochs, Diff, X_star, u_star):
        
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
        
        self.x_f = torch.tensor(X_f, dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_lb = torch.tensor(X_lb, dtype=torch.float32, requires_grad=True).to(device)
        self.x_ub = torch.tensor(X_ub, dtype=torch.float32, requires_grad=True).to(device)
        
        self.gknn1 = GKNN1().to(device)
        self.gknn2 = GKNN2().to(device)
        
        self.it = []; self.l2 = []; self.loss_u = []; self.loss_r = []; self.losses = []; self.lx = []
        
        self.optimizer1 = torch.optim.Adam(self.gknn1.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.optimizer2 = torch.optim.Adam(self.gknn2.parameters(), lr=1e-3, betas=(0.9, 0.999))
        
    def net_u(self, x):
        u = self.gknn1(x) + self.gknn2(x) * torch.exp((x - 1) / self.Diff)
        return u
    
    def net_r(self, x):
        u = self.net_u(x)
        
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        r =  - self.Diff * u_xx + u_x + (1 + self.Diff) * u
        return r
    
    
    def train(self):
        self.gknn1.train()
        self.gknn2.train()
        
        a = timeit.default_timer()
        for epoch in range(self.epochs):
            e = torch.tensor(-(1 + self.Diff) / self.Diff)
            f = torch.tensor(-1)
            start_time = timeit.default_timer()
            u_lb_pred = self.net_u(self.x_lb)
            u_ub_pred = self.net_u(self.x_ub)
            loss_u = torch.mean((u_lb_pred - (torch.tensor(1) + torch.exp(e))) ** 2) + \
                     torch.mean((u_ub_pred - (torch.tensor(1) + torch.exp(f))) ** 2)
            
            r_pred = self.net_r(self.x_f)
            
            r_norm = self.eta * torch.abs(r_pred) / torch.max(torch.abs(r_pred))
            self.rsum = (self.rsum * self.gamma + r_norm).detach()
            loss_r = torch.mean((self.rsum * r_pred) ** 2)
            loss = loss_u + loss_r 
            
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()
            
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
                if epoch % 100000 == 0:
                    b = timeit.default_timer() - a
                    print('Time for 100000 epochs %.2f s' % (b))
                
    def predict(self, X):
        x = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
        
        self.gknn1.eval()
        self.gknn2.eval()
        
        u = self.net_u(x)
        u = u.detach().cpu().numpy()
        return u      
    
    def get_best_u_pred(self):
        return self.best_u_pred  


# In[29]:


#model = PINN(x_in, u_in, epochs, Diff)
#model = GKPINN(x_in, u_in, epochs, Diff)
model = PINN(X_f, X_lb, X_ub, epochs, Diff, X_star, u_star)
model.train()


# In[30]:


a = model.loss_u.index(min(model.loss_u))
b = model.loss_r.index(min(model.loss_r))
a * 100, min(model.loss_u), b * 100, min(model.loss_r)


# In[31]:


# fig = plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
plt.plot(model.loss_u, label='loss_u')
plt.plot(model.loss_r, '--', label='loss_r')
plt.yscale('log')
plt.legend(loc='lower right')
plt.show()


# In[32]:


plt.plot(model.l2)
plt.xscale('log')
plt.yscale('log')
plt.show()
c = model.l2.index(min(model.l2))
c, "{:.3e}".format(min(model.l2)), "{:.3e}".format(model.losses[c]), "{:.3e}".format(model.loss_u[c]), "{:.3e}".format(model.loss_r[c])


# In[33]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def u(x):
    return np.exp(-x) + np.exp((1 + Diff) * (x - 1) / Diff)

def u_x(x):
    return -np.exp(-x) + ((1 + Diff) / Diff) * np.exp((1 + Diff) * (x - 1) / Diff)

X_star = np.linspace(0, 1, nPt).reshape(-1, 1)
u_pred, u_x_pred = model.get_best_u_pred()
u_x_pred = abs(u_x_pred)
# u_pred = model.predict(X_star)
u_star = u(X_star)
u_x_star = u_x(X_star)
u_x_star = abs(u_x_star)

l2_error = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
print('l2_error: %e' % (l2_error))

plt.plot(X_star, u_star, '-', linewidth=1.5, label='Exact')
plt.plot(X_star, u_pred, '--', linewidth=1.5,label='Predicted')
# plt.xlabel('x', fontsize=12)
# plt.ylabel('u', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# plt.plot(X_star, u_star, '-', linewidth=1.5, label='Exact')
plt.plot(X_star, u_x_star, '-', linewidth=1.5,label='Exact')
plt.plot(X_star, u_x_pred, '--', linewidth=1.5,label='Predicted')
# plt.xlabel('x', fontsize=12)
# plt.ylabel('grad_u', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# In[34]:


error_u = u_star - u_pred

plt.tick_params(axis='both', which='major', labelsize=14)
plt.plot(X_star, error_u, label='Error')
plt.legend(loc='upper right', fontsize='large')
plt.show()

